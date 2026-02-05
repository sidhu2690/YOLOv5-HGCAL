import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from model import YOLOv5
from dataloader import create_dataloader
from loss import ComputeLoss


# ======================== Configuration ========================

class Config:
    # Dataset
    train_path = 'data/train/images'
    val_path = 'data/val/images'
    nc = 1                    # number of classes
    in_ch = 3                 # input channels
    
    # Model
    depth = 0.33              # depth multiple (0.33=small, 0.67=medium, 1.0=large)
    width = 0.50              # width multiple (0.50=small, 0.75=medium, 1.0=large)
    
    # Training
    epochs = 100
    batch_size = 16
    imgsz = 640
    optimizer = 'SGD'         # 'SGD', 'Adam', 'AdamW'
    device = '0'              # cuda device or 'cpu'
    workers = 8
    seed = 0
    
    # Save
    save_dir = 'runs/train'
    resume = False


# ======================== Hyperparameters ========================

HYP = {
    # Loss weights
    'box': 0.05,
    'obj': 1.0,
    'cls': 0.5,
    'cls_pw': 1.0,
    'obj_pw': 1.0,
    'anchor_t': 4.0,
    
    # Training
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    
    # Augmentation
    'fliplr': 0.5,
}


# ======================== Helper Functions ========================

def select_device(device=''):
    """Select computing device"""
    if device.lower() == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device(f'cuda:{device}')
    return torch.device('cpu')


def init_seeds(seed=0):
    """Initialize random seeds"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimizer(model, name, lr, momentum, weight_decay):
    """Create optimizer with parameter groups"""
    g_decay, g_no_decay, g_bias = [], [], []
    
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g_bias.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g_no_decay.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g_decay.append(v.weight)
    
    if name == 'Adam':
        optimizer = Adam(g_bias, lr=lr, betas=(momentum, 0.999))
    elif name == 'AdamW':
        optimizer = AdamW(g_bias, lr=lr, betas=(momentum, 0.999))
    else:
        optimizer = SGD(g_bias, lr=lr, momentum=momentum, nesterov=True)
    
    optimizer.add_param_group({'params': g_decay, 'weight_decay': weight_decay})
    optimizer.add_param_group({'params': g_no_decay})
    
    print(f'Optimizer: {name} | {len(g_decay)} weight(decay), {len(g_no_decay)} weight(no decay), {len(g_bias)} bias')
    return optimizer


def save_checkpoint(model, optimizer, epoch, best_fitness, path):
    """Save checkpoint"""
    torch.save({
        'epoch': epoch,
        'best_fitness': best_fitness,
        'model': deepcopy(model).half().state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path, device):
    """Load checkpoint"""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    if optimizer and ckpt.get('optimizer'):
        optimizer.load_state_dict(ckpt['optimizer'])
    print(f'Resumed from epoch {ckpt["epoch"]}')
    return ckpt.get('epoch', 0) + 1, ckpt.get('best_fitness', 0.0)


# ======================== Training Functions ========================

def train_one_epoch(model, dataloader, optimizer, compute_loss, device, epoch, epochs, scaler, warmup_iters):
    """Train for one epoch"""
    model.train()
    mloss = torch.zeros(3, device=device)
    nb = len(dataloader)
    
    pbar = tqdm(enumerate(dataloader), total=nb, desc=f'Epoch {epoch}/{epochs-1}')
    optimizer.zero_grad()
    
    for i, (imgs, targets, paths, _) in pbar:
        ni = i + nb * epoch
        imgs = imgs.to(device).float()
        targets = targets.to(device)
        
        # Warmup
        if ni < warmup_iters:
            xi = [0, warmup_iters]
            for j, pg in enumerate(optimizer.param_groups):
                pg['lr'] = np.interp(ni, xi, [HYP['warmup_bias_lr'] if j == 0 else 0.0, pg['initial_lr']])
                if 'momentum' in pg:
                    pg['momentum'] = np.interp(ni, xi, [HYP['warmup_momentum'], HYP['momentum']])
        
        # Forward
        with autocast(enabled=(device.type != 'cpu')):
            pred = model(imgs)
            loss, loss_items = compute_loss(pred, targets)
        
        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Log
        mloss = (mloss * i + loss_items) / (i + 1)
        mem = f'{torch.cuda.memory_reserved() / 1E9:.2f}G' if torch.cuda.is_available() else 'CPU'
        pbar.set_postfix({'mem': mem, 'box': f'{mloss[0]:.4f}', 'obj': f'{mloss[1]:.4f}', 'cls': f'{mloss[2]:.4f}'})
    
    return mloss


@torch.no_grad()
def validate(model, dataloader, compute_loss, device):
    """Validate model"""
    model.eval()
    val_loss = torch.zeros(3, device=device)
    
    for i, (imgs, targets, _, _) in enumerate(tqdm(dataloader, desc='Validating')):
        imgs = imgs.to(device).float()
        targets = targets.to(device)
        
        pred = model(imgs)
        _, loss_items = compute_loss(pred, targets)
        val_loss = (val_loss * i + loss_items) / (i + 1)
    
    print(f'Val Loss - box: {val_loss[0]:.4f}, obj: {val_loss[1]:.4f}, cls: {val_loss[2]:.4f}')
    return -val_loss.sum().item()  # fitness (higher is better)


# ======================== Main Training ========================

def train(cfg):
    """Main training function"""
    # Setup
    init_seeds(cfg.seed)
    device = select_device(cfg.device)
    print(f'Device: {device}')
    
    # Directories
    save_dir = Path(cfg.save_dir)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    last_pt, best_pt = weights_dir / 'last.pt', weights_dir / 'best.pt'
    
    # Model
    model = YOLOv5(nc=cfg.nc, in_ch=cfg.in_ch, depth_multiple=cfg.depth, width_multiple=cfg.width).to(device)
    model.hyp = HYP
    model.nc = cfg.nc
    
    # Scale anchors
    model.detect.stride = model.detect.stride.to(device)
    model.detect.anchors /= model.detect.stride.view(-1, 1, 1)
    
    print(f'Model: nc={cfg.nc}, depth={cfg.depth}, width={cfg.width}')
    
    # Optimizer
    optimizer = get_optimizer(model, cfg.optimizer, HYP['lr0'], HYP['momentum'], HYP['weight_decay'])
    for pg in optimizer.param_groups:
        pg['initial_lr'] = pg['lr']
    
    # Scheduler
    scheduler = LambdaLR(optimizer, lr_lambda=lambda x: (1 - x / cfg.epochs) * (1 - HYP['lrf']) + HYP['lrf'])
    
    # Resume
    start_epoch, best_fitness = 0, float('-inf')
    if cfg.resume and last_pt.exists():
        start_epoch, best_fitness = load_checkpoint(model, optimizer, last_pt, device)
        scheduler.last_epoch = start_epoch - 1
    
    # Dataloaders
    train_loader, train_dataset = create_dataloader(cfg.train_path, cfg.imgsz, cfg.batch_size, 
                                                     hyp=HYP, augment=True, workers=cfg.workers, shuffle=True)
    val_loader, _ = create_dataloader(cfg.val_path, cfg.imgsz, cfg.batch_size * 2,
                                       hyp=HYP, augment=False, workers=cfg.workers, shuffle=False)
    
    print(f'Train: {len(train_dataset)} images | Val: {len(val_loader.dataset)} images')
    
    # Loss & Scaler
    compute_loss = ComputeLoss(model)
    scaler = GradScaler(enabled=(device.type != 'cpu'))
    warmup_iters = max(round(HYP['warmup_epochs'] * len(train_loader)), 100)
    
    # ==================== Training Loop ====================
    print(f'\nTraining for {cfg.epochs} epochs...\n')
    t0 = time.time()
    
    for epoch in range(start_epoch, cfg.epochs):
        # Train
        train_one_epoch(model, train_loader, optimizer, compute_loss, device, epoch, cfg.epochs, scaler, warmup_iters)
        scheduler.step()
        
        # Validate
        fitness = validate(model, val_loader, compute_loss, device)
        
        # Save
        is_best = fitness > best_fitness
        best_fitness = max(fitness, best_fitness)
        save_checkpoint(model, optimizer, epoch, best_fitness, last_pt)
        if is_best:
            save_checkpoint(model, optimizer, epoch, best_fitness, best_pt)
            print('Saved new best model!')
        
        print(f'Epoch {epoch} done | LR: {optimizer.param_groups[0]["lr"]:.6f}\n')
    
    print(f'\nTraining complete ({(time.time() - t0) / 3600:.2f} hours)')
    print(f'Results saved to {save_dir}')


# ======================== Run ========================

if __name__ == '__main__':
    cfg = Config()
    
    # Modify config here as needed:
    # cfg.train_path = 'path/to/train/images'
    # cfg.val_path = 'path/to/val/images'
    # cfg.nc = 80
    # cfg.epochs = 300
    # cfg.batch_size = 32
    
    train(cfg)
