import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from model import YOLOv5
from dataloader import create_dataloader
from loss import ComputeLoss


# ======================== Config ========================

class Config:
    train_path = 'data/train/images'
    val_path   = 'data/val/images'
    nc         = 1
    in_ch      = 3
    depth      = 0.33
    width      = 0.50
    epochs     = 100
    batch_size = 16
    imgsz      = 640
    workers    = 8
    save_dir   = 'runs/train'
    device     = '0'           # '0' or 'cpu'


HYP = {
    # loss weights
    'box': 0.05,  'obj': 1.0,  'cls': 0.5,
    'cls_pw': 1.0, 'obj_pw': 1.0, 'anchor_t': 4.0,
    # optimiser / schedule
    'lr0': 0.01, 'lrf': 0.01,
    'momentum': 0.937, 'weight_decay': 0.0005,
    'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
}


# ======================== Helpers ========================

def select_device(tag='0'):
    if tag == 'cpu' or not torch.cuda.is_available():
        return torch.device('cpu')
    return torch.device(f'cuda:{tag}')


def build_optimizer(model, lr, momentum, weight_decay):
    """SGD with three param groups: biases | BN weights | other weights."""
    g_bias, g_bn, g_weight = [], [], []
    for m in model.modules():
        if hasattr(m, 'bias') and isinstance(m.bias, nn.Parameter):
            g_bias.append(m.bias)
        if isinstance(m, nn.BatchNorm2d):
            g_bn.append(m.weight)
        elif hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter):
            g_weight.append(m.weight)

    opt = SGD(g_bias, lr=lr, momentum=momentum, nesterov=True)
    opt.add_param_group({'params': g_weight, 'weight_decay': weight_decay})
    opt.add_param_group({'params': g_bn})
    print(f'Optimizer: {len(g_weight)} weight(decay), '
          f'{len(g_bn)} BN(no decay), {len(g_bias)} bias')
    return opt


# ======================== Train / Val ========================

def train_one_epoch(model, loader, optimizer, loss_fn,
                    device, epoch, epochs, warmup_n):
    model.train()
    nb = len(loader)
    mloss = torch.zeros(3, device=device)

    pbar = tqdm(enumerate(loader), total=nb,
                desc=f'Epoch {epoch}/{epochs - 1}')
    for i, (imgs, targets, _, _) in pbar:
        ni = i + nb * epoch
        imgs = imgs.to(device)
        targets = targets.to(device)

        # warmup LR / momentum ramp
        if ni < warmup_n:
            xi = [0, warmup_n]
            for j, pg in enumerate(optimizer.param_groups):
                start_lr = HYP['warmup_bias_lr'] if j == 0 else 0.0
                pg['lr'] = np.interp(ni, xi, [start_lr, pg['initial_lr']])
                if 'momentum' in pg:
                    pg['momentum'] = np.interp(
                        ni, xi, [HYP['warmup_momentum'], HYP['momentum']])

        # forward / backward
        loss, loss_items = loss_fn(model(imgs), targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        mloss = (mloss * i + loss_items) / (i + 1)
        pbar.set_postfix(box=f'{mloss[0]:.4f}',
                         obj=f'{mloss[1]:.4f}',
                         cls=f'{mloss[2]:.4f}')
    return mloss


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    vloss = torch.zeros(3, device=device)
    for i, (imgs, targets, _, _) in enumerate(
            tqdm(loader, desc='Validating')):
        imgs = imgs.to(device)
        targets = targets.to(device)
        _, train_out = model(imgs)          # eval mode → (decoded, raw)
        _, li = loss_fn(train_out, targets)
        vloss = (vloss * i + li) / (i + 1)

    fitness = -vloss.sum().item()           # higher = better
    print(f'  val  box={vloss[0]:.4f}  obj={vloss[1]:.4f}  '
          f'cls={vloss[2]:.4f}')
    return fitness


# ======================== Main ========================

def train(cfg):
    torch.manual_seed(0)
    device = select_device(cfg.device)
    print(f'Device: {device}')

    wdir = Path(cfg.save_dir) / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # --- model ---
    model = YOLOv5(nc=cfg.nc, in_ch=cfg.in_ch,
                   depth_multiple=cfg.depth,
                   width_multiple=cfg.width).to(device)
    model.hyp = HYP
    model.nc = cfg.nc
    model.detect.stride = model.detect.stride.to(device)
    model.detect.anchors /= model.detect.stride.view(-1, 1, 1)

    # --- optimiser & scheduler ---
    optimizer = build_optimizer(model, HYP['lr0'],
                                HYP['momentum'], HYP['weight_decay'])
    for pg in optimizer.param_groups:
        pg['initial_lr'] = pg['lr']

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda x: (1 - x / cfg.epochs) * (1 - HYP['lrf'])
                             + HYP['lrf'])

    # --- data ---
    train_loader, train_ds = create_dataloader(
        cfg.train_path, cfg.imgsz, cfg.batch_size,
        workers=cfg.workers, shuffle=True)
    val_loader, _ = create_dataloader(
        cfg.val_path, cfg.imgsz, cfg.batch_size * 2,
        workers=cfg.workers)

    print(f'Train: {len(train_ds)} imgs | Val: {len(val_loader.dataset)} imgs')

    # --- loss ---
    loss_fn = ComputeLoss(model)
    warmup_n = max(round(HYP['warmup_epochs'] * len(train_loader)), 100)
    best_fitness = float('-inf')

    # --- loop ---
    t0 = time.time()
    for epoch in range(cfg.epochs):
        train_one_epoch(model, train_loader, optimizer, loss_fn,
                        device, epoch, cfg.epochs, warmup_n)
        scheduler.step()

        fitness = validate(model, val_loader, loss_fn, device)

        # checkpoint
        state = {'epoch': epoch,
                 'model': deepcopy(model).float().state_dict()}
        torch.save(state, wdir / 'last.pt')
        if fitness > best_fitness:
            best_fitness = fitness
            torch.save(state, wdir / 'best.pt')
            print('  ✓ new best saved')

        print(f'  lr={optimizer.param_groups[0]["lr"]:.6f}\n')

    hours = (time.time() - t0) / 3600
    print(f'Done ({hours:.2f} h).  Weights → {wdir}')


if __name__ == '__main__':
    train(Config())
