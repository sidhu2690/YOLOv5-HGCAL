"""
Energy prediction evaluation: true vs predicted scatter + histogram
Usage: python plot_energy.py --weights runs/train/run06/weights/best.pt --data data/muon47m.yaml --img 736
"""

import argparse
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from models.common import DetectMultiBackend
from utils.datasets import create_dataloader
from utils.general import check_dataset, check_img_size, non_max_suppression, scale_coords, xywh2xyxy
from utils.torch_utils import select_device
from utils.metrics import box_iou


@torch.no_grad()
def run(weights, data, imgsz=736, batch_size=16, device='', conf_thres=0.25, iou_thres=0.6):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, data=data, fp16=False)
    stride = model.stride
    imgsz = check_img_size(imgsz, s=stride)
    data = check_dataset(data)

    dataloader = create_dataloader(
        data['val'], imgsz, batch_size, stride,
        single_cls=True, pad=0.5, rect=True, workers=8,
        prefix='val: ')[0]

    model.eval()
    model.warmup(imgsz=(1, 16, imgsz, imgsz))

    true_energies = []   # log MeV from labels
    pred_energies = []   # log MeV from model
    matched_confs = []

    for im, targets, paths, shapes in tqdm(dataloader, desc='Evaluating energy'):
        im = im.to(device).float() / 255
        targets = targets.to(device)
        nb, _, height, width = im.shape

        # Inference
        out, _ = model(im, val=True)

        # Scale targets to pixels
        targets[:, 2:6] *= torch.tensor((width, height, width, height), device=device)

        # NMS
        out = non_max_suppression(out, conf_thres, iou_thres, multi_label=True, agnostic=True)

        # Match predictions to ground truth
        for si, pred in enumerate(out):
            # Ground truth for this image
            tidx = targets[:, 0] == si
            gt_labels = targets[tidx]  # [batch, cls, x, y, w, h, energy]
            if len(gt_labels) == 0 or len(pred) == 0:
                continue

            gt_energy = gt_labels[0, 6].item()  # log MeV (same for all boxes in image)
            gt_boxes = xywh2xyxy(gt_labels[:, 2:6])

            # Scale pred boxes to image space
            pred_boxes = pred[:, :4]

            # Find best matching prediction (highest IoU with any GT box)
            iou = box_iou(gt_boxes, pred_boxes)
            if iou.numel() == 0:
                continue

            best_pred_idx = iou.max(0)[0].argmax()
            best_iou = iou[:, best_pred_idx].max()

            if best_iou >= 0.5:
                true_energies.append(gt_energy)
                pred_energies.append(pred[best_pred_idx, 6].item())  # energy column
                matched_confs.append(pred[best_pred_idx, 4].item())

    true_e = np.array(true_energies)
    pred_e = np.array(pred_energies)
    confs = np.array(matched_confs)

    print(f'\nMatched {len(true_e)} detections')
    print(f'True energy range:  {true_e.min():.2f} - {true_e.max():.2f} log(MeV)')
    print(f'Pred energy range:  {pred_e.min():.2f} - {pred_e.max():.2f} log(MeV)')

    # Convert to MeV
    true_mev = np.exp(true_e)
    pred_mev = np.exp(pred_e)

    save_dir = Path(weights).parent.parent
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Plot 1: True vs Pred (log scale) ---
    ax = axes[0, 0]
    ax.scatter(true_e, pred_e, alpha=0.3, s=10, c='blue')
    lims = [min(true_e.min(), pred_e.min()) - 0.5, max(true_e.max(), pred_e.max()) + 0.5]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('True log(Energy/MeV)')
    ax.set_ylabel('Predicted log(Energy/MeV)')
    ax.set_title('True vs Predicted (log space)')
    ax.legend()
    ax.set_aspect('equal')

    # --- Plot 2: True vs Pred (MeV) ---
    ax = axes[0, 1]
    ax.scatter(true_mev, pred_mev, alpha=0.3, s=10, c='green')
    max_mev = max(true_mev.max(), pred_mev.max()) * 1.1
    ax.plot([0, max_mev], [0, max_mev], 'r--', linewidth=2, label='Perfect')
    ax.set_xlabel('True Energy (MeV)')
    ax.set_ylabel('Predicted Energy (MeV)')
    ax.set_title('True vs Predicted (MeV)')
    ax.legend()

    # --- Plot 3: Ratio histogram ---
    ax = axes[0, 2]
    ratio = pred_mev / (true_mev + 1e-6)
    ax.hist(ratio, bins=100, range=(0, 3), color='orange', edgecolor='black')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Perfect (1.0)')
    ax.set_xlabel('Predicted / True Energy')
    ax.set_ylabel('Count')
    ax.set_title(f'Energy Ratio (median={np.median(ratio):.3f})')
    ax.legend()

    # --- Plot 4: Residual in log space ---
    ax = axes[1, 0]
    residual = pred_e - true_e
    ax.hist(residual, bins=100, color='purple', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('log(Pred) - log(True)')
    ax.set_ylabel('Count')
    ax.set_title(f'Log Residual (mean={residual.mean():.3f}, std={residual.std():.3f})')

    # --- Plot 5: Resolution vs True Energy ---
    ax = axes[1, 1]
    n_bins = 10
    bins = np.linspace(true_e.min(), true_e.max(), n_bins + 1)
    bin_centers, bin_stds, bin_means = [], [], []
    for j in range(n_bins):
        mask = (true_e >= bins[j]) & (true_e < bins[j + 1])
        if mask.sum() > 5:
            bin_centers.append((bins[j] + bins[j + 1]) / 2)
            bin_stds.append(residual[mask].std())
            bin_means.append(residual[mask].mean())
    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, fmt='o-', capsize=5, color='teal')
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('True log(Energy/MeV)')
    ax.set_ylabel('Mean ± Std of log residual')
    ax.set_title('Energy Resolution vs True Energy')

    # --- Plot 6: Pred vs confidence ---
    ax = axes[1, 2]
    sc = ax.scatter(confs, pred_e, c=true_e, alpha=0.4, s=10, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='True log(E)')
    ax.set_xlabel('Detection Confidence')
    ax.set_ylabel('Predicted log(Energy/MeV)')
    ax.set_title('Energy vs Confidence')

    plt.tight_layout()
    plt.savefig(save_dir / 'energy_evaluation.png', dpi=200)
    plt.close()
    print(f'Saved to {save_dir / "energy_evaluation.png"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--img', type=int, default=736)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='0')
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.6)
    opt = parser.parse_args()
    run(opt.weights, opt.data, opt.img, opt.batch_size, opt.device, opt.conf_thres, opt.iou_thres)
