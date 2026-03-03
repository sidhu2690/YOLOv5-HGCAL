# save as plot_cm.py
import os
import numpy as np
import matplotlib.pyplot as plt

TEST_LABELS = "data/labels/test/"
PRED_DIR = "runs/detect/test_eval/labels/"
IMG_SIZE = 736

# matrix[pred][true]: 0=em, 1=muon, 2=background
matrix = np.zeros((3, 3), dtype=int)

for f in os.listdir(TEST_LABELS):
    if not f.endswith('.txt'):
        continue

    gt = []
    with open(os.path.join(TEST_LABELS, f)) as fh:
        for line in fh:
            p = line.strip().split()
            gt.append((int(p[0]), float(p[1])*IMG_SIZE, float(p[2])*IMG_SIZE))

    preds = []
    pf = os.path.join(PRED_DIR, f)
    if os.path.exists(pf):
        with open(pf) as fh:
            for line in fh:
                p = line.strip().split()
                preds.append((int(p[0]), float(p[1])*IMG_SIZE, float(p[2])*IMG_SIZE))

    used = set()
    for pcls, px, py in preds:
        best_d, best_i = 999, -1
        for i, (gcls, gx, gy) in enumerate(gt):
            if i in used:
                continue
            d = np.sqrt((px-gx)**2 + (py-gy)**2)
            if d < best_d:
                best_d, best_i = d, i
        if best_d < 15 and best_i >= 0:
            matrix[pcls][gt[best_i][0]] += 1
            used.add(best_i)
        else:
            matrix[pcls][2] += 1  # background FP

    for i, (gcls, gx, gy) in enumerate(gt):
        if i not in used:
            matrix[2][gcls] += 1  # missed

# Print
print("Rows=Predicted, Cols=True")
print(f"{'':>15} {'True EM':>10} {'True Muon':>10} {'BG FP':>10}")
print(f"{'Pred EM':>15} {matrix[0][0]:>10} {matrix[0][1]:>10} {matrix[0][2]:>10}")
print(f"{'Pred Muon':>15} {matrix[1][0]:>10} {matrix[1][1]:>10} {matrix[1][2]:>10}")
print(f"{'Missed':>15} {matrix[2][0]:>10} {matrix[2][1]:>10} {matrix[2][2]:>10}")
print(f"\nBG predicted as EM:   {matrix[0][2]}")
print(f"BG predicted as Muon: {matrix[1][2]}")
print(f"EM missed:            {matrix[2][0]}")
print(f"Muon missed:          {matrix[2][1]}")
print(f"EM as Muon:           {matrix[1][0]}")
print(f"Muon as EM:           {matrix[0][1]}")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
labels_x = ['EM', 'Muon', 'Background FP']
labels_y = ['EM', 'Muon', 'Background FN']

# Normalize by column
norm = matrix.astype(float)
for j in range(3):
    s = norm[:, j].sum()
    if s > 0:
        norm[:, j] /= s

ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        color = 'white' if norm[i][j] > 0.5 else 'black'
        ax.text(j, i, f'{norm[i][j]:.2f}\n({matrix[i][j]})', ha='center', va='center', fontsize=12, color=color)

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(labels_x, fontsize=12)
ax.set_yticklabels(labels_y, fontsize=12)
ax.set_xlabel('True', fontsize=14)
ax.set_ylabel('Predicted', fontsize=14)
ax.set_title('Test Confusion Matrix (nPU=50)', fontsize=14)
plt.tight_layout()
plt.savefig('test_cm.png', dpi=150)
print("\nSaved: test_cm.png")
