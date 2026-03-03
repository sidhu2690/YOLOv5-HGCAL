# save as test_eval.py
import os
import numpy as np

WEIGHTS = "/persistent/data1/ms21080/Data/YOLO_HGCAL/MODEL_050/runs/train/run_02/weights/best.pt"
TEST_IMG = "data/images/test/"
TEST_LABELS = "data/labels/test/"
DATA_YAML = "data/muon47m.yaml"
IMG_SIZE = 736

# Run detection
os.system(f"python detect.py --weights {WEIGHTS} --source {TEST_IMG} --data {DATA_YAML} --img {IMG_SIZE} --device 0 --save-txt --save-conf --nosave --conf-thres 0.25 --project runs/detect --name test_eval --exist-ok")

PRED_DIR = "runs/detect/test_eval/labels/"

# Count
gt_em = 0
gt_muon = 0
pred_em = 0
pred_muon = 0

for f in os.listdir(TEST_LABELS):
    if not f.endswith('.txt'):
        continue
    with open(os.path.join(TEST_LABELS, f)) as fh:
        for line in fh:
            cls = int(line.strip().split()[0])
            if cls == 0: gt_em += 1
            else: gt_muon += 1

    pf = os.path.join(PRED_DIR, f)
    if os.path.exists(pf):
        with open(pf) as fh:
            for line in fh:
                cls = int(line.strip().split()[0])
                if cls == 0: pred_em += 1
                else: pred_muon += 1

print(f"GT  EM: {gt_em}  Muon: {gt_muon}  Total: {gt_em+gt_muon}")
print(f"Pred EM: {pred_em}  Muon: {pred_muon}  Total: {pred_em+pred_muon}")
print(f"Extra EM: {pred_em - gt_em}")
print(f"Extra Muon: {pred_muon - gt_muon}")
print(f"Total ghost: {(pred_em+pred_muon)-(gt_em+gt_muon)}")
