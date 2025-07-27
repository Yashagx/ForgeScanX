import os
import cv2
import numpy as np
from tqdm import tqdm

def calculate_metrics(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0

    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, ~gt).sum()
    FN = np.logical_and(~pred, gt).sum()

    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    iou       = TP / (TP + FP + FN + 1e-6)

    return precision, recall, f1, iou

def evaluate(detected_folder, mask_folder):
    files = [f for f in os.listdir(detected_folder) if f.endswith('.png') or f.endswith('.jpg')]

    total_p, total_r, total_f1, total_iou = 0, 0, 0, 0
    count = 0

    print("🔍 Evaluating detected results against ground truth masks...\n")

    for fname in tqdm(files):
        base = os.path.splitext(fname)[0]
        
        # Match pattern: 123_F_CA1 → 123_B_CA1
        if '_F_' in base:
            mask_name = base.replace('_F_', '_B_')
        elif '_F' in base:
            mask_name = base.replace('_F', '_B')
        else:
            continue

        pred_path = os.path.join(detected_folder, fname)
        gt_path   = os.path.join(mask_folder, mask_name + '.png')

        if not os.path.exists(gt_path):
            continue

        pred = cv2.imread(pred_path, 0)
        gt   = cv2.imread(gt_path, 0)

        if pred is None or gt is None or pred.shape != gt.shape:
            continue

        p, r, f1, iou = calculate_metrics(pred, gt)
        total_p  += p
        total_r  += r
        total_f1 += f1
        total_iou += iou
        count += 1

    if count == 0:
        print("❌ No valid evaluations possible. Check your mask folder or naming conventions.")
        return

    print("\n📊 Evaluation Metrics:")
    print(f"✅ Valid Images Evaluated: {count}")
    print(f"🎯 Precision: {total_p / count:.4f}")
    print(f"📥 Recall:    {total_r / count:.4f}")
    print(f"⭐ F1 Score:  {total_f1 / count:.4f}")
    print(f"📐 IoU:       {total_iou / count:.4f}")

if __name__ == "__main__":
    evaluate(
        detected_folder='data/results/test_output/Binary',
        mask_folder='data/results/test_output/visual'  # Update if needed
    )
