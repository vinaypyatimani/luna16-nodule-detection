"""
3_evaluate.py
=============
Comprehensive evaluation of the trained 3D nodule detector.

Metrics produced:
  1. ROC curve with AUC — standard binary classifier performance
  2. FROC curve — Free-Response ROC, the standard metric for nodule detection
     Plots sensitivity vs. average false positives per scan (FP/scan).
     Clinical target: sensitivity > 0.85 at <= 4 FP/scan.
  3. Confusion matrix with per-class accuracy
  4. Patch visualisation — sample nodule and non-nodule patches
     shown as central axial/coronal/sagittal slices

Why FROC?
  Standard ROC treats all false positives equally. In CT screening,
  a radiologist reviews the full scan — so what matters is how many
  false alarms they see per patient (per scan), not per candidate.
  FROC normalises FP by the number of scans, giving a clinically
  meaningful performance measure.

Outputs:
  results/roc_curve.png          — ROC with AUC
  results/froc_curve.png         — FROC (sensitivity vs FP/scan)
  results/confusion_matrix.png   — confusion matrix heatmap
  results/patch_examples.png     — example patches with predictions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    confusion_matrix, classification_report
)
import os

# ── Re-import model (same architecture as training) ───────────────────────────

class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1    = nn.Conv3d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False)
        self.bn1      = nn.BatchNorm3d(out_ch)
        self.conv2    = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2      = nn.BatchNorm3d(out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm3d(out_ch)
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x)))))
                      + self.shortcut(x))

class NoduleDetector3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True)
        )
        self.stage1 = ResBlock3D(32,  32,  stride=1)
        self.stage2 = ResBlock3D(32,  64,  stride=2)
        self.stage3 = ResBlock3D(64,  128, stride=2)
        self.stage4 = ResBlock3D(128, 256, stride=2)
        self.gap    = nn.AdaptiveAvgPool3d(1)
        self.head   = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x)
        x = self.stage3(x); x = self.stage4(x)
        return self.head(self.gap(x))


# ── FROC computation ──────────────────────────────────────────────────────────

def compute_froc(labels, probs, n_scans, thresholds=None):
    """
    Compute Free-Response ROC curve.

    Args:
        labels     : array of ground truth (0/1)
        probs      : predicted probabilities
        n_scans    : total number of CT scans (for FP/scan normalisation)
        thresholds : probability thresholds to evaluate

    Returns:
        fp_per_scan : array of average FP per scan at each threshold
        sensitivity : array of true positive rate at each threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.99, 100)

    fp_per_scan_list = []
    sensitivity_list = []

    n_pos = (labels == 1).sum()

    for thresh in thresholds:
        pred_bin = (probs >= thresh).astype(int)
        tp = ((pred_bin == 1) & (labels == 1)).sum()
        fp = ((pred_bin == 1) & (labels == 0)).sum()

        sens   = tp / max(n_pos, 1)
        fp_per = fp / max(n_scans, 1)

        sensitivity_list.append(sens)
        fp_per_scan_list.append(fp_per)

    return np.array(fp_per_scan_list), np.array(sensitivity_list)


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate():
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = NoduleDetector3D().to(device)
    model.load_state_dict(
        torch.load("models/nodule_detector_3d.pth", map_location=device)
    )
    model.eval()
    print(f"Model loaded. Device: {device}")

    # Load test data
    patches  = np.load("data/patches.npy")
    labels   = np.load("data/labels.npy")
    idx_test = np.load("data/idx_test.npy")
    meta     = pd.read_csv("data/meta.csv")

    X_test = patches[idx_test]
    y_test = labels[idx_test]
    meta_test = meta.iloc[idx_test].reset_index(drop=True)

    print(f"Test set: {len(X_test)} patches")
    print(f"  Nodules:     {(y_test==1).sum()}")
    print(f"  Non-nodules: {(y_test==0).sum()}")

    # Run inference in batches
    BATCH = 64
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_test), BATCH):
            X_batch = torch.tensor(
                X_test[i:i+BATCH], dtype=torch.float32
            ).to(device)
            probs = model(X_batch).cpu().numpy().flatten()
            all_probs.extend(probs)
    probs = np.array(all_probs)
    preds = (probs >= 0.5).astype(int)

    # ── Classification metrics ──
    auc = roc_auc_score(y_test, probs)
    print("\n" + "=" * 52)
    print("CLASSIFICATION RESULTS")
    print("=" * 52)
    print(classification_report(y_test, preds,
                                  target_names=["Non-nodule", "Nodule"]))
    print(f"ROC-AUC: {auc:.4f}")

    # ── 1. ROC Curve ──
    fpr, tpr, _ = roc_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"3D CNN (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("Sensitivity (True Positive Rate)")
    ax.set_title("ROC Curve — Nodule Detection")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/roc_curve.png", dpi=150)
    print("\nSaved results/roc_curve.png")

    # ── 2. FROC Curve ──
    n_scans = meta_test["seriesuid"].nunique()
    fp_per_scan, sensitivity = compute_froc(y_test, probs, n_scans)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fp_per_scan, sensitivity, color="darkorange", lw=2,
            label="3D CNN")
    # Clinical reference lines
    ax.axvline(x=4.0, color="red", linestyle="--", alpha=0.5,
               label="4 FP/scan target")
    ax.axhline(y=0.85, color="green", linestyle="--", alpha=0.5,
               label="85% sensitivity target")
    ax.set_xlabel("Average False Positives per Scan")
    ax.set_ylabel("Sensitivity")
    ax.set_title("FROC Curve — Lung Nodule Detection\n"
                 "(LUNA16 evaluation standard)")
    ax.set_xlim(0, 20); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/froc_curve.png", dpi=150)
    print("Saved results/froc_curve.png")

    # ── 3. Confusion Matrix ──
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-nodule", "Nodule"])
    ax.set_yticklabels(["Non-nodule", "Nodule"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    print("Saved results/confusion_matrix.png")

    # ── 4. Patch examples ──
    # Show 4 true nodules + 4 true non-nodules with prediction score
    nodule_idx  = np.where((y_test == 1))[0][:4]
    nonnod_idx  = np.where((y_test == 0))[0][:4]

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Example Patches — 3 Orthogonal Views\n"
                 "(Axial / Coronal / Sagittal central slices)",
                 fontsize=12)
    gs = gridspec.GridSpec(2, 4 * 3 + 1, figure=fig,
                           wspace=0.05, hspace=0.3)

    c = 0
    for row, (indices, row_label) in enumerate([
        (nodule_idx,  "NODULE"),
        (nonnod_idx,  "NON-NODULE")
    ]):
        for col, i in enumerate(indices):
            patch = X_test[i, 0]          # (32, 32, 32)
            prob  = probs[i]
            cz, cy, cx = [s // 2 for s in patch.shape]
            views = [
                patch[cz, :, :],    # axial
                patch[:, cy, :],    # coronal
                patch[:, :, cx],    # sagittal
            ]
            view_names = ["Axial", "Coronal", "Sagittal"]
            for v, (view, vname) in enumerate(zip(views, view_names)):
                ax = fig.add_subplot(gs[row, col * 3 + v])
                ax.imshow(view, cmap="gray", vmin=0, vmax=1)
                if v == 0:
                    clr = "green" if (
                        (row_label == "NODULE"     and prob >= 0.5) or
                        (row_label == "NON-NODULE" and prob <  0.5)
                    ) else "red"
                    ax.set_ylabel(
                        f"{row_label}\np={prob:.2f}",
                        fontsize=7, color=clr, rotation=0,
                        labelpad=40
                    )
                ax.set_title(vname, fontsize=6)
                ax.axis("off")

    plt.savefig("results/patch_examples.png", dpi=150, bbox_inches="tight")
    print("Saved results/patch_examples.png")

    print("\nAll done! See results/ folder.")


if __name__ == "__main__":
    evaluate()
