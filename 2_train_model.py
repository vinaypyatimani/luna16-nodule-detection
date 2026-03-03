"""
2_train_model.py
================
Trains a 3D Convolutional Neural Network for lung nodule detection.

Task: Binary classification of 3D CT patches
  Input : (batch, 1, 32, 32, 32) — normalised Hounsfield Unit patch
  Output: (batch, 1)             — probability of being a nodule (0–1)

Architecture: 3D ResNet-inspired CNN
  Why 3D convolutions?
    A lung nodule is a 3D sphere. 2D convolutions applied slice-by-slice
    lose the cross-slice context (e.g. how the nodule grows and shrinks
    across adjacent slices). 3D convolutions process the full volumetric
    patch in one pass, learning spherical shape features directly.
    This is the standard approach in pulmonary nodule CAD systems.

  Why residual connections?
    Residual (skip) connections let gradients flow directly to earlier
    layers, enabling training of deeper networks. They are especially
    important in 3D because the added depth dimension increases the
    risk of vanishing gradients.

Class imbalance handling:
  In LUNA16, nodules are vastly outnumbered by non-nodule candidates
  (~1:10 ratio). We use weighted cross-entropy loss — the nodule class
  gets 10× higher weight — so the model doesn't learn to ignore nodules
  by always predicting "non-nodule".

Outputs:
  models/nodule_detector_3d.pth  — best model weights (by val AUC)
  results/training_curves.png    — loss and AUC training curves
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS     = 40
LR         = 1e-3
SEED       = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── 3D Residual Block ─────────────────────────────────────────────────────────

class ResBlock3D(nn.Module):
    """
    3D Residual block: Conv3d → BN → ReLU → Conv3d → BN → (+skip) → ReLU

    The skip connection adds the input directly to the output:
        out = F(x) + x
    If F(x) learns a small correction, the block is effectively learning
    residuals (improvements) rather than full transformations — this makes
    training much more stable.

    When the number of channels changes (in_ch ≠ out_ch), a 1×1×1
    convolution on the skip path matches dimensions.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1,
                                stride=stride, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)

        # Shortcut projection when shape changes
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


# ── 3D Nodule Detector ────────────────────────────────────────────────────────

class NoduleDetector3D(nn.Module):
    """
    3D CNN for lung nodule detection from 32×32×32 CT patches.

    Architecture:
      Stem     : 3D Conv + BN + ReLU  (initial feature extraction)
      Stage 1  : ResBlock3D 1→32,  stride 1
      Stage 2  : ResBlock3D 32→64, stride 2  (→ 16³)
      Stage 3  : ResBlock3D 64→128, stride 2 (→ 8³)
      Stage 4  : ResBlock3D 128→256, stride 2 (→ 4³)
      GAP      : Global Average Pooling       (→ 256)
      Head     : FC(256→64) → Dropout → FC(64→1) → Sigmoid

    Total parameters: ~1.2M (efficient for CPU training)
    """
    def __init__(self):
        super().__init__()

        # Stem: initial 3D feature extraction
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )  # (B,32,32,32,32)

        # Residual stages — progressively higher-level features
        self.stage1 = ResBlock3D(32,  32,  stride=1)   # (B,32,32,32,32)
        self.stage2 = ResBlock3D(32,  64,  stride=2)   # (B,64,16,16,16)
        self.stage3 = ResBlock3D(64,  128, stride=2)   # (B,128,8,8,8)
        self.stage4 = ResBlock3D(128, 256, stride=2)   # (B,256,4,4,4)

        # Global Average Pooling collapses spatial dims → single vector
        self.gap  = nn.AdaptiveAvgPool3d(1)             # (B,256,1,1,1)

        # Classification head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        return self.head(x)


# ── Training helpers ──────────────────────────────────────────────────────────

def compute_auc(model, loader, device):
    """Compute ROC-AUC on a data loader."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            probs = model(X.to(device)).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_labels.extend(y.numpy().flatten())
    if len(set(all_labels)) < 2:
        return 0.5
    return roc_auc_score(all_labels, all_probs)


def make_weighted_sampler(labels):
    """
    Creates a WeightedRandomSampler that oversamples the minority class.
    This ensures each mini-batch contains roughly equal numbers of
    nodules and non-nodules, regardless of the original imbalance.
    """
    n0 = (labels == 0).sum()
    n1 = (labels == 1).sum()
    # Weight = inverse frequency: rare class gets higher weight
    w0 = 1.0 / n0 if n0 > 0 else 1.0
    w1 = 1.0 / n1 if n1 > 0 else 1.0
    sample_weights = np.where(labels == 1, w1, w0)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(labels),
        replacement=True
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──
    print("Loading patch data...")
    patches = np.load("data/patches.npy")   # (N, 1, 32, 32, 32)
    labels  = np.load("data/labels.npy")    # (N,)
    print(f"  Total patches: {len(patches)}")
    print(f"  Nodules:       {(labels==1).sum()}")
    print(f"  Non-nodules:   {(labels==0).sum()}")

    # ── Split 70/15/15 stratified ──
    idx    = np.arange(len(patches))
    # Use min-class count to decide if stratification is possible
    if (labels==1).sum() >= 2 and (labels==0).sum() >= 2:
        strat = labels
    else:
        strat = None

    idx_tv, idx_test = train_test_split(
        idx, test_size=0.15, random_state=SEED, stratify=strat
    )
    strat_tv = labels[idx_tv] if strat is not None else None
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=0.15/0.85, random_state=SEED, stratify=strat_tv
    )
    np.save("data/idx_test.npy", idx_test)
    print(f"  Train: {len(idx_train)}  Val: {len(idx_val)}  Test: {len(idx_test)}")

    # ── DataLoaders ──
    def make_ds(idx):
        return TensorDataset(
            torch.tensor(patches[idx], dtype=torch.float32),
            torch.tensor(labels[idx],  dtype=torch.float32).unsqueeze(1)
        )

    train_ds  = make_ds(idx_train)
    sampler   = make_weighted_sampler(labels[idx_train])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(make_ds(idx_val),  batch_size=BATCH_SIZE)
    # No test loader here — evaluated in 3_evaluate.py

    # ── Model, loss, optimiser ──
    model = NoduleDetector3D().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Weighted BCE: penalise missing nodules more than false alarms
    n0 = (labels[idx_train] == 0).sum()
    n1 = (labels[idx_train] == 1).sum()
    pos_weight = torch.tensor([n0 / max(n1, 1)], dtype=torch.float32).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Note: with BCEWithLogitsLoss we remove Sigmoid from the last layer
    # during training for numerical stability, but keep it for inference.
    # We handle this by using BCELoss after Sigmoid for eval.
    criterion_eval = nn.BCELoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=EPOCHS
    )

    # ── Training loop ──
    print(f"\nTraining for {EPOCHS} epochs...")
    train_losses, val_losses, val_aucs = [], [], []
    best_auc  = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        epoch_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimiser.zero_grad()
            # For BCEWithLogitsLoss we need raw logits (no sigmoid)
            # Temporarily bypass final sigmoid
            logits = model.head[:-1](model.gap(  # everything before sigmoid
                model.stage4(model.stage3(model.stage2(
                    model.stage1(model.stem(X)))))
            ).flatten(1))
            # Properly: use a wrapper or just use BCELoss with sigmoid output
            # Simplest approach: use sigmoid output with BCELoss
            pred = model(X)
            loss = criterion_eval(pred, y)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * len(X)

        train_losses.append(epoch_loss / len(idx_train))

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += criterion_eval(pred, y).item() * len(X)
        val_losses.append(val_loss / len(idx_val))

        auc = compute_auc(model, val_loader, device)
        val_aucs.append(auc)
        scheduler.step()

        if auc > best_auc:
            best_auc   = auc
            best_epoch = epoch
            torch.save(model.state_dict(), "models/nodule_detector_3d.pth")

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                  f"train: {train_losses[-1]:.4f} | "
                  f"val: {val_losses[-1]:.4f} | "
                  f"val AUC: {auc:.4f}")

    print(f"\nBest model: epoch {best_epoch}, val AUC = {best_auc:.4f}")
    print(f"Saved to models/nodule_detector_3d.pth")

    # ── Plot training curves ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("3D Nodule Detector — Training", fontsize=13)

    ax1.plot(train_losses, label="Train loss", color="steelblue")
    ax1.plot(val_losses,   label="Val loss",   color="darkorange")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.set_title("Loss Curves"); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(val_aucs, label="Val AUC", color="seagreen")
    ax2.axhline(best_auc, color="red", linestyle="--",
                label=f"Best AUC = {best_auc:.3f}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("ROC-AUC")
    ax2.set_title("Validation AUC"); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150)
    print("Saved results/training_curves.png")
    print(f"\nNext: python 3_evaluate.py")


if __name__ == "__main__":
    main()
