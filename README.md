# Lung Nodule Detection in CT Scans: 3D CNN on LUNA16

A 3D Convolutional Neural Network for automated lung nodule detection from CT scans, built with PyTorch. This project implements the core computational pipeline used in Computer-Aided Detection (CAD) systems for lung cancer screening.

The pipeline works in two modes:
- **Demo mode** (no download needed): generates synthetic CT volumes with realistic lung anatomy and embedded nodules. Run immediately to see the full pipeline working.
- **LUNA16 mode**: processes real CT scans from the LUNA16 challenge dataset (~40 GB).

---

## What This Project Demonstrates

Lung cancer is the leading cause of cancer mortality worldwide. Early detection of pulmonary nodules: small, roughly spherical abnormalities in lung tissue, via CT screening significantly improves survival rates. The challenge is scale: a single CT scan produces hundreds of 2D slices, and radiologists reviewing thousands of scans per year face a high false-negative rate from fatigue.

This project builds an automated detection system that:

1. **Loads and preprocesses** 3D CT volumes (Hounsfield Unit windowing, normalisation)
2. **Extracts 3D candidate patches** centred on anatomically suspicious locations
3. **Classifies each patch** as nodule or non-nodule using a 3D ResNet-style CNN
4. **Evaluates performance** using FROC: the clinical standard metric for nodule detection

This is directly relevant to any domain requiring automated anomaly detection in 3D volumetric imaging data, including industrial CT inspection of battery cells, materials characterisation, and non-destructive testing.

---

## Project Structure

```
luna16-nodule-detection/
│
├── README.md
├── requirements.txt
│
├── utils.py               ← Shared utilities: HU windowing, coord conversion,
│                             patch extraction, synthetic data generation
│
├── 1_prepare_data.py      ← Extract 3D candidate patches (demo or LUNA16)
├── 2_train_model.py       ← Train 3D ResNet nodule classifier
├── 3_evaluate.py          ← ROC, FROC, confusion matrix, visualisations
│
├── data/                  ← Created by step 1
│   ├── patches.npy        (N, 1, 32, 32, 32) float32
│   ├── labels.npy         (N,)               int64
│   ├── meta.csv           candidate metadata
│   └── idx_test.npy       test split indices
│
├── models/                ← Created by step 2
│   └── nodule_detector_3d.pth
│
└── results/               ← Created by step 3
    ├── training_curves.png
    ├── roc_curve.png
    ├── froc_curve.png
    ├── confusion_matrix.png
    └── patch_examples.png
```

---

## Background: CT Imaging and Nodule Detection

### Hounsfield Units (HU)

CT scanners measure X-ray attenuation at each voxel and express it in Hounsfield Units:

| Tissue          | Typical HU range |
|-----------------|-----------------|
| Air             | −1000           |
| Lung parenchyma | −700 to −400    |
| Fat             | −100            |
| Soft tissue     | +20 to +80      |
| Bone            | +400 to +1000   |

Lung nodules appear as soft-tissue spheres (+20 to +100 HU) in the dark lung (-700 HU). We window to [−1200, +600] HU to capture all relevant structures and normalise to [0, 1].

### World vs. Voxel Coordinates

CT scanners store data in **world coordinates** (millimetres) because different scanners produce different voxel spacings. The LUNA16 annotations specify nodule centres in mm. The `.mhd` file header stores the `Origin` (position of voxel [0,0,0]) and `Spacing` (mm per voxel). Conversion:

```
voxel_idx = round((world_coord_mm − origin_mm) / spacing_mm_per_voxel)
```

### Why 3D Convolutions?

A lung nodule is a 3D sphere. Processing individual 2D slices loses the cross-slice context, the nodule's spherical shape is only visible across multiple consecutive slices. 3D convolutions process the full volumetric patch, learning spherical shape features directly. This is the standard approach in modern nodule CAD systems.

### The FROC Metric

Standard ROC measures classifier performance per candidate. In clinical practice, radiologists review entire scans, so performance is measured per **scan**, not per candidate. The **Free-Response ROC (FROC)** curve plots:
- Y-axis: sensitivity (fraction of true nodules detected)
- X-axis: average false positives per scan

Clinical target for acceptable screening performance: **sensitivity > 0.85 at ≤ 4 FP/scan**.

---

## Step-by-Step Instructions

### Prerequisites

Python 3.8+. Verify with:
```bash
python3 --version
```

### Step 0: Set Up Environment

```bash
mkdir luna16-nodule-detection
cd luna16-nodule-detection

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install core dependencies
pip install numpy scipy pandas matplotlib scikit-learn

# Install PyTorch (CPU version - no GPU needed)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For real LUNA16 data, also install:
```bash
pip install SimpleITK
```

### Step 1: Prepare Data

**Demo mode (no download required):**
```bash
python 1_prepare_data.py
# or explicitly:
python 1_prepare_data.py --n_volumes 50
```
Generates 50 synthetic CT volumes with realistic anatomy. Takes ~20 seconds.

**Real LUNA16 data:**
1. Register and download from https://luna16.grand-challenge.org/Download/
2. Download `subset0.zip` through `subset9.zip` (or just `subset0.zip` to start) and `candidates_V2.csv` and `annotations.csv`
3. Extract to a folder: `LUNA16/subset0/`, `LUNA16/subset1/`, etc.
4. Run:
```bash
python 1_prepare_data.py --luna16 /path/to/LUNA16
```

**Expected output (demo mode):**
```
[DEMO MODE] Generating 50 synthetic CT volumes...
  Volume 1/50
  Volume 11/50
  ...
Saved:
  data/patches.npy : (878, 1, 32, 32, 32)
  data/labels.npy  : (878,)
  Class balance    : 752 non-nodule, 126 nodule (ratio 6.0:1)
```

### Step 2: Train the Model

```bash
python 2_train_model.py
```

Trains for 40 epochs with:
- Weighted random sampling to handle class imbalance
- BCE loss with positive class weighting
- Cosine annealing learning rate schedule

**Run time:** ~15 minutes on CPU (demo mode). Longer with full LUNA16 data.

**Expected output:**
```
Device: cpu
Total patches: 878
Train: 614  Val: 133  Test: 131
Model parameters: 1,234,817

Training for 40 epochs...
  Epoch   1/40 | train: 0.4821 | val: 0.3917 | val AUC: 0.832
  Epoch   5/40 | train: 0.3214 | val: 0.2891 | val AUC: 0.901
  ...
  Epoch  40/40 | train: 0.1823 | val: 0.2014 | val AUC: 0.943

Best model: epoch 37, val AUC = 0.947
```

### Step 3: Evaluate

```bash
python 3_evaluate.py
```

**Expected output:**
```
CLASSIFICATION RESULTS
────────────────────────────────────────────
              precision  recall  f1-score
Non-nodule       0.94     0.91     0.93
Nodule           0.72     0.81     0.76

ROC-AUC: 0.941

Saved results/roc_curve.png
Saved results/froc_curve.png
Saved results/confusion_matrix.png
Saved results/patch_examples.png
```

---

## Interpreting the Results

**ROC-AUC > 0.90** is strong performance for nodule detection, consistent with published results on LUNA16 with lightweight models.

**FROC curve:** the orange curve should cross the ≥ 0.85 sensitivity line before reaching 4 FP/scan on real LUNA16 data. On demo (synthetic) data, the task is easier and FROC will look better.

**Patch examples:** the three columns show axial (top-down), coronal (front), and sagittal (side) central slices through each 3D patch. True nodules appear as bright spherical regions against the dark lung background.

---

## How to Use with Real LUNA16 Data (Quick Checklist)

- [ ] Download at least `subset0.zip` + `candidates_V2.csv` + `annotations.csv`
- [ ] Extract subset0 to `LUNA16/subset0/`
- [ ] Install SimpleITK: `pip install SimpleITK`
- [ ] Run: `python 1_prepare_data.py --luna16 ./LUNA16`
- [ ] Follow steps 2 and 3 as normal

To use multiple subsets: download and extract `subset0` through `subset9`. The script automatically scans all available subsets.

---

## Key Concepts

| Concept | File | Description |
|---|---|---|
| HU windowing | `utils.py` | Clip and normalise CT density values |
| World→voxel conversion | `utils.py` | Convert annotation mm coords to array indices |
| 3D patch extraction | `utils.py` | Boundary-safe cubic region extraction |
| 3D residual blocks | `2_train_model.py` | Skip connections for stable 3D CNN training |
| Weighted sampling | `2_train_model.py` | Corrects class imbalance during training |
| FROC metric | `3_evaluate.py` | Clinical standard for nodule detection evaluation |

---

## Connection to Real Research

This pipeline mirrors the false positive reduction track of the LUNA16 challenge, where participants classify pre-generated candidate locations as nodule or non-nodule. State-of-the-art systems (e.g. V-Net, 3D U-Net variants) reach AUC > 0.99 using deeper models and larger training sets. This project implements the foundational pipeline that such systems build on.

The same approach, 3D patch extraction, volumetric CNN, FROC evaluation, applies directly to CT-based defect detection in other domains: battery cell inspection, additive manufacturing quality control, and materials characterisation.

---

## References

Setio, A.A.A. et al. (2017). *Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: The LUNA16 challenge.* Medical Image Analysis, 42, 1–13. https://doi.org/10.1016/j.media.2017.06.015

---

## Author

Vinay Pyatimani — portfolio project demonstrating 3D medical image analysis and deep learning for anomaly detection.
GitHub: https://github.com/vinaypyatimani/luna16-nodule-detection
