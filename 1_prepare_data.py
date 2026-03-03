"""
1_prepare_data.py
=================
Prepares the dataset of 3D candidate patches for training.

Two modes:
  --demo   (default) No download needed. Generates synthetic CT volumes
           with realistic lung anatomy and embedded nodules. Ideal for
           testing the pipeline and for demonstrating the approach.

  --luna16 <path>    Real LUNA16 data. Requires the dataset downloaded from
                     https://luna16.grand-challenge.org/Download/
                     Path should contain subset0/ ... subset9/ folders
                     and annotations.csv + candidates_V2.csv

What this script does (both modes):
  1. For each CT volume: load it, apply HU windowing, normalise to [0,1]
  2. For each candidate location: convert world→voxel coordinates
  3. Extract a 32×32×32 voxel cubic patch centred on that location
  4. Label: 1 if nodule, 0 if false positive
  5. Save all patches + labels as .npy files for training

Why 3D patches?
  A 3D patch captures the full spherical morphology of a nodule —
  its extent in all three dimensions simultaneously. This is critical
  because nodules look like spheres in 3D but may appear as circles,
  ellipses, or irregular blobs in individual 2D slices depending on
  the slice plane. 3D convolutions learn this shape directly.

Output:
  data/patches.npy  — shape (N, 1, 32, 32, 32), dtype float32
  data/labels.npy   — shape (N,),                dtype int64
  data/meta.csv     — candidate metadata (volume id, coords, label)

Usage:
  python 1_prepare_data.py            # demo mode (no data needed)
  python 1_prepare_data.py --luna16 /path/to/LUNA16   # real data
  python 1_prepare_data.py --n_volumes 20   # demo: use 20 synthetic volumes
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
from utils import (
    window_and_normalise, world_to_voxel, extract_patch_3d,
    make_synthetic_ct_volume, make_demo_candidates,
    load_mhd_volume, PATCH_SIZE
)

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_N_VOLUMES     = 50    # synthetic volumes in demo mode
NODULES_PER_VOLUME    = (0, 3)  # random range of nodules per volume
N_FP_PER_VOLUME       = 15    # false positive candidates per volume
MAX_CANDIDATES        = 20000  # cap total candidates to keep training fast
SEED                  = 42


# ── Demo mode pipeline ────────────────────────────────────────────────────────

def prepare_demo(n_volumes=DEFAULT_N_VOLUMES):
    """Generate synthetic volumes and extract candidate patches."""
    print(f"[DEMO MODE] Generating {n_volumes} synthetic CT volumes...")
    rng = np.random.default_rng(SEED)

    all_patches = []
    all_labels  = []
    all_meta    = []

    for i in range(n_volumes):
        if i % 10 == 0:
            print(f"  Volume {i+1}/{n_volumes}")

        n_nod = int(rng.integers(NODULES_PER_VOLUME[0],
                                  NODULES_PER_VOLUME[1] + 1))
        volume_hu, nodule_locs, spacing, origin = make_synthetic_ct_volume(
            size=(128, 128, 128),
            n_nodules=n_nod,
            rng=rng
        )
        volume_norm = window_and_normalise(volume_hu)
        volume_id   = f"synthetic_{i:04d}"

        candidates = make_demo_candidates(
            volume_id, nodule_locs, spacing, origin,
            n_false_positives=N_FP_PER_VOLUME, rng=rng
        )

        for cand in candidates:
            # Convert world → voxel
            world = np.array([cand["coordX"], cand["coordY"], cand["coordZ"]])
            vox   = world_to_voxel(world, origin, spacing)
            # Note: LUNA16 coords are (x,y,z), volume is (Z,Y,X)
            center_zyx = (int(vox[2]), int(vox[1]), int(vox[0]))

            patch = extract_patch_3d(volume_norm, center_zyx, PATCH_SIZE)
            all_patches.append(patch)
            all_labels.append(cand["class"])
            all_meta.append({
                "seriesuid": cand["seriesuid"],
                "coordX":    cand["coordX"],
                "coordY":    cand["coordY"],
                "coordZ":    cand["coordZ"],
                "label":     cand["class"]
            })

    return (np.array(all_patches, dtype=np.float32),
            np.array(all_labels, dtype=np.int64),
            pd.DataFrame(all_meta))


# ── LUNA16 real data pipeline ─────────────────────────────────────────────────

def prepare_luna16(luna16_path, max_candidates=MAX_CANDIDATES):
    """
    Process real LUNA16 data.

    Expects:
      luna16_path/
        subset0/ ... subset9/   — .mhd + .raw CT files
        annotations.csv         — confirmed nodule locations
        candidates_V2.csv       — all candidates (nodule + non-nodule)
    """
    print(f"[LUNA16 MODE] Reading data from: {luna16_path}")

    # Load candidate list
    cand_path = os.path.join(luna16_path, "candidates_V2.csv")
    if not os.path.exists(cand_path):
        cand_path = os.path.join(luna16_path, "candidates.csv")
    candidates_df = pd.read_csv(cand_path)
    print(f"  Total candidates: {len(candidates_df)}")
    print(f"  Nodules (class=1): {(candidates_df['class']==1).sum()}")
    print(f"  Non-nodules (class=0): {(candidates_df['class']==0).sum()}")

    # Balance: keep all nodules + equal number of FPs
    nodules = candidates_df[candidates_df["class"] == 1]
    non_nod = candidates_df[candidates_df["class"] == 0].sample(
        n=min(len(nodules) * 10, len(candidates_df[candidates_df["class"] == 0])),
        random_state=SEED
    )
    candidates_df = pd.concat([nodules, non_nod]).sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)

    if len(candidates_df) > max_candidates:
        candidates_df = candidates_df.sample(
            n=max_candidates, random_state=SEED
        ).reset_index(drop=True)

    print(f"  Using {len(candidates_df)} balanced candidates")

    # Find all .mhd files
    mhd_files = {}
    for subset in [f"subset{i}" for i in range(10)]:
        subset_dir = os.path.join(luna16_path, subset)
        if not os.path.isdir(subset_dir):
            continue
        for f in os.listdir(subset_dir):
            if f.endswith(".mhd"):
                uid = f.replace(".mhd", "")
                mhd_files[uid] = os.path.join(subset_dir, f)

    print(f"  Found {len(mhd_files)} CT volumes")

    all_patches = []
    all_labels  = []
    all_meta    = []

    # Group candidates by volume for efficient loading
    grouped  = candidates_df.groupby("seriesuid")
    n_loaded = 0

    for uid, group in grouped:
        if uid not in mhd_files:
            continue

        try:
            volume_hu, origin, spacing = load_mhd_volume(mhd_files[uid])
        except Exception as e:
            print(f"  Warning: could not load {uid}: {e}")
            continue

        volume_norm = window_and_normalise(volume_hu)
        n_loaded   += 1

        if n_loaded % 50 == 0:
            print(f"  Processed {n_loaded} volumes, "
                  f"{len(all_patches)} patches so far")

        for _, row in group.iterrows():
            world     = np.array([row["coordX"], row["coordY"], row["coordZ"]])
            vox       = world_to_voxel(world, origin, spacing)
            center_zyx = (int(vox[2]), int(vox[1]), int(vox[0]))

            patch = extract_patch_3d(volume_norm, center_zyx, PATCH_SIZE)
            all_patches.append(patch)
            all_labels.append(int(row["class"]))
            all_meta.append({
                "seriesuid": uid,
                "coordX": row["coordX"],
                "coordY": row["coordY"],
                "coordZ": row["coordZ"],
                "label": int(row["class"])
            })

    print(f"  Finished. Loaded {n_loaded} volumes, "
          f"{len(all_patches)} patches total.")

    return (np.array(all_patches, dtype=np.float32),
            np.array(all_labels, dtype=np.int64),
            pd.DataFrame(all_meta))


# ── Save and summary ──────────────────────────────────────────────────────────

def save_data(patches, labels, meta):
    os.makedirs("data", exist_ok=True)

    # Add channel dimension: (N, Z, Y, X) → (N, 1, Z, Y, X)
    patches_ch = patches[:, np.newaxis, :, :, :]

    np.save("data/patches.npy", patches_ch)
    np.save("data/labels.npy",  labels)
    meta.to_csv("data/meta.csv", index=False)

    n0 = (labels == 0).sum()
    n1 = (labels == 1).sum()
    print(f"\nSaved:")
    print(f"  data/patches.npy : {patches_ch.shape}  (N, 1, Z, Y, X)")
    print(f"  data/labels.npy  : {labels.shape}")
    print(f"  data/meta.csv    : {len(meta)} rows")
    print(f"  Class balance    : {n0} non-nodule, {n1} nodule "
          f"(ratio {n0/max(n1,1):.1f}:1)")
    print(f"\nNext: python 2_train_model.py")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare LUNA16 candidate patch dataset"
    )
    parser.add_argument(
        "--luna16", type=str, default=None,
        help="Path to LUNA16 data directory (omit for demo mode)"
    )
    parser.add_argument(
        "--n_volumes", type=int, default=DEFAULT_N_VOLUMES,
        help="Number of synthetic volumes in demo mode (default: 50)"
    )
    args = parser.parse_args()

    if args.luna16:
        patches, labels, meta = prepare_luna16(args.luna16)
    else:
        patches, labels, meta = prepare_demo(args.n_volumes)

    save_data(patches, labels, meta)
