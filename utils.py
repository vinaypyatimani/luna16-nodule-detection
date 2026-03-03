"""
utils.py
========
Shared utility functions used across all pipeline scripts.

Covers:
  - Hounsfield Unit (HU) windowing and normalisation
  - World-coordinate to voxel-coordinate conversion
  - 3D patch extraction with boundary padding
  - Loading .mhd CT volumes via SimpleITK (LUNA16 real data)
  - Generating synthetic CT volumes for demo mode

Understanding Hounsfield Units (HU):
  CT scanners measure X-ray attenuation and express it in HU:
    Air         : -1000 HU
    Lung tissue :  -700 to -400 HU
    Fat         :  -100 HU
    Soft tissue :  +20 to +80 HU
    Bone        :  +400 to +1000 HU

  Lung nodules appear as soft-tissue density spherical regions
  (+20 to +100 HU) embedded within the dark lung parenchyma (-700 HU).
  We window to [-1200, 600] HU to capture all relevant lung structures,
  then normalise to [0, 1].
"""

import numpy as np
import os
import warnings

# HU window for lung CT
HU_MIN = -1200.0
HU_MAX =  600.0

# Patch size for 3D candidate extraction (mm-equivalent: ~32mm at 1mm spacing)
PATCH_SIZE = 32   # voxels per side


# ── Hounsfield Unit processing ────────────────────────────────────────────────

def window_and_normalise(volume_hu):
    """
    Clip HU values to lung-relevant window and normalise to [0, 1].

    Args:
        volume_hu : numpy array of any shape, values in HU

    Returns:
        normalised float32 array, same shape, values in [0, 1]
    """
    v = np.clip(volume_hu, HU_MIN, HU_MAX).astype(np.float32)
    return (v - HU_MIN) / (HU_MAX - HU_MIN)


# ── Coordinate conversion ─────────────────────────────────────────────────────

def world_to_voxel(world_coords, origin, spacing):
    """
    Convert world (mm) coordinates to voxel indices.

    LUNA16 annotations are stored in world coordinates (mm) because
    different CT scanners produce different voxel spacings. The .mhd file
    header stores the origin (mm position of voxel [0,0,0]) and the spacing
    (mm per voxel in each axis). Conversion:
        voxel_idx = (world_coord - origin) / spacing

    Args:
        world_coords : array (3,) or (N,3) — [x, y, z] in mm
        origin       : array (3,) — scanner origin in mm
        spacing      : array (3,) — mm/voxel for [x, y, z]

    Returns:
        voxel indices as int array, same leading shape as world_coords
    """
    return np.round((world_coords - origin) / spacing).astype(int)


def voxel_to_world(voxel_coords, origin, spacing):
    """Inverse of world_to_voxel."""
    return voxel_coords * spacing + origin


# ── 3D patch extraction ───────────────────────────────────────────────────────

def extract_patch_3d(volume, center_voxel, patch_size=PATCH_SIZE):
    """
    Extract a cubic 3D patch centred on a candidate voxel location.

    Handles boundary cases by padding with the minimum value (-1000 HU
    equivalent after normalisation ≈ 0) so boundary padding looks like air.

    Args:
        volume       : 3D numpy array (Z, Y, X) — normalised [0,1]
        center_voxel : (z, y, x) integer voxel coordinates
        patch_size   : side length of cubic patch in voxels

    Returns:
        patch : float32 array of shape (patch_size, patch_size, patch_size)
    """
    Z, Y, X = volume.shape
    z, y, x = center_voxel
    h = patch_size // 2

    # Compute crop bounds (clamped to volume)
    z0, z1 = max(0, z - h), min(Z, z + h)
    y0, y1 = max(0, y - h), min(Y, y + h)
    x0, x1 = max(0, x - h), min(X, x + h)

    crop = volume[z0:z1, y0:y1, x0:x1]

    # Padding needed on each side
    pad_z0 = h - (z - z0)
    pad_z1 = patch_size - crop.shape[0] - pad_z0
    pad_y0 = h - (y - y0)
    pad_y1 = patch_size - crop.shape[1] - pad_y0
    pad_x0 = h - (x - x0)
    pad_x1 = patch_size - crop.shape[2] - pad_x0

    pad_width = [
        (max(0, pad_z0), max(0, pad_z1)),
        (max(0, pad_y0), max(0, pad_y1)),
        (max(0, pad_x0), max(0, pad_x1)),
    ]
    patch = np.pad(crop, pad_width, mode='constant', constant_values=0.0)

    # Ensure exact size (handles edge case with rounding)
    patch = patch[:patch_size, :patch_size, :patch_size]
    return patch.astype(np.float32)


# ── LUNA16 real data loading ──────────────────────────────────────────────────

def load_mhd_volume(mhd_path):
    """
    Load a LUNA16 .mhd CT scan using SimpleITK.

    Returns:
        volume  : float32 numpy array (Z, Y, X) in HU
        origin  : array (3,) — world origin [x, y, z] in mm
        spacing : array (3,) — voxel spacing [x, y, z] in mm/voxel

    Requires: pip install SimpleITK
    """
    try:
        import SimpleITK as sitk
    except ImportError:
        raise ImportError(
            "SimpleITK is required to load real .mhd files.\n"
            "Install with: pip install SimpleITK\n"
            "Or run in demo mode: python 1_prepare_data.py --demo"
        )

    itk_img  = sitk.ReadImage(mhd_path)
    volume   = sitk.GetArrayFromImage(itk_img).astype(np.float32)  # (Z, Y, X)
    origin   = np.array(list(itk_img.GetOrigin()))   # (x, y, z) in mm
    spacing  = np.array(list(itk_img.GetSpacing()))  # (x, y, z) mm/voxel

    return volume, origin, spacing


# ── Demo mode: synthetic CT volumes ──────────────────────────────────────────

def make_synthetic_ct_volume(size=(128, 128, 128), n_nodules=0, rng=None,
                              nodule_radius_range=(4, 12), seed=None):
    """
    Generate a synthetic lung CT volume in Hounsfield Units.

    Anatomy modelled:
      - Chest wall     : outer ring of soft-tissue density
      - Lung parenchyma: central region filled with air-density noise
      - Pulmonary vessels: bright branching structures (simplified as
                          a few bright elongated blobs)
      - Nodules        : spherical soft-tissue density blobs inside lung

    This lets the pipeline run fully offline — no LUNA16 download needed.

    Args:
        size              : (Z, Y, X) voxel dimensions
        n_nodules         : number of nodules to embed (0 = normal scan)
        rng               : numpy Generator (or None for default)
        nodule_radius_range: (min, max) nodule radius in voxels
        seed              : random seed if rng not provided

    Returns:
        volume_hu   : float32 (Z, Y, X) array in HU
        nodule_locs : list of (z, y, x) voxel centres for embedded nodules
        spacing     : simulated voxel spacing array [1, 1, 1] mm/voxel
        origin      : simulated origin [0, 0, 0] mm
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    Z, Y, X = size
    cy, cx = Y // 2, X // 2

    # ── Base: air background ──
    volume = np.full(size, -950.0, dtype=np.float32)

    # ── Chest wall ring (each slice) ──
    Y_idx, X_idx = np.mgrid[0:Y, 0:X]
    for z in range(Z):
        r = np.sqrt((X_idx - cx)**2 + (Y_idx - cy)**2)
        chest_wall = (r >= Y * 0.40) & (r < Y * 0.48)
        volume[z][chest_wall]  = rng.uniform(30, 80, volume[z][chest_wall].shape)
        outside = r >= Y * 0.48
        volume[z][outside]     = rng.uniform(40, 60, volume[z][outside].shape)

    # ── Lung tissue: soft noise inside lung region ──
    for z in range(Z):
        r = np.sqrt((X_idx - cx)**2 + (Y_idx - cy)**2)
        lung = r < Y * 0.40
        volume[z][lung] += rng.normal(0, 30, volume[z][lung].shape)

    # ── Pulmonary vessels: a few bright elongated cylinders ──
    for _ in range(rng.integers(3, 7)):
        vx = rng.integers(cx - Y//4, cx + Y//4)
        vy = rng.integers(cy - Y//4, cy + Y//4)
        vr = rng.integers(2, 4)
        r  = np.sqrt((X_idx - vx)**2 + (Y_idx - vy)**2)
        vessel = r <= vr
        for z in range(Z):
            volume[z][vessel] = rng.uniform(30, 80, volume[z][vessel].shape)

    # ── Embed nodules ──
    nodule_locs = []
    Z_idx, Y_idx3, X_idx3 = np.mgrid[0:Z, 0:Y, 0:X]
    for _ in range(n_nodules):
        # Place within lung parenchyma
        nz = rng.integers(Z // 4, 3 * Z // 4)
        ny = rng.integers(cy - Y//4, cy + Y//4)
        nx = rng.integers(cx - X//4, cx + X//4)
        nr = rng.integers(*nodule_radius_range)

        r3d = np.sqrt((X_idx3 - nx)**2 + (Y_idx3 - ny)**2 + (Z_idx - nz)**2)
        nodule_mask = r3d <= nr
        # Soft-tissue HU with slight noise
        volume[nodule_mask] = rng.normal(40, 15, volume[nodule_mask].shape)
        nodule_locs.append((int(nz), int(ny), int(nx)))

    spacing = np.array([1.0, 1.0, 1.0])
    origin  = np.array([0.0, 0.0, 0.0])
    return volume, nodule_locs, spacing, origin


# ── Candidate generation (for demo mode) ─────────────────────────────────────

def make_demo_candidates(volume_id, nodule_locs, spacing, origin,
                          n_false_positives=10, rng=None):
    """
    Generate a candidate list (mimicking candidates_V2.csv structure)
    for a single synthetic volume.

    Candidates = true nodule locations + random false positives.

    Returns:
        candidates : list of dicts with keys:
                     seriesuid, coordX, coordY, coordZ, class
                     (coordinates in world mm — matching LUNA16 CSV format)
    """
    if rng is None:
        rng = np.random.default_rng()

    candidates = []

    # True positives — exact nodule locations converted to world coords
    for (z, y, x) in nodule_locs:
        vox  = np.array([x, y, z], dtype=float)
        world = voxel_to_world(vox, origin, spacing)
        candidates.append({
            "seriesuid": volume_id,
            "coordX": world[0], "coordY": world[1], "coordZ": world[2],
            "class": 1
        })

    # False positives — random locations within lung
    Z, Y, X = 128, 128, 128
    cy, cx  = Y // 2, X // 2
    fp_added = 0
    attempts = 0
    while fp_added < n_false_positives and attempts < 1000:
        attempts += 1
        fz = rng.integers(Z//4, 3*Z//4)
        fy = rng.integers(cy - Y//4, cy + Y//4)
        fx = rng.integers(cx - X//4, cx + X//4)
        # Must not overlap a true nodule
        too_close = any(
            abs(fz - nz) < 8 and abs(fy - ny) < 8 and abs(fx - nx) < 8
            for (nz, ny, nx) in nodule_locs
        )
        if not too_close:
            vox   = np.array([fx, fy, fz], dtype=float)
            world = voxel_to_world(vox, origin, spacing)
            candidates.append({
                "seriesuid": volume_id,
                "coordX": world[0], "coordY": world[1], "coordZ": world[2],
                "class": 0
            })
            fp_added += 1

    return candidates
