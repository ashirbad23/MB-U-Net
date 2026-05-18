"""
Generate representative dataset visualizations for the report.

Outputs saved to:
    assets/
        rgb_patch.png
        gt_yellow_overlay.png
        ndsi.png
        dem.png
        slope.png
        aspect_sin.png
        aspect_cos.png
        profile_curvature.png
        planform_curvature.png
        mean_curvature.png
        gaussian_curvature.png
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from preprocessing import load_data, compute_terrain_features


# ==========================================================
# Configuration
# ==========================================================

DATA_DIR = Path("../data")
OUTPUT_DIR = Path("assets/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Representative glacier patch
Y = 3072
X = 11000

PATCH_SIZE_Y = 1080
PATCH_SIZE_X = 1080


# ==========================================================
# Utility functions
# ==========================================================

def normalize(arr, lower=2, upper=98):
    """Percentile-based normalization to [0, 1]."""
    arr = np.asarray(arr, dtype=np.float32)

    p_low, p_high = np.percentile(arr, (lower, upper))

    if abs(p_high - p_low) < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)

    arr = (arr - p_low) / (p_high - p_low)
    return np.clip(arr, 0, 1)


def save_grayscale(arr, filename, cmap="gray"):
    """Save a single-band image using percentile normalization."""
    arr_norm = normalize(arr)
    plt.imsave(OUTPUT_DIR / filename, arr_norm, cmap=cmap)


def save_diverging(arr, filename, cmap="RdBu_r"):
    """Save signed data centered around zero."""
    arr = np.asarray(arr, dtype=np.float32)
    vmax = np.percentile(np.abs(arr), 98)

    if vmax < 1e-8:
        vmax = 1.0

    plt.imsave(
        OUTPUT_DIR / filename,
        arr,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax
    )


# ==========================================================
# Load data
# ==========================================================

print("Loading data...")
img_mosaic, mask_mosaic, transform, crs = load_data(DATA_DIR)

# Spectral channels: 0-5
# DEM channel: last channel in the loaded mosaic
dem_full = img_mosaic[-1]

print("Computing terrain features...")
terrain = compute_terrain_features(dem_full, transform, crs)

(
    slope_full,            # 0
    aspect_sin_full,       # 1
    aspect_cos_full,       # 2
    profile_curv_full,     # 3
    planform_curv_full,    # 4
    k_max_full,            # 5
    k_min_full,            # 6
    mean_curv_full,        # 7
    gaussian_curv_full,    # 8
    slope_div_full,        # 9
    unsphericity_full,     # 10
) = terrain

# ==========================================================
# Extract representative patch
# ==========================================================

ys = slice(Y, Y + PATCH_SIZE_Y)
xs = slice(X, X + PATCH_SIZE_X)

img_patch = img_mosaic[:, ys, xs]
mask_patch = np.squeeze(mask_mosaic[:, ys, xs]) > 0

# Spectral bands
blue = img_patch[0]
green = img_patch[1]
red = img_patch[2]
nir = img_patch[3]
swir1 = img_patch[4]

# RGB composite
rgb = np.stack([red, green, blue], axis=-1)
rgb = normalize(rgb)
rgb_uint8 = (rgb * 255).astype(np.uint8)

plt.imsave(OUTPUT_DIR / "rgb_patch.png", rgb_uint8)

# ==========================================================
# Ground truth overlay
# ==========================================================

overlay = rgb_uint8.astype(np.float32).copy()
yellow = np.array([255, 255, 0], dtype=np.float32)
alpha = 0.5

overlay[mask_patch] = (
    (1 - alpha) * overlay[mask_patch] +
    alpha * yellow
)

overlay = overlay.astype(np.uint8)

plt.imsave(OUTPUT_DIR / "gt_yellow_overlay.png", overlay)

# ==========================================================
# NDSI = (Green - SWIR1) / (Green + SWIR1)
# ==========================================================

ndsi = (green - swir1) / (green + swir1 + 1e-6)
save_diverging(ndsi, "ndsi.png", cmap="coolwarm")

# ==========================================================
# Extract terrain patches
# ==========================================================

dem_patch = dem_full[ys, xs]
slope_patch = slope_full[ys, xs]
aspect_sin_patch = aspect_sin_full[ys, xs]
aspect_cos_patch = aspect_cos_full[ys, xs]
profile_curv_patch = profile_curv_full[ys, xs]
planform_curv_patch = planform_curv_full[ys, xs]
mean_curv_patch = mean_curv_full[ys, xs]
gaussian_curv_patch = gaussian_curv_full[ys, xs]

# ==========================================================
# Save terrain visualizations
# ==========================================================

# Positive-valued features
save_grayscale(dem_patch, "dem.png", cmap="terrain")
save_grayscale(slope_patch, "slope.png", cmap="viridis")

# Bounded signed features
save_diverging(aspect_sin_patch, "aspect_sin.png", cmap="coolwarm")
save_diverging(aspect_cos_patch, "aspect_cos.png", cmap="coolwarm")

# Signed curvature features
save_diverging(profile_curv_patch, "profile_curvature.png")
save_diverging(planform_curv_patch, "planform_curvature.png")
save_diverging(mean_curv_patch, "mean_curvature.png")
save_diverging(gaussian_curv_patch, "gaussian_curvature.png")

print(f"All visualizations saved to: {OUTPUT_DIR.resolve()}")