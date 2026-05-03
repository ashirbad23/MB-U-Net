from pathlib import Path
import os
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from scipy.ndimage import gaussian_filter
import xdem
import random


PATCH = 512
STRIDE = 512


# --------------------------------------------------
# Load rasters and create mosaics
# --------------------------------------------------

def load_data(data_dir):

    bands_dir = data_dir / "images"
    masks_dir = data_dir / "masks"

    band_files = glob.glob(str(bands_dir / "*.tif"))
    mask_files = glob.glob(str(masks_dir / "*.tif"))

    bands = [rasterio.open(f) for f in band_files]
    masks = [rasterio.open(f) for f in mask_files]

    img_mosaic, img_transform = merge(bands)
    mask_mosaic, mask_transform = merge(masks)

    assert img_transform == mask_transform

    with rasterio.open(band_files[0]) as src:
        transform = src.transform
        crs = src.crs

    return img_mosaic, mask_mosaic, transform, crs


# --------------------------------------------------
# Compute terrain features (CORRECT)
# --------------------------------------------------

def compute_terrain_features(dem, transform, crs):

    dem = gaussian_filter(dem, sigma=1)

    dem_x = xdem.DEM.from_array(dem, transform=transform, crs=crs)

    # --- Base features ---
    slope = dem_x.slope().data
    aspect = dem_x.aspect().data

    aspect_rad = np.deg2rad(aspect)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)

    profile_curv = dem_x.profile_curvature().data
    plan_curv = dem_x.planform_curvature().data

    k_max = dem_x.max_curvature().data
    k_min = dem_x.min_curvature().data

    # --- Fix scaling (xDEM uses *100) ---
    scale = 100.0
    profile_curv /= scale
    plan_curv /= scale
    k_max /= scale
    k_min /= scale

    # --- Derived features ---
    H = (k_max + k_min) / 2          # Mean curvature
    K = k_max * k_min               # Gaussian curvature
    M = (k_max - k_min) / 2         # Unsphericity
    E = 0.5 * (profile_curv - plan_curv)  # Slope azimuth divergence

    # Clean masked values
    def clean(x):
        return np.ma.filled(x, 0)

    return (
        clean(slope),
        clean(aspect_sin),
        clean(aspect_cos),
        clean(profile_curv),
        clean(plan_curv),
        clean(k_max),
        clean(k_min),
        clean(H),
        clean(K),
        clean(E),
        clean(M),
    )


# --------------------------------------------------
# Compute global stats (RAM SAFE)
# --------------------------------------------------

def compute_stats_streaming(img_mosaic, terrain_features):

    terrain_features = list(terrain_features)

    sum_c = None
    sum_sq_c = None
    count = 0

    chunk_size = 256

    _, H, W = img_mosaic.shape

    for y in range(0, H, chunk_size):

        y_end = min(y + chunk_size, H)

        img_chunk = img_mosaic[:, y:y_end, :]

        terrain_chunk = np.stack([
            t[y:y_end, :] for t in terrain_features
        ])

        full = np.concatenate([img_chunk, terrain_chunk], axis=0)

        full = full.astype(np.float32)
        full = full.reshape(full.shape[0], -1)

        if sum_c is None:
            C = full.shape[0]
            sum_c = np.zeros(C, dtype=np.float64)
            sum_sq_c = np.zeros(C, dtype=np.float64)

        sum_c += full.sum(axis=1)
        sum_sq_c += (full ** 2).sum(axis=1)
        count += full.shape[1]

    mean = sum_c / count
    std = np.sqrt(sum_sq_c / count - mean**2)

    std = np.where(std < 1e-6, 1e-6, std)

    return mean.astype(np.float32), std.astype(np.float32)


# --------------------------------------------------
# Patch extraction
# --------------------------------------------------

def extract_patches(img_mosaic, mask_mosaic, terrain_features):

    (
        slope,
        aspect_sin,
        aspect_cos,
        profile_curv,
        plan_curv,
        k_max,
        k_min,
        H,
        K,
        E,
        M
    ) = terrain_features

    os.makedirs("../dataset/images", exist_ok=True)
    os.makedirs("../dataset/masks", exist_ok=True)

    _, H_img, W_img = img_mosaic.shape

    counter = 0

    for y in range(0, H_img - PATCH + 1, STRIDE):
        for x in range(0, W_img - PATCH + 1, STRIDE):

            img_patch = img_mosaic[:, y:y+PATCH, x:x+PATCH]

            terrain_patch = np.stack([
                slope[y:y+PATCH, x:x+PATCH],
                aspect_sin[y:y+PATCH, x:x+PATCH],
                aspect_cos[y:y+PATCH, x:x+PATCH],
                profile_curv[y:y+PATCH, x:x+PATCH],
                plan_curv[y:y+PATCH, x:x+PATCH],
                k_max[y:y+PATCH, x:x+PATCH],
                k_min[y:y+PATCH, x:x+PATCH],
                H[y:y+PATCH, x:x+PATCH],
                K[y:y+PATCH, x:x+PATCH],
                E[y:y+PATCH, x:x+PATCH],
                M[y:y+PATCH, x:x+PATCH],
            ])

            full_patch = np.concatenate([img_patch, terrain_patch], axis=0)

            mask_patch = mask_mosaic[:, y:y+PATCH, x:x+PATCH]

            if mask_patch.sum() == 0:
                if random.random() > 0.3:
                    continue

            r = y // PATCH
            c = x // PATCH

            np.save(f"../dataset/images/img_r{r}_c{c}.npy", full_patch.astype(np.float32))
            np.save(f"../dataset/masks/mask_r{r}_c{c}.npy", mask_patch.astype(np.uint8))

            counter += 1

    print("Saved patches:", counter)


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def main():

    data_dir = Path("../data")

    img_mosaic, mask_mosaic, transform, crs = load_data(data_dir)

    dem = img_mosaic[6]  # DEM band

    terrain_features = compute_terrain_features(dem, transform, crs)

    # --- Compute normalization stats ---
    mean, std = compute_stats_streaming(img_mosaic, terrain_features)

    os.makedirs("../dataset", exist_ok=True)

    np.save("../dataset/mean.npy", mean)
    np.save("../dataset/std.npy", std)

    print("Saved mean/std")

    # --- Extract patches ---
    extract_patches(img_mosaic, mask_mosaic, terrain_features)


if __name__ == "__main__":
    main()