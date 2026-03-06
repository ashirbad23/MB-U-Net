from pathlib import Path
import os
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from scipy.ndimage import gaussian_filter
import xdem


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
# Compute terrain derivatives
# --------------------------------------------------

def compute_terrain_features(dem, transform, crs):

    dem = gaussian_filter(dem, sigma=1)

    dem_x = xdem.DEM.from_array(dem, transform=transform, crs=crs)

    slope = xdem.terrain.slope(dem_x).data
    aspect = xdem.terrain.aspect(dem_x).data

    aspect_rad = np.deg2rad(aspect)
    aspect_sin = np.sin(aspect_rad)
    aspect_cos = np.cos(aspect_rad)

    profile_curv = xdem.terrain.profile_curvature(dem_x).data
    plan_curv = xdem.terrain.planform_curvature(dem_x).data

    dx, dy = np.gradient(dem, 10, 10)

    dxx, dxy = np.gradient(dx, 10, 10)
    dyx, dyy = np.gradient(dy, 10, 10)

    mean_curv = (dxx + dyy) / 2
    gaussian_curv = (dxx * dyy - dxy**2)
    slope_azimuth_divergence = dxx + dyy
    unsphericity = np.sqrt(dxx**2 + 2 * (dxy**2) + dyy**2)

    slope = np.ma.filled(slope, 0)
    profile_curv = np.ma.filled(profile_curv, 0)
    plan_curv = np.ma.filled(plan_curv, 0)
    aspect_sin = np.ma.filled(aspect_sin, 0)
    aspect_cos = np.ma.filled(aspect_cos, 0)

    return (
        slope,
        aspect_sin,
        aspect_cos,
        profile_curv,
        plan_curv,
        mean_curv,
        gaussian_curv,
        slope_azimuth_divergence,
        unsphericity
    )


# --------------------------------------------------
# Patch extraction
# --------------------------------------------------

def extract_patches(img_mosaic, mask_mosaic, terrain_features):

    slope, aspect_sin, aspect_cos, profile_curv, plan_curv, \
    mean_curv, gaussian_curv, slope_azimuth_divergence, unsphericity = terrain_features

    os.makedirs("dataset/images", exist_ok=True)
    os.makedirs("dataset/masks", exist_ok=True)

    _, H, W = img_mosaic.shape

    counter = 0

    for y in range(0, H - PATCH + 1, STRIDE):
        for x in range(0, W - PATCH + 1, STRIDE):

            img_patch = img_mosaic[:, y:y+PATCH, x:x+PATCH]

            terrain_patch = np.stack([
                slope[y:y+PATCH, x:x+PATCH],
                aspect_sin[y:y+PATCH, x:x+PATCH],
                aspect_cos[y:y+PATCH, x:x+PATCH],
                profile_curv[y:y+PATCH, x:x+PATCH],
                plan_curv[y:y+PATCH, x:x+PATCH],
                mean_curv[y:y+PATCH, x:x+PATCH],
                gaussian_curv[y:y+PATCH, x:x+PATCH],
                slope_azimuth_divergence[y:y+PATCH, x:x+PATCH],
                unsphericity[y:y+PATCH, x:x+PATCH]
            ])

            full_patch = np.concatenate([img_patch, terrain_patch], axis=0)

            mask_patch = mask_mosaic[:, y:y+PATCH, x:x+PATCH]

            if mask_patch.sum() < 50:
                continue

            r = y // PATCH
            c = x // PATCH

            np.save(f"dataset/images/img_r{r}_c{c}.npy", full_patch.astype(np.float32))
            np.save(f"dataset/masks/mask_r{r}_c{c}.npy", mask_patch.astype(np.uint8))

            counter += 1

    print("Saved patches:", counter)


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def main():

    data_dir = Path("../data")

    img_mosaic, mask_mosaic, transform, crs = load_data(data_dir)

    dem = img_mosaic[6]

    terrain_features = compute_terrain_features(dem, transform, crs)

    extract_patches(img_mosaic, mask_mosaic, terrain_features)


if __name__ == "__main__":
    main()