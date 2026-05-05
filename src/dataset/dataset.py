import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import glob
import json

from utils.transform import GlacierTransform


class GlacierDataset(Dataset):
    def __init__(self, path: Path, transform=None, patch_size=512, overlap=1.0, mode=None, mode_path=None, bands_used=None):
        super().__init__()

        self.path = Path(path)
        self.transform = transform
        self.patch_size = patch_size
        self.bands_used = bands_used

        assert self.path.exists(), "Dataset path doesn't exist"
        self.mean = np.load(str(self.path / "mean.npy"))
        self.std = np.load(str(self.path / "std.npy"))

        self.samples = []
        self.image_path = self.path / "images"
        self.masks_path = self.path / "masks"

        assert self.image_path.exists(), "Images folder missing"
        assert self.masks_path.exists(), "Masks folder missing"

        if mode is None:
            self.band_files = sorted(glob.glob(str(self.image_path / "*.npy")))
            self.mask_files = sorted(glob.glob(str(self.masks_path / "*.npy")))
        else:
            assert mode_path is not None

            mode_path = Path(mode_path)
            with open(mode_path, "r") as f:
                split = json.load(f)

            if mode.lower() == 'train':
                ids = split['train']
            elif mode.lower() == 'val':
                ids = split['val']
            else:
                raise ValueError("Mode can only be None, train, val")

            self.band_files = sorted([self.image_path / f"img_{i}.npy" for i in ids])
            self.mask_files = sorted([self.masks_path / f"mask_{i}.npy" for i in ids])

        assert len(self.band_files) == len(self.mask_files)

        for b, m in zip(self.band_files, self.mask_files):
            assert Path(b).name.replace("img", "mask") == Path(m).name

        img = np.load(self.band_files[0])
        _, H, W = img.shape

        stride = int(patch_size * overlap)

        for img_path, mask_path in zip(self.band_files, self.mask_files):
            for y in range(0, H - patch_size + 1, stride):
                for x in range(0, W - patch_size + 1, stride):
                    self.samples.append((img_path, mask_path, x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, x, y = self.samples[idx]

        img = np.load(img_path)
        mask = np.load(mask_path)

        img_patch = img[:, y:y + self.patch_size, x:x + self.patch_size]
        mask_patch = mask[:, y:y + self.patch_size, x:x + self.patch_size]

        img_patch = torch.from_numpy(img_patch).float()
        mask_patch = torch.from_numpy(mask_patch).float()

        if self.transform:
            img_patch, mask_patch = self.transform(img_patch, mask_patch, self.mean, self.std)

        img_patch = img_patch[self.bands_used] if self.bands_used is not None else img_patch

        return img_patch, mask_patch


if __name__ == "__main__":
    from pathlib import Path
    import numpy as np
    from tqdm import tqdm

    ROOT = Path(__file__).resolve().parent.parent.parent
    DATASET = ROOT / "dataset"

    # =========================
    # Helper: collect stats
    # =========================
    def analyze_dataset(dataset, title, extreme_thresh=None):
        print(f"\n===== {title} =====")

        channel_vals = None
        extreme_samples = []

        for i in tqdm(range(len(dataset))):
            img, _ = dataset[i]
            img = img.numpy()

            if channel_vals is None:
                channel_vals = [[] for _ in range(img.shape[0])]

            # collect per-channel values
            for c in range(img.shape[0]):
                channel_vals[c].append(img[c].reshape(-1))

            # optional extreme detection
            if extreme_thresh is not None:
                max_val = np.abs(img).max()
                if max_val > extreme_thresh:
                    extreme_samples.append((i, max_val))

        # concat all values
        for c in range(len(channel_vals)):
            channel_vals[c] = np.concatenate(channel_vals[c])

        # print stats
        for c, vals in enumerate(channel_vals):
            p = np.percentile(vals, [0, 1, 50, 99, 99.9, 100])

            print(f"\nChannel {c}")
            print("min      :", p[0])
            print("1%       :", p[1])
            print("median   :", p[2])
            print("99%      :", p[3])
            print("99.9%    :", p[4])
            print("max      :", p[5])

        # extreme summary
        if extreme_thresh is not None:
            print("\nExtreme samples:", len(extreme_samples))
            if extreme_samples:
                print("Top 5 extremes:")
                print(sorted(extreme_samples, key=lambda x: -x[1])[:5])

    # =========================
    # RAW DATA ANALYSIS
    # =========================
    raw_dataset = GlacierDataset(
        path=DATASET,
        patch_size=128,
        overlap=0.5,
        mode=None,
        transform=None,
        bands_used=None
    )
    print(len(raw_dataset))

    analyze_dataset(raw_dataset, "RAW DATA")

    # =========================
    # AFTER TRANSFORM ANALYSIS
    # =========================
    transformed_dataset = GlacierDataset(
        path=DATASET,
        patch_size=512,
        overlap=1,
        mode=None,
        transform=GlacierTransform(),
        bands_used=None
    )

    analyze_dataset(transformed_dataset, "AFTER TRANSFORM", extreme_thresh=20)

    # =========================
    # CLASS IMBALANCE (SEPARATE PASS)
    # =========================

    print("\n===== CLASS IMBALANCE (FAST PASS) =====")

    dataset = GlacierDataset(
        path=DATASET,
        patch_size=512,
        overlap=1,
        mode=None,
        transform=None,  # IMPORTANT → raw masks
        bands_used=None
    )

    total_pixels = 0
    total_fg = 0
    empty_patches = 0

    for i in tqdm(range(len(dataset))):
        _, mask = dataset[i]
        mask = mask.numpy()

        fg = mask.sum()
        total_fg += fg
        total_pixels += mask.size

        if fg == 0:
            empty_patches += 1

    # ---- RESULTS ----
    print("Total pixels      :", total_pixels)
    print("Foreground pixels :", int(total_fg))
    print("Background pixels :", int(total_pixels - total_fg))

    fg_ratio = total_fg / total_pixels
    print("Foreground ratio  :", fg_ratio)

    print("Empty patches     :", empty_patches)
    print("Total patches     :", len(dataset))
    print("Empty patch ratio :", empty_patches / len(dataset))
