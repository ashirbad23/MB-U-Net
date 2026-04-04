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
    ROOT = Path(__file__).resolve().parent
    ROOT = ROOT.parent.parent

    DATASET = ROOT / "dataset"
    CONFIG = ROOT / "config"
    transform = GlacierTransform()
    dataset = GlacierDataset(path=DATASET,
                             patch_size=64,
                             overlap=0.5,
                             mode='train',
                             mode_path=CONFIG / "train_val_split.json",
                             transform=transform,
                             bands_used=[0, 1, 2, 3, 4, 5])
    # img, mask = dataset[1000]
    img, mask = dataset[1000]
    print(torch.unique(mask))
    print(img.shape, mask.shape)
    print(len(dataset))
    for i in range(img.shape[0]):
        print(i, img[i].min(), img[i].max())

    background = 0
    glacier = 0

    for i in range(len(dataset)):
        _, mask = dataset[i]
        if mask.sum() == 0:
            background += 1
        else:
            glacier += 1

    print("background:", background)
    print("glacier:", glacier)

