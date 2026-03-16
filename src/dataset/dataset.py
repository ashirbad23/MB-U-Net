import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import glob
import json


class GlacierDataset(Dataset):
    def __init__(self, path: Path, transform=None, patch_size=512, overlap=1, mode=None, mode_path=None):
        super().__init__()

        self.path = Path(path)
        self.transform = transform
        self.patch_size = patch_size

        assert self.path.exists(), "Dataset path doesn't exist"

        self.samples = []

        if mode is None:
            self.image_path = self.path / "images"
            self.masks_path = self.path / "masks"

            assert self.image_path.exists(), "Images folder missing"
            assert self.masks_path.exists(), "Masks folder missing"

            self.band_files = sorted(glob.glob(str(self.image_path / "*.npy")))
            self.mask_files = sorted(glob.glob(str(self.masks_path / "*.npy")))
        else:
            assert mode_path is not None

            mode_path = Path(mode_path)
            with open(mode_path, "r") as f:
                split = json.load(f)

            if mode.lower() == 'train':
                self.band_files = [Path(p) for p in split['train_bands']]
                self.mask_files = [Path(p) for p in split['train_masks']]
            elif mode.lower() == 'val':
                self.band_files = [Path(p) for p in split['val_bands']]
                self.mask_files = [Path(p) for p in split['val_masks']]
            else:
                raise ValueError("Mode can only be None, train, val")

        assert len(self.band_files) == len(self.mask_files)

        for b, m in zip(self.band_files, self.mask_files):
            assert Path(b).name.replace("img", "mask") == Path(m).name

        img = np.load(self.band_files[0])
        _, H, W = img.shape

        stride = patch_size // overlap

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
        mask_patch = torch.from_numpy(mask_patch).long()

        if self.transform:
            img_patch, mask_patch = self.transform(img_patch, mask_patch)

        return img_patch, mask_patch


if __name__ == "__main__":
    dataset = GlacierDataset(path=Path("/Glacier_Image_Segmentation_Research/Glacier-Analogy/dataset"),
                             patch_size=128,
                             overlap=2,
                             mode='train',
                             mode_path=Path("/Glacier_Image_Segmentation_Research/Glacier-Analogy/config"
                                            "/train_val_split.json"))
    img, mask = dataset[0]
    print(img.shape, mask.shape)
    print(len(dataset))

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

