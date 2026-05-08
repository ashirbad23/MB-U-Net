# src/dataset/test_dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import glob

from utils.transform import GlacierTransform


class GlacierTestDataset(Dataset):
    """
    Inference Dataset

    Loads FULL AOI images and converts them into patches
    while preserving reconstruction metadata.

    Returns:
    {
        "image": Tensor [C,H,W],
        "mask": Tensor [1,H,W],
        "x": int,
        "y": int,
        "image_id": str,
        "orig_h": int,
        "orig_w": int
    }
    """

    def __init__(
            self,
            path: Path,
            transform=None,
            patch_size=128,
            bands_used=None
    ):
        super().__init__()

        self.path = Path(path)

        self.transform = transform
        self.patch_size = patch_size
        self.bands_used = bands_used

        # =========================
        # Paths
        # =========================

        self.image_path = self.path / "images_test"
        self.mask_path = self.path / "masks_test"

        assert self.image_path.exists(), "Images folder missing"
        assert self.mask_path.exists(), "Masks folder missing"

        # =========================
        # Mean / Std
        # =========================

        self.mean = np.load(str(self.path / "mean.npy"))
        self.std = np.load(str(self.path / "std.npy"))

        # =========================
        # File collection
        # =========================

        self.image_files = sorted(
            glob.glob(str(self.image_path / "*.npy"))
        )

        self.mask_files = sorted(
            glob.glob(str(self.mask_path / "*.npy"))
        )

        assert len(self.image_files) == len(self.mask_files)

        # =========================
        # Patch indexing
        # =========================

        self.samples = []

        for img_path, mask_path in zip(self.image_files, self.mask_files):

            img_name = Path(img_path).stem

            img = np.load(img_path)

            _, H, W = img.shape

            # -------------------------
            # NO OVERLAP IN TESTING
            # -------------------------

            stride = self.patch_size

            ys = list(range(0, H, stride))
            xs = list(range(0, W, stride))

            # -------------------------
            # Border alignment
            # Ensures last patch
            # always reaches image end
            # -------------------------

            if ys[-1] + self.patch_size > H:
                ys[-1] = H - self.patch_size

            if xs[-1] + self.patch_size > W:
                xs[-1] = W - self.patch_size

            # remove duplicates
            ys = sorted(list(set(ys)))
            xs = sorted(list(set(xs)))

            for y in ys:
                for x in xs:

                    self.samples.append({
                        "img_path": img_path,
                        "mask_path": mask_path,
                        "x": x,
                        "y": y,
                        "image_id": img_name,
                        "orig_h": H,
                        "orig_w": W
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        img_path = sample["img_path"]
        mask_path = sample["mask_path"]

        x = sample["x"]
        y = sample["y"]

        image_id = sample["image_id"]

        orig_h = sample["orig_h"]
        orig_w = sample["orig_w"]

        # =========================
        # Load arrays
        # =========================

        img = np.load(img_path)
        mask = np.load(mask_path)

        # =========================
        # Extract patch
        # =========================

        img_patch = img[
            :,
            y:y + self.patch_size,
            x:x + self.patch_size
        ]

        mask_patch = mask[
            :,
            y:y + self.patch_size,
            x:x + self.patch_size
        ]

        # =========================
        # Tensor conversion
        # =========================

        img_patch = torch.from_numpy(img_patch).float()
        mask_patch = torch.from_numpy(mask_patch).float()

        # =========================
        # SAME preprocessing
        # as training
        # =========================

        if self.transform is not None:

            img_patch, mask_patch = self.transform(
                img_patch,
                mask_patch,
                self.mean,
                self.std
            )

        # =========================
        # Band selection
        # =========================

        if self.bands_used is not None:
            img_patch = img_patch[self.bands_used]

        return {
            "image": img_patch,
            "mask": mask_patch,
            "x": x,
            "y": y,
            "image_id": image_id,
            "orig_h": orig_h,
            "orig_w": orig_w
        }


if __name__ == "__main__":

    from pathlib import Path

    ROOT = Path(__file__).resolve().parent.parent.parent

    DATASET = ROOT / "dataset"

    dataset = GlacierTestDataset(
        path=DATASET,
        patch_size=128,
        transform=GlacierTransform(
            normalize=True,
            use_rotation=False
        ),
        bands_used=list(range(10))
    )

    print("Total inference patches:", len(dataset))

    sample = dataset[0]

    print("\nSample keys:")
    print(sample.keys())

    print("\nImage shape:")
    print(sample["image"].shape)

    print("\nMask shape:")
    print(sample["mask"].shape)

    print("\nCoords:")
    print(sample["x"], sample["y"])

    print("\nImage ID:")
    print(sample["image_id"])