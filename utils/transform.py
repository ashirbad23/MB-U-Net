import torch
import random


class GlacierTransform:
    def __init__(self,
                 normalize=True,
                 use_rotation=True,
                 use_radiometric=True):
        self.normalize = normalize
        self.use_rotation = use_rotation
        self.use_radiometric = use_radiometric
        self.skip_channels = [8, 9]

    def normalize_img_mask(self, img, mean, std):
        """
        img: (C, H, W)
        """
        for c in range(img.shape[0]):

            # skip sin/cos
            if c in self.skip_channels:
                continue

            # ---- PRE-CLIP (THIS IS THE FIX) ----
            if c in [11, 16]:
                img[c] = torch.clamp(img[c], -1, 1)

            elif c in [10, 12, 13, 14]:
                img[c] = torch.clamp(img[c], -0.1, 0.1)

            elif c == 15:
                img[c] = torch.clamp(img[c], -0.01, 0.01)

            # ---- NORMALIZE ----
            img[c] = (img[c] - mean[c]) / std[c]

        return img

    def rotate(self, img, mask):
        """
        Random 0°, 90°, 180°, 270°
        """
        k = random.randint(0, 3)

        if k > 0:
            img = torch.rot90(img, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])

        return img, mask

    def radiometric_augment(self, img):
        """
        Apply small physically plausible intensity perturbations
        to all channels except aspect sin/cos.
        """

        # Global brightness scaling (±10%)
        scale = random.uniform(0.9, 1.1)

        # Global additive shift (±0.05 normalized units)
        shift = random.uniform(-0.05, 0.05)

        for c in range(img.shape[0]):
            if c in self.skip_channels:
                continue

            img[c] = img[c] * scale + shift

            # Small Gaussian noise
            noise = torch.randn_like(img[c]) * 0.01
            img[c] = img[c] + noise

        return img

    def __call__(self, img, mask, mean, std):
        """
        mean, std: numpy arrays of shape (16,)
        """
        mean = torch.tensor(mean).float()
        std = torch.tensor(std).float()

        # avoid divide by zero
        std[std < 0.1] = 0.1

        # Normalize
        if self.normalize:
            img = self.normalize_img_mask(img, mean, std)

            # ---- FINAL SAFETY CLAMP ----
            img = torch.clamp(img, -5, 5)

        # Augment
        if self.use_rotation:
            img, mask = self.rotate(img, mask)

        if self.use_radiometric:
            img = self.radiometric_augment(img)

        return img, mask
