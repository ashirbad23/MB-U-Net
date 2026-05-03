import torch
import random


class GlacierTransform:
    def __init__(self, normalize=True, use_rotation=True):
        self.normalize = normalize

        self.use_rotation = use_rotation

        # channels to skip (sin/cos)
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

        return img, mask
