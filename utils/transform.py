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
            if c in self.skip_channels:
                continue
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
        std[std < 1e-6] = 1e-6

        # Normalize
        if self.normalize:
            img = self.normalize_img_mask(img, mean, std)

        # Augment
        if self.use_rotation:
            img, mask = self.rotate(img, mask)

        return img, mask
