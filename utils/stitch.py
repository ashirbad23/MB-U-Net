# src/inference/stitch.py

import numpy as np
import torch


class PatchStitcher:
    """
    Reconstructs full probability maps from patch predictions.

    Works with:
    - overlapping patches
    - non-overlapping patches
    - arbitrary image sizes

    IMPORTANT:
    Uses probabilities, NOT binary masks.
    """

    def __init__(self, full_h, full_w):

        self.full_h = full_h
        self.full_w = full_w

        # -----------------------------------
        # Accumulates probabilities
        # -----------------------------------

        self.prob_map = np.zeros(
            (full_h, full_w),
            dtype=np.float32
        )

        # -----------------------------------
        # Counts overlapping contributions
        # -----------------------------------

        self.counter_map = np.zeros(
            (full_h, full_w),
            dtype=np.float32
        )

    def add_patch(self, patch_probs, x, y):
        """
        Add predicted probability patch.

        Args:
            patch_probs:
                shape [H,W] OR [1,H,W]

            x:
                top-left x coordinate

            y:
                top-left y coordinate
        """

        # -----------------------------------
        # Torch -> numpy
        # -----------------------------------

        if torch.is_tensor(patch_probs):
            patch_probs = (
                patch_probs
                .detach()
                .cpu()
                .numpy()
            )

        # -----------------------------------
        # Remove channel dimension if needed
        # -----------------------------------

        if patch_probs.ndim == 3:
            patch_probs = patch_probs.squeeze(0)

        patch_h, patch_w = patch_probs.shape

        # -----------------------------------
        # Add probabilities
        # -----------------------------------

        self.prob_map[
            y:y + patch_h,
            x:x + patch_w
        ] += patch_probs

        # -----------------------------------
        # Count contributions
        # -----------------------------------

        self.counter_map[
            y:y + patch_h,
            x:x + patch_w
        ] += 1.0

    def get_full_probs(self):
        """
        Returns reconstructed probability map.

        Shape:
            [H,W]
        """

        counter = np.clip(self.counter_map, 1e-6, None)

        probs = self.prob_map / counter

        return probs.astype(np.float32)

    def get_binary_mask(self, threshold=0.5):
        """
        Returns thresholded binary mask.

        Shape:
            [H,W]
        """

        probs = self.get_full_probs()

        mask = (probs > threshold).astype(np.uint8)

        return mask

    def reset(self):
        """
        Clears accumulated maps.
        Useful for multiple AOIs.
        """

        self.prob_map.fill(0)
        self.counter_map.fill(0)


if __name__ == "__main__":

    # -----------------------------------
    # Example usage
    # -----------------------------------

    stitcher = PatchStitcher(
        full_h=512,
        full_w=512
    )

    # Fake patch prediction
    patch = np.random.rand(128, 128).astype(np.float32)

    # Add patches
    stitcher.add_patch(patch, x=0, y=0)
    stitcher.add_patch(patch, x=128, y=0)

    probs = stitcher.get_full_probs()
    mask = stitcher.get_binary_mask(threshold=0.5)

    print("Prob map shape:", probs.shape)
    print("Mask shape:", mask.shape)

    print("Prob range:",
          probs.min(),
          probs.max())

    print("Mask unique values:",
          np.unique(mask))