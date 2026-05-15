"""
Integrated Gradients helper utilities.

This module contains reusable functions for explainability.

Workflow:
1. Load top-K image IDs from all_image_metrics.csv
2. Build image_id -> dataset indices mapping
3. Load trained model
4. Create Captum IntegratedGradients object
5. Compute patch attributions
6. Stitch attributions into full-size maps
7. Compute band importance
8. Save CSVs and heatmaps
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients

from src.model.StandardUNet import MultiBranchUNet
from utils.stitch import PatchStitcher


# =====================================================
# PATH HELPERS
# =====================================================

def get_results_dir(config):
    exp_dir = Path(config["explain_exp"])

    dataset_name = config.get(
        "explain_dataset",
        "internal"
    ).lower()

    if dataset_name == "internal":
        return exp_dir / "test_results_internal"

    elif dataset_name == "external":
        return exp_dir / "test_results_external"

    else:
        raise ValueError(
            "explain_dataset must be 'internal' or 'external'"
        )


def get_explain_dir(config):
    """
    Returns:
        runs/exp_xxx/explain/internal
        or
        runs/exp_xxx/explain/external
    """
    exp_dir = Path(config["explain_exp"])

    dataset_name = config.get(
        "explain_dataset",
        "internal"
    ).lower()

    explain_dir = (
        exp_dir
        / "explain"
        / dataset_name
    )

    explain_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return explain_dir


# =====================================================
# TOP-K SELECTION
# =====================================================

def load_top_k_images(config):
    """
    Load top-K images.

    If explain_reference_exp is provided, reuse the
    top_k_selected.csv from that experiment.

    Otherwise, select top-K images from the current
    experiment's all_image_metrics.csv.
    """
    top_k = config.get("top_k", 5)

    reference_exp = config.get("explain_reference_exp", None)
    reference_dataset = config.get(
        "explain_reference_dataset",
        config.get("explain_dataset", "internal")
    )

    exp_dir = Path(config["explain_exp"])
    dataset_name = config.get(
        "explain_dataset",
        "internal"
    ).lower()

    metrics_csv = (
            exp_dir
            / f"test_results_{dataset_name}"
            / "all_image_metrics.csv"
    )

    # -------------------------------------------------
    # CASE 1: Reuse top_k_selected.csv from another experiment
    # -------------------------------------------------
    if reference_exp is not None:

        csv_path = (
            Path(reference_exp)
            / "explain"
            / reference_dataset.lower()
            / "top_k_selected.csv"
        )

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        dft = pd.read_csv(csv_path)
        df = pd.read_csv(metrics_csv)

        # Keep only top_k rows in case the source had more
        df = (
            df[df["image_id"].isin(dft["image_id"])]
            .sort_values("MCC", ascending=False)
            .reset_index(drop=True)
            .head(top_k)
        )

        print(f"Using reference image IDs from: {csv_path}")

        save_dir = get_explain_dir(config)
        df.to_csv(save_dir / "top_k_selected.csv", index=False)

        return df

    # -------------------------------------------------
    # CASE 2: Select top-K from current experiment
    # -------------------------------------------------

    if not metrics_csv.exists():
        raise FileNotFoundError(metrics_csv)

    df = pd.read_csv(metrics_csv)

    df = (
        df
        .sort_values("MCC", ascending=False)
        .reset_index(drop=True)
        .head(top_k)
    )

    # Save selected IDs for reuse
    save_dir = get_explain_dir(config)
    df.to_csv(save_dir / "top_k_selected.csv", index=False)

    return df


def build_image_index_map(dataset):
    """
    Returns:
    {
        image_id: [dataset_index_1, dataset_index_2, ...]
    }
    """
    image_to_indices = {}

    for idx, sample in enumerate(dataset):
        image_id = sample["image_id"]

        if image_id not in image_to_indices:
            image_to_indices[image_id] = []

        image_to_indices[image_id].append(idx)

    return image_to_indices


# =====================================================
# MODEL LOADING
# =====================================================

def load_explain_model(config, device):
    """
    Loads model and threshold from checkpoint.
    """
    exp_dir = Path(config["explain_exp"])

    ckpt_type = config.get(
        "ckpt_type",
        "best"
    )

    ckpt_path = exp_dir / f"{ckpt_type}.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(
        ckpt_path,
        map_location=device
    )

    model = MultiBranchUNet(
        ch_head=config["channel_head"],
        in_ch=config["in_channels"],
        out_ch=config["out_channels"],
        num_levels=config["num_levels"],
        dropout=config["dropout"],
        bands_used=config["bands_used"],
        attn=config["use_attention"]
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    threshold = ckpt["extra"]["best_threshold"]

    return model, threshold


# =====================================================
# INTEGRATED GRADIENTS
# =====================================================

def make_forward_fn(model):
    """
    Returns a tensor of shape [B] for Captum.

    For each input patch, we compute the mean
    probability over all pixels.
    """

    def forward_fn(x):
        logits, _ = model(x)              # [B, 1, H, W]
        probs = torch.sigmoid(logits)     # [B, 1, H, W]

        # Mean over channel and spatial dimensions
        # Output shape: [B]
        return probs.mean(dim=(1, 2, 3))

    return forward_fn

def create_integrated_gradients(model):
    return IntegratedGradients(
        make_forward_fn(model)
    )


def compute_patch_attribution(
        ig,
        image,
        n_steps=32
):
    """
    image: [1, C, H, W]

    Returns:
        attribution [1, C, H, W]
    """
    baseline = torch.zeros_like(image)

    attr = ig.attribute(
        image,
        baselines=baseline,
        n_steps=n_steps
    )

    return attr


# =====================================================
# ATTRIBUTION STITCHING
# =====================================================

def create_band_stitchers(
        num_bands,
        full_h,
        full_w
):
    """
    One PatchStitcher per band.
    """
    return [
        PatchStitcher(
            full_h=full_h,
            full_w=full_w
        )
        for _ in range(num_bands)
    ]


def add_attribution_patch(
        stitchers,
        attr_patch,
        x,
        y
):
    """
    attr_patch: [C, H, W] or [1, C, H, W]
    """
    if torch.is_tensor(attr_patch):
        attr_patch = (
            attr_patch
            .detach()
            .cpu()
            .numpy()
        )

    if attr_patch.ndim == 4:
        attr_patch = attr_patch.squeeze(0)

    # [C, H, W]
    for c in range(attr_patch.shape[0]):
        stitchers[c].add_patch(
            np.abs(attr_patch[c]),
            x=x,
            y=y
        )


def reconstruct_attributions(stitchers):
    """
    Returns:
        full_attr [C, H, W]
    """
    maps = []

    for stitcher in stitchers:
        maps.append(
            stitcher.get_full_probs()
        )

    return np.stack(maps, axis=0)


# =====================================================
# BAND IMPORTANCE
# =====================================================

def compute_band_importance(full_attr):
    """
    full_attr: [C, H, W]

    Returns:
        normalized importance scores [C]
    """
    scores = np.abs(full_attr).mean(
        axis=(1, 2)
    )

    if scores.sum() > 0:
        scores = scores / scores.sum()

    return scores.astype(np.float32)


# =====================================================
# SAVING
# =====================================================

def save_band_importance(
        scores,
        bands_used,
        save_dir
):
    """
    Saves:
    - band_importance.csv
    - band_importance.png
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.DataFrame({
        "band_index": bands_used,
        "importance": scores
    })

    df = df.sort_values(
        "importance",
        ascending=False
    ).reset_index(drop=True)

    df.to_csv(
        save_dir / "band_importance.csv",
        index=False
    )

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(
        [str(b) for b in df["band_index"]],
        df["importance"]
    )
    plt.xlabel("Band Index")
    plt.ylabel("Normalized Importance")
    plt.title("Band Importance")
    plt.tight_layout()
    plt.savefig(
        save_dir / "band_importance.png",
        dpi=300
    )
    plt.close()

    return df


def save_heatmaps(
        full_attr,
        bands_used,
        save_dir
):
    """
    Saves one PNG per band.
    """
    heatmap_dir = Path(save_dir) / "heatmaps"
    heatmap_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for i, band_idx in enumerate(bands_used):
        # Save raw floating-point attribution
        np.save(
            heatmap_dir / f"attr_band_{band_idx:02d}.npy",
            full_attr[i].astype(np.float32)
        )

        # Create visualization
        attr = np.log1p(np.abs(full_attr[i]))

        vmax = np.percentile(attr, 99)
        attr = np.clip(attr, 0, vmax)
        attr = attr / (vmax + 1e-8)

        plt.figure(figsize=(6, 6))
        plt.imshow(attr, cmap="inferno")
        plt.axis("off")
        plt.tight_layout()

        plt.savefig(
            heatmap_dir / f"attr_band_{band_idx:02d}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close()


def save_summary(
        summary_rows,
        config
):
    """
    Saves explain_summary.csv
    """
    explain_dir = get_explain_dir(config)

    df = pd.DataFrame(summary_rows)

    df.to_csv(
        explain_dir / "explain_summary.csv",
        index=False
    )

    return df


def save_patch_indices(
        patch_index_dict,
        config
):
    """
    Saves patch_indices.json
    """
    explain_dir = get_explain_dir(config)

    with open(
        explain_dir / "patch_indices.json",
        "w"
    ) as f:
        json.dump(
            patch_index_dict,
            f,
            indent=4
        )
