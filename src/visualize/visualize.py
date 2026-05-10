# src/visualization/visualization.py

"""
Visualization pipeline.

Phase 1 (implemented):
1. Internal vs External metric comparison
2. Global band importance (averaged across explained images)

Inputs:
- runs/exp_xxx/test_results_internal/test_log.json
- runs/exp_xxx/test_results_external/test_log.json
- runs/exp_xxx/explain/*/band_importance.csv

Outputs:
- runs/exp_xxx/visualizations/metric_comparison.png
- runs/exp_xxx/visualizations/global_band_importance.png
- runs/exp_xxx/visualizations/global_band_importance.csv
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns

sns.set_theme(style="darkgrid", context="talk")


# =====================================================
# HELPERS
# =====================================================

def load_test_metrics(exp_dir, dataset_name):
    """
    dataset_name: 'internal' or 'external'
    """
    log_path = (
            exp_dir
            / f"test_results_{dataset_name}"
            / "test_log.json"
    )

    if not log_path.exists():
        raise FileNotFoundError(log_path)

    with open(log_path, "r") as f:
        metrics = json.load(f)

    return metrics


def get_visualization_dir(exp_dir):
    vis_dir = exp_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    return vis_dir


# =====================================================
# METRIC COMPARISON
# =====================================================

def plot_metric_comparison(exp_dir):
    """
    Compare internal vs external metrics using seaborn.
    """
    internal = load_test_metrics(exp_dir, "internal")
    external = load_test_metrics(exp_dir, "external")

    metrics = ["MCC", "IoU", "Precision", "Recall", "F1"]

    rows = []

    for metric in metrics:
        rows.append({
            "Metric": metric,
            "Score": internal[metric],
            "Dataset": "Internal"
        })
        rows.append({
            "Metric": metric,
            "Score": external[metric],
            "Dataset": "External"
        })

    df = pd.DataFrame(rows)

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=df,
        x="Metric",
        y="Score",
        hue="Dataset"
    )

    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Internal vs External Test Performance")
    plt.tight_layout()

    vis_dir = get_visualization_dir(exp_dir)

    plt.savefig(
        vis_dir / "metric_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# =====================================================
# GLOBAL BAND IMPORTANCE
# =====================================================

def plot_global_band_importance(exp_dir):
    """
    Average band importance across all explained images.
    """
    explain_dir = exp_dir / "explain"

    csv_paths = list(
        explain_dir.glob("*/band_importance.csv")
    )

    if len(csv_paths) == 0:
        raise FileNotFoundError(
            "No band_importance.csv files found."
        )

    dfs = [pd.read_csv(p) for p in csv_paths]

    combined = pd.concat(dfs, ignore_index=True)

    global_df = (
        combined
        .groupby("band_index", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    vis_dir = get_visualization_dir(exp_dir)

    # Save CSV
    global_df.to_csv(
        vis_dir / "global_band_importance.csv",
        index=False
    )

    # Convert band labels to strings for plotting
    global_df["band_label"] = global_df["band_index"].astype(str)

    plt.figure(figsize=(11, 5))

    sns.barplot(
        data=global_df,
        x="band_label",
        y="importance"
    )

    plt.xlabel("Band Index")
    plt.ylabel("Average Importance")
    plt.title("Global Band Importance (Integrated Gradients)")
    plt.tight_layout()

    plt.savefig(
        vis_dir / "global_band_importance.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# Add this function to src/visualization/visualization.py

def plot_segmentation_examples(exp_dir, dataset_name="internal", top_k=5):
    """
    Create 6-panel segmentation figures for the top-K images by MCC.

    Panels:
    1. Ground Truth
    2. Prediction
    3. Probability Map
    4. Error Map
    5. Top Attribution Heatmap
    6. Attribution Overlay
    """
    import cv2

    # -------------------------------------------------
    # Paths
    # -------------------------------------------------

    results_dir = exp_dir / f"test_results_{dataset_name}"
    explain_dir = exp_dir / "explain"

    vis_dir = (
            get_visualization_dir(exp_dir)
            / "segmentation_examples"
    )
    vis_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Load top-K image IDs
    # -------------------------------------------------

    metrics_csv = results_dir / "all_image_metrics.csv"

    if not metrics_csv.exists():
        raise FileNotFoundError(metrics_csv)

    df = pd.read_csv(metrics_csv)
    df = df.sort_values(
        "MCC",
        ascending=False
    ).reset_index(drop=True)

    top_df = df.head(top_k)

    # -------------------------------------------------
    # Process each image
    # -------------------------------------------------

    for _, row in top_df.iterrows():

        image_id = row["image_id"]
        mcc = row["MCC"]

        # -------------------------------
        # Load GT and prediction
        # -------------------------------

        gt_path = results_dir / "gt" / f"gt_{image_id}.png"
        pred_path = results_dir / "preds" / f"pred_{image_id}.png"
        prob_path = results_dir / "probs" / f"prob_{image_id}.npy"

        if not (gt_path.exists() and pred_path.exists() and prob_path.exists()):
            print(f"Skipping {image_id}: missing files")
            continue

        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        prob = np.load(prob_path)

        gt = (gt > 127).astype(np.uint8)
        pred = (pred > 127).astype(np.uint8)

        # -------------------------------
        # Error Map
        # -------------------------------

        # RGB image
        # TP = white
        # FP = red
        # FN = blue
        error = np.zeros(
            (gt.shape[0], gt.shape[1], 3),
            dtype=np.uint8
        )

        tp = (pred == 1) & (gt == 1)
        fp = (pred == 1) & (gt == 0)
        fn = (pred == 0) & (gt == 1)

        error[tp] = [255, 255, 255]
        error[fp] = [255, 0, 0]
        error[fn] = [0, 0, 255]

        # -------------------------------
        # Load top attribution heatmap
        # -------------------------------

        band_csv = (
                explain_dir
                / image_id
                / "band_importance.csv"
        )

        if band_csv.exists():
            band_df = pd.read_csv(band_csv)
            top_band = int(band_df.iloc[0]["band_index"])

            heatmap_path = (
                    explain_dir
                    / image_id
                    / "heatmaps"
                    / f"attr_band_{top_band:02d}.npy"
            )

            if heatmap_path.exists():
                attr = np.load(heatmap_path)
                attr = np.log1p(np.abs(attr))

                vmax = np.percentile(attr, 99)
                attr = np.clip(attr, 0, vmax)
                attr = (attr / (vmax + 1e-8) * 255).astype(np.uint8)
            else:
                attr = np.zeros_like(gt) * 255
        else:
            top_band = -1
            attr = np.zeros_like(gt) * 255

        # Ensure same size
        if attr.shape != gt.shape:
            attr = cv2.resize(
                attr,
                (gt.shape[1], gt.shape[0])
            )

        # -------------------------------
        # Attribution Overlay
        # -------------------------------

        # Convert probability map to grayscale RGB base
        prob_norm = (prob * 255).clip(0, 255).astype(np.uint8)

        base = cv2.cvtColor(
            prob_norm,
            cv2.COLOR_GRAY2RGB
        )

        # Create colored attribution heatmap
        heat = cv2.applyColorMap(
            attr,
            cv2.COLORMAP_INFERNO
        )

        heat = cv2.cvtColor(
            heat,
            cv2.COLOR_BGR2RGB
        )

        # Blend base and heatmap
        overlay = cv2.addWeighted(
            base,  # underlying probability image
            0.6,
            heat,  # attribution heatmap
            0.4,
            0
        )

        # -------------------------------
        # Plot
        # -------------------------------

        fig, axes = plt.subplots(
            2,
            3,
            figsize=(12, 8)
        )

        axes = axes.ravel()

        axes[0].imshow(gt, cmap="gray")
        axes[0].set_title("Ground Truth")

        axes[1].imshow(pred, cmap="gray")
        axes[1].set_title("Prediction")

        axes[2].imshow(prob, cmap="inferno")
        axes[2].set_title("Probability")

        axes[3].imshow(error)
        axes[3].set_title("Error Map")

        axes[4].imshow(attr, cmap="inferno")
        axes[4].set_title(f"Top Band {top_band}")

        axes[5].imshow(overlay)
        axes[5].set_title("Attribution Overlay")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"{image_id} | MCC = {mcc:.4f}",
            fontsize=16
        )

        plt.tight_layout()

        plt.savefig(
            vis_dir / f"{image_id}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()


# =====================================================
# MAIN
# =====================================================

def visualize(config):
    """
    Generate all visualization figures.
    """
    exp_dir = Path(config["visualize_exp"])

    if not exp_dir.exists():
        raise FileNotFoundError(exp_dir)

    print(f"Creating visualizations for: {exp_dir}")

    # 1. Internal vs External metric comparison
    print("Generating metric comparison...")
    plot_metric_comparison(exp_dir)

    # 2. Global band importance
    print("Generating global band importance...")
    plot_global_band_importance(exp_dir)

    print("Generating segmentation examples...")
    plot_segmentation_examples(
        exp_dir,
        dataset_name=config.get("explain_dataset", "internal"),
        top_k=config.get("top_k", 5)
    )

    print("Visualization completed successfully.")
