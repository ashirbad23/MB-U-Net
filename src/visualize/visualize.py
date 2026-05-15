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

def get_visualization_dir(exp_dir, dataset_name=None):
    """
    If dataset_name is None:
        runs/exp_xxx/visualizations/common

    If dataset_name = "internal":
        runs/exp_xxx/visualizations/internal

    If dataset_name = "external":
        runs/exp_xxx/visualizations/external
    """
    if dataset_name is None:
        vis_dir = exp_dir / "visualizations"
    else:
        vis_dir = exp_dir / "visualizations" / dataset_name

    vis_dir.mkdir(parents=True, exist_ok=True)
    return vis_dir


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


# =====================================================
# METRIC COMPARISON
# =====================================================

def plot_metric_comparison(exp_dir):
    """
    Compare internal vs external metrics using seaborn.
    """
    # Common figure comparing both datasets
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

    vis_dir = get_visualization_dir(exp_dir, dataset_name=None)

    plt.savefig(
        vis_dir / "metric_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# =====================================================
# GLOBAL BAND IMPORTANCE
# =====================================================

def plot_global_band_importance(exp_dir, dataset_name="internal"):
    explain_dir = exp_dir / "explain" / dataset_name

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

    vis_dir = get_visualization_dir(
        exp_dir,
        dataset_name=dataset_name
    )

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


def save_selected_examples(exp_dir, dataset_root, dataset_name="internal", top_k=5, refer=False):
    """
    Save comprehensive qualitative examples.

    Output:
        visualizations/<dataset_name>/selected_examples/<image_id>/
            rgb.png
            gt_boundary_on_rgb.png
            pred_boundary_on_rgb.png
            gt_vs_pred_overlay.png
            probability_map.png
            error_map.png
            top_attribution.png
            attribution_overlay.png
            summary_panel.png
    """
    import cv2

    results_dir = exp_dir / f"test_results_{dataset_name}"
    explain_dir = exp_dir / "explain" / dataset_name

    base_dir = (
        get_visualization_dir(
            exp_dir,
            dataset_name=dataset_name
        )
        / "selected_examples"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = results_dir / "all_image_metrics.csv" if refer is None else explain_dir / "top_k_selected.csv"

    if not metrics_csv.exists():
        raise FileNotFoundError(metrics_csv)

    df = pd.read_csv(metrics_csv)
    df = (
        df.sort_values("MCC", ascending=False)
        .reset_index(drop=True)
        .head(top_k)
    )

    dataset_root = Path(dataset_root)

    for _, row in df.iterrows():

        image_id = row["image_id"]
        mcc = float(row["MCC"])

        print(f"Creating selected example for {image_id}")

        image_dir = base_dir / image_id
        image_dir.mkdir(parents=True, exist_ok=True)

        # ==================================================
        # LOAD ORIGINAL MULTI-BAND IMAGE
        # ==================================================

        if dataset_name == "internal":
            image_path = dataset_root / "images" / f"{image_id}.npy"
        else:
            image_path = dataset_root / "images_test" / f"{image_id}.npy"

        if not image_path.exists():
            print(f"Skipping {image_id}: missing {image_path}")
            continue

        image = np.load(image_path)  # [C, H, W]

        # Build RGB from Blue, Green, Red -> RGB
        rgb = np.stack(
            [
                image[2],  # Red
                image[1],  # Green
                image[0],  # Blue
            ],
            axis=-1
        ).astype(np.float32)

        # Percentile stretch
        for c in range(3):
            p2, p98 = np.percentile(rgb[..., c], (2, 98))
            rgb[..., c] = np.clip(rgb[..., c], p2, p98)
            rgb[..., c] = (
                rgb[..., c] - p2
            ) / (p98 - p2 + 1e-8)

        rgb = (rgb * 255).astype(np.uint8)

        # ==================================================
        # LOAD GT, PREDICTION, PROBABILITY
        # ==================================================

        gt_path = results_dir / "gt" / f"gt_{image_id}.png"
        pred_path = results_dir / "preds" / f"pred_{image_id}.png"
        prob_path = results_dir / "probs" / f"prob_{image_id}.npy"

        if not (
            gt_path.exists()
            and pred_path.exists()
            and prob_path.exists()
        ):
            print(f"Skipping {image_id}: missing GT/PRED/PROB")
            continue

        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        prob = np.load(prob_path)

        gt = (gt > 127).astype(np.uint8)
        pred = (pred > 127).astype(np.uint8)

        # ==================================================
        # CONTOURS
        # ==================================================

        gt_contours, _ = cv2.findContours(
            gt,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        pred_contours, _ = cv2.findContours(
            pred,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # GT boundary
        gt_overlay = rgb.copy()
        cv2.drawContours(
            gt_overlay,
            gt_contours,
            -1,
            (0, 255, 0),
            2
        )

        # Prediction boundary
        pred_overlay = rgb.copy()
        cv2.drawContours(
            pred_overlay,
            pred_contours,
            -1,
            (255, 0, 0),
            2
        )

        # Combined GT + Prediction
        combined_overlay = rgb.copy()
        cv2.drawContours(
            combined_overlay,
            gt_contours,
            -1,
            (0, 255, 0),
            2
        )
        cv2.drawContours(
            combined_overlay,
            pred_contours,
            -1,
            (255, 0, 0),
            2
        )

        # ==================================================
        # ERROR MAP OVERLAY ON RGB
        # FP = Red, FN = Blue
        # ==================================================

        error_overlay = rgb.copy()

        fp = (pred == 1) & (gt == 0)
        fn = (pred == 0) & (gt == 1)

        fp_mask = np.zeros_like(gt, dtype=np.uint8)
        fn_mask = np.zeros_like(gt, dtype=np.uint8)

        fp_mask[fp] = 1
        fn_mask[fn] = 1

        fp_contours, _ = cv2.findContours(
            fp_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        fn_contours, _ = cv2.findContours(
            fn_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            error_overlay,
            fp_contours,
            -1,
            (255, 0, 0),   # red
            2
        )

        cv2.drawContours(
            error_overlay,
            fn_contours,
            -1,
            (0, 0, 255),   # blue
            2
        )

        # ==================================================
        # LOAD TOP INTEGRATED GRADIENTS HEATMAP
        # ==================================================

        top_band = -1
        attr_uint8 = np.zeros_like(gt, dtype=np.uint8)
        attribution_overlay = rgb.copy()

        band_csv = (
            explain_dir
            / image_id
            / "band_importance.csv"
        )

        if band_csv.exists():

            band_df = pd.read_csv(band_csv)
            top_band = int(
                band_df.iloc[0]["band_index"]
            )

            attr_path = (
                explain_dir
                / image_id
                / "heatmaps"
                / f"attr_band_{top_band:02d}.npy"
            )

            if attr_path.exists():
                attr = np.load(attr_path)
                attr = np.log1p(np.abs(attr))

                vmax = np.percentile(attr, 99)
                attr = np.clip(attr, 0, vmax)

                attr_uint8 = (
                    attr / (vmax + 1e-8) * 255
                ).astype(np.uint8)

                if attr_uint8.shape != gt.shape:
                    attr_uint8 = cv2.resize(
                        attr_uint8,
                        (gt.shape[1], gt.shape[0])
                    )

                heat = cv2.applyColorMap(
                    attr_uint8,
                    cv2.COLORMAP_INFERNO
                )

                heat = cv2.cvtColor(
                    heat,
                    cv2.COLOR_BGR2RGB
                )

                attribution_overlay = cv2.addWeighted(
                    rgb,
                    0.65,
                    heat,
                    0.35,
                    0
                )

        # ==================================================
        # SAVE INDIVIDUAL IMAGES
        # ==================================================

        plt.imsave(image_dir / "rgb.png", rgb)
        plt.imsave(image_dir / "gt_boundary_on_rgb.png", gt_overlay)
        plt.imsave(image_dir / "pred_boundary_on_rgb.png", pred_overlay)
        plt.imsave(image_dir / "gt_vs_pred_overlay.png", combined_overlay)
        plt.imsave(image_dir / "probability_map.png", prob, cmap="turbo")
        plt.imsave(image_dir / "error_map.png", error_overlay)
        plt.imsave(image_dir / "top_attribution.png", attr_uint8, cmap="inferno")
        plt.imsave(image_dir / "attribution_overlay.png", attribution_overlay)

        # ==================================================
        # SUMMARY PANEL (2 x 4)
        # ==================================================

        fig, axes = plt.subplots(
            2,
            4,
            figsize=(24, 12)
        )

        axes = axes.ravel()

        axes[0].imshow(rgb)
        axes[0].set_title("RGB Composite")

        axes[1].imshow(gt_overlay)
        axes[1].set_title("GT Boundary")

        axes[2].imshow(pred_overlay)
        axes[2].set_title("Prediction Boundary")

        axes[3].imshow(combined_overlay)
        axes[3].set_title("GT (Green) vs Pred (Red)")

        axes[4].imshow(prob, cmap="turbo")
        axes[4].set_title("Probability Map")

        axes[5].imshow(error_overlay)
        axes[5].set_title("Error Map (FP Red, FN Blue)")

        axes[6].imshow(attr_uint8, cmap="inferno")
        axes[6].set_title(f"Top IG Band {top_band}")

        axes[7].imshow(attribution_overlay)
        axes[7].set_title("IG Overlay on RGB")

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"{image_id} | MCC = {mcc:.4f}",
            fontsize=22
        )

        plt.tight_layout()

        plt.savefig(
            image_dir / "summary_panel.png",
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
    dataset_name = config.get("explain_dataset", "internal").lower()

    plot_global_band_importance(
        exp_dir,
        dataset_name=dataset_name
    )

    print("Generating selected examples...")
    save_selected_examples(
        exp_dir,
        dataset_root=config['dataset'],
        dataset_name=dataset_name,
        top_k=config.get("top_k", 5),
        refer=True if config["explain_reference_exp"] is not None else False
    )

    print("Visualization completed successfully.")
