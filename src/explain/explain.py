# src/explain/explain.py

import torch
from pathlib import Path

from src.dataset.test_dataset import GlacierTestDataset
from utils.transform import GlacierTransform

from utils.ig_helper import (
    load_explain_model,
    load_top_k_images,
    build_image_index_map,
    create_integrated_gradients,
    compute_patch_attribution,
    create_band_stitchers,
    add_attribution_patch,
    reconstruct_attributions,
    compute_band_importance,
    save_band_importance,
    save_heatmaps,
    save_summary,
    save_patch_indices,
    get_explain_dir
)


def explain(config):
    """
    Explainability pipeline using Integrated Gradients.

    Steps:
    1. Load trained model.
    2. Read all_image_metrics.csv from test.py.
    3. Select top-K images by MCC.
    4. Build image_id -> dataset indices mapping.
    5. Run Integrated Gradients on each 128x128 patch.
    6. Stitch attribution maps across patches.
    7. Compute band importance.
    8. Save CSVs and heatmaps.
    """

    # =====================================================
    # CONFIG
    # =====================================================

    device = torch.device(config["device"])
    top_k = config.get("top_k", 5)

    print(f"Device: {device}")
    print(f"Top-K images: {top_k}")

    # =====================================================
    # LOAD MODEL
    # =====================================================

    model, threshold = load_explain_model(
        config,
        device
    )

    print(f"Loaded model (threshold = {threshold:.4f})")

    # =====================================================
    # LOAD TOP-K IMAGES
    # =====================================================

    top_df = load_top_k_images(config)

    print("\nSelected images:")
    print(top_df[["image_id", "MCC"]])

    # =====================================================
    # DATASET
    # =====================================================

    transform = GlacierTransform(
        normalize=True,
        use_rotation=False,
        use_radiometric=False
    )

    dataset_name = config.get(
        "explain_dataset",
        "internal"
    ).lower()

    if dataset_name == "internal":
        dataset = GlacierTestDataset(
            path=Path(config["dataset"]),
            patch_size=config["patch_size"],
            bands_used=config["bands_used"],
            transform=transform,
            mode="test",
            mode_path=config["split_path"]
        )
    elif dataset_name == "external":
        dataset = GlacierTestDataset(
            path=Path(config["dataset"]),
            patch_size=config["patch_size"],
            bands_used=config["bands_used"],
            transform=transform
        )
    else:
        raise ValueError(
            "explain_dataset must be 'internal' or 'external'"
        )

    print(f"\nLoaded {dataset_name} dataset")
    print(f"Total patches: {len(dataset)}")

    # =====================================================
    # BUILD IMAGE -> PATCH INDICES MAP
    # =====================================================

    image_index_map = build_image_index_map(dataset)

    # Save for future debugging/reference
    patch_index_dict = {}

    for _, row in top_df.iterrows():
        image_id = row["image_id"]

        patch_index_dict[image_id] = {
            "mcc": float(row["MCC"]),
            "num_patches": len(image_index_map[image_id]),
            "dataset_indices": image_index_map[image_id]
        }

    save_patch_indices(
        patch_index_dict,
        config
    )

    # =====================================================
    # CAPTUM
    # =====================================================

    ig = create_integrated_gradients(model)

    # =====================================================
    # PROCESS EACH IMAGE
    # =====================================================

    summary_rows = []

    for _, row in top_df.iterrows():

        image_id = row["image_id"]
        image_mcc = float(row["MCC"])

        print("\n" + "=" * 60)
        print(f"Explaining {image_id}")
        print(f"MCC: {image_mcc:.4f}")
        print("=" * 60)

        indices = image_index_map[image_id]

        # Get image size from first patch
        first_sample = dataset[indices[0]]

        full_h = int(first_sample["orig_h"])
        full_w = int(first_sample["orig_w"])

        num_bands = len(config["bands_used"])

        # One stitcher per band
        band_stitchers = create_band_stitchers(
            num_bands=num_bands,
            full_h=full_h,
            full_w=full_w
        )

        # -------------------------------------------------
        # PROCESS ALL PATCHES OF THIS IMAGE
        # -------------------------------------------------

        for idx in indices:

            sample = dataset[idx]

            image = sample["image"].unsqueeze(0).to(device)

            x = int(sample["x"])
            y = int(sample["y"])

            # Compute attribution for one patch
            attr = compute_patch_attribution(
                ig,
                image,
                n_steps=config.get("ig_steps", 32)
            )

            # Stitch attribution patch
            add_attribution_patch(
                band_stitchers,
                attr,
                x=x,
                y=y
            )

        # -------------------------------------------------
        # RECONSTRUCT FULL ATTRIBUTION MAPS
        # -------------------------------------------------

        full_attr = reconstruct_attributions(
            band_stitchers
        )  # [C, H, W]

        # -------------------------------------------------
        # COMPUTE BAND IMPORTANCE
        # -------------------------------------------------

        scores = compute_band_importance(
            full_attr
        )

        # -------------------------------------------------
        # SAVE RESULTS
        # -------------------------------------------------

        image_save_dir = (
            get_explain_dir(config)
            / image_id
        )

        band_df = save_band_importance(
            scores=scores,
            bands_used=config["bands_used"],
            save_dir=image_save_dir
        )

        save_heatmaps(
            full_attr=full_attr,
            bands_used=config["bands_used"],
            save_dir=image_save_dir
        )

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        best_row = band_df.iloc[0]

        summary_rows.append({
            "image_id": image_id,
            "MCC": image_mcc,
            "num_patches": len(indices),
            "most_important_band": int(best_row["band_index"]),
            "importance": float(best_row["importance"])
        })

        print(
            f"Most important band: "
            f"{int(best_row['band_index'])}"
        )

    # =====================================================
    # SAVE OVERALL SUMMARY
    # =====================================================

    save_summary(
        summary_rows,
        config
    )

    print("\nExplainability completed successfully.")