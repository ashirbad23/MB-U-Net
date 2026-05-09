import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.test_dataset import GlacierTestDataset
from src.model.StandardUNet import MultiBranchUNet

from utils.transform import GlacierTransform
from utils.stitch import PatchStitcher
from utils.metrics import compute_metrics_direct, find_best_threshold


# =====================================================
# HELPERS
# =====================================================

def get_latest_experiment(base_dir):

    if not os.path.exists(base_dir):
        return None

    exps = [
        d for d in os.listdir(base_dir)
        if d.startswith("exp_")
    ]

    if len(exps) == 0:
        return None

    exps = sorted(exps)

    return os.path.join(base_dir, exps[-1])


def setup_logger(exp_dir):

    log_file = os.path.join(
        exp_dir,
        "test_metrics.log"
    )

    logger = logging.getLogger("test_logger")

    logger.setLevel(logging.INFO)

    if not logger.handlers:

        fh = logging.FileHandler(log_file)
        ch = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s - %(message)s"
        )

        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# =====================================================
# TEST
# =====================================================

@torch.no_grad()
def test(config):

    device = torch.device(config["device"])

    # =================================================
    # EXPERIMENT
    # =================================================

    exp_dir = (
            config.get("test_exp")
            or get_latest_experiment(
        config["run_base_dir"]
    )
    )

    if exp_dir is None:
        raise ValueError("No experiment found")

    print(f"Using experiment: {exp_dir}")

    logger = setup_logger(exp_dir)

    # =================================================
    # CHECKPOINT
    # =================================================

    ckpt_name = config.get(
        "ckpt_type",
        "best"
    )

    ckpt_path = os.path.join(
        exp_dir,
        f"{ckpt_name}.pth"
    )

    if not os.path.exists(ckpt_path):
        raise ValueError(
            f"{ckpt_name}.pth not found"
        )

    logger.info(
        f"Loading checkpoint: {ckpt_path}"
    )

    ckpt = torch.load(
        ckpt_path,
        map_location=device
    )

    threshold = config.get(
        "test_threshold",
        ckpt["extra"]["best_threshold"]
    )

    logger.info(
        f"Best Threshold: {threshold:.3f}"
    )

    # =================================================
    # DATA
    # =================================================

    transform = GlacierTransform(
        normalize=True,
        use_rotation=False
    )

    dataset = GlacierTestDataset(
        path=Path(config["dataset"]),
        patch_size=config["patch_size"],
        bands_used=config["bands_used"],
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    logger.info(
        f"Total patches: {len(dataset)}"
    )

    # =================================================
    # MODEL
    # =================================================

    model = MultiBranchUNet(
        ch_head=config["channel_head"],
        in_ch=config["in_channels"],
        out_ch=config["out_channels"],
        num_levels=config["num_levels"],
        dropout=config["dropout"],
        bands_used=config["bands_used"],
        attn=config["use_attention"]
    ).to(device)

    model.load_state_dict(
        ckpt["model"]
    )

    model.eval()

    logger.info("Model loaded successfully")

    # =================================================
    # SAVE DIRS
    # =================================================

    save_dir = os.path.join(
        exp_dir,
        "test_results"
    )

    pred_dir = os.path.join(
        save_dir,
        "preds"
    )

    prob_dir = os.path.join(
        save_dir,
        "probs"
    )

    gt_dir = os.path.join(
        save_dir,
        "gt"
    )

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(prob_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    # =================================================
    # STITCHERS
    # =================================================

    stitchers = {}
    gt_maps = {}

    # =================================================
    # INFERENCE
    # =================================================

    logger.info("Starting inference...")

    for batch in tqdm(
            loader,
            dynamic_ncols=True
    ):

        images = batch["image"].to(device)

        masks = batch["mask"]

        xs = batch["x"]
        ys = batch["y"]

        image_ids = batch["image_id"]

        orig_hs = batch["orig_h"]
        orig_ws = batch["orig_w"]

        # ---------------------------------------------
        # FORWARD
        # ---------------------------------------------

        with autocast(
                enabled=(device.type == "cuda")
        ):

            preds, _ = model(images)

            probs = torch.sigmoid(preds)

        probs = probs.cpu()

        # =============================================
        # PATCH RECONSTRUCTION
        # =============================================

        B = probs.shape[0]

        for i in range(B):

            image_id = image_ids[i]

            x = int(xs[i])
            y = int(ys[i])

            orig_h = int(orig_hs[i])
            orig_w = int(orig_ws[i])

            # -----------------------------------------
            # CREATE STITCHER
            # -----------------------------------------

            if image_id not in stitchers:

                stitchers[image_id] = PatchStitcher(
                    full_h=orig_h,
                    full_w=orig_w
                )

                gt_maps[image_id] = np.zeros(
                    (orig_h, orig_w),
                    dtype=np.uint8
                )

            # -----------------------------------------
            # ADD PATCH
            # -----------------------------------------

            stitchers[image_id].add_patch(
                probs[i],
                x=x,
                y=y
            )

            # -----------------------------------------
            # GT PATCH
            # -----------------------------------------

            gt_patch = (
                masks[i]
                .squeeze(0)
                .numpy()
            )

            h, w = gt_patch.shape

            gt_maps[image_id][
                y:y+h,
                x:x+w
            ] = gt_patch

    # =================================================
    # FINAL RECONSTRUCTION
    # =================================================

    logger.info("Saving outputs...")

    total_metrics = {
        "IoU": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "MCC": []
    }

    for image_id in stitchers.keys():

        # ---------------------------------------------
        # FULL PROBABILITY MAP
        # ---------------------------------------------

        full_probs = stitchers[
            image_id
        ].get_full_probs()

        # ---------------------------------------------
        # FINAL BINARY MASK
        # ---------------------------------------------

        pred_mask = stitchers[
            image_id
        ].get_binary_mask(
            threshold=threshold
        )

        gt_mask = gt_maps[image_id]

        # ---------------------------------------------
        # METRICS
        # ---------------------------------------------

        pred_tensor = torch.from_numpy(
            pred_mask
        ).unsqueeze(0).unsqueeze(0).float()

        gt_tensor = torch.from_numpy(
            gt_mask
        ).unsqueeze(0).unsqueeze(0).float()

        metrics = compute_metrics_direct(
            pred_tensor,
            gt_tensor
        )

        for k in total_metrics:
            total_metrics[k].append(metrics[k])

        logger.info(
            f"{image_id} | "
            f"MCC: {metrics['MCC']:.4f} | "
            f"F1: {metrics['F1']:.4f}"
        )

        # ---------------------------------------------
        # SAVE PROBS
        # ---------------------------------------------

        np.save(
            os.path.join(
                prob_dir,
                f"prob_{image_id}.npy"
            ),
            full_probs.astype(np.float32)
        )

        # ---------------------------------------------
        # SAVE PRED
        # ---------------------------------------------

        cv2.imwrite(
            os.path.join(
                pred_dir,
                f"pred_{image_id}.png"
            ),
            pred_mask * 255
        )

        # ---------------------------------------------
        # SAVE GT
        # ---------------------------------------------

        cv2.imwrite(
            os.path.join(
                gt_dir,
                f"gt_{image_id}.png"
            ),
            gt_mask * 255
        )

    # =================================================
    # FINAL METRICS
    # =================================================

    logger.info("========== FINAL RESULTS ==========")

    final_results = {}

    for k in total_metrics:

        avg = np.mean(total_metrics[k])

        final_results[k] = float(avg)

        logger.info(f"{k}: {avg:.4f}")

    # =================================================
    # SAVE JSON
    # =================================================

    log_path = os.path.join(
        exp_dir,
        "test_log.json"
    )

    with open(log_path, "w") as f:

        json.dump(
            final_results,
            f,
            indent=4
        )

    logger.info(
        f"Saved test log to {log_path}"
    )
