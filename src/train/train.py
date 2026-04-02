import os
import json
import csv
import logging
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dataset.dataset import GlacierDataset
from utils.transform import GlacierTransform
from utils.loss import FocalDiceLoss
from utils.metrics import compute_metrics
from utils.scheduler import GradualWarmupScheduler
from utils.save_model import save_model

from src.model.model import SUnet

import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ================= EXPERIMENT MANAGEMENT =================

def create_experiment_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)

    existing = [d for d in os.listdir(base_dir) if d.startswith("exp_")]
    ids = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]

    next_id = max(ids) + 1 if ids else 1
    exp_name = f"exp_{next_id:03d}"

    exp_path = os.path.join(base_dir, exp_name)
    os.makedirs(exp_path, exist_ok=True)

    return exp_path


def get_latest_experiment(base_dir):
    if not os.path.exists(base_dir):
        return None

    exps = [d for d in os.listdir(base_dir) if d.startswith("exp_")]
    if not exps:
        return None

    exps = sorted(exps)
    return os.path.join(base_dir, exps[-1])


# ================= LOGGER =================

def get_logger(log_dir):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_dir / "train.log")
    console_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ================= CSV =================

def init_csv(log_dir):
    csv_path = Path(log_dir) / "metrics.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss",
            "IoU", "Precision", "Recall", "F1", "MCC"
        ])

    return csv_path


def log_csv(csv_path, epoch, train_loss, val_loss, metrics):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            train_loss,
            val_loss,
            metrics["IoU"],
            metrics["Precision"],
            metrics["Recall"],
            metrics["F1"],
            metrics["MCC"]
        ])


# ================= CHECKPOINT =================

def load_checkpoint(path, model, optimizer, scheduler, device):
    if not os.path.exists(path):
        return 0, -1

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    return checkpoint["epoch"] + 1, checkpoint.get("best_metric", -1)


# ================= VALIDATION =================

@torch.no_grad()
def validate(model, config, device, dataset):
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True
    )

    model.eval()
    loss_fn = FocalDiceLoss()

    total_loss = 0
    total_metrics = {k: 0 for k in ["IoU", "Precision", "Recall", "F1", "MCC"]}
    count = 0

    for bands, masks in tqdm(loader, desc="Validation", dynamic_ncols=True):
        bands = bands.to(device)
        masks = masks.to(device)

        preds, _ = model(bands)
        loss = loss_fn(preds, masks)

        total_loss += loss.item()
        metrics = compute_metrics(preds, masks)

        for k in total_metrics:
            total_metrics[k] += metrics[k]

        count += 1

    avg_loss = total_loss / count
    avg_metrics = {k: v / count for k, v in total_metrics.items()}

    return avg_loss, avg_metrics


# ================= TRAIN =================

def train(config: dict):
    set_seed(config["seed"])
    device = torch.device(config['device'])

    # ===== EXP DIR =====
    if config.get("resume", False):
        exp_dir = config.get("resume_path") or get_latest_experiment(config["run_base_dir"])
        if exp_dir is None:
            raise ValueError("No experiment found to resume")
    else:
        exp_dir = create_experiment_dir(config["run_base_dir"])

    # ===== LOGGER =====
    logger = get_logger(exp_dir)
    csv_path = init_csv(exp_dir)

    logger.info(f"Experiment dir: {exp_dir}")

    # ===== SAVE CONFIG =====
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # ===== DATA =====
    train_dataset = GlacierDataset(
        path=Path(config['dataset']),
        patch_size=config["patch_size"],
        overlap=config["overlap_train"],
        transform=GlacierTransform(),
        mode='train',
        mode_path=Path(config['split_path']),
        bands_used=config["bands_used"]
    )

    val_dataset = GlacierDataset(
        path=Path(config['dataset']),
        patch_size=config["patch_size"],
        overlap=config["overlap_val"],
        mode='val',
        mode_path=Path(config['split_path']),
        bands_used=config["bands_used"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # ===== MODEL =====
    model = SUnet(
        ch_head=config["channel_head"],
        in_ch=config["in_channels"],
        out_ch=config["out_channels"],
        num_res_blocks=config["num_res_blocks"],
        attn=config["use_attention"],
        se=config["use_se"],
        dropout=config["dropout"]
    ).to(device)

    # ===== LOSS =====
    loss_fn = FocalDiceLoss()

    # ===== OPTIMIZER =====
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-4
    )

    # ===== SCHEDULER =====
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=config["epochs"]
    )

    scheduler = GradualWarmupScheduler(
        optimizer=optimizer,
        multiplier=config["warmup_multiplier"],
        warm_epoch=config["epochs"] // 5,
        after_scheduler=cosine
    )

    # ===== RESUME =====
    latest_path = os.path.join(exp_dir, "latest.pth")
    start_epoch, best_mcc = load_checkpoint(
        latest_path, model, optimizer, scheduler, device
    )

    logger.info(f"Start epoch: {start_epoch}, Best MCC: {best_mcc:.4f}")

    # ================= TRAIN LOOP =================
    for epoch in range(start_epoch, config["epochs"]):

        model.train()
        total_train_loss = 0

        pbar = tqdm(train_loader, dynamic_ncols=True)

        for bands, masks in pbar:
            bands = bands.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            preds, _ = model(bands)
            loss = loss_fn(preds, masks)

            if not torch.isfinite(loss):
                logger.warning("Skipping non-finite loss")
                continue

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # ===== VALIDATION =====
        val_loss, val_metrics = validate(model, config, device, val_dataset)

        # ===== LOGGING =====
        logger.info(f"Epoch {epoch}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(
            f"IoU: {val_metrics['IoU']:.4f} | "
            f"Precision: {val_metrics['Precision']:.4f} | "
            f"Recall: {val_metrics['Recall']:.4f} | "
            f"F1: {val_metrics['F1']:.4f} | "
            f"MCC: {val_metrics['MCC']:.4f}"
        )

        log_csv(csv_path, epoch, avg_train_loss, val_loss, val_metrics)

        # ===== SAVE LATEST =====
        save_model(
            exp_dir,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            tag="latest",
            best_metric=best_mcc,
            loss=avg_train_loss
        )

        # ===== SAVE BEST =====
        if val_metrics["MCC"] > best_mcc:
            best_mcc = val_metrics["MCC"]

            logger.info(f"New best MCC: {best_mcc:.4f}")

            save_model(
                exp_dir,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                tag="best",
                best_metric=best_mcc,
                loss=avg_train_loss
            )

    logger.info("Training complete")
