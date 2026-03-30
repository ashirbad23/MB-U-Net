import os
from pathlib import Path
import csv
import logging

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


# ================= LOGGER =================
def get_logger(log_path):
    log_path = Path(log_path)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_path / "train.log")
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ================= CSV LOGGER =================
def init_csv(log_path):
    csv_path = Path(log_path) / "metrics.csv"

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss",
            "IoU", "Precision", "Recall", "F1", "MCC"
        ])

    return csv_path


def log_csv(csv_path, epoch, train_loss, val_loss, metrics):
    with open(csv_path, mode="a", newline="") as f:
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


# ================= VALIDATION =================
@torch.no_grad()
def validate(model, config, device):
    dataset = GlacierDataset(
        path=Path(config['dataset']),
        patch_size=config["patch_size"],
        overlap=config["overlap_val"],
        mode='val',
        mode_path=Path(config['split_path']),
        bands_used=config["bands_used"]
    )

    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True
    )

    model.eval()
    loss_fn = FocalDiceLoss()

    total_loss = 0
    total_metrics = {
        "IoU": 0,
        "Precision": 0,
        "Recall": 0,
        "F1": 0,
        "MCC": 0
    }
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
    device = torch.device(config['device'])

    logger = get_logger(config["checkpoint"])
    csv_path = init_csv(config["checkpoint"])

    # ---- log config ----
    logger.info("==== CONFIG ====")
    for k, v in config.items():
        logger.info(f"{k}: {v}")
    logger.info("================")

    # ---- TRAIN DATA ----
    train_dataset = GlacierDataset(
        path=Path(config['dataset']),
        patch_size=config["patch_size"],
        overlap=config["overlap_train"],
        transform=GlacierTransform(),
        mode='train',
        mode_path=Path(config['split_path']),
        bands_used=config["bands_used"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # ---- MODEL ----
    model = SUnet(
        ch_head=config["channel_head"],
        in_ch=config["in_channels"],
        out_ch=config["out_channels"],
        num_res_blocks=config["num_res_blocks"],
        attn=config["use_attention"],
        se=config["use_se"],
        dropout=config["dropout"]
    ).to(device)

    # ---- LOSS ----
    loss_fn = FocalDiceLoss()

    # ---- OPTIMIZER ----
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-4
    )

    # ---- SCHEDULER ----
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

    best_mcc = -1

    # ================= TRAIN LOOP =================
    for epoch in range(config["epochs"]):

        model.train()
        pbar = tqdm(train_loader, dynamic_ncols=True)

        total_train_loss = 0

        for bands, masks in pbar:
            bands = bands.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            preds, _ = model(bands)

            loss = loss_fn(preds, masks)
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pbar.set_postfix({"epoch": epoch, "loss": loss.item()})

        scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # ---- VALIDATION ----
        val_loss, val_metrics = validate(model, config, device)

        # ---- LOGGING ----
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

        # ---- CSV LOG ----
        log_csv(csv_path, epoch, avg_train_loss, val_loss, val_metrics)

        # ---- SAVE BEST ----
        if val_metrics["MCC"] > best_mcc:
            best_mcc = val_metrics["MCC"]

            logger.info(f"New best MCC: {best_mcc:.4f} — saving model")

            save_model(
                config["checkpoint"],
                model,
                optimizer,
                scheduler,
                epoch,
                best_mcc
            )