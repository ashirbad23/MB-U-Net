import os
import json
import csv
import logging
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

from src.dataset.dataset import GlacierDataset
from utils.transform import GlacierTransform
from utils.loss import FocalDiceLoss, BCEDiceLoss
from utils.metrics import compute_metrics, find_best_threshold
from utils.scheduler import GradualWarmupScheduler
from utils.save_model import save_model
from utils.sampler import GlacierBalancedSampler

from src.model.model import SUnetSimple
from src.model.StandardUNet import SUnet, MultiBranchUNet

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
            "IoU", "Precision", "Recall", "F1", "MCC", "lr"
        ])

    return csv_path


def log_csv(csv_path, epoch, train_loss, val_loss, metrics, lr):
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
            metrics["MCC"],
            lr
        ])


# ================= CHECKPOINT =================

def load_checkpoint(path, model, optimizer, scheduler, device, scaler=None):
    if not os.path.exists(path):
        return 0, -1

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])

    if scaler and "scaler" in checkpoint and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    return checkpoint["epoch"] + 1, checkpoint.get("best_metric", -1)


# ================= VALIDATION =================

@torch.no_grad()
def validate(model, loss_fn, config, device, dataset):
    loader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True
    )

    model.eval()

    total_loss = 0
    count = 0

    all_preds = []
    all_targets = []

    for bands, masks in tqdm(loader, desc="Validation", dynamic_ncols=True):
        bands = bands.to(device)
        masks = masks.to(device)

        with autocast(enabled=(device.type == "cuda")):
            preds, _ = model(bands)
            loss = loss_fn(preds, masks)

        total_loss += loss.item()
        count += 1

        # 🔥 collect for global evaluation
        all_preds.append(preds.detach().cpu())
        all_targets.append(masks.detach().cpu())

    avg_loss = total_loss / count

    # 🔥 concatenate everything
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # ---- metrics @ 0.5 (for reference) ----
    metrics_05 = compute_metrics(all_preds, all_targets, threshold=0.5)

    # ---- BEST threshold ----
    best_t, best_metrics = find_best_threshold(all_preds, all_targets)

    return avg_loss, metrics_05, best_metrics, best_t


# ================= TRAIN =================

def train(config: dict):
    set_seed(config["seed"])
    device = torch.device(config['device'])

    scaler = GradScaler(enabled=(device.type == "cuda"))

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
        transform=GlacierTransform(use_rotation=False),
        mode='train',
        mode_path=Path(config['split_path']),
        bands_used=config["bands_used"]
    )

    val_dataset = GlacierDataset(
        path=Path(config['dataset']),
        patch_size=config["patch_size"],
        overlap=config["overlap_val"],
        transform=GlacierTransform(use_rotation=False, use_radiometric=False),
        mode='val',
        mode_path=Path(config['split_path']),
        bands_used=config["bands_used"]
    )

    # sampler = GlacierBalancedSampler(
    #     dataset=train_dataset,
    #     batch_size=config['batch_size'],
    #     cache_path=config['sampler_cache']
    # )

    train_loader = DataLoader(
        train_dataset,
        # batch_sampler=sampler,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        pin_memory=True,
    )

    # ===== MODEL =====
    # model = SUnetSimple(
    #     ch_head=config["channel_head"],
    #     in_ch=config["in_channels"],
    #     out_ch=config["out_channels"],
    #     num_res_blocks=config["num_res_blocks"],
    #     attn=config["use_attention"],
    #     se=config["use_se"],
    #     dropout=config["dropout"]
    # ).to(device)

    # model = SUnet(
    #     ch_head=config["channel_head"],
    #     in_ch=config["in_channels"],
    #     out_ch=config["out_channels"],
    #     num_levels=config["num_levels"],
    #     attn=config["use_attention"],
    #     se=config["use_se"],
    #     dropout=config["dropout"]
    # ).to(device)

    model = MultiBranchUNet(
        ch_head=config["channel_head"],
        in_ch=config["in_channels"],
        out_ch=config["out_channels"],
        num_levels=config["num_levels"],
        bands_used=config["bands_used"],
        attn=config["use_attention"],
        dropout=config["dropout"]
    ).to(device)

    # ===== LOSS =====
    loss_fn = BCEDiceLoss(
        dice_weight=config['dice_weight']
    )

    # ===== OPTIMIZER =====
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["w_decay"]
    )

    # ===== SCHEDULER =====
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=50,
        eta_min=config["min_lr"]
    )

    # scheduler = GradualWarmupScheduler(
    #     optimizer=optimizer,
    #     multiplier=config["warmup_multiplier"],
    #     warm_epoch=config["warm_epochs"],
    #     after_scheduler=cosine
    # )

    # ===== RESUME =====
    latest_path = os.path.join(exp_dir, "latest.pth")
    start_epoch, best_mcc = load_checkpoint(
        latest_path, model, optimizer, scheduler, device, scaler
    )

    logger.info(f"Start epoch: {start_epoch}, Best MCC: {best_mcc:.4f}")

    patience = config.get("patience", 8)
    epochs_no_improve = 0

    # ================= TRAIN LOOP =================
    for epoch in range(start_epoch, config["epochs"]):

        model.train()
        total_train_loss = 0

        pbar = tqdm(train_loader, dynamic_ncols=True)

        accum_steps = config.get("accum_steps", 1)
        optimizer.zero_grad()

        for i, (bands, masks) in enumerate(pbar):

            bands = bands.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast(enabled=(device.type == "cuda")):
                preds, _ = model(bands)
                raw_loss = loss_fn(preds, masks)

            if not torch.isfinite(raw_loss):
                logger.warning(f"Skipping non-finite loss; Loss: {raw_loss.item()}; Batch: {i}")
                optimizer.zero_grad()
                continue

            loss = raw_loss / accum_steps
            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()

            total_train_loss += raw_loss.item()
            pbar.set_postfix({"loss": f"{raw_loss.item():.4f}"})

        # leftover step
        if len(train_loader) % accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)

        # ===== VALIDATION =====
        val_loss, metrics_05, best_metrics, best_t = validate(model, loss_fn, config, device, val_dataset)
        lr = optimizer.param_groups[0]["lr"]

        # ===== LOGGING =====
        logger.info(f"Epoch {epoch}")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Best Threshold: {best_t:.3f}")
        logger.info(
            f"IoU: {best_metrics['IoU']:.4f} | "
            f"Precision: {best_metrics['Precision']:.4f} | "
            f"Recall: {best_metrics['Recall']:.4f} | "
            f"F1: {best_metrics['F1']:.4f} | "
        )
        logger.info(
            f"[0.5] MCC: {metrics_05['MCC']:.4f} | "
            f"[best] MCC: {best_metrics['MCC']:.4f} | "
            f"T: {best_t:.3f}"
        )
        logger.info(f"LR: {lr}")

        log_csv(csv_path, epoch, avg_train_loss, val_loss, best_metrics, lr)

        scheduler.step()

        # ===== SAVE LATEST =====
        save_model(
            exp_dir,
            model,
            optimizer,
            scheduler,
            epoch,
            tag="latest",
            best_metric=best_mcc,
            loss=avg_train_loss,
            scaler=scaler,
            extra={"best_threshold": best_t}
        )

        # ===== SAVE BEST =====
        if best_metrics["MCC"] > best_mcc:
            best_mcc = best_metrics["MCC"]
            epochs_no_improve = 0

            logger.info(f"New best MCC: {best_mcc:.4f}")

            save_model(
                exp_dir,
                model,
                optimizer,
                scheduler,
                epoch,
                tag="best",
                best_metric=best_mcc,
                loss=avg_train_loss,
                scaler=scaler,
                extra={"best_threshold": best_t}
            )
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    logger.info("Training complete")
