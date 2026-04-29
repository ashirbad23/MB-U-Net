import torch
from pathlib import Path
from src.train.train import train
import os

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "dataset"
CONFIG = ROOT / "config"
CHECKPOINTS = ROOT / "model_checkpoints"

model_config = {
    "mode": "train",
    "seed": 42,
    # ================= PATHS =================
    "dataset": str(ROOT / "dataset"),
    "split_path": str(ROOT / "config" / "train_val_split.json"),
    "run_base_dir": str(ROOT / "runs"),

    # ================= RESUME =================
    "resume": False,
    "resume_path": None,

    # ================= DEVICE =================
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # ================= DATA =================
    "patch_size": 64,
    "overlap_train": 0.5,
    "overlap_val": 1,

    # ================= BANDS =================
    "bands_used": None,

    # ================= MODEL =================
    "in_channels": 18,
    "out_channels": 1,
    "channel_head": 32,
    "num_res_blocks": 2,
    "dropout": 0.1,
    "use_attention": False,
    "use_se": True,

    # ================= TRAINING =================
    "batch_size": 8,
    "num_workers": 8,
    "epochs": 10,

    # ================= OPTIMIZATION =================
    "learning_rate": 1e-4,
    "min_lr": 1e-5,
    "warmup_multiplier": 1.5,
    "warm_epochs": 1,

    # ================= LOSS =================
    "num_classes": 2,
    "accum_steps": 2
}

if __name__ == "__main__":
    if model_config["mode"] == "train":
        train(model_config)
