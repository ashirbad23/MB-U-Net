import torch
from pathlib import Path
from src.train.train import train
from src.inference.test import test
import os

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "dataset"
CONFIG = ROOT / "config"
CHECKPOINTS = ROOT / "model_checkpoints"
CACHE = ROOT / "cache"

model_config = {
    "mode": "train",
    "seed": 42,

    # =====================================================
    # PATHS
    # =====================================================

    "dataset": str(ROOT / "dataset"),
    "split_path": str(ROOT / "config" / "train_val_split.json"),
    "run_base_dir": str(ROOT / "runs"),

    # =====================================================
    # RESUME
    # =====================================================

    "resume": False,
    "resume_path": None,

    # =====================================================
    # DEVICE
    # =====================================================

    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # =====================================================
    # DATA
    # =====================================================

    "patch_size": 128,

    "overlap_train": 0.5,
    "overlap_val": 1,

    # =====================================================
    # BANDS
    # =====================================================

    "bands_used": list(range(18)),

    # =====================================================
    # MODEL
    # =====================================================

    "in_channels": 18,
    "out_channels": 1,

    "channel_head": [16, 8, 4],
    "num_levels": 4,

    "dropout": 0.15,

    "use_attention": False,
    "use_se": False,

    # =====================================================
    # TRAINING
    # =====================================================

    "batch_size": 16,
    "num_workers": 4,

    "epochs": 50,

    # =====================================================
    # OPTIMIZATION
    # =====================================================

    "learning_rate": 1e-4,
    "min_lr": 1e-5,

    "patience": 10,

    "w_decay": 5e-5,

    # =====================================================
    # LOSS
    # =====================================================

    "accum_steps": 2,

    "dice_weight": 0.25,

    # =====================================================
    # CACHE
    # =====================================================

    "sampler_cache": str(
        CACHE / "sampler_128_10bands.json"
    ),

    # =====================================================
    # TEST
    # =====================================================

    "test_exp": str(ROOT
                    / "runs"
                    / "exp_010"),

    "ckpt_type": "best"
}

if __name__ == "__main__":
    if model_config["mode"] == "train":
        train(model_config)
    elif model_config["mode"] == "test":
        test(model_config)
