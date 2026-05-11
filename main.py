import torch
from pathlib import Path
from src.train.train import train
from src.inference.test import test
from src.explain.explain import explain
from src.visualize.visualize import visualize
import os

ROOT = Path(__file__).resolve().parent

DATASET = ROOT / "dataset"
CONFIG = ROOT / "config"
CHECKPOINTS = ROOT / "model_checkpoints"
CACHE = ROOT / "cache"

model_config = {
    "mode": "visualize",
    "seed": 42,

    # =====================================================
    # PATHS
    # =====================================================

    "dataset": str(ROOT / "dataset"),
    "split_path": str(ROOT / "config" / "train_val_test_split.json"),
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
        CACHE / "sampler_128_10bands_new_aug.json"
    ),

    # =====================================================
    # TEST
    # =====================================================

    "test_exp": str(ROOT
                    / "runs"
                    / "exp_003"),
    # "test_threshold": 0.60,

    "ckpt_type": "best",
    # =====================================================
    # EXPLAINABILITY
    # =====================================================

    "explain_exp": str(
        ROOT
        / "runs"
        / "exp_003"  # change to your target experiment
    ),

    # Which dataset to explain:
    # "internal" -> uses test_results_internal
    # "external" -> uses test_results_external
    "explain_dataset": "internal",

    # Number of best images (sorted by MCC)
    "top_k": 20,

    # Integrated Gradients settings
    "ig_steps": 64,

    # =====================================================
    # VISUALIZATION
    # =====================================================

    "visualize_exp": str(
        ROOT
        / "runs"
        / "exp_003"  # change to your experiment folder
    ),
}

if __name__ == "__main__":
    if model_config["mode"] == "train":
        train(model_config)
    elif model_config["mode"] == "test":
        test(model_config)
    elif model_config["mode"] == "explain":
        explain(model_config)
    elif model_config["mode"] == "visualize":
        visualize(model_config)
