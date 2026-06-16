import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import yaml
from datetime import datetime

# Add src/ to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from config import (
    RANDOM_SEED, PROCESSED_DIR, METRICS_DIR,
    PREPROCESSING_MODES, EFFICIENTNET_VERSIONS
)
from preprocess_images import preprocess_dataset
from train import train_kfold


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"Seed set to {seed}")


# ── YAML loader ───────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load experiment configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ── Preprocessing check ───────────────────────────────────────────────────────

def ensure_preprocessed(preprocessing_mode: str) -> None:
    """
    Check if preprocessed images already exist for this mode.
    If not, run the preprocessing pipeline.
    """
    out_dir = os.path.join(PROCESSED_DIR, preprocessing_mode)

    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        print(f"Preprocessed images already exist for mode '{preprocessing_mode}'. Skipping.")
        return

    print(f"Preprocessed images not found for mode '{preprocessing_mode}'. Running preprocessing...")
    import pandas as pd
    from config import METADATA_CSV
    df        = pd.read_csv(METADATA_CSV)
    image_ids = df["image_id"].tolist()
    preprocess_dataset(preprocessing_mode, image_ids)


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_experiment(config: dict, device: torch.device) -> dict:
    """
    Run a single experiment defined by a config dict.
    Automatically detects if unimodal or multimodal based on 'metadata_cols'.
    """
    efficientnet_version = config['model']
    preprocessing_mode   = config['preprocessing']
    metadata_cols        = config.get('metadata_cols', None)
    is_multimodal        = metadata_cols is not None and len(metadata_cols) > 0

    print(f"\n{'='*60}")
    print(f"EXPERIMENT START")
    print(f"  Mode:          {'Multimodal' if is_multimodal else 'Unimodal'}")
    print(f"  Model:         EfficientNet-{efficientnet_version.upper()}")
    print(f"  Preprocessing: {preprocessing_mode}")
    if is_multimodal:
        print(f"  Metadata:      {metadata_cols}")
    print(f"  Device:        {device}")
    print(f"  Time:          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # ── Seed ──────────────────────────────────────────────────────────────────
    set_seed(RANDOM_SEED)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    ensure_preprocessed(preprocessing_mode)

    # ── Training ──────────────────────────────────────────────────────────────
    if is_multimodal:
        from train import train_kfold_multimodal
        summary = train_kfold_multimodal(
            preprocessing_mode = preprocessing_mode,
            metadata_cols      = metadata_cols,
            device             = device,
            phase1_epochs      = config.get('phase1_epochs', 5),
            max_epochs         = config.get('max_epochs', 30),
            lr_phase1          = config.get('lr_phase1', 1e-3),
            lr_phase2          = config.get('lr_phase2', 1e-4),
            patience           = config.get('patience', 7),
            batch_size         = config.get('batch_size', 32),
        )
    else:
        from train import train_kfold
        summary = train_kfold(
            preprocessing_mode   = preprocessing_mode,
            efficientnet_version = efficientnet_version,
            device               = device,
            phase1_epochs        = config.get('phase1_epochs', 5),
            max_epochs           = config.get('max_epochs', 30),
            lr_phase1            = config.get('lr_phase1', 1e-3),
            lr_phase2            = config.get('lr_phase2', 1e-4),
            patience             = config.get('patience', 7),
            batch_size           = config.get('batch_size', 32),
        )

    # ── Save experiment record ────────────────────────────────────────────────
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config':    config,
        'results':   summary,
    }

    os.makedirs(METRICS_DIR, exist_ok=True)
    if is_multimodal:
        metadata_str    = '_'.join(metadata_cols)
        experiment_name = f'multimodal_b0_none_{metadata_str}'
    else:
        experiment_name = f'efficientnet_{efficientnet_version}_{preprocessing_mode}'

    record_path = os.path.join(METRICS_DIR, f'{experiment_name}_record.json')
    with open(record_path, 'w') as f:
        json.dump(record, f, indent=2)
    print(f'\nExperiment record saved to {record_path}')

    return record


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run a skin lesion classification experiment.")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the experiment YAML config file."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use: 'cuda' or 'cpu'. Auto-detected if not specified."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load config ───────────────────────────────────────────────────────────
    config = load_config(args.config)
    print(f"\nLoaded config from {args.config}:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # ── Run ───────────────────────────────────────────────────────────────────
    run_experiment(config, device)