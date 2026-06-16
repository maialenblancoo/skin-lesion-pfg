import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    XAI_DIR, FIGURES_DIR, METRICS_DIR,
    NUM_CLASSES, BATCH_SIZE, NUM_WORKERS,
    UNCERTAINTY_THRESHOLD, CLASS_NAMES_FULL, IDX_TO_CLASS
)
from model import build_model
from dataset import load_test
from transforms import get_val_transforms


# ── MC Dropout helpers ────────────────────────────────────────────────────────

def enable_dropout(model: torch.nn.Module) -> None:
    """
    Set model to eval mode but keep Dropout layers active.
    This is the key trick for MC Dropout inference.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_inference(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    n_passes: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run N stochastic forward passes on a single image.

    Args:
        model:        Trained model with dropout layers.
        image_tensor: Single image tensor (C, H, W).
        n_passes:     Number of MC forward passes.
        device:       Torch device.

    Returns:
        probs_mc: Array of shape (n_passes, NUM_CLASSES) with softmax probs.
    """
    enable_dropout(model)
    image_batch = image_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
    probs_mc    = []

    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(image_batch)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
            probs_mc.append(probs)

    return np.array(probs_mc)  # (n_passes, NUM_CLASSES)


# ── Uncertainty metrics ───────────────────────────────────────────────────────

def predictive_entropy(probs_mc: np.ndarray) -> float:
    """
    Compute predictive entropy from MC samples.
    Higher entropy = higher uncertainty.

    Args:
        probs_mc: (n_passes, NUM_CLASSES)

    Returns:
        Scalar entropy value.
    """
    mean_probs = probs_mc.mean(axis=0)  # (NUM_CLASSES,)
    entropy    = -np.sum(mean_probs * np.log(mean_probs + 1e-8))
    return float(entropy)


def mean_variance(probs_mc: np.ndarray) -> float:
    """
    Compute mean variance across classes from MC samples.

    Args:
        probs_mc: (n_passes, NUM_CLASSES)

    Returns:
        Scalar mean variance.
    """
    return float(probs_mc.var(axis=0).mean())


# ── Full uncertainty evaluation ───────────────────────────────────────────────

def run_uncertainty(
    model_path: str,
    preprocessing_mode: str,
    efficientnet_version: str,
    device: torch.device,
    n_passes: int = 30,
):
    """
    Run MC Dropout uncertainty estimation on the full test set.
    Flags ambiguous cases for specialist review.

    Args:
        model_path:           Path to best trained model .pth.
        preprocessing_mode:   Preprocessing mode used during training.
        efficientnet_version: EfficientNet variant.
        device:               Torch device.
        n_passes:             Number of MC Dropout forward passes.

    Returns:
        DataFrame with per-sample uncertainty metrics.
    """
    experiment_name = f"efficientnet_{efficientnet_version}_{preprocessing_mode}"
    print(f"\nRunning MC Dropout uncertainty — {experiment_name}")
    print(f"Forward passes per sample: {n_passes}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model(efficientnet_version, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # ── Load test set ─────────────────────────────────────────────────────────
    test_dataset = load_test(preprocessing_mode, get_val_transforms())
    print(f"Test samples: {len(test_dataset)}")

    # ── Per-sample uncertainty ────────────────────────────────────────────────
    records = []

    for idx in tqdm(range(len(test_dataset)), desc="MC Dropout"):
        image_tensor, true_label = test_dataset[idx]

        probs_mc    = mc_dropout_inference(model, image_tensor, n_passes, device)
        mean_probs  = probs_mc.mean(axis=0)
        pred_label  = int(np.argmax(mean_probs))
        entropy     = predictive_entropy(probs_mc)
        variance    = mean_variance(probs_mc)
        confidence  = float(mean_probs.max())
        flagged     = entropy > UNCERTAINTY_THRESHOLD

        records.append({
            "sample_idx":   idx,
            "true_label":   true_label,
            "true_class":   CLASS_NAMES_FULL[IDX_TO_CLASS[true_label]],
            "pred_label":   pred_label,
            "pred_class":   CLASS_NAMES_FULL[IDX_TO_CLASS[pred_label]],
            "confidence":   confidence,
            "entropy":      entropy,
            "variance":     variance,
            "correct":      true_label == pred_label,
            "flagged":      flagged,
        })

    df = pd.DataFrame(records)

    # ── Summary ───────────────────────────────────────────────────────────────
    n_flagged   = df["flagged"].sum()
    pct_flagged = 100 * n_flagged / len(df)

    print(f"\n{'─'*50}")
    print(f"  Total samples:   {len(df)}")
    print(f"  Flagged:         {n_flagged} ({pct_flagged:.1f}%)")
    print(f"  Mean entropy:    {df['entropy'].mean():.4f}")
    print(f"  Mean confidence: {df['confidence'].mean():.4f}")
    print(f"{'─'*50}")

    print("\nFlagged cases by class:")
    flagged_by_class = (
        df[df["flagged"]]
        .groupby("true_class")
        .size()
        .sort_values(ascending=False)
    )
    print(flagged_by_class.to_string())

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(XAI_DIR, exist_ok=True)
    csv_path = os.path.join(XAI_DIR, f"{experiment_name}_uncertainty.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nUncertainty results saved to {csv_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_uncertainty_distribution(df, experiment_name)
    plot_uncertainty_by_class(df, experiment_name)

    return df


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_uncertainty_distribution(df: pd.DataFrame, experiment_name: str):
    """Plot histogram of predictive entropy across the test set."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["entropy"], bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(
        UNCERTAINTY_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
        label=f"Threshold = {UNCERTAINTY_THRESHOLD}"
    )
    ax.set_xlabel("Predictive Entropy", fontsize=12)
    ax.set_ylabel("Number of samples", fontsize=12)
    ax.set_title(f"Uncertainty Distribution — {experiment_name}", fontsize=13)
    ax.legend()
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, f"{experiment_name}_uncertainty_dist.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Uncertainty distribution plot saved to {save_path}")


def plot_uncertainty_by_class(df: pd.DataFrame, experiment_name: str):
    """Boxplot of entropy per true class."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    classes      = df["true_class"].unique()
    entropy_data = [df[df["true_class"] == c]["entropy"].values for c in classes]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(entropy_data, labels=classes, patch_artist=True)
    ax.axhline(
        UNCERTAINTY_THRESHOLD, color="red", linestyle="--", linewidth=1.5,
        label=f"Threshold = {UNCERTAINTY_THRESHOLD}"
    )
    ax.set_xlabel("True Class", fontsize=12)
    ax.set_ylabel("Predictive Entropy", fontsize=12)
    ax.set_title(f"Uncertainty by Class — {experiment_name}", fontsize=13)
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, f"{experiment_name}_uncertainty_by_class.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Uncertainty by class plot saved to {save_path}")


if __name__ == "__main__":
    import glob
    from config import MODELS_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pth"))
    if not model_files:
        print("No models found in outputs/models/. Train first.")
    else:
        model_path = sorted(model_files)[0]
        filename   = os.path.basename(model_path).replace(".pth", "")
        parts      = filename.split("_")
        version    = parts[1]
        mode       = "_".join(parts[2:-1])

        run_uncertainty(
            model_path=model_path,
            preprocessing_mode=mode,
            efficientnet_version=version,
            device=device,
            n_passes=30,
        )