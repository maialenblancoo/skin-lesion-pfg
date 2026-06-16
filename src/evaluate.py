import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)
from tqdm import tqdm

from config import (
    SPLITS_DIR, MODELS_DIR, METRICS_DIR, FIGURES_DIR,
    NUM_CLASSES, BATCH_SIZE, NUM_WORKERS,
    CLASS_NAMES_FULL, IDX_TO_CLASS
)
from dataset import load_test
from model import build_model
from transforms import get_val_transforms


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, loader, device):
    """
    Run inference on a DataLoader.

    Returns:
        labels: Ground truth labels (N,)
        preds:  Predicted class indices (N,)
        probs:  Softmax probabilities (N, NUM_CLASSES)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Inference", leave=False):
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)

            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(labels, preds, experiment_name: str):
    """Save a normalized confusion matrix figure."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    class_names = [CLASS_NAMES_FULL[IDX_TO_CLASS[i]] for i in range(NUM_CLASSES)]
    cm = confusion_matrix(labels, preds, normalize="true")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {experiment_name}", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, f"{experiment_name}_confusion_matrix.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {save_path}")


def plot_roc_curves(labels, probs, experiment_name: str):
    """Save ROC curves for all classes in a single figure."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    class_names = [CLASS_NAMES_FULL[IDX_TO_CLASS[i]] for i in range(NUM_CLASSES)]
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (name, color) in enumerate(zip(class_names, colors)):
        binary_labels = (labels == i).astype(int)
        fpr, tpr, _   = roc_curve(binary_labels, probs[:, i])
        roc_auc       = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=1.8,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {experiment_name}", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, f"{experiment_name}_roc_curves.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"ROC curves saved to {save_path}")


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate_model(
    model_path: str,
    preprocessing_mode: str,
    efficientnet_version: str,
    device: torch.device,
):
    """
    Evaluate a trained model on the fixed test set.
    Saves metrics, confusion matrix and ROC curves.

    Args:
        model_path:           Path to the saved .pth model weights.
        preprocessing_mode:   Preprocessing mode used during training.
        efficientnet_version: EfficientNet variant ('b0', 'b1', 'b2').
        device:               Torch device.

    Returns:
        Dictionary with all computed metrics.
    """
    experiment_name = f"efficientnet_{efficientnet_version}_{preprocessing_mode}"
    print(f"\nEvaluating: {experiment_name}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model(efficientnet_version, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # ── Load test set ─────────────────────────────────────────────────────────
    test_dataset = load_test(preprocessing_mode, get_val_transforms())
    test_loader  = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    labels, preds, probs = run_inference(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy = accuracy_score(labels, preds)
    recall   = recall_score(labels, preds, average="macro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    auc_macro = roc_auc_score(labels, probs, multi_class="ovr", average="macro")

    # Per-class metrics
    f1_per_class     = f1_score(labels, preds, average=None, zero_division=0)
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)

    print(f"\n{'─'*50}")
    print(f"  Accuracy:      {accuracy:.4f}")
    print(f"  Recall macro:  {recall:.4f}")
    print(f"  F1 macro:      {f1_macro:.4f}")
    print(f"  AUC macro:     {auc_macro:.4f}")
    print(f"{'─'*50}")
    print("\nPer-class metrics:")
    for i in range(NUM_CLASSES):
        class_name = CLASS_NAMES_FULL[IDX_TO_CLASS[i]]
        print(f"  {class_name:<25} Recall: {recall_per_class[i]:.4f} | F1: {f1_per_class[i]:.4f}")

    print(f"\n{classification_report(labels, preds, target_names=list(CLASS_NAMES_FULL.values()), zero_division=0)}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(labels, preds, experiment_name)
    plot_roc_curves(labels, probs, experiment_name)

    # ── Save metrics CSV ──────────────────────────────────────────────────────
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics = {
        "experiment":        experiment_name,
        "model":             efficientnet_version,
        "preprocessing":     preprocessing_mode,
        "accuracy":          accuracy,
        "recall_macro":      recall,
        "f1_macro":          f1_macro,
        "auc_macro":         auc_macro,
    }
    for i in range(NUM_CLASSES):
        class_key = IDX_TO_CLASS[i]
        metrics[f"recall_{class_key}"] = recall_per_class[i]
        metrics[f"f1_{class_key}"]     = f1_per_class[i]

    metrics_df   = pd.DataFrame([metrics])
    metrics_path = os.path.join(METRICS_DIR, f"{experiment_name}_test_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")

    return metrics


if __name__ == "__main__":
    import glob

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Evaluate all saved models in outputs/models/
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pth"))
    if not model_files:
        print("No models found in outputs/models/. Train first.")
    else:
        all_metrics = []
        for model_path in sorted(model_files):
            filename = os.path.basename(model_path)
            # Parse filename: efficientnet_b0_none_fold0.pth
            parts    = filename.replace(".pth", "").split("_")
            version  = parts[1]
            # preprocessing mode may have underscores too, fold is last
            fold_str = parts[-1]           # fold0
            mode     = "_".join(parts[2:-1])  # none / dullrazor / colorconstancy / both

            metrics = evaluate_model(
                model_path=model_path,
                preprocessing_mode=mode,
                efficientnet_version=version,
                device=device,
            )
            all_metrics.append(metrics)

        # Save combined comparison table
        comparison_df   = pd.DataFrame(all_metrics).sort_values("auc_macro", ascending=False)
        comparison_path = os.path.join(METRICS_DIR, "all_models_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nFull comparison saved to {comparison_path}")