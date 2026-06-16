import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score
)
from tqdm import tqdm

from config import (
    SPLITS_DIR, MODELS_DIR, METRICS_DIR,
    N_FOLDS, BATCH_SIZE, NUM_WORKERS,
    NUM_CLASSES, RANDOM_SEED
)
from dataset import load_fold
from model import build_model, freeze_backbone_layers, unfreeze_all_layers
from transforms import get_train_transforms, get_val_transforms


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(labels, preds, probs):
    """Compute accuracy, recall, F1 and AUC from predictions."""
    accuracy = accuracy_score(labels, preds)
    recall   = recall_score(labels, preds, average="macro", zero_division=0)
    f1       = f1_score(labels, preds, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0
    return {"accuracy": accuracy, "recall": recall, "f1": f1, "auc": auc}


# ── One epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    """Run a single train or validation epoch."""
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)

            total_loss  += loss.item() * len(labels)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    metrics["loss"] = avg_loss
    return metrics


# ── Train one fold ─────────────────────────────────────────────────────────────

def train_fold(
    fold: int,
    preprocessing_mode: str,
    efficientnet_version: str,
    device: torch.device,
    class_weights: torch.Tensor,
    phase1_epochs: int = 5,
    max_epochs: int = 30,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 1e-4,
    patience: int = 7,
    batch_size: int = 32,
):
    """
    Train a single fold in two phases:
      Phase 1 — frozen backbone, train classifier only.
      Phase 2 — full fine-tuning with early stopping.

    Returns a DataFrame with per-epoch metrics.
    """
    print(f"\n{'='*60}")
    print(f"Fold {fold} | EfficientNet-{efficientnet_version.upper()} | Preprocessing: {preprocessing_mode}")
    print(f"{'='*60}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_dataset, val_dataset = load_fold(
        fold, preprocessing_mode,
        get_train_transforms(), get_val_transforms()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # ── Model & loss ──────────────────────────────────────────────────────────
    model     = build_model(efficientnet_version, pretrained=True)
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    history    = []
    best_auc   = 0.0
    best_epoch = 0
    no_improve = 0

    # ── Model save path ───────────────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(
        MODELS_DIR,
        f"efficientnet_{efficientnet_version}_{preprocessing_mode}_fold{fold}.pth"
    )

    # ── Phase 1: frozen backbone ──────────────────────────────────────────────
    print(f"\n--- Phase 1: Frozen backbone ({phase1_epochs} epochs) ---")
    freeze_backbone_layers(model)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_phase1
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs)

    for epoch in range(phase1_epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_metrics   = run_epoch(model, val_loader,   criterion, None,      device, is_train=False)
        scheduler.step()

        row = {"phase": 1, "epoch": epoch + 1}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}":   v for k, v in val_metrics.items()})
        history.append(row)

        print(
            f"  Epoch {epoch+1}/{phase1_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc   = val_metrics["auc"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)

    # ── Phase 2: full fine-tuning ─────────────────────────────────────────────
    phase2_epochs = max_epochs - phase1_epochs
    print(f"\n--- Phase 2: Full fine-tuning (max {phase2_epochs} epochs, patience={patience}) ---")
    unfreeze_all_layers(model)
    optimizer = AdamW(model.parameters(), lr=lr_phase2, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs)

    for epoch in range(phase2_epochs):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_metrics   = run_epoch(model, val_loader,   criterion, None,      device, is_train=False)
        scheduler.step()

        row = {"phase": 2, "epoch": phase1_epochs + epoch + 1}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}":   v for k, v in val_metrics.items()})
        history.append(row)

        print(
            f"  Epoch {phase1_epochs+epoch+1}/{max_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f}"
        )

        if val_metrics["auc"] > best_auc:
            best_auc   = val_metrics["auc"]
            best_epoch = phase1_epochs + epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Best model saved (AUC: {best_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {phase1_epochs+epoch+1} (no improvement for {patience} epochs)")
                break

    print(f"\nFold {fold} best Val AUC: {best_auc:.4f} at epoch {best_epoch}")
    return pd.DataFrame(history), best_auc, model_path


# ── Full K-Fold training ───────────────────────────────────────────────────────

def train_kfold(
    preprocessing_mode: str,
    efficientnet_version: str,
    device: torch.device,
    phase1_epochs: int = 5,
    max_epochs: int = 30,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 1e-4,
    patience: int = 7,
    batch_size: int = 32,
):

    """
    Run full K-Fold training for a given preprocessing mode and model version.
    Saves per-fold metrics and a summary CSV.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Load class weights
    weights_path  = os.path.join(SPLITS_DIR, "class_weights.npy")
    class_weights = torch.tensor(np.load(weights_path), dtype=torch.float32)

    os.makedirs(METRICS_DIR, exist_ok=True)
    experiment_name = f"efficientnet_{efficientnet_version}_{preprocessing_mode}"

    fold_results = []

    for fold in range(N_FOLDS):
        history_df, best_auc, model_path = train_fold(
            fold=fold,
            preprocessing_mode=preprocessing_mode,
            efficientnet_version=efficientnet_version,
            device=device,
            class_weights=class_weights,
            phase1_epochs=phase1_epochs,
            max_epochs=max_epochs,
            lr_phase1=lr_phase1,
            lr_phase2=lr_phase2,
            patience=patience,
            batch_size=batch_size,
        )

        # Save per-fold history
        history_path = os.path.join(METRICS_DIR, f"{experiment_name}_fold{fold}_history.csv")
        history_df.to_csv(history_path, index=False)

        fold_results.append({
            "fold":       fold,
            "best_auc":   best_auc,
            "model_path": model_path,
        })

    # Summary across folds
    aucs = [r["best_auc"] for r in fold_results]
    summary = {
        "experiment":  experiment_name,
        "model":       efficientnet_version,
        "preprocessing": preprocessing_mode,
        "mean_auc":    np.mean(aucs),
        "std_auc":     np.std(aucs),
        "fold_aucs":   str(aucs),
    }

    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(METRICS_DIR, f"{experiment_name}_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print(f"K-Fold complete for {experiment_name}")
    print(f"Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"Summary saved to {summary_path}")
    print(f"{'='*60}")

    return summary


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Quick test with b0 + none preprocessing
    train_kfold(
        preprocessing_mode="none",
        efficientnet_version="b0",
        device=device,
    )

    # ── Multimodal training ───────────────────────────────────────────────────────

def run_epoch_multimodal(model, loader, criterion, optimizer, device, is_train: bool):
    """Run a single train or validation epoch for the multimodal model."""
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, metadata, labels in tqdm(loader, leave=False):
            images   = images.to(device)
            metadata = metadata.to(device)
            labels   = labels.to(device)

            logits = model(images, metadata)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            preds = np.argmax(probs, axis=1)

            total_loss  += loss.item() * len(labels)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    metrics['loss'] = avg_loss
    return metrics


def train_fold_multimodal(
    fold: int,
    preprocessing_mode: str,
    metadata_cols: list,
    device: torch.device,
    class_weights: torch.Tensor,
    phase1_epochs: int = 5,
    max_epochs: int = 30,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 1e-4,
    patience: int = 7,
    batch_size: int = 32,
):
    """
    Train a single fold of the multimodal model in two phases.

    Returns:
        history_df, best_auc, model_path
    """
    from model import MultimodalModel
    from dataset import load_fold_multimodal

    metadata_str = '_'.join(metadata_cols)
    experiment_name = f'multimodal_b0_none_{metadata_str}'

    print(f'\n{"="*60}')
    print(f'Fold {fold} | Multimodal | Metadata: {metadata_cols}')
    print(f'{"="*60}')

    # ── Data ──────────────────────────────────────────────────────────────────
    train_dataset, val_dataset, metadata_dim = load_fold_multimodal(
        fold, preprocessing_mode, metadata_cols,
        get_train_transforms(), get_val_transforms()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # ── Model & loss ──────────────────────────────────────────────────────────
    model     = MultimodalModel(
        metadata_dim=metadata_dim,
        efficientnet_version='b0',
        pretrained=True
    )
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    history    = []
    best_auc   = 0.0
    best_epoch = 0
    no_improve = 0

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(
        MODELS_DIR,
        f'{experiment_name}_fold{fold}.pth'
    )

    # ── Phase 1: frozen backbone ──────────────────────────────────────────────
    print(f'\n--- Phase 1: Frozen backbone ({phase1_epochs} epochs) ---')
    model.freeze_backbone()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_phase1
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs)

    for epoch in range(phase1_epochs):
        train_metrics = run_epoch_multimodal(model, train_loader, criterion, optimizer, device, is_train=True)
        val_metrics   = run_epoch_multimodal(model, val_loader,   criterion, None,      device, is_train=False)
        scheduler.step()

        row = {'phase': 1, 'epoch': epoch + 1}
        row.update({f'train_{k}': v for k, v in train_metrics.items()})
        row.update({f'val_{k}':   v for k, v in val_metrics.items()})
        history.append(row)

        print(
            f'  Epoch {epoch+1}/{phase1_epochs} | '
            f'Train Loss: {train_metrics["loss"]:.4f} | '
            f'Val AUC: {val_metrics["auc"]:.4f} | '
            f'Val Recall: {val_metrics["recall"]:.4f}'
        )

        if val_metrics['auc'] > best_auc:
            best_auc   = val_metrics['auc']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_path)

    # ── Phase 2: full fine-tuning ─────────────────────────────────────────────
    phase2_epochs = max_epochs - phase1_epochs
    print(f'\n--- Phase 2: Full fine-tuning (max {phase2_epochs} epochs, patience={patience}) ---')
    model.unfreeze_all()
    optimizer = AdamW(model.parameters(), lr=lr_phase2, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs)

    for epoch in range(phase2_epochs):
        train_metrics = run_epoch_multimodal(model, train_loader, criterion, optimizer, device, is_train=True)
        val_metrics   = run_epoch_multimodal(model, val_loader,   criterion, None,      device, is_train=False)
        scheduler.step()

        row = {'phase': 2, 'epoch': phase1_epochs + epoch + 1}
        row.update({f'train_{k}': v for k, v in train_metrics.items()})
        row.update({f'val_{k}':   v for k, v in val_metrics.items()})
        history.append(row)

        print(
            f'  Epoch {phase1_epochs+epoch+1}/{max_epochs} | '
            f'Train Loss: {train_metrics["loss"]:.4f} | '
            f'Val AUC: {val_metrics["auc"]:.4f} | '
            f'Val Recall: {val_metrics["recall"]:.4f}'
        )

        if val_metrics['auc'] > best_auc:
            best_auc   = val_metrics['auc']
            best_epoch = phase1_epochs + epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f'  ✓ Best model saved (AUC: {best_auc:.4f})')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'  Early stopping at epoch {phase1_epochs+epoch+1}')
                break

    print(f'\nFold {fold} best Val AUC: {best_auc:.4f} at epoch {best_epoch}')
    return pd.DataFrame(history), best_auc, model_path


def train_kfold_multimodal(
    preprocessing_mode: str,
    metadata_cols: list,
    device: torch.device,
    phase1_epochs: int = 5,
    max_epochs: int = 30,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 1e-4,
    patience: int = 7,
    batch_size: int = 32,
):
    """
    Run full K-Fold training for the multimodal model.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    weights_path  = os.path.join(SPLITS_DIR, 'class_weights.npy')
    class_weights = torch.tensor(np.load(weights_path), dtype=torch.float32)

    metadata_str    = '_'.join(metadata_cols)
    experiment_name = f'multimodal_b0_none_{metadata_str}'

    os.makedirs(METRICS_DIR, exist_ok=True)
    fold_results = []

    for fold in range(N_FOLDS):
        history_df, best_auc, model_path = train_fold_multimodal(
            fold=fold,
            preprocessing_mode=preprocessing_mode,
            metadata_cols=metadata_cols,
            device=device,
            class_weights=class_weights,
            phase1_epochs=phase1_epochs,
            max_epochs=max_epochs,
            lr_phase1=lr_phase1,
            lr_phase2=lr_phase2,
            patience=patience,
            batch_size=batch_size,
        )

        history_path = os.path.join(METRICS_DIR, f'{experiment_name}_fold{fold}_history.csv')
        history_df.to_csv(history_path, index=False)

        fold_results.append({
            'fold':       fold,
            'best_auc':   best_auc,
            'model_path': model_path,
        })

    aucs    = [r['best_auc'] for r in fold_results]
    summary = {
        'experiment':    experiment_name,
        'model':         'b0',
        'preprocessing': preprocessing_mode,
        'metadata':      metadata_str,
        'mean_auc':      np.mean(aucs),
        'std_auc':       np.std(aucs),
        'fold_aucs':     str(aucs),
    }

    summary_df   = pd.DataFrame([summary])
    summary_path = os.path.join(METRICS_DIR, f'{experiment_name}_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print(f'\n{"="*60}')
    print(f'K-Fold complete for {experiment_name}')
    print(f'Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}')
    print(f'{"="*60}')

    return summary