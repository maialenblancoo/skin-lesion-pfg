import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    XAI_DIR, NUM_CLASSES, BATCH_SIZE,
    CLASS_NAMES_FULL, IDX_TO_CLASS
)
from model import build_model
from transforms import get_val_transforms
from dataset import load_test


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_map(saliency_map: np.ndarray) -> np.ndarray:
    """Normalize a saliency map to [0, 1]."""
    s_min, s_max = saliency_map.min(), saliency_map.max()
    if s_max - s_min < 1e-8:
        return np.zeros_like(saliency_map)
    return (saliency_map - s_min) / (s_max - s_min)


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized image tensor (C, H, W) back to RGB uint8."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = std * img + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def overlay_heatmap(image_rgb: np.ndarray, heatmap: np.ndarray, alpha=0.5, colormap=None) -> np.ndarray:
    import cv2 as _cv2
    if colormap is None:
        colormap = _cv2.COLORMAP_JET
    """Overlay a heatmap on an RGB image."""
    heatmap_uint8 = (normalize_map(heatmap) * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap_color + (1 - alpha) * image_rgb).astype(np.uint8)
    return overlay


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """Grad-CAM implementation for EfficientNet (timm)."""

    def __init__(self, model: torch.nn.Module):
        self.model      = model
        self.gradients  = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        # Target: last conv block of EfficientNet
        target_layer = self.model.blocks[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        image_tensor = image_tensor.unsqueeze(0).requires_grad_(True)

        logits = self.model(image_tensor)
        score  = logits[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = F.relu(cam).cpu().detach().numpy()
        cam     = cv2.resize(cam, (image_tensor.shape[-1], image_tensor.shape[-2]))
        return normalize_map(cam)


# ── Grad-CAM++ ────────────────────────────────────────────────────────────────

class GradCAMPlusPlus:
    """Grad-CAM++ implementation for EfficientNet (timm)."""

    def __init__(self, model: torch.nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.blocks[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        image_tensor = image_tensor.unsqueeze(0).requires_grad_(True)

        logits = self.model(image_tensor)
        score  = logits[0, class_idx]
        score.backward()

        grads = self.gradients          # (1, C, H, W)
        acts  = self.activations        # (1, C, H, W)

        grads_sq  = grads ** 2
        grads_cu  = grads ** 3
        denom     = 2 * grads_sq + (acts * grads_cu).sum(dim=(2, 3), keepdim=True) + 1e-8
        alpha     = grads_sq / denom
        weights   = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1).squeeze()
        cam = F.relu(cam).cpu().detach().numpy()
        cam = cv2.resize(cam, (image_tensor.shape[-1], image_tensor.shape[-2]))
        return normalize_map(cam)


# ── Vanilla Saliency ──────────────────────────────────────────────────────────

def vanilla_saliency(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_idx: int
) -> np.ndarray:
    """
    Vanilla Saliency: gradient of the class score w.r.t. input image.
    Returns a grayscale saliency map.
    """
    model.zero_grad()
    inp = image_tensor.unsqueeze(0).requires_grad_(True)

    logits = model(inp)
    score  = logits[0, class_idx]
    score.backward()

    saliency = inp.grad.data.abs().squeeze()       # (C, H, W)
    saliency = saliency.max(dim=0)[0].cpu().numpy()      # max across channels
    return normalize_map(saliency)


# ── SmoothGrad ────────────────────────────────────────────────────────────────

def smoothgrad(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_idx: int,
    n_samples: int = 20,
    noise_level: float = 0.15,
) -> np.ndarray:
    """
    SmoothGrad: average saliency over N noisy versions of the input.
    Reduces noise in vanilla saliency maps.
    """
    accumulated = np.zeros(image_tensor.shape[1:])  # (H, W)
    std = noise_level * (image_tensor.max() - image_tensor.min()).item()

    for _ in range(n_samples):
        noise      = torch.randn_like(image_tensor) * std
        noisy_inp  = (image_tensor + noise).unsqueeze(0).requires_grad_(True)

        model.zero_grad()
        logits = model(noisy_inp)
        score  = logits[0, class_idx]
        score.backward()

        saliency    = noisy_inp.grad.data.abs().squeeze()
        saliency    = saliency.max(dim=0)[0].cpu().numpy()
        accumulated += saliency

    averaged = accumulated / n_samples
    return normalize_map(averaged)


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_xai(
    image_tensor: torch.Tensor,
    gradcam_map: np.ndarray,
    gradcam_pp_map: np.ndarray,
    saliency_map: np.ndarray,
    smoothgrad_map: np.ndarray,
    true_label: int,
    pred_label: int,
    save_path: str,
):
    """
    Save a figure with 5 panels:
    Original | Grad-CAM | Grad-CAM++ | Vanilla Saliency | SmoothGrad
    """
    image_rgb = tensor_to_rgb(image_tensor)
    h, w      = image_rgb.shape[:2]

    panels = [
        ("Original",         image_rgb),
        ("Grad-CAM",         overlay_heatmap(image_rgb, gradcam_map)),
        ("Grad-CAM++",       overlay_heatmap(image_rgb, gradcam_pp_map)),
        ("Vanilla Saliency", overlay_heatmap(image_rgb, saliency_map)),
        ("SmoothGrad",       overlay_heatmap(image_rgb, smoothgrad_map)),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    true_name = CLASS_NAMES_FULL[IDX_TO_CLASS[true_label]]
    pred_name = CLASS_NAMES_FULL[IDX_TO_CLASS[pred_label]]
    correct   = "✓" if true_label == pred_label else "✗"
    fig.suptitle(
        f"True: {true_name}  |  Pred: {pred_name}  {correct}",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main XAI pipeline ─────────────────────────────────────────────────────────

def run_xai(
    model_path: str,
    preprocessing_mode: str,
    efficientnet_version: str,
    device: torch.device,
    n_samples: int = 20,
):
    """
    Run all XAI techniques on n_samples from the test set.

    Args:
        model_path:           Path to the best trained model .pth.
        preprocessing_mode:   Preprocessing mode used during training.
        efficientnet_version: EfficientNet variant.
        device:               Torch device.
        n_samples:            Number of test images to explain.
    """
    experiment_name = f"efficientnet_{efficientnet_version}_{preprocessing_mode}"
    out_dir = os.path.join(XAI_DIR, experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model(efficientnet_version, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # ── Load test samples ─────────────────────────────────────────────────────
    test_dataset = load_test(preprocessing_mode, get_val_transforms())
    indices      = np.random.choice(len(test_dataset), n_samples, replace=False)

    # ── Initialize XAI methods ────────────────────────────────────────────────
    gradcam    = GradCAM(model)
    gradcam_pp = GradCAMPlusPlus(model)

    print(f"\nRunning XAI for {experiment_name} on {n_samples} samples...")

    for i, idx in enumerate(tqdm(indices, desc="XAI")):
        image_tensor, true_label = test_dataset[idx]
        image_tensor = image_tensor.to(device)

        # Predicted class
        with torch.no_grad():
            logits     = model(image_tensor.unsqueeze(0))
            pred_label = logits.argmax(dim=1).item()

        # Use predicted class for explanations
        target_class = pred_label

        gc_map  = gradcam(image_tensor, target_class)
        gcpp_map = gradcam_pp(image_tensor, target_class)
        sal_map  = vanilla_saliency(model, image_tensor, target_class)
        sg_map   = smoothgrad(model, image_tensor, target_class)

        save_path = os.path.join(out_dir, f"sample_{i:03d}_idx{idx}.png")
        visualize_xai(
            image_tensor=image_tensor.cpu(),
            gradcam_map=gc_map,
            gradcam_pp_map=gcpp_map,
            saliency_map=sal_map,
            smoothgrad_map=sg_map,
            true_label=true_label,
            pred_label=pred_label,
            save_path=save_path,
        )

    print(f"\nXAI figures saved to {out_dir}")


if __name__ == "__main__":
    import glob
    from config import MODELS_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run XAI on the best model (highest AUC from metrics)
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pth"))
    if not model_files:
        print("No models found in outputs/models/. Train first.")
    else:
        # For now use the first model found
        model_path = sorted(model_files)[0]
        filename   = os.path.basename(model_path).replace(".pth", "")
        parts      = filename.split("_")
        version    = parts[1]
        mode       = "_".join(parts[2:-1])

        run_xai(
            model_path=model_path,
            preprocessing_mode=mode,
            efficientnet_version=version,
            device=device,
            n_samples=20,
        )