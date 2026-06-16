"""
inference.py — Final inference pipeline for the multimodal skin-lesion system.

This module is the single source of truth for how a prediction is produced, shared
by the notebooks and by app.py (Streamlit). It reproduces, end to end:

    TTA (5 augmentations) per model
        -> selective weighted ensemble (localization 0.7 / sex_age 0.3)
        -> melanoma-specific decision threshold (0.30)

and, as a SEPARATE step (it does not change the diagnosis), MC-Dropout uncertainty,
replicating notebook 12 (flat 0.7/0.3 ensemble, NO TTA, T=30 stochastic passes).

Design notes
------------
* Metadata encoding is taken from MultimodalSkinLesionDataset so the app and the
  training pipeline can never drift apart (same category order, same age/90 scaling).
* The image backbone has no dropout, so image features are deterministic; MC-Dropout
  only re-runs the (tiny) fusion head T times, which is practically free.
* The 0.7/0.3 weights and the 0.30 threshold must be selected on validation
  (fold 0 val), never on test — see notebooks 09 (corrected) and 10.

Target: Python 3.8 (no `X | Y` type unions, no dict `|` merge).
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CLASSES, MODELS_DIR
from model import MultimodalModel
from dataset import MultimodalSkinLesionDataset as _DS


# ── Constants ─────────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

MEL_IDX = CLASSES.index('mel')   # 4
BCC_IDX = CLASSES.index('bcc')   # 1

# Final system hyper-parameters (chosen on validation, see nb09 corrected / nb10)
W_PRIMARY     = 0.6    # localization model (primary) — selected on validation (nb09)
W_SECONDARY   = 0.4    # sex_age model (secondary)
MEL_THRESHOLD = 0.30   # melanoma decision threshold
AGE_MEAN      = 51.9   # fold-0 train age mean, used to impute missing age

# Default model file names (fold 0)
LOC_MODEL_FILE = 'multimodal_b0_none_localization_fold0.pth'
SA_MODEL_FILE  = 'multimodal_b0_none_sex_age_fold0.pth'

# Metadata columns per model (must match training)
METADATA_COLS_LOC = ['localization']
METADATA_COLS_SA  = ['sex', 'age']

# Single source of truth for the metadata encoding (imported from the dataset)
SEX_CATEGORIES = _DS.SEX_CATEGORIES        # ['male', 'female', 'unknown']
LOC_CATEGORIES = _DS.LOC_CATEGORIES        # 15 anatomical zones, fixed order

# TTA: same 5 transforms used in notebook 08
TTA_TRANSFORMS = [
    A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
    A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.HorizontalFlip(p=1.0), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
    A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.VerticalFlip(p=1.0), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
    A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.Rotate(limit=(90, 90), p=1.0), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
    A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.Rotate(limit=(270, 270), p=1.0), A.Normalize(mean=MEAN, std=STD), ToTensorV2()]),
]
_VAL_TRANSFORM = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])


# ── Metadata encoding ─────────────────────────────────────────────────────────
def encode_metadata(sex, age, localization, metadata_cols, age_mean=AGE_MEAN):
    """
    Encode raw clinical inputs into the fixed-length float vector the model expects.
    Mirrors MultimodalSkinLesionDataset._encode_metadata exactly (same order).

    Args:
        sex:           e.g. 'male' / 'female' / None
        age:           number or None (missing -> age_mean)
        localization:  e.g. 'back' / 'face' / None
        metadata_cols: ['localization'] or ['sex', 'age']
        age_mean:      imputation value for missing age

    Returns:
        np.ndarray (metadata_dim,) float32
    """
    features = []  # type: List[float]

    if 'sex' in metadata_cols:
        sex_val = str(sex).lower() if sex is not None else 'unknown'
        if sex_val not in SEX_CATEGORIES:
            sex_val = 'unknown'
        features.extend([1.0 if sex_val == c else 0.0 for c in SEX_CATEGORIES])

    if 'age' in metadata_cols:
        if age is None or (isinstance(age, float) and np.isnan(age)):
            age = age_mean
        features.append(float(age) / 90.0)

    if 'localization' in metadata_cols:
        loc_val = str(localization).lower() if localization is not None else 'unknown'
        if loc_val not in LOC_CATEGORIES:
            loc_val = 'unknown'
        features.extend([1.0 if loc_val == c else 0.0 for c in LOC_CATEGORIES])

    return np.array(features, dtype=np.float32)


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models(models_dir=MODELS_DIR, loc_file=LOC_MODEL_FILE, sa_file=SA_MODEL_FILE,
                device=None):
    """
    Load both trained models once (use @st.cache_resource around this in the app).

    Returns:
        dict with keys: 'loc', 'sa', 'device'
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # metadata dims: localization -> 15, sex+age -> 4
    model_loc = MultimodalModel(metadata_dim=len(LOC_CATEGORIES),
                                efficientnet_version='b0', pretrained=False)
    model_loc.load_state_dict(torch.load(os.path.join(models_dir, loc_file),
                                         map_location=device, weights_only=False))
    model_loc = model_loc.to(device).eval()

    model_sa = MultimodalModel(metadata_dim=len(SEX_CATEGORIES) + 1,
                               efficientnet_version='b0', pretrained=False)
    model_sa.load_state_dict(torch.load(os.path.join(models_dir, sa_file),
                                        map_location=device, weights_only=False))
    model_sa = model_sa.to(device).eval()

    return {'loc': model_loc, 'sa': model_sa, 'device': device}


# ── Image helpers ─────────────────────────────────────────────────────────────
def load_image_rgb(path):
    """Read an image file from disk as RGB uint8 (H, W, 3). Matches dataset.py (BGR->RGB)."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError('Image not found: ' + str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pil_to_rgb(pil_image):
    """Convert a PIL image (e.g. from a Streamlit upload) to RGB uint8 numpy."""
    return np.array(pil_image.convert('RGB'))


# ── Core inference (single image) ─────────────────────────────────────────────
def _tta_probs_single(model, image_rgb, metadata_vec, device):
    """Average softmax probabilities over the 5 TTA transforms for one image. -> (7,)"""
    batch = torch.stack([t(image=image_rgb)['image'] for t in TTA_TRANSFORMS])  # (5,3,H,W)
    meta  = torch.tensor(metadata_vec, dtype=torch.float32).unsqueeze(0).repeat(batch.shape[0], 1)
    model.eval()
    with torch.no_grad():
        logits = model(batch.to(device), meta.to(device))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
    return probs.mean(axis=0)


def _std_probs_single(model, image_rgb, metadata_vec, device):
    """Standard (no-TTA) softmax probabilities for one image. -> (7,)"""
    img  = _VAL_TRANSFORM(image=image_rgb)['image'].unsqueeze(0)
    meta = torch.tensor(metadata_vec, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(img.to(device), meta.to(device))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
    return probs[0]


def selective_weighted_ensemble(probs_primary, probs_secondary,
                                w1=W_PRIMARY, w2=W_SECONDARY):
    """
    Selective weighted ensemble (identical rule to notebook 09).
    Accepts (N, 7) or (7,) arrays. If the two models agree on argmax -> use primary;
    otherwise -> weighted average.

    Returns:
        final_probs (same shape as input), agree_mask (N,) or scalar bool
    """
    pp = np.atleast_2d(probs_primary).astype(np.float64)
    ps = np.atleast_2d(probs_secondary).astype(np.float64)

    agree = np.argmax(pp, axis=1) == np.argmax(ps, axis=1)
    final = w1 * pp + w2 * ps
    final[agree] = pp[agree]

    if np.ndim(probs_primary) == 1:
        return final[0], bool(agree[0])
    return final, agree


def apply_melanoma_threshold(probs, threshold=MEL_THRESHOLD):
    """
    Melanoma-specific decision rule (identical to notebook 10).
    Accepts (N, 7) or (7,). If P(mel) >= threshold -> predict melanoma,
    else argmax over the remaining classes.

    Returns:
        preds (N,) int  OR  scalar int for a single sample
    """
    p2d = np.atleast_2d(probs)
    preds = []
    for i in range(len(p2d)):
        if p2d[i, MEL_IDX] >= threshold:
            preds.append(MEL_IDX)
        else:
            remaining = p2d[i].copy()
            remaining[MEL_IDX] = -1.0
            preds.append(int(np.argmax(remaining)))
    preds = np.array(preds)
    return int(preds[0]) if np.ndim(probs) == 1 else preds


def predict(image_rgb, sex, age, localization, models,
            w1=W_PRIMARY, w2=W_SECONDARY, threshold=MEL_THRESHOLD, use_tta=True):
    """
    Full deterministic prediction for one lesion.

    Args:
        image_rgb:    RGB uint8 numpy array (H, W, 3)
        sex, age, localization: raw clinical inputs (strings / number / None)
        models:       dict from load_models()
        use_tta:      True -> TTA (final system); False -> standard single pass

    Returns:
        dict with final_probs, pred_idx/pred_class (after threshold),
        argmax_idx/argmax_class (before threshold), agree, probs_loc, probs_sa
    """
    device = models['device']
    meta_loc = encode_metadata(sex, age, localization, METADATA_COLS_LOC)
    meta_sa  = encode_metadata(sex, age, localization, METADATA_COLS_SA)

    infer = _tta_probs_single if use_tta else _std_probs_single
    probs_loc = infer(models['loc'], image_rgb, meta_loc, device)
    probs_sa  = infer(models['sa'],  image_rgb, meta_sa,  device)

    final_probs, agree = selective_weighted_ensemble(probs_loc, probs_sa, w1, w2)
    pred_idx    = apply_melanoma_threshold(final_probs, threshold)   # final decision
    argmax_idx  = int(np.argmax(final_probs))

    return {
        'final_probs':   final_probs,
        'pred_idx':      pred_idx,
        'pred_class':    CLASSES[pred_idx],
        'argmax_idx':    argmax_idx,
        'argmax_class':  CLASSES[argmax_idx],
        'agree':         agree,
        'probs_loc':     probs_loc,
        'probs_sa':      probs_sa,
    }


# ── Uncertainty (MC Dropout) — SEPARATE from the diagnosis ─────────────────────
def enable_mc_dropout(model):
    """eval() everywhere, then re-enable ONLY Dropout layers (BatchNorm stays in eval)."""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def mc_dropout_uncertainty(image_rgb, sex, age, localization, models,
                           T=30, w1=W_PRIMARY, w2=W_SECONDARY,
                           entropy_threshold=None):
    """
    MC-Dropout uncertainty for one lesion, replicating notebook 12:
    flat 0.7/0.3 ensemble, NO TTA, T stochastic passes through the fusion head only
    (image features are deterministic and computed once).

    NOTE: this does NOT change the diagnosis; it is an auxiliary "how reliable" signal.
    `entropy_threshold` must be calibrated on validation; if given, a boolean flag is
    returned. Normalised entropy is in [0, 1].

    Returns:
        dict: mean_probs, pred_idx, pred_class, entropy, pred_std, per_class_std,
              high_uncertainty (None unless entropy_threshold is provided)
    """
    device = models['device']
    meta_loc = encode_metadata(sex, age, localization, METADATA_COLS_LOC)
    meta_sa  = encode_metadata(sex, age, localization, METADATA_COLS_SA)

    img = _VAL_TRANSFORM(image=image_rgb)['image'].unsqueeze(0).to(device)

    # Deterministic backbone features (no dropout in the backbone) — computed once.
    with torch.no_grad():
        feat_loc = models['loc'].backbone(img)   # (1, 1280)
        feat_sa  = models['sa'].backbone(img)

    ml = torch.tensor(meta_loc, dtype=torch.float32).unsqueeze(0).to(device)
    ms = torch.tensor(meta_sa,  dtype=torch.float32).unsqueeze(0).to(device)

    enable_mc_dropout(models['loc'])
    enable_mc_dropout(models['sa'])

    samples = []  # T x (7,)
    with torch.no_grad():
        for _ in range(T):
            zl = models['loc'].metadata_branch(ml)
            pl = torch.softmax(models['loc'].classifier(torch.cat([feat_loc, zl], dim=1)), dim=1)
            zs = models['sa'].metadata_branch(ms)
            ps = torch.softmax(models['sa'].classifier(torch.cat([feat_sa, zs], dim=1)), dim=1)
            ens = w1 * pl + w2 * ps                       # flat ensemble (matches nb12)
            samples.append(ens.cpu().numpy()[0])

    models['loc'].eval()
    models['sa'].eval()

    samples = np.stack(samples, axis=0)                   # (T, 7)
    mean_probs    = samples.mean(axis=0)                  # (7,)
    per_class_std = samples.std(axis=0)                   # (7,)
    pred_idx      = int(np.argmax(mean_probs))
    entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-12)) / np.log(len(CLASSES)))

    high = None if entropy_threshold is None else bool(entropy >= entropy_threshold)

    return {
        'mean_probs':       mean_probs,
        'pred_idx':         pred_idx,
        'pred_class':       CLASSES[pred_idx],
        'entropy':          entropy,
        'pred_std':         float(per_class_std[pred_idx]),
        'per_class_std':    per_class_std,
        'high_uncertainty': high,
    }
