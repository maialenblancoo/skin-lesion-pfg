import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import (
    IMG_PART1_DIR, IMG_PART2_DIR, PROCESSED_DIR,
    METADATA_CSV, PREPROCESSING_MODES
)


# ── DullRazor ─────────────────────────────────────────────────────────────────

def dullrazor(image: np.ndarray) -> np.ndarray:
    """
    Remove hair artifacts from dermoscopic images using the DullRazor algorithm.
    1. Convert to grayscale and apply blackhat morphological filter to detect hair.
    2. Threshold to create a hair mask.
    3. Inpaint the masked regions.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    cleaned = cv2.inpaint(image, hair_mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)

    return cleaned


# ── Color Constancy (Gray World) ──────────────────────────────────────────────

def color_constancy(image: np.ndarray) -> np.ndarray:
    """
    Apply Gray World color constancy to normalize illumination.
    Scales each channel so that its mean equals the overall image mean.
    """
    image_float = image.astype(np.float32)
    mean_per_channel = image_float.mean(axis=(0, 1))
    overall_mean = mean_per_channel.mean()

    scale = overall_mean / (mean_per_channel + 1e-6)
    corrected = np.clip(image_float * scale, 0, 255).astype(np.uint8)

    return corrected


# ── Preprocessing pipeline ────────────────────────────────────────────────────

def preprocess_image(image: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply the selected preprocessing mode to a single image.

    Args:
        image: BGR image as numpy array (H, W, 3).
        mode:  One of 'none', 'dullrazor', 'colorconstancy', 'both'.

    Returns:
        Preprocessed image as numpy array.
    """
    if mode == "none":
        return image
    elif mode == "dullrazor":
        return dullrazor(image)
    elif mode == "colorconstancy":
        return color_constancy(image)
    elif mode == "both":
        image = dullrazor(image)
        image = color_constancy(image)
        return image
    else:
        raise ValueError(
            f"Unknown preprocessing mode: '{mode}'. "
            f"Choose from: none, dullrazor, colorconstancy, both."
        )


def find_image_path(image_id: str):
    """Return the full path of an image given its ID, searching both part folders."""
    for folder in [IMG_PART1_DIR, IMG_PART2_DIR]:
        path = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None


# ── Batch preprocessing ───────────────────────────────────────────────────────

def preprocess_dataset(mode: str, image_ids: list) -> None:
    """
    Preprocess all images in image_ids with the given mode and save
    them to PROCESSED_DIR/<mode>/.

    Args:
        mode:      Preprocessing mode.
        image_ids: List of image IDs to process.
    """
    out_dir = os.path.join(PROCESSED_DIR, mode)
    os.makedirs(out_dir, exist_ok=True)

    skipped = 0
    for image_id in tqdm(image_ids, desc=f"[{mode}]"):
        out_path = os.path.join(out_dir, f"{image_id}.jpg")

        # Skip already processed images
        if os.path.exists(out_path):
            continue

        src_path = find_image_path(image_id)
        if src_path is None:
            skipped += 1
            continue

        image = cv2.imread(src_path)
        if image is None:
            skipped += 1
            continue

        processed = preprocess_image(image, mode)
        cv2.imwrite(out_path, processed)

    if skipped > 0:
        print(f"  [{mode}] Warning: {skipped} images skipped.")
    print(f"  [{mode}] Done → {out_dir}")


if __name__ == "__main__":
    print("Loading image IDs from metadata...")
    df = pd.read_csv(METADATA_CSV)
    image_ids = df["image_id"].tolist()
    print(f"Total images to process: {len(image_ids)}")

    for mode in PREPROCESSING_MODES:
        print(f"\nPreprocessing mode: '{mode}'...")
        preprocess_dataset(mode, image_ids)

    print("\nAll preprocessing modes completed.")
    print(f"Processed images saved to: {PROCESSED_DIR}")