import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from config import (
    METADATA_CSV, SPLITS_DIR, N_FOLDS, RANDOM_SEED, CLASS_TO_IDX,
    IMG_PART1_DIR, IMG_PART2_DIR
)


def load_metadata():
    """Load and prepare the HAM10000 metadata CSV."""
    df = pd.read_csv(METADATA_CSV)

    def find_image_path(image_id):
        for folder in [IMG_PART1_DIR, IMG_PART2_DIR]:
            path = os.path.join(folder, f"{image_id}.jpg")
            if os.path.exists(path):
                return path
        return None

    df["image_path"] = df["image_id"].apply(find_image_path)
    missing = df["image_path"].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} images not found on disk.")

    df = df.dropna(subset=["image_path"])
    df["label"] = df["dx"].map(CLASS_TO_IDX)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return df


def split_test_set(df, test_size=0.15):
    """
    Split off a fixed test set at the patient level (patient-aware).
    Ensures no patient appears in both train/val and test.
    """
    # Get unique patients and their dominant label
    patient_labels = (
        df.groupby("lesion_id")["label"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
    )

    train_val_patients, test_patients = train_test_split(
        patient_labels["lesion_id"],
        test_size=test_size,
        stratify=patient_labels["label"],
        random_state=RANDOM_SEED
    )

    train_val_df = df[df["lesion_id"].isin(train_val_patients)].reset_index(drop=True)
    test_df      = df[df["lesion_id"].isin(test_patients)].reset_index(drop=True)

    return train_val_df, test_df


def make_patient_aware_splits(train_val_df):
    """
    Create K-Fold splits on the train+val set ensuring that images
    from the same patient are never split across train and validation.
    """
    os.makedirs(SPLITS_DIR, exist_ok=True)

    patient_labels = (
        train_val_df.groupby("lesion_id")["label"]
        .agg(lambda x: x.value_counts().idxmax())
        .reset_index()
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(patient_labels["lesion_id"], patient_labels["label"])
    ):
        train_patients = patient_labels.iloc[train_idx]["lesion_id"].values
        val_patients   = patient_labels.iloc[val_idx]["lesion_id"].values

        train_df = train_val_df[train_val_df["lesion_id"].isin(train_patients)].reset_index(drop=True)
        val_df   = train_val_df[train_val_df["lesion_id"].isin(val_patients)].reset_index(drop=True)

        train_df.to_csv(os.path.join(SPLITS_DIR, f"fold_{fold}_train.csv"), index=False)
        val_df.to_csv(os.path.join(SPLITS_DIR, f"fold_{fold}_val.csv"),   index=False)

        print(f"Fold {fold} → Train: {len(train_df)} | Val: {len(val_df)}")

    print(f"\nK-Fold splits saved to {SPLITS_DIR}")


def compute_class_weights(df):
    """
    Compute class weights for Weighted Cross Entropy.
    Returns a list of weights ordered by CLASS_TO_IDX.
    """
    counts    = df["label"].value_counts().sort_index()
    total     = len(df)
    n_classes = len(counts)

    weights = total / (n_classes * counts)
    weights = weights / weights.sum() * n_classes

    print("\nClass weights for Weighted Cross Entropy:")
    for idx, w in enumerate(weights):
        print(f"  Class {idx}: {w:.4f}")

    return weights.values.tolist()


if __name__ == "__main__":
    print("Loading metadata...")
    df = load_metadata()
    print(f"Total samples: {len(df)}")
    print(f"\nClass distribution:\n{df['dx'].value_counts()}\n")

    print("Splitting off fixed test set (15%)...")
    train_val_df, test_df = split_test_set(df, test_size=0.15)
    print(f"Train+Val: {len(train_val_df)} | Test: {len(test_df)}")

    # Save test set
    test_df.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False)
    print(f"Test set saved to {SPLITS_DIR}/test.csv")

    print("\nCreating patient-aware K-Fold splits on train+val...")
    make_patient_aware_splits(train_val_df)

    print("\nComputing class weights (based on train+val only)...")
    weights = compute_class_weights(train_val_df)

    weights_path = os.path.join(SPLITS_DIR, "class_weights.npy")
    np.save(weights_path, np.array(weights))
    print(f"Class weights saved to {weights_path}")