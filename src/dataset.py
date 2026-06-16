import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from config import PROCESSED_DIR, CLASS_TO_IDX


class SkinLesionDataset(Dataset):
    """
    PyTorch Dataset for the HAM10000 skin lesion dataset.

    Args:
        df:           DataFrame with at least 'image_id' and 'label' columns.
        preprocessing_mode: One of 'none', 'dullrazor', 'colorconstancy', 'both'.
        transform:    Albumentations transform pipeline (train or val).
    """

    def __init__(self, df: pd.DataFrame, preprocessing_mode: str, transform=None):
        self.df                 = df.reset_index(drop=True)
        self.preprocessing_mode = preprocessing_mode
        self.transform          = transform
        self.image_dir          = os.path.join(PROCESSED_DIR, preprocessing_mode)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        image_id = row["image_id"]
        label    = int(row["label"])

        # Load preprocessed image
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Convert BGR → RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms (resize, augmentation, normalization)
        if self.transform is not None:
            augmented = self.transform(image=image)
            image     = augmented["image"]

        return image, label

    def get_labels(self):
        """Return all labels as a list (useful for computing class weights)."""
        return self.df["label"].tolist()


def load_fold(fold: int, preprocessing_mode: str, train_transform, val_transform):
    """
    Load train and validation DataFrames for a given fold and preprocessing mode.

    Args:
        fold:               Fold index (0 to N_FOLDS-1).
        preprocessing_mode: Preprocessing mode string.
        train_transform:    Albumentations pipeline for training.
        val_transform:      Albumentations pipeline for validation.

    Returns:
        train_dataset, val_dataset
    """
    from config import SPLITS_DIR

    train_df = pd.read_csv(os.path.join(SPLITS_DIR, f"fold_{fold}_train.csv"))
    val_df   = pd.read_csv(os.path.join(SPLITS_DIR, f"fold_{fold}_val.csv"))

    train_dataset = SkinLesionDataset(train_df, preprocessing_mode, train_transform)
    val_dataset   = SkinLesionDataset(val_df,   preprocessing_mode, val_transform)

    return train_dataset, val_dataset


def load_test(preprocessing_mode: str, val_transform):
    """
    Load the fixed test set.

    Args:
        preprocessing_mode: Preprocessing mode string.
        val_transform:      Albumentations pipeline (no augmentation).

    Returns:
        test_dataset
    """
    from config import SPLITS_DIR

    test_df = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
    return SkinLesionDataset(test_df, preprocessing_mode, val_transform)

class MultimodalSkinLesionDataset(Dataset):
    """
    PyTorch Dataset for multimodal training (image + clinical metadata).

    Args:
        df:                 DataFrame with image_id, label and clinical columns.
        preprocessing_mode: One of 'none', 'dullrazor', 'colorconstancy', 'both'.
        metadata_cols:      List of raw metadata column names to use.
                            e.g. ['sex', 'age', 'localization']
        transform:          Albumentations transform pipeline.
        age_mean:           Mean age for imputation (computed from train set).
    """

    # Fixed encoding maps
    SEX_CATEGORIES = ['male', 'female', 'unknown']
    LOC_CATEGORIES = [
        'abdomen', 'acral', 'back', 'chest', 'ear', 'face',
        'foot', 'genital', 'hand', 'lower extremity', 'neck',
        'scalp', 'trunk', 'unknown', 'upper extremity'
    ]

    def __init__(
        self,
        df: pd.DataFrame,
        preprocessing_mode: str,
        metadata_cols: list,
        transform=None,
        age_mean: float = None,
    ):
        self.df                 = df.reset_index(drop=True)
        self.preprocessing_mode = preprocessing_mode
        self.metadata_cols      = metadata_cols
        self.transform          = transform
        self.image_dir          = os.path.join(PROCESSED_DIR, preprocessing_mode)
        self.age_mean           = age_mean if age_mean is not None else 51.9

    def __len__(self):
        return len(self.df)

    def _encode_metadata(self, row) -> np.ndarray:
        """
        Encode clinical metadata into a fixed-length float vector.

        Encoding:
            sex          → one-hot (3): male, female, unknown
            age          → scalar normalized by 90, missing → age_mean/90
            localization → one-hot (15 categories)
        """
        features = []

        if 'sex' in self.metadata_cols:
            sex_val = str(row.get('sex', 'unknown')).lower()
            if sex_val not in self.SEX_CATEGORIES:
                sex_val = 'unknown'
            one_hot = [1.0 if sex_val == c else 0.0 for c in self.SEX_CATEGORIES]
            features.extend(one_hot)

        if 'age' in self.metadata_cols:
            age = row.get('age', None)
            if pd.isna(age) or age is None:
                age = self.age_mean
            features.append(float(age) / 90.0)

        if 'localization' in self.metadata_cols:
            loc_val = str(row.get('localization', 'unknown')).lower()
            if loc_val not in self.LOC_CATEGORIES:
                loc_val = 'unknown'
            one_hot = [1.0 if loc_val == c else 0.0 for c in self.LOC_CATEGORIES]
            features.extend(one_hot)

        return np.array(features, dtype=np.float32)

    def get_metadata_dim(self) -> int:
        """Return the dimension of the metadata vector."""
        dim = 0
        if 'sex'          in self.metadata_cols: dim += 3
        if 'age'          in self.metadata_cols: dim += 1
        if 'localization' in self.metadata_cols: dim += 15
        return dim

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        image_id = row['image_id']
        label    = int(row['label'])

        # Load preprocessed image
        image_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        image      = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f'Image not found: {image_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)['image']

        # Encode metadata
        metadata = self._encode_metadata(row)
        metadata = torch.tensor(metadata, dtype=torch.float32)

        return image, metadata, label

    def get_labels(self):
        return self.df['label'].tolist()


def load_fold_multimodal(
    fold: int,
    preprocessing_mode: str,
    metadata_cols: list,
    train_transform,
    val_transform,
):
    """
    Load train and validation multimodal datasets for a given fold.

    Returns:
        train_dataset, val_dataset, metadata_dim
    """
    from config import SPLITS_DIR

    train_df = pd.read_csv(os.path.join(SPLITS_DIR, f'fold_{fold}_train.csv'))
    val_df   = pd.read_csv(os.path.join(SPLITS_DIR, f'fold_{fold}_val.csv'))

    # Compute age mean from train set only (avoid data leakage)
    age_mean = train_df['age'].mean()

    train_dataset = MultimodalSkinLesionDataset(
        train_df, preprocessing_mode, metadata_cols, train_transform, age_mean
    )
    val_dataset = MultimodalSkinLesionDataset(
        val_df, preprocessing_mode, metadata_cols, val_transform, age_mean
    )

    metadata_dim = train_dataset.get_metadata_dim()
    return train_dataset, val_dataset, metadata_dim


def load_test_multimodal(
    preprocessing_mode: str,
    metadata_cols: list,
    val_transform,
    age_mean: float = 51.9,
):
    """
    Load the fixed test set for multimodal evaluation.

    Returns:
        test_dataset
    """
    from config import SPLITS_DIR

    test_df = pd.read_csv(os.path.join(SPLITS_DIR, 'test.csv'))
    return MultimodalSkinLesionDataset(
        test_df, preprocessing_mode, metadata_cols, val_transform, age_mean
    )