import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import IMAGE_SIZE


def get_train_transforms():
    """
    Augmentation pipeline for training.
    Includes geometric and color transforms to improve generalization.
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),

        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.3),

        # Color augmentations
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),

        # Normalization (ImageNet stats, since we use pretrained EfficientNet)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms():
    """
    Minimal pipeline for validation and test — no augmentation, only resize and normalize.
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])