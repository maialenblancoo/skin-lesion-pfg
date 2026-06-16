import os

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
SPLITS_DIR      = os.path.join(DATA_DIR, "splits")

IMG_PART1_DIR   = os.path.join(RAW_DIR, "HAM10000_images_part_1")
IMG_PART2_DIR   = os.path.join(RAW_DIR, "HAM10000_images_part_2")
METADATA_CSV    = os.path.join(RAW_DIR, "HAM10000_metadata.csv")

OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
MODELS_DIR      = os.path.join(OUTPUTS_DIR, "models")
METRICS_DIR     = os.path.join(OUTPUTS_DIR, "metrics")
FIGURES_DIR     = os.path.join(OUTPUTS_DIR, "figures")
XAI_DIR         = os.path.join(OUTPUTS_DIR, "xai")

# ── Classes ───────────────────────────────────────────────────────────────────
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# Full names for visualizations
CLASS_NAMES_FULL = {
    "akiec": "Actinic Keratosis",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesion",
}

# ── Image ─────────────────────────────────────────────────────────────────────
IMAGE_SIZE = 224  # B0 and B1; B2 uses 260 but we start with 224
CHANNELS   = 3

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-5
NUM_WORKERS   = 4

# ── K-Fold ────────────────────────────────────────────────────────────────────
N_FOLDS     = 5
RANDOM_SEED = 42

# ── Models to experiment with ─────────────────────────────────────────────────
EFFICIENTNET_VERSIONS = ["b0", "b1", "b2"]

# ── Preprocessing modes to experiment with ────────────────────────────────────
PREPROCESSING_MODES = ["none", "dullrazor", "colorconstancy", "both"]

# ── Uncertainty rejection ─────────────────────────────────────────────────────
UNCERTAINTY_THRESHOLD = 0.15  # cases below this threshold are flagged for review