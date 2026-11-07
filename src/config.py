from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "modelo.pt"

# Hiperparámetros
SEED = 42
INPUT_DIM = 2
HIDDEN_DIM = 16
NUM_CLASSES = 2
LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCHS = 50
TEST_SIZE = 0.2
