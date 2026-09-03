import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "knowledge.pdf"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
NVIDIA_MODEL_NAME = "openai/gpt-oss-20b"

TOP_K = 3

# OCR Configuration: "auto" (fallback for scanned pages), "force" (always OCR), or "off"
OCR_MODE = os.getenv("OCR_MODE", "auto")
OCR_LANGUAGES = ["en"]

