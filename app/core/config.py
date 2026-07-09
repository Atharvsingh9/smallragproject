from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR=BASE_DIR/ "data"
PDF_PATH=DATA_DIR/ "knowledge.pdf"

CHUNK_SIZE=500
CHUNK_OVERLAP=50

EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"
GROQ_MODEL_NAME="llama-3.3-70b-versatile"

TOP_K=3
