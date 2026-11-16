import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", PROJECT_ROOT / "data"))     # Colab: /content/data (로컬 디스크)
INDEX_DIR = DATA_DIR / "index"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"