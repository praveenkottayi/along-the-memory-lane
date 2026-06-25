"""
Central config for Along the Memory Lane.
All paths and model settings live here.

Call ensure_dirs() once at the start of each script to create required directories.
Do not rely on import-time side effects.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MEMORY_STORE_DIR = BASE_DIR / "memory_store"

RAW_JOURNAL_DIR = RAW_DIR / "journal"
RAW_BLOG_DIR = RAW_DIR / "blog"
RAW_NOTES_DIR = RAW_DIR / "notes"

# Ollama models
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b-instruct-q4_0"

# ChromaDB
CHROMA_COLLECTION = "memories"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def ensure_dirs():
    """Create required data directories. Call once at script startup."""
    for d in [RAW_JOURNAL_DIR, RAW_BLOG_DIR, RAW_NOTES_DIR, PROCESSED_DIR, MEMORY_STORE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
