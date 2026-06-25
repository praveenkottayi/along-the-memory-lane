"""
Central config for Along the Memory Lane.
All paths and model settings live here.
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
LLM_MODEL = "llama3.1"  # switch to llama3.1:70b if you want more power

# ChromaDB
CHROMA_COLLECTION = "memories"

# Chunking
CHUNK_SIZE = 512       # tokens
CHUNK_OVERLAP = 64

# Ensure directories exist
for d in [RAW_JOURNAL_DIR, RAW_BLOG_DIR, RAW_NOTES_DIR, PROCESSED_DIR, MEMORY_STORE_DIR]:
    d.mkdir(parents=True, exist_ok=True)
