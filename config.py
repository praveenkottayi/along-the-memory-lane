"""
Central config for Along the Memory Lane.
All paths and model settings live here.

Call ensure_dirs() once at the start of each script to create required directories.
Do not rely on import-time side effects.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env so ANTHROPIC_API_KEY (and any other secrets) are available to the
# Anthropic SDK without exporting them in the shell. Safe no-op if .env is absent.
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MEMORY_STORE_DIR = BASE_DIR / "memory_store"

RAW_JOURNAL_DIR = RAW_DIR / "journal"
RAW_BLOG_DIR = RAW_DIR / "blog"
RAW_NOTES_DIR = RAW_DIR / "notes"

# Ollama models (local, used for embedding + chat)
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b-instruct-q4_0"

# Handwriting OCR — TEMPORARY: runs through the Anthropic API while the pipeline
# is being trialled. Goal is fully local OCR (see VISION.md), but every local
# vision model tried so far failed on cursive: Apple Vision (garbled), llava
# (garbage), moondream (empty), Tesseract (garbage), and llama3.2-vision won't
# even load — its 'mllama' architecture is unsupported by the Ollama llama.cpp
# runner. Claude reads the handwriting cleanly, so it's the stopgap until
# local vision works. Needs ANTHROPIC_API_KEY in the environment.
OCR_VISION_MODEL = "claude-opus-4-8"

# Glossary — single source of truth for personal names & abbreviations.
# Lives in glossary.txt (gitignored to keep personal names private). It powers
# BOTH the LLM system prompt (so answers read correctly) AND query expansion in
# the app (so "Anu" actually retrieves the right chunks). Falls back to the
# JOURNAL_GLOSSARY env var if the file is absent.
GLOSSARY_FILE = BASE_DIR / "glossary.txt"


def _load_glossary():
    """Parse glossary.txt into (text_block, shorthand->expansion map).

    Each non-comment line is `shorthand = Expanded form`. The text block is used
    for the system prompt; the map drives query expansion before retrieval.
    """
    mapping = {}
    if GLOSSARY_FILE.exists():
        for line in GLOSSARY_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip().lower(), value.strip()
            if key and value:
                mapping[key] = value
    if mapping:
        text = "; ".join(f"{k} = {v}" for k, v in mapping.items())
    else:
        # Fall back to the legacy single-line env var.
        text = os.getenv("JOURNAL_GLOSSARY", "")
    return text, mapping


GLOSSARY, GLOSSARY_MAP = _load_glossary()

_default_system_prompt = (
    "You are a personal memory assistant helping the author recall entries "
    "from their own journals and blog posts.\n"
    "When answering, refer to the author in second person (\"you wrote...\", \"you felt...\").\n"
    + (f"Use the following glossary to resolve names and abbreviations:\n{GLOSSARY}\n" if GLOSSARY else "")
    + "Always ground your answer in the retrieved excerpts. If nothing relevant was found, say so honestly."
)

SYSTEM_PROMPT = os.getenv("JOURNAL_SYSTEM_PROMPT", _default_system_prompt)

# ChromaDB
CHROMA_COLLECTION = "memories"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def ensure_dirs():
    """Create required data directories. Call once at script startup."""
    for d in [RAW_JOURNAL_DIR, RAW_BLOG_DIR, RAW_NOTES_DIR, PROCESSED_DIR, MEMORY_STORE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
