"""
Ingest processed text files into ChromaDB via LlamaIndex.

Usage:
    python scripts/ingest.py                  # full ingest (warns if data exists)
    python scripts/ingest.py --incremental    # only ingest new files

Reads all .txt files from data/processed/, parses their front-matter header
into document metadata (date, source, title, etc.), chunks the body, embeds
with nomic-embed-text (via Ollama), and stores in ChromaDB.

Metadata is propagated to every chunk so the UI filters (date range, source)
work correctly. This relies on parse_front_matter() in common.py — the same
format used by all ingestion scripts (fetch_wordpress_api, ocr_journals, etc.).

Incremental mode tracks ingested files in memory_store/ingested.json
so re-running only adds new content — safe to run after adding journals.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from common import parse_front_matter
from config import (
    CHROMA_COLLECTION,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBED_MODEL,
    LLM_MODEL,
    MEMORY_STORE_DIR,
    PROCESSED_DIR,
    ensure_dirs,
)

TRACKING_FILE = MEMORY_STORE_DIR / "ingested.json"


def setup_llama_index():
    """Configure LlamaIndex to use local Ollama models."""
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120.0)
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP


def get_chroma_collection():
    """Initialize ChromaDB persistent client and return client + collection."""
    client = chromadb.PersistentClient(path=str(MEMORY_STORE_DIR))
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    return client, collection


def load_ingested_files() -> set[str]:
    """Load the set of already-ingested file paths from tracking file."""
    if TRACKING_FILE.exists():
        return set(json.loads(TRACKING_FILE.read_text()))
    return set()


def save_ingested_files(ingested: set[str]):
    """Persist the set of ingested file paths to tracking file."""
    TRACKING_FILE.write_text(json.dumps(sorted(ingested), indent=2))


def get_all_processed_files() -> list[Path]:
    """Return all .txt files in the processed directory."""
    return sorted(PROCESSED_DIR.rglob("*.txt"))


def load_documents(files: list[Path]) -> list[Document]:
    """Read processed .txt files and return LlamaIndex Documents with metadata.

    Each file uses the front-matter format written by front_matter() in common.py:
        ---
        date: 2015-06-12
        source: journal
        title: Journal — 2015-06-12
        ---
        Body text...

    The metadata dict is attached to the Document so LlamaIndex propagates it
    to every chunk stored in ChromaDB — enabling date and source filtering in
    the UI.  Files with no front-matter are ingested as plain text with an
    empty metadata dict (graceful fallback).
    """
    docs = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        metadata, body = parse_front_matter(text)
        # Always record the source file path for debugging
        metadata["file_path"] = str(f)
        if body.strip():
            docs.append(Document(text=body, metadata=metadata))
        else:
            print(f"  Warning: empty body in {f.name}, skipping.")
    return docs


def ingest(incremental: bool = False):
    ensure_dirs()

    print("Setting up models...")
    setup_llama_index()

    print("Connecting to ChromaDB...")
    client, collection = get_chroma_collection()

    all_files = get_all_processed_files()
    if not all_files:
        print(f"No .txt files found in {PROCESSED_DIR}.")
        print("Run fetch_wordpress_api.py or ocr_journals.py first.")
        sys.exit(1)

    if incremental:
        # Only ingest files not previously ingested
        ingested = load_ingested_files()
        new_files = [f for f in all_files if str(f) not in ingested]

        if not new_files:
            print("No new files to ingest. Everything is up to date.")
            return

        print(f"Found {len(new_files)} new file(s) to ingest "
              f"({len(ingested)} already ingested):")
        for f in new_files:
            print(f"  + {f.relative_to(PROCESSED_DIR)}")

        files_to_ingest = new_files

    else:
        # Full ingest — warn if data already exists
        existing_count = collection.count()
        if existing_count > 0:
            print(f"\nWarning: collection already has {existing_count} entries.")
            print("Re-ingesting will create duplicate chunks and corrupt search results.")
            response = input("Clear existing data and re-ingest? [y/N]: ").strip().lower()
            if response == "y":
                client.delete_collection(CHROMA_COLLECTION)
                collection = client.get_or_create_collection(CHROMA_COLLECTION)
                # Reset tracking file too
                if TRACKING_FILE.exists():
                    TRACKING_FILE.unlink()
                print("Cleared existing collection.")
            else:
                print("Aborted. Use --incremental to add only new files.")
                sys.exit(0)

        files_to_ingest = all_files
        print(f"Ingesting {len(files_to_ingest)} files...")

    # Load files with front-matter parsed into document metadata
    documents = load_documents(files_to_ingest)
    print(f"Loaded {len(documents)} documents")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Chunking and embedding (this may take a while)...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    # Update tracking file
    ingested = load_ingested_files()
    ingested.update(str(f) for f in files_to_ingest)
    save_ingested_files(ingested)

    print(f"\nDone. {len(documents)} documents ingested.")
    print(f"Total tracked files: {len(ingested)}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Ingest processed files into ChromaDB.")
    ap.add_argument(
        "--incremental",
        action="store_true",
        help="Only ingest new files not previously ingested. Safe to run repeatedly."
    )
    args = ap.parse_args()
    ingest(incremental=args.incremental)


if __name__ == "__main__":
    main()
