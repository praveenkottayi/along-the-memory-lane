"""
Ingest processed text files into ChromaDB via LlamaIndex.

Usage:
    python scripts/ingest.py                  # full rebuild (asks before wiping)
    python scripts/ingest.py --yes            # full rebuild, no prompt
    python scripts/ingest.py --incremental    # sync: add new, update changed, remove deleted

Reads all .txt files from data/processed/, parses their front-matter header
into document metadata (date, source, title, etc.), chunks the body, embeds
with nomic-embed-text (via Ollama), and stores in ChromaDB.

Metadata is propagated to every chunk so the UI filters (date range, source)
work correctly. This relies on parse_front_matter() in common.py — the same
format used by all ingestion scripts (fetch_wordpress_api, ocr_journals, etc.).

Incremental mode keeps a content fingerprint (SHA-256) of every ingested file
in memory_store/ingested.json, so re-running only touches what changed:
new files are added, edited files (e.g. an updated blog post or a re-described
photo) have their old chunks replaced, and deleted files have their chunks
removed. Safe to run as often as you like.
"""
import hashlib
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


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ingested_files() -> dict[str, str | None]:
    """Load {file path: content hash} for already-ingested files.

    Older tracking files are a plain list of paths with no hashes; those
    entries get hash None, so the next sync re-embeds them once.
    """
    if not TRACKING_FILE.exists():
        return {}
    data = json.loads(TRACKING_FILE.read_text())
    if isinstance(data, list):
        return {path: None for path in data}
    return data


def save_ingested_files(ingested: dict[str, str | None]):
    """Persist {file path: content hash} to the tracking file."""
    TRACKING_FILE.write_text(json.dumps(dict(sorted(ingested.items())), indent=2))


def delete_file_chunks(collection, path: str):
    """Remove every chunk that came from one source file."""
    collection.delete(where={"file_path": path})


def has_front_matter(path: Path) -> bool:
    """True if the file carries the front-matter header (date, source, ...).

    Files without it are not memory entries but leftovers — e.g. full-journal
    transcriptions that split_journal_txt.py splits into per-entry files, or
    raw photo description dumps. Indexing them duplicates entries as undated
    chunks that can't be filtered by date or source.
    """
    metadata, _ = parse_front_matter(path.read_text(encoding="utf-8"))
    return bool(metadata.get("source"))


def get_all_processed_files() -> list[Path]:
    """Return all .txt files in the processed directory that have front matter."""
    files, skipped = [], []
    for f in sorted(PROCESSED_DIR.rglob("*.txt")):
        (files if has_front_matter(f) else skipped).append(f)
    if skipped:
        # Mostly photo description caches (photos/<album>/descriptions/) — expected.
        print(f"  Skipping {len(skipped)} file(s) without front matter, e.g. "
              f"{skipped[0].relative_to(PROCESSED_DIR)}")
    return files


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
    the UI.  Files with no front-matter are skipped by get_all_processed_files().
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


def embed_files(collection, files: list[Path]) -> int:
    """Chunk, embed and store files. Returns the number of documents ingested."""
    documents = load_documents(files)
    if not documents:
        return 0
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    print(f"Chunking and embedding {len(documents)} document(s)...")
    VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress=True)
    return len(documents)


def sync(collection, all_files: list[Path]):
    """Bring ChromaDB in line with data/processed/: add, update, remove."""
    tracked = load_ingested_files()
    current = {str(f): file_hash(f) for f in all_files}

    new = [p for p in current if p not in tracked]
    changed = [p for p in current if p in tracked and tracked[p] != current[p]]
    removed = [p for p in tracked if p not in current]

    if not (new or changed or removed):
        print("Everything is up to date.")
        return

    print(f"New: {len(new)}   Changed: {len(changed)}   Removed: {len(removed)}")
    for label, paths in (("+", new), ("~", changed), ("-", removed)):
        for p in paths[:20]:
            print(f"  {label} {Path(p).relative_to(PROCESSED_DIR) if p.startswith(str(PROCESSED_DIR)) else p}")
        if len(paths) > 20:
            print(f"  {label} ... and {len(paths) - 20} more")

    # Old chunks of changed and removed files go first, so nothing is duplicated.
    for p in changed + removed:
        delete_file_chunks(collection, p)
        tracked.pop(p, None)

    embed_files(collection, [Path(p) for p in new + changed])
    for p in new + changed:
        tracked[p] = current[p]
    save_ingested_files(tracked)
    print(f"\nDone. {collection.count()} chunks from {len(tracked)} files.")


def rebuild(client, collection, all_files: list[Path], assume_yes: bool):
    """Wipe the collection and ingest everything from scratch."""
    if collection.count() > 0:
        print(f"\nCollection has {collection.count()} chunks; a full rebuild wipes them.")
        if not assume_yes:
            response = input("Clear existing data and re-ingest? [y/N]: ").strip().lower()
            if response != "y":
                print("Aborted. Use --incremental to sync only what changed.")
                sys.exit(0)
        client.delete_collection(CHROMA_COLLECTION)
        collection = client.get_or_create_collection(CHROMA_COLLECTION)
        print("Cleared existing collection.")

    embed_files(collection, all_files)
    save_ingested_files({str(f): file_hash(f) for f in all_files})
    print(f"\nDone. {collection.count()} chunks from {len(all_files)} files.")


def ingest(incremental: bool = False, assume_yes: bool = False):
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
        sync(collection, all_files)
    else:
        rebuild(client, collection, all_files, assume_yes)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Ingest processed files into ChromaDB.")
    ap.add_argument(
        "--incremental",
        action="store_true",
        help="Sync only what changed: add new files, update edited ones, remove deleted ones."
    )
    ap.add_argument("--yes", action="store_true", help="Full rebuild without the confirmation prompt.")
    args = ap.parse_args()
    ingest(incremental=args.incremental, assume_yes=args.yes)


if __name__ == "__main__":
    main()
