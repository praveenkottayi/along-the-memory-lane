"""
Ingest processed text files into ChromaDB via LlamaIndex.

Usage:
    python scripts/ingest.py

Reads all .txt files from data/processed/, chunks them,
embeds with nomic-embed-text (via Ollama), stores in ChromaDB.

Re-running this script will warn if data already exists and ask
before proceeding to avoid duplicate entries.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

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


def setup_llama_index():
    """Configure LlamaIndex to use local Ollama models."""
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120.0)
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP


def get_chroma_collection():
    """Initialize ChromaDB persistent client and return collection."""
    client = chromadb.PersistentClient(path=str(MEMORY_STORE_DIR))
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    return client, collection


def ingest():
    ensure_dirs()

    print("Setting up models...")
    setup_llama_index()

    print("Connecting to ChromaDB...")
    client, collection = get_chroma_collection()

    # Warn if collection already has data to prevent duplicates
    existing_count = collection.count()
    if existing_count > 0:
        print(f"\nWarning: collection already has {existing_count} entries.")
        print("Re-ingesting will create duplicate chunks and corrupt search results.")
        response = input("Clear existing data and re-ingest? [y/N]: ").strip().lower()
        if response == "y":
            client.delete_collection(CHROMA_COLLECTION)
            collection = client.get_or_create_collection(CHROMA_COLLECTION)
            print("Cleared existing collection.")
        else:
            print("Aborted. Existing data kept.")
            sys.exit(0)

    # Load all processed text files
    if not any(PROCESSED_DIR.rglob("*.txt")):
        print(f"No .txt files found in {PROCESSED_DIR}.")
        print("Run fetch_wordpress_api.py or parse_wordpress.py first.")
        sys.exit(1)

    print(f"Loading documents from {PROCESSED_DIR}...")
    reader = SimpleDirectoryReader(
        input_dir=str(PROCESSED_DIR),
        recursive=True,
        required_exts=[".txt"],
    )
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Chunking and embedding (this may take a while)...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"\nDone. {len(documents)} documents ingested into ChromaDB at {MEMORY_STORE_DIR}")


if __name__ == "__main__":
    ingest()
