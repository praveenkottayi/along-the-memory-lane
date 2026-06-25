"""
Ingest processed text files into ChromaDB via LlamaIndex.

Usage:
    python scripts/ingest.py

Reads all .txt files from data/processed/, chunks them,
embeds with nomic-embed-text (via Ollama), stores in ChromaDB.
"""
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR, MEMORY_STORE_DIR, EMBED_MODEL, LLM_MODEL, CHROMA_COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


def setup_llama_index():
    """Configure LlamaIndex to use local Ollama models."""
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120.0)
    Settings.chunk_size = CHUNK_SIZE
    Settings.chunk_overlap = CHUNK_OVERLAP


def get_chroma_store():
    """Initialize ChromaDB persistent client and collection."""
    client = chromadb.PersistentClient(path=str(MEMORY_STORE_DIR))
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    return ChromaVectorStore(chroma_collection=collection), client


def ingest():
    print("Setting up models...")
    setup_llama_index()

    print("Connecting to ChromaDB...")
    vector_store, _ = get_chroma_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load all processed text files
    if not any(PROCESSED_DIR.rglob("*.txt")):
        print(f"No .txt files found in {PROCESSED_DIR}. Run parse_wordpress.py first.")
        sys.exit(1)

    print(f"Loading documents from {PROCESSED_DIR}...")
    reader = SimpleDirectoryReader(
        input_dir=str(PROCESSED_DIR),
        recursive=True,
        required_exts=[".txt"],
    )
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents")

    print("Chunking and embedding (this may take a while)...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    print(f"\nDone. {len(documents)} documents ingested into ChromaDB at {MEMORY_STORE_DIR}")
    return index


if __name__ == "__main__":
    ingest()
