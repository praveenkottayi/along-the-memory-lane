"""
Retrieval + answer pipeline for Along the Memory Lane.

Shared by the Streamlit app (app/app.py) and the evaluation script
(scripts/evaluate.py), so the eval always measures exactly what the app runs.
Every retrieval improvement (filters, hybrid search, LangGraph routing...)
should land here first and be checked with the eval before the app uses it.
"""
import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import FilterCondition, FilterOperator, MetadataFilter, MetadataFilters
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    CHROMA_COLLECTION,
    EMBED_MODEL,
    GLOSSARY_MAP,
    LLM_MODEL,
    MEMORY_STORE_DIR,
    SYSTEM_PROMPT,
)

ALL_SOURCES = ["blog", "journal", "photo", "notes"]

# Defaults match the app's sidebar defaults.
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.4  # 0.5 dropped good chunks: eval hit rate 0.52 -> 0.83 at 0.4


class MemoryStoreError(RuntimeError):
    """Raised when the ChromaDB memory store is missing or empty."""


def load_index() -> VectorStoreIndex:
    """Configure the local Ollama models and open the existing ChromaDB index."""
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120.0, system_prompt=SYSTEM_PROMPT)

    client = chromadb.PersistentClient(path=str(MEMORY_STORE_DIR))
    try:
        collection = client.get_collection(CHROMA_COLLECTION)
    except Exception as e:
        raise MemoryStoreError("Memory store not found. Run `python scripts/ingest.py` first.") from e

    if collection.count() == 0:
        raise MemoryStoreError("Memory store is empty. Run `python scripts/ingest.py` first.")

    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store)


def expand_query(query: str) -> str:
    """Expand shorthand names/abbreviations in query using the shared glossary.

    Source of truth is glossary.txt (loaded in config as GLOSSARY_MAP), so the
    same entries drive both retrieval expansion here and the LLM system prompt.
    """
    words = query.split()
    expanded = [GLOSSARY_MAP.get(w.lower().rstrip("?.,"), w) for w in words]
    return " ".join(expanded)


def build_filters(date_from=None, date_to=None, sources=None) -> MetadataFilters | None:
    """Build ChromaDB metadata filters from an optional date range and source list."""
    filter_list = []
    if date_from:
        filter_list.append(MetadataFilter(key="date", value=str(date_from), operator=FilterOperator.GTE))
    if date_to:
        filter_list.append(MetadataFilter(key="date", value=str(date_to), operator=FilterOperator.LTE))
    if sources and len(sources) < len(ALL_SOURCES):
        # Use OR so selecting "blog + journal" returns chunks from either source,
        # not chunks that are simultaneously both (which would always be empty).
        source_filters = [
            MetadataFilter(key="source", value=src, operator=FilterOperator.EQ)
            for src in sources
        ]
        filter_list.append(MetadataFilters(filters=source_filters, condition=FilterCondition.OR))
    return MetadataFilters(filters=filter_list) if filter_list else None


def search(
    index: VectorStoreIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: MetadataFilters | None = None,
) -> list[NodeWithScore]:
    """Vector search for an (already expanded) query.

    Fetches more than top_k so the score threshold can still leave enough.
    """
    fetch_k = min(top_k * 2, 20)
    retriever = index.as_retriever(similarity_top_k=fetch_k, filters=filters)
    return retriever.retrieve(query)


def apply_threshold(
    fetched: list[NodeWithScore],
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[NodeWithScore]:
    """Keep the chunks the LLM will see: score >= threshold, best top_k.

    Applied BEFORE the LLM. None-scored chunks are excluded — they have no
    relevance signal.
    """
    kept = [n for n in fetched if n.score is not None and n.score >= score_threshold]
    return kept[:top_k]


def retrieve(
    index: VectorStoreIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    filters: MetadataFilters | None = None,
) -> tuple[list[NodeWithScore], list[NodeWithScore]]:
    """Retrieve chunks for a query: glossary expansion → search → threshold.

    Returns (kept, fetched): `fetched` is everything the vector search returned,
    `kept` is what passes the score threshold, capped at top_k — the chunks the
    LLM will actually see. graph.py runs the same three steps as graph nodes.
    """
    fetched = search(index, expand_query(query), top_k, filters)
    return apply_threshold(fetched, top_k, score_threshold), fetched


def synthesize(query: str, nodes: list[NodeWithScore]) -> str:
    """Answer the query from the given chunks only."""
    synthesizer = get_response_synthesizer(response_mode="tree_summarize")
    return str(synthesizer.synthesize(query, nodes=nodes))
