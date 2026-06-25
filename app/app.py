"""
Along the Memory Lane — Streamlit UI
Query your memories by date, event, or natural language.

Run: streamlit run app/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
import streamlit as st
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import CHROMA_COLLECTION, EMBED_MODEL, LLM_MODEL, MEMORY_STORE_DIR

st.set_page_config(page_title="Along the Memory Lane", page_icon="📖", layout="wide")

st.title("📖 Along the Memory Lane")
st.caption("Query 13 years of memories — journals, blogs, notes")


@st.cache_resource
def load_index():
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120.0)

    client = chromadb.PersistentClient(path=str(MEMORY_STORE_DIR))
    try:
        collection = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        st.error("Memory store not found. Run `python scripts/ingest.py` first.")
        st.stop()

    if collection.count() == 0:
        st.error("Memory store is empty. Run `python scripts/ingest.py` first.")
        st.stop()

    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store)


# Sidebar filters
with st.sidebar:
    st.header("Filters")
    date_from = st.date_input("From date", value=None)
    date_to = st.date_input("To date", value=None)
    source_filter = st.multiselect(
        "Source",
        options=["blog", "journal", "notes"],
        default=["blog", "journal", "notes"],
    )
    top_k = st.slider("Results to retrieve", min_value=3, max_value=20, value=5)
    score_threshold = st.slider(
        "Min relevance score",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Chunks below this score are excluded before Llama sees them. Higher = stricter."
    )

# Main query
query = st.text_input(
    "What do you want to remember?",
    placeholder="e.g. 'What was I doing in Goa in 2015?' or 'memories about my first job'"
)

if query:
    with st.spinner("Searching memories..."):
        try:
            index = load_index()

            # Build metadata filters
            filter_list = []
            if date_from:
                filter_list.append(MetadataFilter(key="date", value=str(date_from), operator=FilterOperator.GTE))
            if date_to:
                filter_list.append(MetadataFilter(key="date", value=str(date_to), operator=FilterOperator.LTE))
            if source_filter and len(source_filter) < 3:
                for src in source_filter:
                    filter_list.append(MetadataFilter(key="source", value=src, operator=FilterOperator.EQ))

            # Retrieve more than top_k so we can apply score threshold and still have enough
            fetch_k = min(top_k * 2, 20)

            retriever = index.as_retriever(similarity_top_k=fetch_k)
            nodes = retriever.retrieve(query)

            # Apply score threshold BEFORE sending to LLM
            filtered_nodes = [n for n in nodes if n.score is None or n.score >= score_threshold]
            filtered_nodes = filtered_nodes[:top_k]

            if not filtered_nodes:
                st.warning("No memories found above the relevance threshold. Try lowering the score or broadening the query.")
            else:
                # Synthesize answer from filtered nodes only
                from llama_index.core.response_synthesizers import get_response_synthesizer
                synthesizer = get_response_synthesizer(response_mode="tree_summarize")
                response = synthesizer.synthesize(query, nodes=filtered_nodes)

                st.markdown("### Answer")
                st.write(str(response))

                st.markdown("---")
                st.caption(f"{len(filtered_nodes)} of {len(nodes)} chunks passed the relevance threshold ({score_threshold})")
                st.markdown("### Source Memories")

                for node in filtered_nodes:
                    meta = node.metadata or {}
                    date = meta.get("date", "Unknown date")
                    source = meta.get("source", "unknown")
                    title = meta.get("title", "Untitled")

                    with st.expander(f"[{source.upper()}] {date} — {title}"):
                        st.write(node.text[:800] + ("..." if len(node.text) > 800 else ""))
                        if node.score:
                            st.caption(f"Relevance score: {node.score:.3f}")

        except ConnectionError:
            st.error("Could not connect to Ollama.")
            st.info("Make sure Ollama is running — open the Ollama menu bar app.")
        except Exception as e:
            st.error(f"Unexpected error: {type(e).__name__}: {e}")

else:
    st.info("Type a question above to search your memories.")

    with st.expander("Example queries"):
        st.markdown("""
- *What was I feeling in January 2014?*
- *Memories about travel*
- *What did I write about my family in 2018?*
- *Moments of happiness*
- *What were my goals in 2016?*
        """)
