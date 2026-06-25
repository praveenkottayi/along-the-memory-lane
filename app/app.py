"""
Along the Memory Lane — Streamlit UI
Query your memories by date, event, or natural language.

Run: streamlit run app/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from config import MEMORY_STORE_DIR, EMBED_MODEL, LLM_MODEL, CHROMA_COLLECTION

st.set_page_config(page_title="Along the Memory Lane", page_icon="📖", layout="wide")

st.title("📖 Along the Memory Lane")
st.caption("Query 13 years of memories — journals, blogs, notes")


@st.cache_resource
def load_index():
    Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL)
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=120.0)

    client = chromadb.PersistentClient(path=str(MEMORY_STORE_DIR))
    collection = client.get_or_create_collection(CHROMA_COLLECTION)
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

# Main query
query = st.text_input("What do you want to remember?",
                       placeholder="e.g. 'What was I doing in Goa in 2015?' or 'memories about my first job'")

if query:
    with st.spinner("Searching memories..."):
        try:
            index = load_index()

            # Build metadata filters
            filters = {}
            if date_from:
                filters["date_from"] = str(date_from)
            if date_to:
                filters["date_to"] = str(date_to)

            query_engine = index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="tree_summarize",
            )

            response = query_engine.query(query)

            st.markdown("### 🧠 Answer")
            st.write(str(response))

            # Show source documents
            if hasattr(response, "source_nodes") and response.source_nodes:
                st.markdown("---")
                st.markdown("### 📄 Source Memories")
                for i, node in enumerate(response.source_nodes):
                    meta = node.metadata or {}
                    date = meta.get("date", "Unknown date")
                    source = meta.get("source", "unknown")
                    title = meta.get("title", "Untitled")

                    with st.expander(f"[{source.upper()}] {date} — {title}"):
                        st.write(node.text[:800] + ("..." if len(node.text) > 800 else ""))
                        st.caption(f"Relevance score: {node.score:.3f}" if node.score else "")

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Make sure Ollama is running: `ollama serve`")

else:
    st.info("Type a question above to search your memories.")

    with st.expander("💡 Example queries"):
        st.markdown("""
- *What was I feeling in January 2014?*
- *Memories about travel*
- *What did I write about my family in 2018?*
- *Moments of happiness*
- *What were my goals in 2016?*
        """)
