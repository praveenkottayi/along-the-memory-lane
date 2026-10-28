"""
Along the Memory Lane — Streamlit UI
Query your memories by date, event, or natural language.

Run: streamlit run app/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from graph import ask_stream
from rag import ALL_SOURCES, DEFAULT_SCORE_THRESHOLD, DEFAULT_TOP_K, MemoryStoreError, build_filters
from rag import load_index as _load_index

st.set_page_config(page_title="Along the Memory Lane", page_icon="📖", layout="wide")

st.title("📖 Along the Memory Lane")
st.caption("Query 13 years of memories — journals, blogs, photos")


@st.cache_resource
def load_index():
    try:
        return _load_index()
    except MemoryStoreError as e:
        st.error(str(e))
        st.stop()


# Sidebar filters
with st.sidebar:
    st.header("Filters")
    date_from = st.date_input("From date", value=None)
    date_to = st.date_input("To date", value=None)
    source_filter = st.multiselect(
        "Source",
        options=ALL_SOURCES,
        default=ALL_SOURCES,
    )
    top_k = st.slider("Results to retrieve", min_value=3, max_value=20, value=DEFAULT_TOP_K)
    score_threshold = st.slider(
        "Min relevance score",
        min_value=0.0, max_value=1.0, value=DEFAULT_SCORE_THRESHOLD, step=0.05,
        help="Chunks below this score are excluded before Llama sees them. Higher = stricter."
    )

# Main query
query = st.text_input(
    "What do you want to remember?",
    placeholder="e.g. 'What was I doing in Goa in 2015?' or 'memories about my first job'"
)

def step_message(node: str, state: dict) -> str:
    """One line per finished graph step, shown in the progress box."""
    if node == "expand":
        changed = state["expanded_query"] != query
        return f"🔁 Glossary: {state['expanded_query']!r}" if changed else "🔁 Glossary: no names to expand"
    if node == "search":
        return f"🔍 Found {len(state['fetched'])} candidate chunks"
    if node == "threshold":
        return f"✂️ {len(state['chunks'])} passed the relevance threshold ({score_threshold})"
    if node == "answer":
        return "🤖 Answer written"
    return node


if query:
    try:
        index = load_index()
        filters = build_filters(date_from, date_to, source_filter)

        # Run the LangGraph pipeline (graph.py) step by step, so each step shows as it finishes.
        state = {}
        with st.status("Searching memories...", expanded=False) as status:
            for node, changes in ask_stream(index, query, top_k, score_threshold, filters):
                state.update(changes or {})
                st.write(step_message(node, state))
                if node == "threshold" and state["chunks"]:
                    status.update(label="Writing the answer...")
            found = bool(state.get("chunks"))
            status.update(label="Done" if found else "Nothing found", state="complete")

        filtered_nodes, nodes = state.get("chunks", []), state.get("fetched", [])
        if not filtered_nodes:
            st.warning("No memories found above the relevance threshold. Try lowering the score or broadening the query.")
        else:
            st.markdown("### Answer")
            st.write(state["answer"])

            st.markdown("---")
            st.caption(f"{len(filtered_nodes)} of {len(nodes)} chunks passed the relevance threshold ({score_threshold})")
            st.markdown("### Source Memories")

            for node in filtered_nodes:
                meta = node.metadata or {}
                date = meta.get("date", "Unknown date")
                source = meta.get("source", "unknown")
                title = meta.get("title", "Untitled")

                with st.expander(f"[{source.upper()}] {date} — {title}"):
                    if source == "photo":
                        image_path = meta.get("image_path")
                        if image_path and Path(image_path).exists():
                            st.image(image_path, use_container_width=True)
                        elif image_path:
                            st.caption(f"📷 Image not found at: {image_path}")
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
