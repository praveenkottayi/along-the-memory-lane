# Along the Memory Lane 📖

A fully local, private AI assistant for querying years of personal journals, blogs and photos.
**The goal is that nothing leaves your machine.** Today one step — handwriting OCR — is
temporarily running through Claude vision while the local pipeline is being tuned; it is
slated to move to a local vision model (see [Privacy](#privacy)).

→ See [VISION.md](./VISION.md) for the story behind this project.
→ See [study-material.md](./study-material.md) for the theory: RAG, evaluation, tuning lessons and LangGraph.

---

## Architecture

### 1. Getting memories in — `scripts/update.py`

```mermaid
flowchart TD
    subgraph SOURCES["📦 Your raw memories"]
        B["🌐 Blog posts<br/>(WordPress.com)"]
        J["📓 Handwritten journals<br/>(scanned PDF)"]
        P["📷 Photos<br/>(JPEG / HEIC albums)"]
    end

    subgraph BLOG["blog step"]
        WP["fetch_wordpress_api.py<br/>only posts newer than<br/>the latest saved one"]
    end

    subgraph JOURNAL["journals step"]
        OCR["Handwriting OCR — run by hand<br/>Claude vision (API) or<br/>Claude-chat transcript"]
        SPLIT["parse_claude_transcript.py<br/>one file per dated entry"]
    end

    subgraph PHOTO["photos step — ingest_photos.py"]
        EXIF["EXIF date<br/>+ resize to 1024 px"]
        VIS["👁️ Local vision model<br/>gemma4:12b — rich, ~50 s<br/>moondream — fast, ~1 s"]
        CACHE["Description cache<br/>tagged with the model"]
    end

    TXT["📑 data/processed/*.txt<br/>front matter: date · source · title<br/>(photos also: image_path)"]

    subgraph INDEX["index step — ingest.py --incremental"]
        SYNC["SHA-256 sync<br/>add new · update edited · remove deleted<br/>skip files without front matter"]
        EMBED["🧠 Chunk (512 tokens)<br/>+ embed with nomic-embed-text"]
        STORE[("🗄️ ChromaDB<br/>memory_store/")]
    end

    CHECK["📏 eval step<br/>quick retrieval check"]

    B --> WP --> TXT
    J --> OCR --> SPLIT --> TXT
    P --> EXIF --> VIS --> CACHE --> TXT
    TXT --> SYNC --> EMBED --> STORE --> CHECK
```

### 2. Answering a question — `graph.py` (LangGraph), shared by the app and the eval

```mermaid
flowchart LR
    Q["💬 You ask<br/>(Streamlit app)"] --> EXP["🔁 Glossary<br/>expansion"]
    EXP --> RET["🔍 Semantic search<br/>top 10 chunks"]
    STORE[("🗄️ ChromaDB")] --> RET
    RET --> TH["✂️ Score ≥ 0.4<br/>keep best 5"]
    TH --> LLM["🤖 Llama 3.1 8B<br/>answers from these<br/>chunks only"]
    LLM --> ANS["✅ Answer + sources<br/>(photos shown as images)"]

    subgraph EVAL["📏 scripts/evaluate.py"]
        QS["eval/questions.csv<br/>questions + expected entries"]
        SC["Scorecard → history.csv<br/>hit rate · recall · MRR<br/>key facts · LLM judge"]
    end

    QS -. same pipeline .-> EXP
    TH -. retrieved entries .-> SC
    LLM -. answers .-> SC
```

Everything runs locally on Ollama except handwriting OCR (see [Privacy](#privacy)).

---

## How It Works

Every memory — blog post, journal entry or photo — is turned into **plain text
with a small metadata header**, and from then on all sources are treated the same.

### 1. Everything becomes text

| Source | How it becomes text | Script |
|---|---|---|
| Blog | Posts fetched from the WordPress API (only ones newer than the latest saved post), HTML stripped, images downloaded | `fetch_wordpress_api.py` |
| Handwritten journal | PDF → page images → handwriting OCR, or a Claude-chat transcript → split into **one file per dated entry** | `pdf_to_images.py` + `ocr_journals.py`, or `parse_claude_transcript.py` |
| Photo | A **local vision model** looks at the photo and writes a 3–5 sentence description; the date comes from the photo's EXIF data (falls back to the filename, then the folder name) | `ingest_photos.py` |

Each file in `data/processed/` starts with a front-matter header:

```
---
date: 2025-10-19
source: journal          # blog | journal | photo
title: Journal — 2025-10-19
---                      # photos also carry image_path, so the app can show the image
Body text…
```

**How photos are "read":** the image itself is never searched. `ingest_photos.py`
sends each photo to the vision model in `PHOTO_VISION_MODEL` (via Ollama, on your
machine) with `PHOTO_DESCRIPTION_PROMPT`, which asks what is happening, the setting,
the mood and notable objects. That description is cached in
`data/processed/photos/<album>/descriptions/` (so re-runs skip photos already
described) and written out as a normal text entry pointing back to the image. The
app shows the original image next to the answer. Photos are shrunk to 1024 px on the
longest side first (`PHOTO_MAX_SIDE`), which makes the model ~40% faster.

#### Choosing the photo model

Measured on this machine, same photos, 1024 px:

| Model | Time per photo | Description quality |
|---|---|---|
| `gemma4:12b` (default) | ~50 s | 3–5 rich sentences: occasion, setting, mood, objects |
| `moondream` | ~1 s | 1–2 short sentences, sometimes wrong (e.g. "urn of water on table") |

So 200 photos take about **3 hours with Gemma** or **4 minutes with Moondream**.
Good descriptions are what make photos findable, so Gemma is the default; a
practical pattern is Moondream now for quick coverage, Gemma overnight to upgrade.
Any other Ollama vision model works too — pull it and pass its name.

```bash
python scripts/ingest_photos.py --all --model moondream          # fast, new photos only
python scripts/ingest_photos.py --all --model gemma4:12b --refresh   # upgrade descriptions from other models
python scripts/ingest_photos.py --all --inspect                  # which model described each photo
python scripts/update.py --photo-model moondream                 # same options via update.py
```

The cache records which model wrote each description. Without `--refresh`, an
existing description is kept whichever model wrote it; with `--refresh`, photos
described by a different model are redone. Run `update.py` (or
`ingest.py --incremental`) afterwards and the changed descriptions replace the old
ones in the index.

### 2. Indexing (`scripts/ingest.py`)

- Reads every `.txt` with a front-matter header. Files without one — full-journal
  transcriptions, description caches — are skipped, because indexing them would
  duplicate entries as chunks with no date.
- Splits each entry into chunks of ~512 tokens (64 overlap).
- Embeds each chunk with `nomic-embed-text` and stores it in ChromaDB; the header
  (date, source, title) is attached to every chunk so it can be filtered and shown.
- `--incremental` keeps a SHA-256 fingerprint of every file in
  `memory_store/ingested.json`: new files are added, edited files have their old
  chunks replaced, deleted files are removed. A full rebuild is only needed after
  changing the embedding model or chunk size.

### 3. Answering a question (`graph.py` + `rag.py`)

The pipeline is a small [LangGraph](https://langchain-ai.github.io/langgraph/) graph
(`graph.py`). Each step is a node that calls a plain function in `rag.py`; the
nodes share one *state* (question, candidates, kept chunks, answer). The graph
below is generated from the code (`python graph.py`):

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	expand(expand)
	search(search)
	threshold(threshold)
	answer(answer)
	__end__([<p>__end__</p>]):::last
	__start__ --> expand;
	expand --> search;
	search --> threshold;
	threshold -.-> __end__;
	threshold -.-> answer;
	answer --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

The dotted lines are a **conditional edge**: if nothing passes the threshold the
graph ends without calling the LLM. Steps:

1. **Glossary expansion** — shorthand in the question is rewritten using
   `glossary.txt` (e.g. a nickname → "nickname (relation)").
2. **Semantic search** — the question is embedded and the 2 × `top_k` closest
   chunks are fetched from ChromaDB (optionally filtered by date range / source).
3. **Score threshold** — chunks with similarity below `score_threshold` (0.4) are
   dropped; the best `top_k` (5) remaining are kept.
4. **Answer** — Llama 3.1 writes the answer from those chunks only
   (`tree_summarize`), with a system prompt telling it to say so when nothing
   relevant was found.

The app streams the graph step by step, so you see each step as it finishes.
The same graph is used by the Streamlit app and the evaluation script, so the
eval always measures exactly what the app runs. LlamaIndex still does all the
indexing and retrieval; LangGraph only orchestrates the steps. LangSmith (the
cloud tracing service) is not used.

---

## Evaluation & Tuning

Settings are tuned against a **private question set**, not by feel.

### The question set

`eval/questions.csv` (gitignored — it contains personal details) holds real
questions with their ground truth:

| Column | Meaning |
|---|---|
| `type` | lookup, date, person, place, aggregate, timeline, vague, emotion, photo, followup, unanswerable |
| `expected_source` / `expected_dates` | which entries *should* be retrieved |
| `key_facts` | short facts a correct answer must contain (`;`-separated) |
| `reference_answer` | a model answer to compare against |

Include a few **unanswerable** questions (things never written about) — the
assistant should say it doesn't know rather than invent something.

### Running it

```bash
python scripts/evaluate.py --retrieval-only --threshold 0.4 --label try04   # ~1 s, no LLM
python scripts/evaluate.py --judge --label baseline                         # full run, ~5–7 min
python scripts/evaluate.py --ids q01,q05 --label quick                      # subset
python scripts/evaluate.py --pipeline linear --label compare                # the pre-graph rag.py pipeline
```

`--pipeline graph` (the default — what the app runs) or `--pipeline linear` (the plain
`rag.py` functions) lets you check that a change to the graph didn't change
behaviour by accident.

Each run writes `eval/results/<timestamp>_<label>/` (`results.csv`, a readable
`answers.md`, `summary.json`) and adds a line to `eval/results/history.csv` for
comparing runs.

### Metrics

| Metric | What it tells you |
|---|---|
| **Hit rate** | share of questions where at least one expected entry reached the LLM |
| **Recall** | share of all expected entries retrieved (matters for "list everything" questions) |
| **MRR** | how high the first correct entry ranks (1st = 1.0, 2nd = 0.5…) |
| **Precision** | share of retrieved chunks that are relevant (vs. noise) |
| **Hit rate before threshold** | if much higher than hit rate, the threshold is discarding good chunks |
| **Key-fact coverage** | share of key facts found in the answer (rough word overlap) |
| **Refusals** | says "I don't know" on unanswerable questions (good) vs. on answerable ones (bad) |
| **Judge correctness / faithfulness** | optional 1–5 scores from the local LLM (`--judge`): matches the reference? any invented details? |

Retrieval metrics are deterministic; judge scores from an 8B model vary by about
±1 per question between runs, so read them as a trend.

### Tuning log

| Change | Hit rate | Recall | MRR | Judge correctness | Notes |
|---|---|---|---|---|---|
| Baseline (threshold 0.5) | 0.52 | 0.42 | 0.45 | 2.6 / 5 | 14 of 32 questions retrieved nothing; hit rate *before* threshold was 0.90 |
| Threshold 0.5 → 0.4 | 0.83 | 0.68 | 0.68 | 3.7 / 5 | the threshold was the main bottleneck; answers ~2× slower (more context) |
| Skip files without front matter | **0.86** | **0.73** | **0.76** | 3.5 / 5 | removed 63 undated duplicate chunks; unanswerable questions still refused 3/3 |
| +2 new blog posts | 0.83 | 0.70 | 0.74 | — | not a pipeline change: a new post about birds seen from a window now outranks the older post about a balcony nest — similar topic, different event |
| Step 0: pipeline as a LangGraph graph | 0.83 | 0.70 | 0.74 | 3.4 / 5 | same retrieval as the plain pipeline on all 32 questions; judge score equal (3.4 vs 3.4); answers still vary run to run (LLM sampling), not because of the graph |

Still weak: questions hinging on a **name or place** (meaning-based search misses
exact words), **date-specific** questions (nearby entries get mixed up), and
**similar topics** (the newest post on a theme crowds out the specific one). These
are what the next steps of the [Roadmap](#roadmap) target. New content can shift
results, which is why `update.py` ends with an eval check.

---

## Quick Start

### Prerequisites
- macOS (Apple Silicon) or Linux
- Python 3.11+ — [uv](https://docs.astral.sh/uv/) recommended
- [Ollama](https://ollama.com) for local embeddings + chat
- An [Anthropic API key](https://console.anthropic.com) — only needed to OCR handwritten journals

### 1. Install Ollama and pull the local models
```bash
brew install ollama
ollama serve          # keep running in the background
ollama pull llama3.1:8b-instruct-q4_0   # matches LLM_MODEL in config.py
ollama pull nomic-embed-text             # matches EMBED_MODEL in config.py
ollama pull gemma4:12b                   # matches PHOTO_VISION_MODEL — for photo descriptions
```

### 2. Set up the Python environment
```bash
git clone https://github.com/praveenkottayi/along-the-memory-lane.git
cd along-the-memory-lane
uv sync                       # or: python -m venv .venv && pip install -r requirements.txt
```

### 3. Configure secrets and your glossary
```bash
cp .env.example .env          # add ANTHROPIC_API_KEY (only used for journal OCR)
cp glossary.example.txt glossary.txt   # add your names/abbreviations
```
The glossary is the single source of truth for personal shorthand. It feeds both
the LLM system prompt and **query expansion** — so a search for "Anu" is rewritten
to "Anu (wife)" before retrieval and actually finds the right entries.

### 4. Ingest your memories

> **Already set up? Just run `python scripts/update.py`** — see [Updating your memories](#updating-your-memories).
> The steps below are what it runs, for first-time setup or running one part by hand.


**Blog (WordPress.com):**
```bash
python scripts/fetch_wordpress_api.py   # new posts + images from BLOG_SITE in config.py
#   Flags:
#     --all               re-fetch every post (only needed if you edited old posts)
#     --site <domain>     a different WordPress.com site
# Alternative, from a local XML export:
# python scripts/parse_wordpress.py --input data/raw/blog/wordpress_export.xml
python scripts/ingest.py --incremental
```

**Handwritten journals** — two ways to transcribe, can be combined:
```bash
# A) Automated pipeline: PDF → page images → Claude OCR → entry .txt files
python scripts/pdf_to_images.py --pdf "data/raw/journal/16_2025-08_2026-01.pdf"
python scripts/ocr_journals.py --journal data/raw/journal/16_2025-08_2026-01
#   Flags:
#     --all               process every folder under data/raw/journal/ in one go
#     --inspect           review per-page header detection (uses cached sidecars, no API calls)
#   Each page's OCR result is cached as page_001.ocr.txt next to the image.
#   Re-running the script skips cached pages — safe to resume after interruption.

# B) Claude-chat transcript: transcribe a PDF in Claude chat (it produces a
#    ===== PAGE N | DATE | DATE_HEADER ===== format), save as
#    data/processed/journal/#<journal_name>.txt, then parse it:
python scripts/parse_claude_transcript.py        # converts all # files to entry .txt files
#   Flags:
#     --file <path>       process a single transcript file
#     --dry-run           preview what would be written without touching anything

# After using either method, patch any missing title fields:
python scripts/migrate_add_title.py              # safe to re-run; skips files that already have title
#   Flags:
#     --dry-run           preview which files would be patched

# Sync ChromaDB — adds new entries, updates edited ones (safe to run repeatedly):
python scripts/ingest.py --incremental
```

**Photos:**
```bash
# Drop a folder of photos (JPEG or HEIC) under data/raw/photos/<album-name>/
# e.g. data/raw/photos/2015-goa/

python scripts/ingest_photos.py --album data/raw/photos/2015-goa
#   Flags:
#     --all               process every album folder under data/raw/photos/
#     --inspect           print cached descriptions without calling the vision model
#     --dry-run           preview dates and cache status, no model calls, no writes
#     --model <name>      use another Ollama vision model for this run (e.g. moondream — fast)
#     --refresh           re-describe photos whose description came from a different model
#   Each description is cached in data/processed/photos/<album>/descriptions/.
#   Re-running skips already-described photos. HEIC (iPhone) and JPEG both supported.

python scripts/ingest.py --incremental   # add photo descriptions to ChromaDB
```

**Rebuilding ChromaDB from scratch** (only needed after changing `EMBED_MODEL`,
`CHUNK_SIZE` or `CHUNK_OVERLAP` — for new or edited content, `--incremental` is enough):
```bash
python scripts/ingest.py          # confirm wipe when prompted → full rebuild
python scripts/ingest.py --yes    # same, without the prompt
```

### 5. Run the app
```bash
streamlit run app/app.py
```

---

## Updating Your Memories

Wrote a new blog post, transcribed a journal, or added photos? One command:

```bash
python scripts/update.py
```

| Step | What it does | Script |
|---|---|---|
| `blog` | Fetches posts newer than your latest saved one from `BLOG_SITE` (if offline, it warns and carries on) | `fetch_wordpress_api.py` |
| `journals` | Turns `#<journal>.txt` transcripts into entries, adds missing titles | `parse_claude_transcript.py`, `migrate_add_title.py` |
| `photos` | Describes **new** photos in every album with the local vision model (cached, so only new photos are processed). With Gemma that's ~50 s per photo, so a first run on 200 photos takes ~3 hours; `--photo-model moondream` takes minutes | `ingest_photos.py --all` |
| `index` | Syncs ChromaDB: adds new files, **updates edited ones**, removes deleted ones | `ingest.py --incremental` |
| `eval` | ~1 s retrieval check against your question set, added to `eval/results/history.csv` | `evaluate.py --retrieval-only` |

```bash
python scripts/update.py --skip-photos          # quick update, skip the slow step
python scripts/update.py --only blog --only index   # just these steps
python scripts/update.py --full-eval            # end with the full eval (answers + judge)
python scripts/update.py --photo-model moondream                   # fast photo descriptions
python scripts/update.py --photo-model gemma4:12b --refresh-photos # upgrade other models' descriptions
```

Every step is safe to re-run: unchanged files are skipped. The index sync knows
what changed because `memory_store/ingested.json` stores a content fingerprint
(SHA-256) of every ingested file.

Not included: **handwriting OCR** (`pdf_to_images.py` + `ocr_journals.py`), which
calls the Anthropic API — run it by hand when you scan a new journal, then run
`update.py`.

---

## Hyperparameters

Model, data and indexing settings live in `config.py`; retrieval defaults live in `rag.py`. Every script picks them up automatically. Re-run the eval after changing any of them.

| Parameter | Where | Default | What it controls |
|-----------|-------------------|---------|-----------------|
| `BLOG_SITE` | `config.py` (or env var / `--site`) | your WordPress.com domain | Blog fetched by `fetch_wordpress_api.py` and `update.py`. |
| `EMBED_MODEL` | `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model. Must match what was used when the index was built — changing it requires a full re-ingest. |
| `LLM_MODEL` | `LLM_MODEL` | `llama3.1:8b-instruct-q4_0` | Ollama chat model used to synthesise answers. Safe to swap without re-ingesting. |
| `PHOTO_VISION_MODEL` | `PHOTO_VISION_MODEL` (or env var / `--model`) | `gemma4:12b` | Ollama vision model for photo descriptions. Fully local. See [Choosing the photo model](#choosing-the-photo-model). |
| `PHOTO_MAX_SIDE` | `PHOTO_MAX_SIDE` | `1024` | Photos are shrunk to this many pixels on the longest side before description. Smaller = faster. |
| `PHOTO_DESCRIPTION_PROMPT` | `PHOTO_DESCRIPTION_PROMPT` | *(see config.py)* | Prompt sent to the vision model for each photo. Edit to change what the description focuses on. |
| `OCR_VISION_MODEL` | `OCR_VISION_MODEL` | `claude-opus-4-8` | Claude model used for handwriting OCR. Only called by `ocr_journals.py`. |
| `CHUNK_SIZE` | `CHUNK_SIZE` | `512` | Tokens per chunk when splitting documents before embedding. Larger = more context per result but lower precision. Journal entries are narrative — try 768–1024. |
| `CHUNK_OVERLAP` | `CHUNK_OVERLAP` | `64` | Token overlap between consecutive chunks. Prevents a sentence being cut mid-thought at a boundary. ~10–15% of chunk size is a good rule of thumb. |
| `CHROMA_COLLECTION` | `CHROMA_COLLECTION` | `memories` | ChromaDB collection name. Change only if you want parallel indexes for testing. |
| `score_threshold` | `rag.py` `DEFAULT_SCORE_THRESHOLD` (UI slider) | `0.4` | Minimum similarity score for a chunk to be sent to the LLM. Higher = stricter. 0.5 discarded too many good chunks (see [Tuning log](#tuning-log)). |
| `top_k` | `rag.py` `DEFAULT_TOP_K` (UI slider) | `5` | How many chunks (after threshold filtering) to pass to the LLM. More = richer context, slower response. |

> **Note on chunk_size / chunk_overlap:** changing these requires wiping and rebuilding ChromaDB (`python scripts/ingest.py` → confirm wipe). The embedding and chunking happen together at ingest time.

---

## Project Structure

```
along-the-memory-lane/
├── VISION.md                   ← Project intention (start here)
├── README.md                   ← Technical setup (you are here)
├── study-material.md           ← The theory behind it all (RAG, eval, LangGraph) — shareable
├── config.py                   ← All paths, models, prompt + glossary loading
├── rag.py                      ← Retrieval + answer functions (glossary, search, threshold, answer)
├── graph.py                    ← LangGraph pipeline: state + nodes + edges over rag.py (python graph.py → diagram)
├── pyproject.toml / uv.lock    ← Dependencies (requirements.txt mirrors these)
├── .env.example                ← Template for ANTHROPIC_API_KEY
├── glossary.example.txt        ← Template for personal names/abbreviations
│
├── data/                       ← gitignored — your personal content
│   ├── raw/journal|blog|photos/ ← scanned PDFs, blog images, photo albums (raw/photos/<album>/)
│   └── processed/              ← .txt entries with front matter: blog/, journal/<journal>/, photos/<album>/
│                                  (photos/<album>/descriptions/ = vision-model cache, not indexed)
│
├── memory_store/               ← ChromaDB vector index + ingested.json fingerprints (gitignored)
│
├── eval/                       ← gitignored — private question set + run results
│   ├── questions.csv
│   └── results/                ← one folder per run + history.csv
│
├── scripts/
│   ├── common.py                    ← shared helpers: front-matter r/w, journal text parsers, slug/HTML utils
│   ├── fetch_wordpress_api.py       ← fetch new posts + images from a WordPress.com site (--all, --site)
│   ├── parse_wordpress.py           ← alternative: parse a local WordPress XML export
│   ├── pdf_to_images.py             ← scanned journal PDF → page images
│   ├── ocr_journals.py              ← Claude-vision OCR of journal images → entry .txt files
│   │                                   (--all, --inspect, page-level caching via .ocr.txt sidecars)
│   ├── parse_claude_transcript.py   ← parse Claude-chat transcripts (#journal.txt) → entry .txt files
│   │                                   (--file, --dry-run)
│   ├── migrate_add_title.py         ← one-time: stamp title into journal .txt files missing it
│   │                                   (--dry-run); safe to re-run
│   ├── split_journal_txt.py         ← older alternative to parse_claude_transcript.py
│   ├── ingest_photos.py             ← Phase 3: EXIF date + local vision description → .txt
│   │                                   (--album, --all, --model, --refresh, --inspect, --dry-run; HEIC + JPEG)
│   ├── update.py                    ← one command: blog → journals → photos → index → eval check
│   │                                   (--skip-X, --only X, --photo-model, --refresh-photos, --full-eval)
│   ├── ingest.py                    ← chunk + embed + store in ChromaDB
│   │                                   (--incremental syncs new/changed/deleted; --yes; skips files without front matter)
│   └── evaluate.py                  ← score the pipeline on eval/questions.csv
│                                       (--pipeline graph|linear, --retrieval-only, --judge, --ids, --label)
│
└── app/
    └── app.py                  ← Streamlit query UI
```

---

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | WordPress blog ingestion + local RAG query UI | ✅ Complete |
| 2 | Handwritten journal OCR (PDF → Claude vision → entries) | ✅ Complete (local OCR deferred) |
| 3 | Photo search — local vision descriptions → ChromaDB (same pipeline as journals) | ✅ Pipeline complete — most photos still need (re)describing with Gemma |
| — | One-command updates (`update.py`), incremental fetch + index sync | ✅ Complete |
| 4 | "On This Day" feature + timeline browser | ⬜ Planned (Roadmap step 7) |

---

## Roadmap

Step 1 is on `main`. Step 0 and steps 2–7 are built on the `langgraph` branch, one concept
at a time, each measured with the eval before it's kept and merged into `main`. Design
rules: everything stays local on Ollama; keep the graph shallow (every LLM step
costs seconds on an 8B model); use plain rules where possible (dates) and the LLM
only for language.

| # | Step | What it adds | Status |
|---|---|---|---|
| 0 | Pipeline as a LangGraph graph | same behaviour, rebuilt as nodes + state; app streams each step | ✅ Done — retrieval identical on all eval questions |
| 1 | Evaluation set + harness | private questions with expected entries; hit rate, recall, MRR, answer scores | ✅ Done (+ threshold tuning, dedupe) |
| 2 | Question → filters (self-query) | "in October 2025" becomes a date filter automatically | ⬜ Next |
| 3 | Hybrid search + whole-entry answers | keyword search (BM25) alongside meaning search, so names and places are found; answers use the full entry | ⬜ |
| 4 | LangGraph router | separate routes for lookup, aggregate ("where all did I travel…") and timeline ("how did my thinking change…") questions | ⬜ |
| 5 | Corrective loop | check relevance → rewrite the query → retry, at most twice | ⬜ |
| 6 | Conversation memory | follow-ups like "and the year after?" | ⬜ |
| 7 | Ingest-time enrichment + "On this day" | tag entries with people, places and mood; query by emotion or person | ⬜ |

---

## Privacy

The goal is uncompromising: **nothing leaves your machine.**

**Fully local today — never leaves your machine:**
- Text embeddings and chat (Ollama: `nomic-embed-text` + `llama3.1`)
- Photo descriptions (Ollama vision model — photos never leave your machine)
- The evaluation question set and results in `eval/`
- The ChromaDB vector index in `memory_store/`
- Every query, retrieval, and generated answer
- `data/`, `memory_store/`, `eval/`, `.env`, and `glossary.txt` are gitignored

**Temporary exception — handwriting OCR:**
Local vision models tried so far (Apple Vision, llava, moondream, Tesseract,
llama3.2-vision) couldn't reliably read cursive, so while the RAG side is being
trialled, journal page **images** are transcribed via the Claude API. This is a
stopgap, not the destination — the plan is to move OCR to a local vision
model once one reads the handwriting well. Even now it runs only when you
OCR journals, sends images only (never the index or your queries), and is skipped
entirely if you ingest only typed sources (blogs, notes).
