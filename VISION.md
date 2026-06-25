# Along the Memory Lane — Project Vision

> *"The life of a man is a journey; a journey that must be travelled however bad the roads and the accommodations."* — Oliver Goldsmith

---

## What is this?

Imagine you kept a diary every day for 13 years. You wrote about your travels, your relationships, your highs and lows, your dreams, and your ordinary Tuesday afternoons. You also wrote blog posts, jotted quick notes, and took thousands of photos.

Now imagine being able to sit down and ask:

- *"What was I feeling in the summer of 2014?"*
- *"What were my goals when I turned 30?"*
- *"Show me everything I wrote about my time in Goa."*
- *"What did I think about my career back in 2018?"*

And getting a thoughtful, accurate answer — drawn from your own words, your own memories.

That is **Along the Memory Lane**.

---

## The Problem

Personal memories are scattered:
- Handwritten journals sitting in a shelf, unread
- Blog posts buried in a WordPress archive
- Loose notes in notebooks
- Photos with no context

You *know* the memories exist. But finding them is nearly impossible. You'd have to physically flip through journals, scroll through years of blog posts, or hope your memory serves you well.

---

## The Solution

This project builds a **personal AI memory assistant** that:

1. **Ingests** all your memories — journals (via OCR), blog posts, notes, and photos
2. **Understands** the content using AI, not just keyword matching
3. **Lets you query** in plain English, by date, by event, by emotion, by person
4. **Runs entirely on your own computer** — your most private thoughts never leave your machine

Think of it as a search engine for your life, powered by AI, that only you can access.

---

## Why fully local / private?

Journals contain your rawest, most honest thoughts. They are not for the cloud. This project uses:
- **Ollama** — runs AI models locally on your Mac
- **ChromaDB** — stores your memory index on your hard drive
- No internet connection needed after setup
- No third-party service ever sees your data

---

## What data goes in?

| Source | Format | Status |
|--------|--------|--------|
| Blog posts | Fetched via WordPress.com REST API (posts + images) | ✅ Done |
| Personal journals (13 years) | Handwritten → scan → OCR | Phase 2 |
| Written notes | Handwritten → scan → OCR | Phase 2 |
| Photos | JPEG/PNG with date metadata | Phase 3 |

---

## How it works (plain English)

```
Your memories (journals, blogs, notes)
        ↓
   Scan / Export
        ↓
   Convert to text (OCR for handwriting)
        ↓
   Break into chunks + understand meaning (AI embeddings)
        ↓
   Store in a local search index (ChromaDB)
        ↓
   You ask a question in plain English
        ↓
   AI finds the most relevant memories
        ↓
   AI reads them and gives you a meaningful answer
        ↓
   You also see the original source excerpts
```

---

## Roadmap

### Phase 1 — Blogs ✅ Complete
- [x] Project scaffolding and configuration
- [x] WordPress.com API fetcher (posts + images downloaded)
- [x] Text cleaning, chunking, embedding pipeline
- [x] ChromaDB vector store (local, persistent)
- [x] Streamlit query UI
- [x] Relevance score threshold filter (pre-LLM)
- [x] Date range and source filters
- [x] Code review, security audit, cleanup

### Phase 2 — Handwritten Journals (next)
- [ ] Scan pages with any phone camera → sync to Mac (Google Photos / USB)
- [ ] Apple Vision OCR on Mac (processes images locally via Python)
- [ ] Date extraction from handwritten headers
- [ ] Merge journal entries into the same index as blogs

### Phase 3 — Photos
- [ ] CLIP image embeddings for visual search
- [ ] Link photos to journal entries by date
- [ ] Search by visual content + date

### Phase 4 — Polish
- [ ] "On this day" feature — what were you doing exactly N years ago?
- [ ] Timeline browser by year and month
- [ ] Export memory summaries as PDFs
- [ ] Migrate to Qdrant for multimodal (text + image) named vectors

---

## What I'm learning through this project

- **RAG (Retrieval-Augmented Generation)** — the technique of combining a search index with a language model to answer questions grounded in specific documents
- **Local LLMs** — running powerful AI models entirely on personal hardware
- **Vector embeddings** — how AI represents meaning mathematically to enable semantic search
- **OCR** — converting handwriting to machine-readable text
- **Multimodal AI** — combining text and image understanding

---

## Tech stack at a glance

| What | Tool |
|------|------|
| AI brain | Llama 3.1 (via Ollama) |
| Memory search | nomic-embed-text + ChromaDB |
| RAG framework | LlamaIndex |
| Scanning | Any phone camera → sync to Mac |
| Handwriting OCR | Apple Vision (runs on Mac, via Python) |
| UI | Streamlit |
| Language | Python |

---

*This is a personal project. The data is private. The memories are mine.*
