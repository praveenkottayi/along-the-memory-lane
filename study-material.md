# Study Material — Building and Improving a Local RAG

The theory behind *Along the Memory Lane*: what each technique is, why it's used,
and what we learned by measuring it. Written to be read on its own — no need to
know the code. Examples are generic; numbers come from this project's evaluation
(32 questions over ~350 chunks of blogs, journal entries and photo descriptions).

**Contents**

1. [RAG in one page](#1-rag-in-one-page)
2. [Getting data in: ingestion](#2-getting-data-in-ingestion)
3. [Retrieval: finding the right chunks](#3-retrieval-finding-the-right-chunks)
4. [Generation: writing the answer](#4-generation-writing-the-answer)
5. [Evaluation: measuring instead of guessing](#5-evaluation-measuring-instead-of-guessing)
6. [Tuning lessons from this project](#6-tuning-lessons-from-this-project)
7. [Keeping the index up to date](#7-keeping-the-index-up-to-date)
8. [LlamaIndex, LangChain and LangGraph](#8-llamaindex-langchain-and-langgraph)
9. [LangGraph concepts](#9-langgraph-concepts)
10. [Step 0: porting to a graph safely](#10-step-0-porting-to-a-graph-safely)
11. [What comes next (roadmap theory)](#11-what-comes-next-roadmap-theory)
12. [Running everything locally](#12-running-everything-locally)
13. [Interview cheat sheet](#13-interview-cheat-sheet)
14. [Glossary](#14-glossary)

---

## 1. RAG in one page

**Retrieval-Augmented Generation (RAG)**: instead of asking a language model to
answer from memory, you first **retrieve** relevant passages from your own
documents and then ask the model to **generate** an answer *from those passages*.

```
your documents ──► chunk ──► embed ──► vector database        (once, at ingest)

question ──► embed ──► find closest chunks ──► LLM writes answer from them   (every query)
```

Why not just ask the LLM?

| Problem with a plain LLM | How RAG fixes it |
|---|---|
| It never saw your private data | Your data is retrieved and put in front of it |
| It "hallucinates" — invents plausible facts | It's told to answer only from the retrieved text, and you can show the sources |
| Its knowledge is frozen at training time | Add a document, re-index, and it's searchable immediately |
| Sending everything in one prompt is too long / slow | Only the few most relevant chunks are sent |

**RAG vs fine-tuning**: fine-tuning changes the model's weights (teaches it a
style or skill); RAG supplies *knowledge* at question time. For "answer questions
about my documents", RAG is cheaper, updatable and citeable.

**RAG vs long context**: newer models accept very long prompts, but stuffing
everything in is slow, costly, and models get worse at finding a detail buried in
the middle ("lost in the middle"). Retrieval keeps the prompt small and focused —
essential on a small local model.

---

## 2. Getting data in: ingestion

### 2.1 Everything becomes text

Retrieval works on text, so every source is converted first:

| Source | Conversion |
|---|---|
| Blog posts | Fetched from an API, HTML stripped |
| Handwritten journals | Scanned → OCR (optical character recognition) → split into one file per dated entry |
| Photos | A **vision model** writes a description of each photo ("captioning"); that description is what gets searched |

That last one is a simple form of **multimodal RAG**: images become searchable
through text descriptions. The quality of the description decides whether a
photo can be found at all — a vague caption ("a table with a cup") makes the
photo effectively invisible. Bigger vision models describe better but are slower
(here: ~50 s per photo vs ~1 s for a tiny model).

### 2.2 Metadata (front matter)

Each text file starts with a small header:

```
---
date: 2025-10-19
source: journal
title: Journal — 2025-10-19
---
```

This **metadata** travels with every chunk into the database. It enables
**filtering** ("only journals", "only October 2025"), lets the app show where an
answer came from, and lets the evaluation check *which entry* was retrieved.
Files without metadata were found to be harmful (see [§6](#6-tuning-lessons-from-this-project)).

### 2.3 Chunking

Long documents are split into **chunks** (here ~512 tokens with 64 overlapping).

- **Too large**: one chunk covers many topics; its meaning gets blurred and it
  matches many questions weakly.
- **Too small**: a chunk loses context ("she said yes" — to what?).
- **Overlap**: the end of one chunk repeats at the start of the next, so a
  sentence cut at a boundary still appears whole somewhere.

A common refinement is **small-to-big / parent-document retrieval**: search small
chunks (precise matching) but give the LLM the whole entry they came from
(complete context).

### 2.4 Embeddings

An **embedding model** turns text into a list of numbers (a vector, here 768 of
them) such that texts with similar *meaning* get nearby vectors. "I went for a
run" and "jogging this morning" end up close even with no words in common.

Similarity is usually **cosine similarity** — the angle between two vectors.
Scores here roughly range 0.3–0.7; what counts as "relevant" depends on the model,
which is why the cut-off must be tuned, not guessed.

The same embedding model must be used at ingest and query time — changing it
means re-embedding everything.

### 2.5 Vector database

Stores each chunk's vector, text and metadata, and answers "which stored vectors
are closest to this one?" quickly. This project uses **ChromaDB**, which runs
locally as files on disk.

---

## 3. Retrieval: finding the right chunks

### 3.1 Top-k and the score threshold

- **top-k**: how many chunks go to the LLM (here 5). More = more context, but
  slower answers and more noise.
- **score threshold**: drop chunks below a similarity score (here 0.4). Too high
  and good chunks are thrown away; too low and noise gets through.
- Fetch more than you need (here 2 × k = 10), *then* filter, so the threshold
  still leaves enough.

### 3.2 Query expansion

Personal shorthand (nicknames, abbreviations) doesn't embed well. A **glossary**
rewrites the question before searching — e.g. a nickname becomes
"nickname (relation)" — so it matches how entries are written. The same glossary
is given to the LLM so answers read correctly.

### 3.3 Where pure semantic search fails

Embeddings capture *meaning*, which is also their weakness:

| Question type | Why semantic search struggles |
|---|---|
| Names, places, rare words (a small village, a friend's name) | A rare proper noun barely moves the vector; the chunk that mentions it may not rank first |
| Dates ("in October 2025") | Embeddings don't do date arithmetic; nearby dates look alike |
| Two posts on a similar topic | The newer or longer one can outrank the specific one you meant |
| "List everything" questions | 5 chunks can't cover a whole year |

These are exactly what the roadmap steps target ([§11](#11-what-comes-next-roadmap-theory)).

---

## 4. Generation: writing the answer

The retrieved chunks and the question go to the LLM (here Llama 3.1 8B, local)
with a **system prompt**: answer only from these excerpts, refer to the author as
"you", say so honestly if nothing relevant was found.

**Synthesis mode**: *tree_summarize* answers from each group of chunks and then
combines the partial answers — robust when the chunks don't fit in one prompt.

**Grounding / faithfulness**: an answer is *faithful* if every claim is supported
by the retrieved text. Showing the source chunks under the answer lets the user
check.

**Refusal**: for something never written about, the right answer is "I don't
know". A system that always answers something is worse than one that admits
gaps.

**Temperature**: LLMs sample words with some randomness. At the default
temperature the same input gives slightly different wording each run — in this
project only 4 of 32 answers were word-for-word identical between two runs with
identical inputs. Setting temperature to 0 makes output (nearly) repeatable,
which helps evaluation.

---

## 5. Evaluation: measuring instead of guessing

The most important habit in this project: **every change is measured on a fixed
question set before it's kept.** Without it, "it seems better" is a guess.

### 5.1 The evaluation set (ground truth)

15–40 real questions, each with:

| Field | Purpose |
|---|---|
| question | what a user would actually ask |
| type | lookup, date, person, place, aggregate, timeline, vague, emotion, photo, follow-up, unanswerable |
| expected entries | which documents *should* be retrieved (the ground truth) |
| key facts | short facts a correct answer must contain |
| reference answer | a model answer to compare with |

Tips:
- **Write questions from documents you know**, so the answer is certain.
- **Mix question types** — each later improvement targets a type, and the
  per-type scores show whether it worked.
- **Include unanswerable questions** — they test whether the system refuses
  instead of inventing.
- **Keep it private** if it contains personal details (here it's gitignored).
- **Update it when the data changes** — a new document may become a valid
  answer to an old question.

### 5.2 Retrieval metrics (deterministic)

| Metric | Question it answers | Example |
|---|---|---|
| **Hit rate @k** | Did at least one expected entry reach the LLM? | 0.83 = yes for 83% of questions |
| **Recall @k** | What share of *all* expected entries came back? | matters for "list everything" questions |
| **MRR** (mean reciprocal rank) | How high did the first correct entry rank? | 1st → 1.0, 2nd → 0.5, 3rd → 0.33 |
| **Precision @k** | What share of retrieved chunks were relevant? | low = the LLM reads lots of noise |
| **Hit rate before threshold** | Was the right chunk found but then filtered out? | a big gap vs hit rate = threshold too strict |

These are **deterministic**: same index, same query → same result. Trust them
most.

### 5.3 Answer metrics

| Metric | How |
|---|---|
| **Key-fact coverage** | share of key facts whose words appear in the answer — cheap, rough |
| **LLM-as-a-judge** | another LLM call scores the answer 1–5 for **correctness** (vs the reference) and **faithfulness** (no invented details) |
| **Refusal accuracy** | refuses on unanswerable questions (good) vs on answerable ones (bad) |
| **Manual score** | you read the answer and score it — the gold standard at small scale |
| **Latency** | seconds per answer — matters a lot on a local model |

**Caveats of LLM-as-a-judge**: a small model judging itself is noisy (±1 point per
question between runs here) and can be fooled by confident wording. Read judge
averages as trends; check surprising cases by hand. Tools like RAGAS, DeepEval or
TruLens package these metrics; the ideas are the same.

**Evaluate the evaluator**: in this project the eval once reported a refusal as
a failure because the detector didn't know the phrase "not able to find". When
a number looks wrong, read the actual answers.

### 5.4 The loop

```
baseline ──► change ONE thing ──► re-run eval ──► compare ──► keep or revert
```

One change at a time, so you know what caused the difference. Keep a history of
runs (here `history.csv`) to see the trend.

---

## 6. Tuning lessons from this project

| Change | Hit rate | What it taught |
|---|---|---|
| Baseline, threshold 0.5 | 0.52 | 14 of 32 questions retrieved *nothing* |
| Threshold 0.5 → 0.4 | 0.83 | "hit rate before threshold" was 0.90 — the right chunks were found and then thrown away. **The cheapest fix was a number.** |
| Remove files without metadata | 0.86 | 63 duplicate, undated chunks (a full journal transcript indexed alongside its per-day entries) were crowding out the real entries. **Data quality beats clever algorithms.** |
| +2 new blog posts | 0.83 | A new post on a similar topic outranked the specific older one. **New data can make old answers worse** — re-run the eval after updates. |
| Pipeline rebuilt as a LangGraph graph | 0.83 | Identical retrieval on all 32 questions — a refactor proven safe by measurement. |

Other observations:
- A duplicate chunk can *accidentally* produce a right answer from the wrong
  source; removing it exposed real misses. Look at *what* was retrieved, not just
  whether the answer looked right.
- More context (lower threshold) roughly doubled answer time — every quality gain
  has a latency cost on an 8B model.

---

## 7. Keeping the index up to date

**Incremental ingestion**: re-embedding everything on every update is slow, so
only process what changed.

- **Change detection by content hash**: store a fingerprint (SHA-256) of every
  ingested file. On the next run: new file → add; same path, different hash →
  delete its old chunks and re-add; missing file → delete its chunks. Tracking
  only file *names* misses edits — a real bug found here.
- **Incremental fetching**: ask the source only for items newer than the latest
  one you have (e.g. blog posts published after the newest saved post).
- **Caching expensive steps**: photo descriptions are cached per photo and tagged
  with the model that wrote them, so re-runs skip finished work and switching
  models can selectively redo old descriptions.
- **Idempotent steps**: running a step twice gives the same result, so a whole
  update can be re-run safely after any failure.
- **One command** (`update.py`) chains the steps and ends with an eval check, so
  quality regressions from new data are noticed.

---

## 8. LlamaIndex, LangChain and LangGraph

| Tool | What it is | Analogy |
|---|---|---|
| **LlamaIndex** | Library for indexing documents and retrieving from them: loaders, chunking, embeddings, vector-store integrations, retrievers, answer synthesis | The **librarian** — finds the right pages |
| **LangChain** | A large toolbox of LLM building blocks: prompt templates, model wrappers, retrievers, tools, many integrations | A **box of parts** |
| **LangGraph** | A framework for the **control flow** of LLM apps: steps as nodes, decisions as edges, loops, shared state, memory | The **manager** — decides what to do next |

They're independent. LangGraph doesn't need LangChain's components; a LangGraph
node can call any Python function — including LlamaIndex retrieval. This project
keeps **LlamaIndex for retrieval** and adds **LangGraph for orchestration**.
(LlamaIndex has its own orchestration, *Workflows*, which would be a valid
alternative.)

### When do you need an orchestration framework?

A **fixed pipeline** (same steps every time) is fine as plain functions. You need
more when the app must:

| Need | Example |
|---|---|
| **Branch** | "list everything" questions take a different path from lookups |
| **Loop** | results look poor → rewrite the query → search again |
| **Remember** | "and the year after?" needs the previous question |
| **Show progress / trace** | see what each step did |

All of this *can* be written in plain Python (if/else, for-loops, a list of
messages). A framework earns its place as the flow grows: each feature becomes
one node instead of more nested if/else, and memory/checkpointing come built in.

---

## 9. LangGraph concepts

| Concept | Meaning | In this project |
|---|---|---|
| **State** | One shared record passed through the graph; every node reads it and returns the fields it changed | `question`, `expanded_query`, `fetched`, `chunks`, `answer` |
| **Node** | A plain function: state in → updates out | `expand`, `search`, `threshold`, `answer` |
| **Edge** | "After A, run B" | expand → search → threshold |
| **Conditional edge** | A small *router* function looks at the state and names the next node | after `threshold`: chunks found → `answer`, none → end (skip the LLM) |
| **START / END** | Entry and exit points | |
| **Runtime context** | Dependencies that aren't data (database handles, clients), passed in per call rather than stored in state | the vector index |
| **Compile** | Turns the graph definition into a runnable object; checks it's well-formed | done once at import |
| **Invoke** | Run to the end, get the final state | `ask(question)` |
| **Stream** | Run step by step, receiving each node's updates as it finishes | the app shows 🔁 → 🔍 → ✂️ → 🤖 |
| **Checkpointer / thread** | Saves the state after each step under a conversation id, enabling memory and resuming | coming in step 6 |
| **Visualisation** | The graph can export itself as a Mermaid diagram | the README diagram is generated from code |

The mental shift: **instead of functions passing values to each other, every
step reads and writes one shared state.** Adding a feature = adding a field and a
node.

```
START → expand → search → threshold ─┬─ chunks found ─► answer → END
                                     └─ nothing found ─────────► END
```

---

## 10. Step 0: porting to a graph safely

Before adding new behaviour, the existing pipeline was rebuilt as a graph with
**exactly the same behaviour**, and that was proven with the evaluation:

- Nodes call the existing functions — nothing was rewritten.
- Both versions were evaluated back to back on the same data.
- **Pass criteria**: retrieval identical (it's deterministic, so any difference
  means a wiring bug); answer scores within noise; no extra latency.
- Result: identical retrieved entries, order and scores for all 32 questions;
  judge score equal; graph overhead ~0 ms.

This is the general rule for refactors: **change structure *or* behaviour, never
both at once**, and verify the structural change with numbers. Every later step is
then a small, measurable change to a known-good graph.

---

## 11. What comes next (roadmap theory)

| Step | Technique | Idea |
|---|---|---|
| 2 | **Self-query / metadata extraction** | Turn "what did I do in October 2025?" into a date filter automatically. Dates are parsed with plain rules (reliable, instant); the LLM is used only for language. |
| 3 | **Hybrid search** | Combine **BM25** (classic keyword ranking — great for names and rare words) with vector search (great for meaning). Merge the two ranked lists with **Reciprocal Rank Fusion (RRF)**: each document scores Σ 1/(k + rank) across lists. |
| 3 | **Parent-document retrieval** | Match on small chunks, answer from the whole entry. |
| 3 | *(optional)* **Reranking** | A cross-encoder model rescores the top candidates by reading question and chunk together — more accurate, slower. |
| 4 | **Router / intent classification** | Classify the question (lookup, aggregate, timeline) and send it down a different path; e.g. aggregate = fetch many entries in a date range, extract, combine (**map-reduce**). |
| 5 | **Corrective RAG (CRAG)** | Grade retrieved chunks for relevance; if poor, rewrite the query and retry (capped at 2 retries to bound latency). Related: **Self-RAG**, **reflection**. |
| 6 | **Conversation memory** | A checkpointer stores the conversation per thread; follow-ups are rewritten into standalone questions before retrieval. |
| 7 | **Ingest-time enrichment** | Tag every entry with people, places and mood once at ingest (**entity extraction / NER**), so queries can filter by them; enables "on this day". **Contextual retrieval** (prepending a one-line summary to each chunk before embedding) fits here too. |

Not planned but worth knowing: **GraphRAG** (build an entity–relationship graph
and retrieve over it), **HyDE** (embed a generated hypothetical answer instead of
the question), **semantic caching** (reuse answers to near-identical questions),
**guardrails** (input/output checks).

Design rules for a small local model: keep the graph **shallow** (every LLM step
costs seconds), use **rules where rules work** (dates), and add an LLM step only
when the eval shows it pays off.

---

## 12. Running everything locally

| Piece | Local choice |
|---|---|
| LLM | Llama 3.1 8B (4-bit **quantized**) via **Ollama** |
| Embeddings | nomic-embed-text via Ollama |
| Vision (photo descriptions) | gemma4 12B or moondream via Ollama |
| Vector DB | ChromaDB on disk |
| Orchestration | LangGraph (a library — runs in-process) |
| Tracing | none in the cloud — LangSmith is deliberately not used |

**Quantization** stores model weights in fewer bits (e.g. 4 instead of 16) so an
8B model fits in laptop memory and runs faster, with a small quality cost.

**Privacy trade-offs** are explicit: here only handwriting OCR uses a cloud API
(local models couldn't read cursive yet), and only for page images — never the
index or the questions.

---

## 13. Interview cheat sheet

**The story in 30 seconds**

> I built a fully local, multimodal RAG over years of my own journals, blogs and
> photos. I started with a measured baseline on a private evaluation set, found
> the relevance threshold was discarding the right chunks (hit rate 0.52 → 0.83),
> removed duplicate undated chunks (→ 0.86), made ingestion incremental with
> content hashing, and ported the pipeline to LangGraph with identical retrieval
> proven by the eval. Next: self-query filters, hybrid BM25 + vector search, a
> router for aggregate and timeline questions, a corrective retry loop and
> conversation memory — each kept only if the numbers improve.

**Likely questions**

| Question | Short answer |
|---|---|
| How do you evaluate a RAG? | Separately for retrieval (hit rate, recall, MRR, precision on a ground-truth set) and generation (faithfulness, correctness, refusals); deterministic metrics first, LLM-as-judge as a trend |
| Why did your answers improve? | Mostly retrieval and data quality: threshold tuning and removing duplicates — not prompt tweaks |
| RAG vs fine-tuning? | RAG supplies knowledge at query time (updatable, citeable); fine-tuning changes behaviour/style |
| RAG vs long context? | Long prompts are slow, costly and lose details in the middle; retrieval keeps context small — vital on a local 8B model |
| Why hybrid search? | Embeddings miss exact names and rare words; BM25 catches them; RRF merges the rankings |
| Why LangGraph? | The flow needs branching, retries and memory; a graph keeps that readable and gives checkpointing — but a fixed pipeline doesn't need it |
| How do you avoid hallucination? | Answer only from retrieved text, show sources, test refusals with unanswerable questions, measure faithfulness |
| How do you keep it fast? | Shallow graph, rules instead of LLM calls where possible, small top-k, quantized model, caching |
| How do you handle updates? | Incremental fetch + hash-based sync (add/update/delete), cached captions, eval check after every update |

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **Aggregate question** | Needs many documents combined ("where all did I travel in 2016?") |
| **BM25** | Classic keyword-based ranking function used by search engines |
| **Checkpointer** | LangGraph component that saves state between steps/runs (memory) |
| **Chunk** | A piece of a document, the unit that is embedded and retrieved |
| **Conditional edge** | A graph edge whose destination is chosen at run time from the state |
| **Cosine similarity** | Similarity of two vectors by angle; the usual retrieval score |
| **CRAG** | Corrective RAG — grade retrieval, retry with a rewritten query if poor |
| **Embedding** | Vector of numbers representing the meaning of a text |
| **Faithfulness / groundedness** | Every claim in the answer is supported by the retrieved text |
| **Front matter / metadata** | Structured fields (date, source, title) attached to each document and chunk |
| **Ground truth** | The known correct answer/documents for an evaluation question |
| **Hallucination** | A fluent but unsupported or false statement from an LLM |
| **Hit rate @k** | Share of questions where a correct document is in the top k |
| **Hybrid search** | Keyword (BM25) + semantic (vector) search combined |
| **Idempotent** | Running it again gives the same result — safe to retry |
| **LLM-as-a-judge** | Using an LLM to score another LLM's output |
| **MRR** | Mean reciprocal rank — rewards putting the right document first |
| **Multimodal RAG** | RAG over more than text, e.g. images via captions |
| **Node** | One step of a LangGraph graph |
| **Ollama** | Tool for running open models locally |
| **Parent-document retrieval** | Search small chunks, return the larger document they belong to |
| **Quantization** | Storing model weights in fewer bits to save memory and time |
| **Query expansion / rewriting** | Changing the question before search to match documents better |
| **Recall @k** | Share of all relevant documents that appear in the top k |
| **Reranker / cross-encoder** | Model that rescores candidate chunks by reading them with the question |
| **Router** | Node that classifies a question and picks the path through the graph |
| **RRF** | Reciprocal Rank Fusion — merges ranked lists by summing 1/(k + rank) |
| **Self-query** | Extracting metadata filters (e.g. dates) from the question itself |
| **State** | The shared data record a LangGraph graph passes between nodes |
| **Temperature** | Randomness of LLM output; 0 ≈ deterministic |
| **Threshold** | Minimum similarity score for a chunk to be used |
| **Top-k** | Number of chunks passed to the LLM |
| **Vector database** | Database that finds the stored vectors closest to a query vector |
