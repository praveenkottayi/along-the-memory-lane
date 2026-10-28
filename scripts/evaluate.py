"""
Evaluate the RAG pipeline against the private question set.

Usage:
    python scripts/evaluate.py                         # full run: retrieval + answers
    python scripts/evaluate.py --retrieval-only        # fast: skip the LLM, retrieval metrics only
    python scripts/evaluate.py --judge                 # also score answers with the local LLM as judge
    python scripts/evaluate.py --ids q01,q05,q13       # run a subset
    python scripts/evaluate.py --label hybrid --top-k 8 --threshold 0.4
    python scripts/evaluate.py --pipeline graph        # run the LangGraph version (graph.py)

Reads eval/questions.csv (gitignored — personal), runs every question through
the same pipeline the app uses — the straight-line rag.py functions
(`--pipeline linear`) or the LangGraph graph in graph.py (`--pipeline graph`) —
and writes a timestamped run folder:

    eval/results/<YYYYmmdd-HHMM>_<label>/
        results.csv    one row per question (metrics, answer, retrieved chunks)
        answers.md     human-readable: question, expected, answer, sources
        summary.json   the averaged scorecard
    eval/results/history.csv   one line per run, to compare runs over time

Metrics
    Retrieval (answerable questions, scored on the chunks the LLM sees):
        hit        at least one expected entry was retrieved
        recall     share of expected entry dates that were retrieved
        mrr        1 / rank of the first expected entry (0 if none)
        precision  share of retrieved chunks that are expected entries
        hit_fetch  hit before the score threshold — if this is higher than
                   `hit`, the threshold is throwing away good chunks
        undated    chunks with no date metadata (e.g. the combined journal file)
    Answers:
        fact_coverage  share of key_facts found in the answer (word overlap — rough)
        refused        the system said it couldn't find anything (correct only
                       for unanswerable questions)
        judge_*        optional 1–5 scores from the local LLM (--judge)
        manual_score   left blank for you to fill in by hand (1–5)
"""
import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from config import BASE_DIR, LLM_MODEL
from rag import DEFAULT_SCORE_THRESHOLD, DEFAULT_TOP_K, MemoryStoreError, load_index, retrieve, synthesize

EVAL_DIR = BASE_DIR / "eval"
QUESTIONS_FILE = EVAL_DIR / "questions.csv"
RESULTS_DIR = EVAL_DIR / "results"
HISTORY_FILE = RESULTS_DIR / "history.csv"

console = Console()

# Phrases that mean "I couldn't find it" — used to detect refusals.
REFUSAL_PATTERNS = re.compile(
    r"no (record|mention|information|entries|entry|memories|excerpts?)|"
    r"(not|n't) (find|found|mention|contain|provide|include)|"
    r"nothing (relevant|about|was found)|"
    r"(don't|do not|cannot|can't) (know|recall|see)|"
    r"(unable|not able) to (find|locate)|no relevant|"
    r"(isn't|is not|there's no|there is no) any (mention|record|information)",
    re.IGNORECASE,
)

STOPWORDS = set(
    "a an the and or of to in on at for with from by is was were be been my your i you "
    "it its this that as about into after before up down out over all some any".split()
)

JUDGE_PROMPT = """You are grading an answer from a personal-memory assistant.

Question: {question}
Reference answer: {reference}
Key facts a good answer contains: {key_facts}

Retrieved excerpts the assistant was given:
{context}

Assistant's answer:
{answer}

Score two things from 1 (worst) to 5 (best):
- correctness: how well the answer matches the reference answer and key facts.
- faithfulness: whether every claim in the answer is supported by the retrieved excerpts (5 = no invented details).

Reply with JSON only: {{"correctness": <1-5>, "faithfulness": <1-5>, "reason": "<one short sentence>"}}"""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def split_list(value: str) -> list[str]:
    """'a; b; c' -> ['a', 'b', 'c']"""
    return [v.strip() for v in (value or "").split(";") if v.strip()]


def load_questions(ids: set[str] | None) -> list[dict]:
    if not QUESTIONS_FILE.exists():
        console.print(f"[red]Question file not found:[/red] {QUESTIONS_FILE}")
        sys.exit(1)
    with QUESTIONS_FILE.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    questions = []
    for row in rows:
        if ids and row["id"] not in ids:
            continue
        questions.append({
            **row,
            "sources": set(split_list(row["expected_source"])) - {"none"},
            "dates": set(split_list(row["expected_dates"])),
            "facts": split_list(row["key_facts"]),
            "answerable": row["type"] != "unanswerable",
        })
    return questions


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def node_info(node) -> dict:
    meta = node.metadata or {}
    return {
        "date": str(meta.get("date", "")),
        "source": meta.get("source", ""),
        "title": meta.get("title", ""),
        "file": Path(meta.get("file_path", "")).name,
        "score": round(node.score, 3) if node.score is not None else None,
    }


def is_expected(info: dict, q: dict) -> bool:
    if info["date"] not in q["dates"]:
        return False
    return not q["sources"] or info["source"] in q["sources"]


def retrieval_metrics(kept: list[dict], fetched: list[dict], q: dict) -> dict:
    relevant = [is_expected(n, q) for n in kept]
    found_dates = {n["date"] for n, rel in zip(kept, relevant) if rel}
    first_rank = next((i + 1 for i, rel in enumerate(relevant) if rel), None)
    return {
        "hit": int(any(relevant)),
        "recall": round(len(found_dates) / len(q["dates"]), 3),
        "mrr": round(1 / first_rank, 3) if first_rank else 0.0,
        "precision": round(sum(relevant) / len(kept), 3) if kept else 0.0,
        "hit_fetch": int(any(is_expected(n, q) for n in fetched)),
    }


def words(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    return {t.rstrip("s") if len(t) > 3 else t for t in tokens if t not in STOPWORDS}


def fact_coverage(answer: str, facts: list[str]) -> float:
    """Share of key facts whose words mostly (>= 60%) appear in the answer."""
    if not facts:
        return 0.0
    answer_words = words(answer)
    hits = 0
    for fact in facts:
        fact_words = words(fact)
        if fact_words and len(fact_words & answer_words) / len(fact_words) >= 0.6:
            hits += 1
    return round(hits / len(facts), 3)


def judge(judge_llm, q: dict, answer: str, kept_nodes) -> dict:
    if not answer:
        # Nothing passed the threshold: a miss, but it can't be unfaithful.
        return {"judge_correctness": 1, "judge_faithfulness": "", "judge_reason": "no answer (nothing retrieved)"}
    context = "\n---\n".join(n.text[:1200] for n in kept_nodes) or "(nothing retrieved)"
    prompt = JUDGE_PROMPT.format(
        question=q["question"], reference=q["reference_answer"],
        key_facts="; ".join(q["facts"]), context=context, answer=answer or "(no answer)",
    )
    try:
        data = json.loads(str(judge_llm.complete(prompt)))
        clamp = lambda v: min(5, max(1, int(v)))
        return {
            "judge_correctness": clamp(data.get("correctness", 1)),
            "judge_faithfulness": clamp(data.get("faithfulness", 1)),
            "judge_reason": str(data.get("reason", "")),
        }
    except Exception as e:  # a bad judge reply shouldn't kill the run
        return {"judge_correctness": "", "judge_faithfulness": "", "judge_reason": f"judge error: {e}"}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_linear(index, q: dict, args) -> tuple[list, list, str, float, float]:
    """Straight-line pipeline: rag.retrieve() then rag.synthesize()."""
    t0 = time.perf_counter()
    kept_nodes, fetched_nodes = retrieve(index, q["question"], args.top_k, args.threshold)
    retrieval_s = time.perf_counter() - t0

    answer, answer_s = "", 0.0
    if not args.retrieval_only and kept_nodes:
        t0 = time.perf_counter()
        answer = synthesize(q["question"], kept_nodes)
        answer_s = time.perf_counter() - t0
    return kept_nodes, fetched_nodes, answer, retrieval_s, answer_s


def run_graph(index, q: dict, args) -> tuple[list, list, str, float, float]:
    """LangGraph pipeline, streamed node by node so each step can be timed.

    With --retrieval-only the stream is stopped after the threshold node, so
    the LLM never runs.
    """
    from graph import ask_stream

    state, timings = {}, {}
    t0 = time.perf_counter()
    for node, changes in ask_stream(index, q["question"], args.top_k, args.threshold):
        now = time.perf_counter()
        timings[node], t0 = now - t0, now
        state.update(changes or {})
        if args.retrieval_only and node == "threshold":
            break
    retrieval_s = sum(v for k, v in timings.items() if k != "answer")
    return state.get("chunks", []), state.get("fetched", []), state.get("answer", ""), retrieval_s, timings.get("answer", 0.0)


def run_question(index, q: dict, args, judge_llm) -> dict:
    runner = run_graph if args.pipeline == "graph" else run_linear
    kept_nodes, fetched_nodes, answer, retrieval_s, answer_s = runner(index, q, args)

    kept = [node_info(n) for n in kept_nodes]
    fetched = [node_info(n) for n in fetched_nodes]

    row = {
        "id": q["id"], "type": q["type"], "question": q["question"],
        "expected_dates": q["expected_dates"],
        "n_kept": len(kept), "n_fetched": len(fetched),
        "undated": sum(1 for n in kept if not n["date"]),
        "retrieved": " | ".join(f"{n['source'] or '?'} {n['date'] or 'undated'} ({n['score']})" for n in kept),
        "retrieval_s": round(retrieval_s, 2),
    }
    if q["answerable"] and q["dates"]:
        row.update(retrieval_metrics(kept, fetched, q))

    if not args.retrieval_only:
        # The app shows a "nothing found" warning when no chunk passes the threshold.
        refused = (not kept_nodes) or bool(REFUSAL_PATTERNS.search(answer))
        row.update({
            "answer": answer,
            "answer_s": round(answer_s, 2),
            "refused": int(refused),
            "fact_coverage": fact_coverage(answer, q["facts"]) if q["answerable"] else "",
        })
        if judge_llm and q["answerable"]:
            row.update(judge(judge_llm, q, answer, kept_nodes))
        row["manual_score"] = ""

    row["_kept"] = kept  # for answers.md, not written to the CSV
    return row


def mean(values) -> float | None:
    values = [v for v in values if v not in ("", None)]
    return round(sum(values) / len(values), 3) if values else None


def summarize(rows: list[dict], questions: list[dict], args) -> dict:
    by_id = {q["id"]: q for q in questions}
    scored = [r for r in rows if "hit" in r]
    unanswerable = [r for r in rows if not by_id[r["id"]]["answerable"]]
    answerable = [r for r in rows if by_id[r["id"]]["answerable"]]
    kept_total = sum(r["n_kept"] for r in rows)

    summary = {
        "run": args.run_name, "label": args.label, "pipeline": args.pipeline, "model": LLM_MODEL,
        "top_k": args.top_k, "threshold": args.threshold, "questions": len(rows),
        "hit_rate": mean(r["hit"] for r in scored),
        "recall": mean(r["recall"] for r in scored),
        "mrr": mean(r["mrr"] for r in scored),
        "precision": mean(r["precision"] for r in scored),
        "hit_rate_before_threshold": mean(r["hit_fetch"] for r in scored),
        "empty_retrievals": sum(1 for r in rows if r["n_kept"] == 0),
        "undated_chunk_share": round(sum(r["undated"] for r in rows) / kept_total, 3) if kept_total else None,
        "avg_retrieval_s": mean(r["retrieval_s"] for r in rows),
    }
    if not args.retrieval_only:
        summary.update({
            "fact_coverage": mean(r["fact_coverage"] for r in answerable),
            "correct_refusals": f"{sum(r['refused'] for r in unanswerable)}/{len(unanswerable)}",
            "false_refusals": f"{sum(r['refused'] for r in answerable)}/{len(answerable)}",
            "avg_answer_s": mean(r["answer_s"] for r in rows if r["answer_s"]),
        })
        if args.judge:
            summary["judge_correctness"] = mean(r.get("judge_correctness") for r in answerable)
            summary["judge_faithfulness"] = mean(r.get("judge_faithfulness") for r in answerable)

    # Per-type breakdown — shows which kinds of question the pipeline struggles with.
    per_type = {}
    for r in scored:
        per_type.setdefault(r["type"], []).append(r)
    summary["by_type"] = {
        t: {"n": len(rs), "hit_rate": mean(r["hit"] for r in rs), "recall": mean(r["recall"] for r in rs)}
        for t, rs in sorted(per_type.items())
    }
    return summary


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(run_dir: Path, rows: list[dict], questions: list[dict], summary: dict):
    run_dir.mkdir(parents=True, exist_ok=True)

    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields and not k.startswith("_")]
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    by_id = {q["id"]: q for q in questions}
    lines = [f"# Eval run {summary['run']}\n"]
    for r in rows:
        q = by_id[r["id"]]
        status = "" if "hit" not in r else (" ✅" if r["hit"] else " ❌")
        lines.append(f"## {r['id']} · {r['type']}{status}\n")
        lines.append(f"**Q:** {q['question']}\n")
        lines.append(f"**Expected:** {q['expected_source']} {q['expected_dates']}  ")
        lines.append(f"**Reference:** {q['reference_answer']}\n")
        if "answer" in r:
            lines.append(f"**Answer** ({r['answer_s']}s):\n\n> " + (r["answer"] or "_(nothing passed the threshold)_").replace("\n", "\n> ") + "\n")
        lines.append("**Retrieved:**\n")
        for n in r["_kept"]:
            mark = "✓" if is_expected(n, q) else "·"
            lines.append(f"- {mark} {n['source'] or '?'} {n['date'] or 'undated'} — {n['title'] or n['file']} ({n['score']})")
        if not r["_kept"]:
            lines.append("- _(nothing)_")
        lines.append("")
    (run_dir / "answers.md").write_text("\n".join(lines), encoding="utf-8")

    flat = {k: v for k, v in summary.items() if k != "by_type"}
    new_file = not HISTORY_FILE.exists()
    existing_fields = []
    if not new_file:
        with HISTORY_FILE.open(newline="", encoding="utf-8") as f:
            existing_fields = next(csv.reader(f), [])
    fields = existing_fields + [k for k in flat if k not in existing_fields]
    if fields != existing_fields and not new_file:
        # New metric columns appeared — rewrite the file with the wider header.
        with HISTORY_FILE.open(newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
        with HISTORY_FILE.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(old)
    with HISTORY_FILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow(flat)


def print_summary(summary: dict, run_dir: Path):
    table = Table(title=f"Scorecard — {summary['run']}", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for k, v in summary.items():
        if k in ("by_type", "run"):
            continue
        table.add_row(k, str(v))
    console.print(table)

    by_type = Table(title="Retrieval by question type")
    for col in ("type", "n", "hit_rate", "recall"):
        by_type.add_column(col)
    for t, s in summary["by_type"].items():
        by_type.add_row(t, str(s["n"]), str(s["hit_rate"]), str(s["recall"]))
    console.print(by_type)
    console.print(f"\nSaved to [bold]{run_dir.relative_to(BASE_DIR)}[/bold] (results.csv, answers.md, summary.json)")


def main():
    ap = argparse.ArgumentParser(description="Evaluate the RAG pipeline on eval/questions.csv.")
    ap.add_argument("--label", default="run", help="name for this run, e.g. baseline, hybrid")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    ap.add_argument("--ids", help="comma-separated question ids to run, e.g. q01,q05")
    ap.add_argument("--pipeline", choices=["linear", "graph"], default="graph",
                    help="graph = LangGraph (graph.py, what the app runs; default), linear = plain rag.py functions")
    ap.add_argument("--retrieval-only", action="store_true", help="skip answer generation (fast)")
    ap.add_argument("--judge", action="store_true", help="score answers with the local LLM as judge")
    args = ap.parse_args()
    if args.judge and args.retrieval_only:
        ap.error("--judge needs answers; drop --retrieval-only")

    args.run_name = f"{datetime.now():%Y%m%d-%H%M}_{args.label}"
    questions = load_questions(set(args.ids.split(",")) if args.ids else None)

    try:
        index = load_index()
    except MemoryStoreError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    judge_llm = None
    if args.judge:
        from llama_index.llms.ollama import Ollama
        judge_llm = Ollama(model=LLM_MODEL, request_timeout=180.0, json_mode=True, temperature=0)

    rows = []
    for i, q in enumerate(questions, 1):
        console.print(f"[dim]{i}/{len(questions)}[/dim] {q['id']} {q['question']}")
        try:
            rows.append(run_question(index, q, args, judge_llm))
        except ConnectionError:
            console.print("[red]Could not connect to Ollama — is it running?[/red]")
            sys.exit(1)

    summary = summarize(rows, questions, args)
    run_dir = RESULTS_DIR / args.run_name
    write_results(run_dir, rows, questions, summary)
    print_summary(summary, run_dir)


if __name__ == "__main__":
    main()
