"""
One command to bring everything up to date: new blog posts, journal
transcripts, photos → ChromaDB → quick eval check.

Usage:
    python scripts/update.py                    # everything
    python scripts/update.py --skip-photos      # skip the slow photo descriptions
    python scripts/update.py --only blog        # just one step (repeatable: --only blog --only index)
    python scripts/update.py --full-eval        # finish with the full eval (answers + judge, ~5–7 min)

Steps (each is an existing script, run in order):
    blog      fetch_wordpress_api.py          fetch posts newer than the latest saved one (config.BLOG_SITE)
    journals  parse_claude_transcript.py      turn '#<journal>.txt' transcripts into entries
              migrate_add_title.py            stamp missing titles
    photos    ingest_photos.py --all          describe new photos with the local vision model
                                              (cached — only new photos hit the model, but the
                                              first run can take 1–2 hours)
    index     ingest.py --incremental         add new, update changed, remove deleted chunks
    eval      evaluate.py --retrieval-only    ~1 s sanity check; skipped if eval/questions.csv is missing

Handwriting OCR (pdf_to_images.py + ocr_journals.py) is not included: it calls
the Anthropic API and needs a new PDF, so run it by hand when you scan a journal.

A failed blog fetch (e.g. offline) is reported and the update continues;
any other failed step stops the run.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BASE_DIR, BLOG_SITE

SCRIPTS = BASE_DIR / "scripts"
STEPS = ["blog", "journals", "photos", "index", "eval"]


def run(label: str, *args: str) -> bool:
    """Run one script with the current Python; return True on success."""
    cmd = [sys.executable, str(SCRIPTS / args[0]), *args[1:]]
    print(f"\n{'=' * 70}\n▶ {label}\n  {' '.join(Path(c).name if i < 2 else c for i, c in enumerate(cmd))}\n{'=' * 70}")
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=BASE_DIR)
    status = "✓" if result.returncode == 0 else f"✗ (exit {result.returncode})"
    print(f"{status} {label} — {time.perf_counter() - start:.0f}s")
    return result.returncode == 0


def main():
    ap = argparse.ArgumentParser(description="Fetch new data and update the memory index.")
    ap.add_argument("--only", action="append", choices=STEPS, help="run only these steps")
    for step in STEPS:
        ap.add_argument(f"--skip-{step}", action="store_true", help=f"skip the {step} step")
    ap.add_argument("--full-eval", action="store_true", help="run the full eval (answers + judge) instead of the quick check")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keep our headers in order with the child scripts' output

    selected = [s for s in STEPS if (not args.only or s in args.only) and not getattr(args, f"skip_{s}")]
    print(f"Steps: {', '.join(selected) or '(none)'}")
    started = time.perf_counter()
    summary = []

    def step(name: str, ok: bool, fatal: bool = True):
        summary.append((name, ok))
        if not ok and fatal:
            print(f"\nStopping: the {name} step failed. Fix it and re-run — finished steps are cached.")
            sys.exit(1)

    if "blog" in selected:
        ok = run(f"Fetch new blog posts from {BLOG_SITE}", "fetch_wordpress_api.py", "--site", BLOG_SITE)
        if not ok:
            print("  Blog fetch failed (offline?). Continuing with the posts already on disk.")
        step("blog", ok, fatal=False)

    if "journals" in selected:
        step("journals", run("Parse journal transcripts", "parse_claude_transcript.py")
             and run("Add missing journal titles", "migrate_add_title.py"))

    if "photos" in selected:
        step("photos", run("Describe new photos", "ingest_photos.py", "--all"))

    if "index" in selected:
        step("index", run("Sync ChromaDB", "ingest.py", "--incremental"))

    if "eval" in selected:
        if not (BASE_DIR / "eval" / "questions.csv").exists():
            print("\nSkipping eval: eval/questions.csv not found.")
        elif args.full_eval:
            step("eval", run("Full evaluation", "evaluate.py", "--judge", "--label", "update"), fatal=False)
        else:
            step("eval", run("Quick retrieval check", "evaluate.py", "--retrieval-only", "--label", "update"), fatal=False)

    print(f"\n{'=' * 70}\nUpdate finished in {(time.perf_counter() - started) / 60:.1f} min")
    for name, ok in summary:
        print(f"  {'✓' if ok else '✗'} {name}")
    if "eval" in selected:
        print("Compare with earlier runs in eval/results/history.csv")


if __name__ == "__main__":
    main()
