"""
One-time migration: add a 'title' field to journal .txt files that are missing it.

Usage:
    python scripts/migrate_add_title.py           # patch all journal files
    python scripts/migrate_add_title.py --dry-run # preview, no writes

Background:
    The old ocr_journals.py didn't include 'title' in the front-matter header.
    The UI and ingest pipeline now expect a consistent schema with 'title'
    across all sources (blogs already have titles from WordPress; journals get
    a synthesized "Journal — YYYY-MM-DD" title).

    This script is safe to re-run: files that already have a title are skipped.
    Blog files are also skipped (they already have titles from fetch_wordpress_api.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from common import front_matter, parse_front_matter
from config import PROCESSED_DIR


def add_title_to_file(path: Path, dry_run: bool = False) -> bool:
    """Add 'title' to a journal .txt if missing. Returns True if file was changed."""
    text = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(text)

    if not meta:
        # No front-matter at all — skip (not our format)
        return False

    if meta.get("source") != "journal":
        # Only patch journal files
        return False

    if "title" in meta:
        # Already has a title — nothing to do
        return False

    date = meta.get("date", "unknown")
    meta["title"] = f"Journal — {date}"

    # Rebuild the file with title inserted after date (keeps field order readable)
    ordered_keys = ["date", "title", "source", "journal", "time", "day_of_week",
                    "location", "work_mode"]
    ordered_meta = {k: meta[k] for k in ordered_keys if k in meta}
    # Append any extra keys not in the order list
    for k, v in meta.items():
        if k not in ordered_meta:
            ordered_meta[k] = v

    new_text = front_matter({k: v for k, v in ordered_meta.items() if v not in (None, "")}, body)

    if not dry_run:
        path.write_text(new_text, encoding="utf-8")

    return True


def main():
    ap = argparse.ArgumentParser(
        description="Add missing 'title' field to journal .txt files."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview which files would be patched without writing anything",
    )
    args = ap.parse_args()

    journal_dir = PROCESSED_DIR / "journal"
    if not journal_dir.exists():
        print(f"No journal directory found at {journal_dir}")
        sys.exit(0)

    # Find all .txt files under data/processed/journal/, skip # transcript files
    all_files = [
        f for f in sorted(journal_dir.rglob("*.txt"))
        if not f.name.startswith("#")
    ]

    patched = 0
    skipped = 0

    for path in all_files:
        changed = add_title_to_file(path, dry_run=args.dry_run)
        if changed:
            print(f"  {'[dry-run] would patch' if args.dry_run else 'patched'}: {path.relative_to(PROCESSED_DIR)}")
            patched += 1
        else:
            skipped += 1

    print(f"\n{'Would patch' if args.dry_run else 'Patched'} {patched} file(s), "
          f"skipped {skipped} (already have title or non-journal).")

    if args.dry_run:
        print("[dry-run] No files were written.")
    elif patched > 0:
        print("Done. Run: python scripts/ingest.py to rebuild ChromaDB.")


if __name__ == "__main__":
    main()
