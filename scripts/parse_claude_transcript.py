"""
Parse a Claude-chat journal transcript into standard front-matter .txt files.

Usage:
    python scripts/parse_claude_transcript.py                    # all # files
    python scripts/parse_claude_transcript.py --file <path>      # single file
    python scripts/parse_claude_transcript.py --dry-run          # preview, no writes

Background:
    When ocr_journals.py couldn't be run (or for higher-quality transcriptions),
    journals were transcribed via Claude chat and saved with a '#' prefix:
        data/processed/journal/#16_2025-08_2026-01.txt

    These files use a page-delimited format:
        ===== PAGE N | DATE: YYYY-MM-DD | DATE_HEADER: <text or (continues)> =====
        <page text>

    This script converts them to the same front-matter .txt format that
    ocr_journals.py produces, so ingest.py picks them up uniformly.

    Output goes to data/processed/journal/<journal_name>/ — the same folder
    as pipeline-generated files. Claude-chat versions take precedence (overwrite)
    since they tend to be higher quality.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import front_matter, parse_day, parse_location, parse_time, parse_work_mode
from config import PROCESSED_DIR, ensure_dirs

# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

# Matches: ===== PAGE 3 | DATE: 2025-08-09 | DATE_HEADER: Aug 9th – 8:45am – Saturday =====
PAGE_RE = re.compile(
    r"^===== PAGE (\d+) \| DATE: (.+?) \| DATE_HEADER: (.+?) =====$",
    re.MULTILINE,
)

# DATE_HEADER values that mean "this page continues the previous entry"
CONTINUATION_MARKERS = {"(continues)", "(none)", "(cover)"}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_transcript(path: Path) -> list[tuple[dict, list[str]]]:
    """Parse a Claude-chat transcript into a list of (metadata, pages) entries.

    Each entry groups all consecutive pages that belong to the same journal
    entry — i.e. until the next page with a real DATE_HEADER (not a
    continuation marker).
    """
    text = path.read_text(encoding="utf-8")

    # Split on page markers. PAGE_RE has 3 capture groups, so re.split gives:
    # [pre_text, page_num, date, date_header, content, page_num, date, ...]
    parts = PAGE_RE.split(text)

    entries: list[tuple[dict, list[str]]] = []
    current_meta: dict | None = None
    current_pages: list[str] = []

    # parts[0] is the file header (# comments) — skip it
    i = 1
    while i + 3 < len(parts):
        _page_num = parts[i].strip()
        date = parts[i + 1].strip()
        date_header = parts[i + 2].strip()
        content = parts[i + 3].strip()
        i += 4

        is_continuation = date_header.lower() in CONTINUATION_MARKERS
        is_decorative = date.lower() in ("(cover)", "(none)", "")

        if is_decorative:
            # Cover or undated decorative page — append text if any, else skip
            if content and current_pages is not None:
                current_pages.append(content)
            continue

        if is_continuation:
            if content and current_meta is not None:
                current_pages.append(content)
            continue

        # Real entry header — flush previous and start new entry
        if current_meta is not None and current_pages:
            entries.append((current_meta, current_pages))

        current_meta = _build_metadata(date, date_header, path.stem.lstrip("#"))
        current_pages = [content] if content else []

    # Flush last entry
    if current_meta is not None and current_pages:
        entries.append((current_meta, current_pages))

    return entries


def _build_metadata(date: str, date_header: str, journal_name: str) -> dict:
    """Build a metadata dict from a page's DATE and DATE_HEADER fields."""
    return {
        "date": date,
        "title": f"Journal — {date}",
        "source": "journal",
        "journal": journal_name,
        "time": parse_time(date_header),
        "day_of_week": parse_day(date_header),
        "location": parse_location(date_header),
        "work_mode": parse_work_mode(date_header),
    }


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_entries(
    entries: list[tuple[dict, list[str]]],
    out_dir: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Write entries to disk. Returns (saved, overwritten) counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    overwritten = 0
    date_counts: dict[str, int] = {}

    for meta, pages in entries:
        date = meta["date"]
        date_counts[date] = date_counts.get(date, 0) + 1
        count = date_counts[date]

        filename = f"{date}.txt" if count == 1 else f"{date}_{count:02d}.txt"
        dest = out_dir / filename
        existed = dest.exists()

        content = "\n\n--- page break ---\n\n".join(pages)
        text = front_matter(
            {k: v for k, v in meta.items() if v not in (None, "")},
            content,
        )

        if not dry_run:
            dest.write_text(text, encoding="utf-8")

        action = "overwrite" if existed else "write"
        print(f"  [{action}] {filename}")
        saved += 1
        if existed:
            overwritten += 1

    return saved, overwritten


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_file(path: Path, dry_run: bool = False):
    """Parse a single Claude-chat transcript and write output files."""
    journal_name = path.stem.lstrip("#")
    out_dir = PROCESSED_DIR / "journal" / journal_name

    print(f"\nParsing: {path.name} → {out_dir.relative_to(PROCESSED_DIR)}/")
    entries = parse_transcript(path)
    print(f"  Found {len(entries)} entries")

    saved, overwritten = write_entries(entries, out_dir, dry_run=dry_run)
    print(f"  {'Would write' if dry_run else 'Wrote'} {saved} files "
          f"({overwritten} overwrote pipeline versions)")


def find_transcripts(processed_journal_dir: Path) -> list[Path]:
    """Find all Claude-chat transcripts (files starting with #) in the journal dir."""
    return sorted([
        f for f in processed_journal_dir.iterdir()
        if f.is_file() and f.suffix == ".txt" and f.name.startswith("#")
    ])


def main():
    ap = argparse.ArgumentParser(
        description="Convert Claude-chat journal transcripts to front-matter .txt files."
    )
    source = ap.add_mutually_exclusive_group()
    source.add_argument(
        "--file",
        help="Path to a single transcript file (e.g. data/processed/journal/#16_...txt)"
    )
    source.add_argument(
        "--all",
        action="store_true",
        help="Process all # transcript files under data/processed/journal/"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be written without touching any files",
    )
    args = ap.parse_args()

    ensure_dirs()
    journal_dir = PROCESSED_DIR / "journal"

    if args.file:
        path = Path(args.file)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        process_file(path, dry_run=args.dry_run)
    else:
        transcripts = find_transcripts(journal_dir)
        if not transcripts:
            print(f"No # transcript files found in {journal_dir}")
            sys.exit(1)
        print(f"Found {len(transcripts)} transcript(s)")
        for t in transcripts:
            process_file(t, dry_run=args.dry_run)

    if args.dry_run:
        print("\n[dry-run] No files were written.")
    else:
        print("\nDone. Run: python scripts/migrate_add_title.py && python scripts/ingest.py")


if __name__ == "__main__":
    main()
