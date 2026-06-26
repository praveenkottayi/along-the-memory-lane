"""
Split a Claude-transcribed journal .txt file into one file per entry.

The input format is:
  ===== PAGE n | DATE: YYYY-MM-DD | DATE_HEADER: <text> =====
  ... page content ...

Undated pages inherit the most recent date above them (same entry).
Output: data/processed/journal/<journal_name>/<date>.txt

Usage:
    python scripts/split_journal_txt.py --input data/processed/journal/_16.txt
    python scripts/split_journal_txt.py --input data/processed/journal/_16.txt --journal 16_2025-08_2026-01
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from common import front_matter
from config import PROCESSED_DIR, ensure_dirs

PAGE_PATTERN = re.compile(
    r"^===== PAGE (\d+) \| DATE: (.+?) \| DATE_HEADER: (.+?)(?:\s*\| TOPIC: .+?)? =====$"
)

DATE_INVALID = {"(cover)", "(none)", "(blank)", "(unknown)", ""}
# These DATE_HEADER values mean "continuation of previous entry, not a new one"
CONTINUATION_MARKERS = {"(continues)", "(none)", "(cover)", "(blank)", "(unknown)", ""}


def parse_date_header(raw: str) -> dict:
    """Extract time and day_of_week from the DATE_HEADER field if present."""
    meta = {}
    # e.g. "Aug 9th – 8:45am – Saturday"
    time_m = re.search(r"\b(\d{1,2}:\d{2}\s*(?:am|pm)?)\b", raw, re.IGNORECASE)
    if time_m:
        meta["time"] = time_m.group(1)
    day_m = re.search(
        r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b",
        raw, re.IGNORECASE
    )
    if day_m:
        meta["day_of_week"] = day_m.group(1).capitalize()
    return meta


def format_entry(date: str, journal: str, pages: list[dict]) -> str:
    """Format one journal entry as a structured .txt file (front-matter + pages)."""
    # Pull time/day from the first page header that carries them.
    extra = {}
    for p in pages:
        if not extra:
            extra = parse_date_header(p.get("date_header", ""))

    content_parts = [p["text"].strip() for p in pages if p["text"].strip()]
    content = "\n\n--- page break ---\n\n".join(content_parts)

    return front_matter(
        {
            "date": date,
            "source": "journal",
            "journal": journal,
            "time": extra.get("time"),
            "day_of_week": extra.get("day_of_week"),
        },
        content,
    )


def split_journal(input_path: Path, output_dir: Path, journal_name: str):
    raw = input_path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    # Group lines into pages
    pages = []
    current = None

    for line in lines:
        m = PAGE_PATTERN.match(line.strip())
        if m:
            if current is not None:
                pages.append(current)
            page_num = int(m.group(1))
            date_raw = m.group(2).strip()
            date_header = m.group(3).strip()
            current = {
                "page": page_num,
                "date": date_raw if date_raw.lower() not in DATE_INVALID else None,
                "date_header": date_header if date_header.lower() not in DATE_INVALID else "",
                "text": "",
            }
        elif current is not None:
            current["text"] += line + "\n"

    if current is not None:
        pages.append(current)

    print(f"Found {len(pages)} pages")

    # Group pages into entries: new entry only when DATE_HEADER has real content.
    # Continuation pages have DATE_HEADER: (none) even though DATE is inherited.
    entries = []  # list of (date, [pages])
    current_date = None
    current_pages = []

    for p in pages:
        has_header = (
            p["date_header"]
            and p["date_header"].lower() not in CONTINUATION_MARKERS
        )
        if has_header:
            # New entry — save previous
            if current_pages and current_date:
                entries.append((current_date, current_pages))
            current_date = p["date"]
            current_pages = [p]
        else:
            # Continuation page — append to current entry
            if current_date:
                current_pages.append(p)
            # else: cover / pre-first-entry pages — skip

    if current_pages and current_date:
        entries.append((current_date, current_pages))

    print(f"Detected {len(entries)} entries")

    # Write one .txt per entry
    output_dir.mkdir(parents=True, exist_ok=True)
    date_counts: dict[str, int] = {}
    saved = 0

    for date, entry_pages in entries:
        date_counts[date] = date_counts.get(date, 0) + 1
        count = date_counts[date]
        filename = f"{date}.txt" if count == 1 else f"{date}_{count:02d}.txt"
        dest = output_dir / filename
        dest.write_text(format_entry(date, journal_name, entry_pages), encoding="utf-8")
        page_nums = [p["page"] for p in entry_pages]
        print(f"  {filename}  ({len(entry_pages)} pages: {page_nums[0]}–{page_nums[-1]})")
        saved += 1

    print(f"\nSaved {saved} entries to {output_dir}")
    print("Next: python scripts/ingest.py --incremental")


def main():
    ap = argparse.ArgumentParser(description="Split a Claude-transcribed journal .txt into per-entry files.")
    ap.add_argument("--input", required=True, help="Path to the single .txt file from Claude")
    ap.add_argument("--journal", default=None,
                    help="Journal folder name (default: inferred from input filename)")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    journal_name = args.journal or input_path.stem.lstrip("#_").replace(" ", "_")
    output_dir = PROCESSED_DIR / "journal" / journal_name

    ensure_dirs()
    print(f"Input : {input_path}")
    print(f"Output: {output_dir}")
    print(f"Journal: {journal_name}\n")

    split_journal(input_path, output_dir, journal_name)


if __name__ == "__main__":
    main()
