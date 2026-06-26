"""
OCR pipeline for handwritten journal images.

Usage:
    # OCR a single journal
    python scripts/ocr_journals.py --journal data/raw/journal/16_2025-08_2026-01

    # OCR all journals under data/raw/journal/ (skips fully-cached ones)
    python scripts/ocr_journals.py --all

    # Inspect entry grouping on cached sidecar files (no API calls)
    python scripts/ocr_journals.py --journal data/raw/journal/16_2025-08_2026-01 --inspect
    python scripts/ocr_journals.py --all --inspect

For each journal folder:
  - Reads images in order (page_001.jpg, page_002.jpg, ...)
  - Uses Claude vision (Anthropic API) to transcribe the handwriting, caching
    the result as a sidecar <image>.ocr.txt so reruns skip already-done pages
  - Detects entry headers (date, time, day, location, WFH)
  - Groups consecutive pages into a single entry
  - Saves one .txt file per entry to data/processed/journal/

Local vision models couldn't read cursive (Apple Vision, llava, moondream,
Tesseract) and llama3.2-vision won't load on this Ollama build, so OCR runs
through the Anthropic API. Requires ANTHROPIC_API_KEY in the environment.
"""
import argparse
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

from common import (
    front_matter,
    parse_date,
    parse_day,
    parse_location,
    parse_time,
    parse_work_mode,
)
from config import OCR_VISION_MODEL, PROCESSED_DIR, RAW_JOURNAL_DIR, ensure_dirs

# ---------------------------------------------------------------------------
# Handwriting OCR via Claude vision
# ---------------------------------------------------------------------------

OCR_PROMPT = (
    "Transcribe every word of handwritten text in this journal page image "
    "exactly as written. Preserve line breaks. Do not summarise, interpret, "
    "or add anything. If a word is unclear, write your best guess in brackets, "
    "e.g. [unclear]. Output only the transcribed text, nothing else."
)

MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    """Lazily create the Anthropic client (reads ANTHROPIC_API_KEY from env)."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def ocr_image(image_path: Path) -> str:
    """Use Claude vision to transcribe a handwritten journal page.

    Results are cached as a sidecar <image>.ocr.txt next to the image file.
    Re-running the script loads the cache instead of calling the API, so:
      - Interrupted runs can resume from where they left off.
      - You can manually correct bad transcriptions by editing the .ocr.txt.
      - The API is never called twice for the same page.
    """
    sidecar = image_path.with_suffix(".ocr.txt")
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8")

    media_type = MEDIA_TYPES.get(image_path.suffix.lower(), "image/jpeg")
    image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")

    try:
        response = get_client().messages.create(
            model=OCR_VISION_MODEL,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }],
        )
    except anthropic.APIError as e:
        print(f"  Warning: Claude OCR error for {image_path.name}: {e}")
        return ""

    result = "".join(b.text for b in response.content if b.type == "text").strip()
    sidecar.write_text(result, encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Entry header detection
# ---------------------------------------------------------------------------
# parse_date / parse_time / parse_day / parse_location / parse_work_mode
# are imported from common.py — edit them there, not here.


def is_entry_header(text: str) -> bool:
    """Return True if this text looks like a journal entry header (has a date)."""
    # Check only the first 5 lines — headers are at the top of the page
    first_lines = "\n".join(text.strip().splitlines()[:5])
    return parse_date(first_lines) is not None


# ---------------------------------------------------------------------------
# Entry grouping and saving
# ---------------------------------------------------------------------------

def extract_metadata(header_text: str, journal_folder: str) -> dict:
    """Extract all available metadata from the entry header text."""
    first_lines = "\n".join(header_text.strip().splitlines()[:6])
    date = parse_date(first_lines)
    return {
        "date": date or "unknown",
        "time": parse_time(first_lines),
        "day_of_week": parse_day(first_lines),
        "location": parse_location(first_lines),
        "work_mode": parse_work_mode(first_lines),
        "source": "journal",
        "journal": journal_folder,
    }


def format_entry(metadata: dict, pages_text: list[str]) -> str:
    """Format a journal entry as a structured .txt file (front-matter + pages).

    Title is synthesized from the date so every source has a consistent
    metadata schema (blog posts have titles; journals get one generated here).
    """
    date = metadata["date"]
    title = f"Journal — {date}"
    content = "\n\n--- page break ---\n\n".join(pages_text)
    return front_matter(
        {
            "date": date,
            "title": title,
            "source": metadata["source"],
            "journal": metadata["journal"],
            "time": metadata.get("time"),
            "day_of_week": metadata.get("day_of_week"),
            "location": metadata.get("location"),
            "work_mode": metadata.get("work_mode"),
        },
        content,
    )


def get_journal_images(journal_dir: Path) -> list[Path]:
    """Return sorted page images from a journal folder, skipping the cover."""
    return sorted([
        f for f in journal_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        and f.stem.lower() != "cover"
    ])


def group_pages_into_entries(
    images: list[Path],
    journal_name: str,
    ocr_fn,
) -> list[tuple[dict, list[str]]]:
    """OCR pages and group them into (metadata, pages_text) entries.

    Pages before the first detected header are saved as a preamble entry
    with date "unknown" rather than being silently dropped.
    """
    entries: list[tuple[dict, list[str]]] = []
    current_pages: list[str] = []
    current_meta: dict | None = None

    for i, img_path in enumerate(images):
        text = ocr_fn(img_path, i, len(images))

        if not text.strip():
            print(f"    Warning: no text extracted from {img_path.name}")
            if current_meta is not None:
                current_pages.append("[unreadable page]")
            continue

        if is_entry_header(text):
            # Save the previous entry before starting a new one
            if current_meta is not None and current_pages:
                entries.append((current_meta, current_pages))

            current_meta = extract_metadata(text, journal_name)
            current_pages = [text]
            print(f"    New entry: {current_meta['date']} "
                  f"{current_meta.get('time', '')} "
                  f"{current_meta.get('location', '')}")
        else:
            # Continuation page — or preamble before the first header
            if current_meta is None:
                # Pages before the first dated entry: save as preamble, don't drop
                current_meta = {
                    "date": "unknown",
                    "source": "journal",
                    "journal": journal_name,
                }
                current_pages = []
            current_pages.append(text)

    # Flush the last entry
    if current_meta is not None and current_pages:
        entries.append((current_meta, current_pages))

    return entries


def save_entries(entries: list[tuple[dict, list[str]]], out_dir: Path) -> tuple[int, int]:
    """Write entries to disk, returning (saved, unknown_count)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    unknown_count = 0

    for meta, pages in entries:
        date = meta["date"]
        if date == "unknown":
            unknown_count += 1
            filename = f"unknown_{unknown_count:03d}.txt"
        else:
            filename = f"{date}.txt"
            dest = out_dir / filename
            if dest.exists():
                # Multiple entries on the same date: append a counter
                suffix = 2
                while (out_dir / f"{date}_{suffix:02d}.txt").exists():
                    suffix += 1
                filename = f"{date}_{suffix:02d}.txt"

        (out_dir / filename).write_text(format_entry(meta, pages), encoding="utf-8")
        saved += 1

    return saved, unknown_count


# ---------------------------------------------------------------------------
# Process modes
# ---------------------------------------------------------------------------

def process_journal(journal_dir: Path, output_dir: Path):
    """OCR all pages in a journal folder and save one .txt per entry."""
    journal_name = journal_dir.name
    out_dir = output_dir / "journal" / journal_name

    images = get_journal_images(journal_dir)
    if not images:
        print(f"No images found in {journal_dir}")
        return

    print(f"\nProcessing journal: {journal_name} ({len(images)} pages)")

    def ocr_fn(img_path, i, total):
        print(f"  OCR: {img_path.name} ({i+1}/{total})")
        return ocr_image(img_path)

    entries = group_pages_into_entries(images, journal_name, ocr_fn)
    saved, unknown_count = save_entries(entries, out_dir)

    print(f"  Saved {saved} entries to {out_dir}")
    if unknown_count:
        print(f"  Warning: {unknown_count} page(s) had no detectable date header")


def inspect_journal(journal_dir: Path):
    """Show entry grouping from cached OCR sidecars — no API calls.

    Prints per-page classification (HEADER/continuation) and the date detected,
    so you can tune header detection patterns without spending API tokens.
    Pages with no sidecar cache are flagged so you know to run OCR first.
    """
    journal_name = journal_dir.name
    images = get_journal_images(journal_dir)

    if not images:
        print(f"No images found in {journal_dir}")
        return

    print(f"\nInspecting: {journal_name} ({len(images)} pages)")
    uncached = 0

    for img_path in images:
        sidecar = img_path.with_suffix(".ocr.txt")
        if not sidecar.exists():
            print(f"  {img_path.name}: [no OCR cache — run without --inspect first]")
            uncached += 1
            continue

        text = sidecar.read_text(encoding="utf-8")
        first_lines = "\n".join(text.strip().splitlines()[:5])
        date = parse_date(first_lines)

        if date:
            loc = parse_location(first_lines)
            time_ = parse_time(first_lines)
            extras = " ".join(filter(None, [time_, loc]))
            print(f"  {img_path.name}: HEADER  → {date} {extras}".rstrip())
        else:
            # Show first non-empty line as a hint for manual inspection
            preview = next((l.strip() for l in text.splitlines() if l.strip()), "")
            print(f"  {img_path.name}: continuation  ({preview[:60]})")

    if uncached:
        print(f"\n  {uncached} page(s) not yet OCR'd. Run without --inspect to process them.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="OCR journal images and extract entries."
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--journal",
        help="Path to a single journal folder (e.g. data/raw/journal/01_2013-01_2013-08)"
    )
    source.add_argument(
        "--all",
        action="store_true",
        help="Process all journal folders under data/raw/journal/",
    )
    ap.add_argument(
        "--inspect",
        action="store_true",
        help=(
            "Show per-page header detection results from cached .ocr.txt sidecars "
            "without calling the API. Run OCR first if sidecars are missing."
        ),
    )
    args = ap.parse_args()

    ensure_dirs()

    if args.all:
        journal_dirs = sorted([
            d for d in RAW_JOURNAL_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        if not journal_dirs:
            print(f"No journal folders found under {RAW_JOURNAL_DIR}")
            sys.exit(1)
        print(f"Found {len(journal_dirs)} journal folder(s)")

        for journal_dir in journal_dirs:
            if args.inspect:
                inspect_journal(journal_dir)
            else:
                # process_journal always runs — it skips OCR API calls for pages
                # that already have a .ocr.txt sidecar, but re-runs entry grouping
                # and saves .txt files. Safe to call repeatedly.
                process_journal(journal_dir, PROCESSED_DIR)
    else:
        journal_dir = Path(args.journal)
        if not journal_dir.exists():
            print(f"Journal folder not found: {journal_dir}")
            sys.exit(1)

        if args.inspect:
            inspect_journal(journal_dir)
        else:
            process_journal(journal_dir, PROCESSED_DIR)
            print("\nDone. Now run: python scripts/ingest.py --incremental")


if __name__ == "__main__":
    main()
