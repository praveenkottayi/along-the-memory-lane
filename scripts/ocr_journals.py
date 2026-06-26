"""
OCR pipeline for handwritten journal images.

Usage:
    python scripts/ocr_journals.py --journal data/raw/journal/16_2025-08_2026-01

For each journal folder:
  - Reads images in order (page_001.jpg, page_002.jpg, ...)
  - Uses Claude vision (Anthropic API) to transcribe the handwriting
  - Detects entry headers (date, time, day, location, WFH)
  - Groups consecutive pages into a single entry
  - Saves one .txt file per entry to data/processed/journal/

Local vision models couldn't read cursive (Apple Vision, llava, moondream,
Tesseract) and llama3.2-vision won't load on this Ollama build, so OCR runs
through the Anthropic API. Requires ANTHROPIC_API_KEY in the environment.
"""
import argparse
import base64
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

from common import front_matter
from config import OCR_VISION_MODEL, PROCESSED_DIR, ensure_dirs

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
    """Use Claude vision to transcribe a handwritten journal page."""
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

    return "".join(b.text for b in response.content if b.type == "text").strip()


# ---------------------------------------------------------------------------
# Entry header detection
# ---------------------------------------------------------------------------

# Date patterns — handles various formats commonly used in handwriting
DATE_PATTERNS = [
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|"
    r"July|August|September|October|November|December)\s+(\d{4})\b",
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
    r"\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\b",
]

TIME_PATTERN = r"\b(\d{1,2})[:\.](\d{2})\s*(AM|PM|am|pm)?\b"

DAY_PATTERN = r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"

LOCATION_KEYWORDS = [
    "Mumbai", "Bangalore", "Delhi", "Chennai", "Goa", "Hyderabad", "Pune",
    "Kochi", "Kerala", "office", "home", "cafe", "hotel", "airport",
]

WFH_PATTERN = r"\bWFH\b|\bWork\s+from\s+Home\b|\bwork\s+from\s+home\b"

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_date(text: str) -> str | None:
    """Try to extract a YYYY-MM-DD date from OCR text."""
    # Format: "15th April 2013" or "April 15, 2013"
    m = re.search(DATE_PATTERNS[0], text, re.IGNORECASE)
    if m:
        day, month_name, year = m.group(1), m.group(2), m.group(3)
        month = MONTH_MAP.get(month_name.lower())
        if month:
            return f"{year}-{month:02d}-{int(day):02d}"

    m = re.search(DATE_PATTERNS[1], text, re.IGNORECASE)
    if m:
        month_name, day, year = m.group(1), m.group(2), m.group(3)
        month = MONTH_MAP.get(month_name.lower())
        if month:
            return f"{year}-{month:02d}-{int(day):02d}"

    # Format: "15/04/2013" or "15-04-2013"
    m = re.search(DATE_PATTERNS[2], text)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        if len(y) == 2:
            y = "20" + y
        return f"{y}-{int(mo):02d}-{int(d):02d}"

    return None


def parse_time(text: str) -> str | None:
    """Extract time string from OCR text."""
    m = re.search(TIME_PATTERN, text, re.IGNORECASE)
    if m:
        h, mi, meridiem = m.group(1), m.group(2), m.group(3) or ""
        return f"{h}:{mi} {meridiem}".strip()
    return None


def parse_day(text: str) -> str | None:
    """Extract day of week from OCR text."""
    m = re.search(DAY_PATTERN, text, re.IGNORECASE)
    return m.group(1).capitalize() if m else None


def parse_location(text: str) -> str | None:
    """Extract location hint from OCR text."""
    for loc in LOCATION_KEYWORDS:
        if re.search(rf"\b{loc}\b", text, re.IGNORECASE):
            return loc
    return None


def parse_work_mode(text: str) -> str | None:
    """Detect WFH or similar work mode markers."""
    if re.search(WFH_PATTERN, text, re.IGNORECASE):
        return "WFH"
    return None


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
    """Format a journal entry as a structured .txt file (front-matter + pages)."""
    content = "\n\n--- page break ---\n\n".join(pages_text)
    return front_matter(
        {
            "date": metadata["date"],
            "source": metadata["source"],
            "journal": metadata["journal"],
            "time": metadata.get("time"),
            "day_of_week": metadata.get("day_of_week"),
            "location": metadata.get("location"),
            "work_mode": metadata.get("work_mode"),
        },
        content,
    )


def process_journal(journal_dir: Path, output_dir: Path):
    """OCR all pages in a journal folder and save one .txt per entry."""
    journal_name = journal_dir.name
    out_dir = output_dir / "journal" / journal_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect images in sorted order, skip cover
    images = sorted([
        f for f in journal_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        and f.stem.lower() != "cover"
    ])

    if not images:
        print(f"No images found in {journal_dir}")
        return

    print(f"\nProcessing journal: {journal_name} ({len(images)} pages)")

    entries = []           # list of (metadata, [page_texts])
    current_pages = []     # page texts for current entry
    current_meta = None

    for i, img_path in enumerate(images):
        print(f"  OCR: {img_path.name} ({i+1}/{len(images)})")
        text = ocr_image(img_path)

        if not text.strip():
            print(f"    Warning: no text extracted from {img_path.name}")
            if current_pages:
                current_pages.append("[unreadable page]")
            continue

        if is_entry_header(text):
            # Save previous entry before starting new one
            if current_pages and current_meta:
                entries.append((current_meta, current_pages))

            current_meta = extract_metadata(text, journal_name)
            current_pages = [text]
            print(f"    New entry detected: {current_meta['date']} "
                  f"{current_meta.get('time', '')} "
                  f"{current_meta.get('location', '')}")
        else:
            # Continuation of current entry
            if current_pages is None:
                # Pages before first detected header — label as preamble
                current_meta = {
                    "date": "unknown",
                    "source": "journal",
                    "journal": journal_name,
                }
                current_pages = []
            current_pages.append(text)

    # Save last entry
    if current_pages and current_meta:
        entries.append((current_meta, current_pages))

    # Write entries to disk
    saved = 0
    unknown_count = 0
    for meta, pages in entries:
        date = meta["date"]
        if date == "unknown":
            unknown_count += 1
            filename = f"unknown_{unknown_count:03d}.txt"
        else:
            filename = f"{date}.txt"
            # Handle multiple entries on same date
            dest = out_dir / filename
            if dest.exists():
                filename = f"{date}_{saved+1:02d}.txt"

        (out_dir / filename).write_text(
            format_entry(meta, pages), encoding="utf-8"
        )
        saved += 1

    print(f"  Saved {saved} entries to {out_dir}")
    if unknown_count:
        print(f"  Warning: {unknown_count} pages had no detectable date header")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="OCR journal images and extract entries."
    )
    ap.add_argument(
        "--journal",
        required=True,
        help="Path to a journal folder (e.g. data/raw/journal/01_2013-01_2013-08)"
    )
    args = ap.parse_args()

    journal_dir = Path(args.journal)
    if not journal_dir.exists():
        print(f"Journal folder not found: {journal_dir}")
        sys.exit(1)

    ensure_dirs()
    process_journal(journal_dir, PROCESSED_DIR)
    print("\nDone. Now run: python scripts/ingest.py --incremental")


if __name__ == "__main__":
    main()
