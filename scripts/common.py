"""
Shared helpers for the ingestion scripts.

Covers three areas:
  - Text format: front_matter() / parse_front_matter() — the contract every
    processed .txt file uses to carry metadata into ChromaDB.
  - Journal text parsing: parse_date/time/day/location/work_mode — used by
    ocr_journals.py and parse_claude_transcript.py so the logic lives once.
  - Utilities: make_slug, html_to_text — shared across WordPress and journal
    scripts.
"""
import re

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Journal header parsing — shared by ocr_journals.py and parse_claude_transcript.py
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|"
    r"July|August|September|October|November|December)\s+(\d{4})\b",
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
    r"\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\b",
]

_TIME_PATTERN = r"\b(\d{1,2})[:\.](\d{2})\s*(AM|PM|am|pm)?\b"
_DAY_PATTERN = r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
_WFH_PATTERN = r"\bWFH\b|\bWork\s+from\s+Home\b|\bwork\s+from\s+home\b"

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

LOCATION_KEYWORDS = [
    "Mumbai", "Bangalore", "Delhi", "Chennai", "Goa", "Hyderabad", "Pune",
    "Kochi", "Kerala", "office", "home", "cafe", "hotel", "airport",
]


def parse_date(text: str) -> str | None:
    """Extract a YYYY-MM-DD date from journal header text."""
    m = re.search(_DATE_PATTERNS[0], text, re.IGNORECASE)
    if m:
        day, month_name, year = m.group(1), m.group(2), m.group(3)
        month = _MONTH_MAP.get(month_name.lower())
        if month:
            return f"{year}-{month:02d}-{int(day):02d}"

    m = re.search(_DATE_PATTERNS[1], text, re.IGNORECASE)
    if m:
        month_name, day, year = m.group(1), m.group(2), m.group(3)
        month = _MONTH_MAP.get(month_name.lower())
        if month:
            return f"{year}-{month:02d}-{int(day):02d}"

    m = re.search(_DATE_PATTERNS[2], text)
    if m:
        d, mo, y = m.group(1), m.group(2), m.group(3)
        if len(y) == 2:
            y = "20" + y
        return f"{y}-{int(mo):02d}-{int(d):02d}"

    return None


def parse_time(text: str) -> str | None:
    """Extract time string from journal header text."""
    m = re.search(_TIME_PATTERN, text, re.IGNORECASE)
    if m:
        h, mi, meridiem = m.group(1), m.group(2), m.group(3) or ""
        return f"{h}:{mi} {meridiem}".strip()
    return None


def parse_day(text: str) -> str | None:
    """Extract day of week from journal header text."""
    m = re.search(_DAY_PATTERN, text, re.IGNORECASE)
    return m.group(1).capitalize() if m else None


def parse_location(text: str) -> str | None:
    """Extract location hint from journal header text."""
    for loc in LOCATION_KEYWORDS:
        if re.search(rf"\b{loc}\b", text, re.IGNORECASE):
            return loc
    return None


def parse_work_mode(text: str) -> str | None:
    """Detect WFH or similar work mode markers."""
    return "WFH" if re.search(_WFH_PATTERN, text, re.IGNORECASE) else None


def make_slug(title: str, max_len: int = 60) -> str:
    """Turn a post/entry title into a filesystem-safe slug.

    Lowercases, drops punctuation, collapses whitespace to single hyphens,
    and trims to `max_len` characters.
    """
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:max_len].strip("-")


def html_to_text(html: str) -> str:
    """Strip HTML tags and collapse runs of 3+ blank lines down to one."""
    soup = BeautifulSoup(html or "", "lxml")
    text = soup.get_text(separator="\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Parse a ---\\nkey: value\\n--- header from a processed .txt file.

    Returns (metadata_dict, body_text). If no valid front-matter is found,
    returns ({}, original_text) so callers can handle either format safely.

    This is the inverse of front_matter() — together they form the contract
    that every source (blog, journal, and future sources) uses to carry
    metadata through the pipeline into ChromaDB.
    """
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    header = text[4:end]
    body = text[end + 5:]  # skip the closing "\n---\n"
    metadata: dict[str, str] = {}
    for line in header.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key and value:
                metadata[key] = value
    return metadata, body


def front_matter(fields: dict, content: str) -> str:
    """Wrap `content` in a `---`-delimited `key: value` metadata header.

    Empty/None values are omitted, so callers can pass optional fields freely.
    This is the single format every processed .txt file uses, which is what the
    ingest step reads back as document metadata.
    """
    lines = [f"{k}: {v}" for k, v in fields.items() if v not in (None, "")]
    return "---\n" + "\n".join(lines) + "\n---\n\n" + content.rstrip() + "\n"
