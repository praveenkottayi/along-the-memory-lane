"""
Shared helpers for the ingestion scripts.

These three functions were duplicated across the WordPress and journal scripts;
keeping a single copy here means slug rules, HTML cleaning, and the `.txt` file
format stay consistent for every source that feeds data/processed/.
"""
import re

from bs4 import BeautifulSoup


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


def front_matter(fields: dict, content: str) -> str:
    """Wrap `content` in a `---`-delimited `key: value` metadata header.

    Empty/None values are omitted, so callers can pass optional fields freely.
    This is the single format every processed .txt file uses, which is what the
    ingest step reads back as document metadata.
    """
    lines = [f"{k}: {v}" for k, v in fields.items() if v not in (None, "")]
    return "---\n" + "\n".join(lines) + "\n---\n\n" + content.rstrip() + "\n"
