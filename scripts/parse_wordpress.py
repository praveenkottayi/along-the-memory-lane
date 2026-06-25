"""
Parse WordPress XML export into structured text files.

Usage:
    python scripts/parse_wordpress.py --input data/raw/blog/wordpress_export.xml

Each post becomes a .txt file in data/processed/ with a metadata header.
"""
import argparse
import re
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DIR


def clean_html(html: str) -> str:
    """Strip HTML tags and clean whitespace."""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_wordpress_export(xml_path: Path) -> list[dict]:
    """Extract posts from WordPress WXR export file."""
    with open(xml_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml-xml")

    posts = []
    for item in soup.find_all("item"):
        post_type = item.find("post_type")
        status = item.find("status")

        # Only published posts/pages
        if not post_type or post_type.text not in ("post", "page"):
            continue
        if not status or status.text != "publish":
            continue

        title = item.find("title")
        pub_date = item.find("pubDate")
        content = item.find("encoded")  # wp:content or content:encoded
        if not content:
            content = item.find("content:encoded")

        if not title or not content:
            continue

        # Parse date
        try:
            dt = date_parser.parse(pub_date.text) if pub_date else datetime(1970, 1, 1)
        except Exception:
            dt = datetime(1970, 1, 1)

        posts.append({
            "title": title.text.strip(),
            "date": dt.strftime("%Y-%m-%d"),
            "datetime": dt.isoformat(),
            "content": clean_html(content.text),
            "source": "blog",
        })

    return posts


def save_posts(posts: list[dict], output_dir: Path):
    """Save each post as a .txt file with metadata header."""
    output_dir = output_dir / "blog"
    output_dir.mkdir(parents=True, exist_ok=True)

    for post in posts:
        # Filename: date + slugified title
        slug = re.sub(r"[^\w\s-]", "", post["title"].lower())
        slug = re.sub(r"[\s_-]+", "-", slug)[:60]
        filename = f"{post['date']}_{slug}.txt"

        content = f"""---
date: {post['date']}
datetime: {post['datetime']}
source: blog
title: {post['title']}
---

{post['content']}
"""
        (output_dir / filename).write_text(content, encoding="utf-8")

    print(f"Saved {len(posts)} posts to {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to WordPress XML export")
    args = ap.parse_args()

    xml_path = Path(args.input)
    if not xml_path.exists():
        print(f"File not found: {xml_path}")
        sys.exit(1)

    print(f"Parsing {xml_path}...")
    posts = parse_wordpress_export(xml_path)
    print(f"Found {len(posts)} published posts")
    save_posts(posts, PROCESSED_DIR)


if __name__ == "__main__":
    main()
