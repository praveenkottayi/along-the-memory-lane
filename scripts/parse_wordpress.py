"""
Parse a WordPress XML export file into structured text files.

Usage:
    python scripts/parse_wordpress.py --input data/raw/blog/wordpress_export.xml

Each published post becomes a .txt file in data/processed/blog/ with a metadata header.
Use this script if you have a local WordPress XML export instead of a live site.
"""
import argparse
import sys
from datetime import datetime
from html import unescape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from common import front_matter, html_to_text, make_slug
from config import PROCESSED_DIR, ensure_dirs


def parse_wordpress_export(xml_path: Path) -> list[dict]:
    """Extract published posts and pages from a WordPress WXR export file."""
    with open(xml_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml-xml")

    posts = []
    for item in soup.find_all("item"):
        post_type = item.find("post_type")
        status = item.find("status")

        if not post_type or post_type.text not in ("post", "page"):
            continue
        if not status or status.text != "publish":
            continue

        title = item.find("title")
        pub_date = item.find("pubDate")
        content = item.find("encoded")

        if not title or not content:
            continue

        try:
            dt = date_parser.parse(pub_date.text) if pub_date else datetime(1970, 1, 1)
        except Exception:
            print(f"  Warning: could not parse date for '{title.text.strip()}', using epoch.")
            dt = datetime(1970, 1, 1)

        posts.append({
            "title": unescape(title.text.strip()),
            "date": dt.strftime("%Y-%m-%d"),
            "datetime": dt.isoformat(),
            "content": html_to_text(content.text),
            "source": "blog",
        })

    return posts


def save_posts(posts: list[dict], output_dir: Path):
    """Save each post as a .txt file with a metadata header."""
    output_dir = output_dir / "blog"
    output_dir.mkdir(parents=True, exist_ok=True)

    for post in posts:
        filename = f"{post['date']}_{make_slug(post['title'])}.txt"
        content = front_matter(
            {
                "date": post["date"],
                "datetime": post["datetime"],
                "source": "blog",
                "title": post["title"],
            },
            post["content"],
        )
        (output_dir / filename).write_text(content, encoding="utf-8")

    print(f"Saved {len(posts)} posts to {output_dir}")


def main():
    ap = argparse.ArgumentParser(description="Parse a WordPress XML export into text files.")
    ap.add_argument("--input", required=True, help="Path to WordPress XML export file")
    args = ap.parse_args()

    xml_path = Path(args.input)
    if not xml_path.exists():
        print(f"File not found: {xml_path}")
        sys.exit(1)

    ensure_dirs()
    print(f"Parsing {xml_path}...")
    posts = parse_wordpress_export(xml_path)
    print(f"Found {len(posts)} published posts")
    save_posts(posts, PROCESSED_DIR)


if __name__ == "__main__":
    main()
