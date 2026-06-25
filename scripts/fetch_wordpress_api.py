"""
Fetch all posts from a WordPress.com blog via the public REST API.
No login or export needed — works for any public WordPress.com site.

Usage:
    python scripts/fetch_wordpress_api.py --site praveenkottayi.com

For each post, saves:
  data/processed/blog/<date>_<slug>.txt   ← cleaned text with metadata header
  data/raw/blog/images/<date>_<slug>/     ← all images from that post
"""
import argparse
import re
import sys
import time
from html import unescape
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from config import PROCESSED_DIR, RAW_BLOG_DIR, ensure_dirs


def extract_images(html: str) -> list[str]:
    """Extract unique image URLs from post HTML, stripping CDN resize params."""
    soup = BeautifulSoup(html or "", "lxml")
    urls = []
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src")
        if src and src.startswith("http"):
            src = src.split("?")[0]  # strip WordPress CDN resize params
            urls.append(src)
    return list(dict.fromkeys(urls))  # deduplicate, preserve order


def clean_html(html: str, image_paths: list[str]) -> str:
    """Strip HTML tags, normalise whitespace, append local image references."""
    soup = BeautifulSoup(html or "", "lxml")
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if image_paths:
        refs = "\n".join(f"  - {p}" for p in image_paths)
        text += f"\n\n[images]\n{refs}"

    return text


def download_images(image_urls: list[str], folder: Path) -> list[str]:
    """Download images into folder. Returns list of local paths for saved images."""
    folder.mkdir(parents=True, exist_ok=True)
    saved = []

    for url in image_urls:
        try:
            filename = Path(urlparse(url).path).name
            if not filename or "." not in filename:
                filename = f"image_{len(saved) + 1}.jpg"
            dest = folder / filename

            if dest.exists():
                saved.append(str(dest))
                continue

            resp = requests.get(url, timeout=20, stream=True)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            saved.append(str(dest))
            time.sleep(0.1)
        except Exception as e:
            print(f"    Warning: could not download {url}: {e}")

    return saved


def make_slug(title: str) -> str:
    """Convert a post title into a safe filename slug."""
    slug = re.sub(r"[^\w\s-]", "", title.lower())
    slug = re.sub(r"[\s_-]+", "-", slug).strip("-")
    return slug[:60].strip("-")


def fetch_all_posts(site: str) -> list[dict]:
    """Paginate through all published posts using WordPress.com REST API v1.1."""
    base_url = f"https://public-api.wordpress.com/rest/v1.1/sites/{site}/posts"
    posts = []
    offset = 0
    per_page = 100

    print(f"Fetching posts from {site}...")

    while True:
        params = {
            "number": per_page,
            "offset": offset,
            "status": "publish",
            "fields": "ID,title,date,content,slug,excerpt",
        }

        for attempt in range(3):
            try:
                resp = requests.get(base_url, params=params, timeout=30)
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"Failed after 3 attempts at offset {offset}: {e}")
                    raise
                print(f"Retrying (attempt {attempt + 2}/3)...")
                time.sleep(2)

        data = resp.json()
        batch = data.get("posts", [])
        if not batch:
            break

        posts.extend(batch)
        print(f"  Fetched {len(posts)} posts so far...")

        if len(batch) < per_page:
            break

        offset += per_page
        time.sleep(0.5)

    return posts


def save_posts(posts: list[dict], processed_dir: Path, images_dir: Path):
    """Save each post as a .txt file with metadata header, and download its images."""
    text_dir = processed_dir / "blog"
    text_dir.mkdir(parents=True, exist_ok=True)

    for post in posts:
        try:
            dt = date_parser.parse(post["date"])
        except Exception:
            dt = None

        date_str = dt.strftime("%Y-%m-%d") if dt else "unknown"
        datetime_str = dt.isoformat() if dt else ""

        title = unescape(post.get("title", "Untitled").strip())
        raw_content = post.get("content", "")
        post_id = f"{date_str}_{make_slug(title)}"

        # Download images
        image_urls = extract_images(raw_content)
        local_image_paths = []
        if image_urls:
            post_image_dir = images_dir / post_id
            print(f"  Downloading {len(image_urls)} image(s) for: {title[:50]}")
            local_image_paths = download_images(image_urls, post_image_dir)

        content = clean_html(raw_content, local_image_paths)
        text = f"""---
date: {date_str}
datetime: {datetime_str}
source: blog
title: {title}
images: {len(local_image_paths)}
---

{content}
"""
        (text_dir / f"{post_id}.txt").write_text(text, encoding="utf-8")

    print(f"\nSaved {len(posts)} posts to {text_dir}")
    print(f"Images saved to {images_dir}")


def main():
    ap = argparse.ArgumentParser(description="Fetch all posts from a WordPress.com site.")
    ap.add_argument("--site", required=True, help="WordPress.com site domain (e.g. praveenkottayi.com)")
    args = ap.parse_args()

    ensure_dirs()
    posts = fetch_all_posts(args.site)
    print(f"\nTotal posts found: {len(posts)}")

    images_dir = RAW_BLOG_DIR / "images"
    save_posts(posts, PROCESSED_DIR, images_dir)
    print("\nDone. Now run: python scripts/ingest.py")


if __name__ == "__main__":
    main()
