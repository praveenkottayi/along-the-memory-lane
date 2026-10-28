"""
Fetch all posts from a WordPress.com blog via the public REST API.
No login or export needed — works for any public WordPress.com site.

Usage:
    python scripts/fetch_wordpress_api.py                      # new posts only (site from config.BLOG_SITE)
    python scripts/fetch_wordpress_api.py --all                # re-fetch every post (e.g. after editing old ones)
    python scripts/fetch_wordpress_api.py --site example.com

By default only posts published after the newest post already saved in
data/processed/blog/ are fetched — old posts are rarely edited, so there's no
need to download everything each time. With no saved posts it fetches all.

For each post, saves:
  data/processed/blog/<date>_<slug>.txt   ← cleaned text with metadata header
  data/raw/blog/images/<date>_<slug>/     ← all images from that post
"""
import argparse
import sys
import time
from html import unescape
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser

from common import front_matter, html_to_text, make_slug, parse_front_matter
from config import BLOG_SITE, PROCESSED_DIR, RAW_BLOG_DIR, ensure_dirs


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


def latest_saved_post(processed_dir: Path) -> str | None:
    """Return the ISO datetime of the newest blog post already on disk, or None."""
    latest = None
    for f in (processed_dir / "blog").glob("*.txt"):
        metadata, _ = parse_front_matter(f.read_text(encoding="utf-8"))
        dt = metadata.get("datetime")
        if dt and (latest is None or date_parser.parse(dt) > date_parser.parse(latest)):
            latest = dt
    return latest


def fetch_all_posts(site: str, after: str | None = None) -> list[dict]:
    """Paginate through published posts using WordPress.com REST API v1.1.

    `after` (ISO 8601) limits the fetch to posts published after that moment.
    """
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
        if after:
            params["after"] = after

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

        content = html_to_text(raw_content)
        if local_image_paths:
            refs = "\n".join(f"  - {p}" for p in local_image_paths)
            content += f"\n\n[images]\n{refs}"

        text = front_matter(
            {
                "date": date_str,
                "datetime": datetime_str,
                "source": "blog",
                "title": title,
                "images": len(local_image_paths),
            },
            content,
        )
        (text_dir / f"{post_id}.txt").write_text(text, encoding="utf-8")

    print(f"\nSaved {len(posts)} posts to {text_dir}")
    print(f"Images saved to {images_dir}")


def main():
    ap = argparse.ArgumentParser(description="Fetch all posts from a WordPress.com site.")
    ap.add_argument("--site", default=BLOG_SITE, help=f"WordPress.com site domain (default: {BLOG_SITE}, from config.BLOG_SITE)")
    ap.add_argument("--all", action="store_true", help="re-fetch every post, not just new ones")
    args = ap.parse_args()

    ensure_dirs()
    after = None if args.all else latest_saved_post(PROCESSED_DIR)
    if after:
        print(f"Fetching posts published after {after} (use --all to re-fetch everything)")
    posts = fetch_all_posts(args.site, after=after)
    print(f"\n{'New' if after else 'Total'} posts found: {len(posts)}")
    if not posts:
        print("Nothing new — blog is up to date.")
        return

    images_dir = RAW_BLOG_DIR / "images"
    save_posts(posts, PROCESSED_DIR, images_dir)
    print("\nDone. Now run: python scripts/ingest.py --incremental")


if __name__ == "__main__":
    main()
