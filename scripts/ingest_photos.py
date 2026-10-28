"""
Photo ingestion pipeline — Phase 3.

Converts a folder of personal photos into searchable memories alongside
journals and blogs. Fully local: no API calls, no data leaves the machine.

How it works:
  1. Extract the date from EXIF metadata (falls back to filename then folder name).
  2. Run the local vision model (PHOTO_VISION_MODEL, via Ollama) to describe the photo.
  3. Cache the description in data/processed/photos/<album>/descriptions/,
     tagged with the model that wrote it.
  4. Write a standard front-matter .txt to data/processed/photos/<album>/
  5. Run `python scripts/ingest.py --incremental` to embed and store in ChromaDB.

Because descriptions are embedded with the same nomic-embed-text model used for
journals and blogs, a single query retrieves memories from all sources at once.

Usage:
    # Describe and process a single album
    python scripts/ingest_photos.py --album data/raw/photos/2015-goa

    # Process all albums under data/raw/photos/
    python scripts/ingest_photos.py --all

    # Preview what would be written without touching files (no Ollama calls)
    python scripts/ingest_photos.py --album data/raw/photos/2015-goa --dry-run

    # Show cached descriptions (and which model wrote each) without re-running vision
    python scripts/ingest_photos.py --album data/raw/photos/2015-goa --inspect

    # Use a different vision model for this run (new photos only)
    python scripts/ingest_photos.py --all --model moondream

    # Re-describe photos whose description came from a different model
    python scripts/ingest_photos.py --all --model gemma4:12b --refresh

Prerequisites:
    ollama pull gemma4:12b        # once, to download the default model (config.PHOTO_VISION_MODEL)
    pip install pillow-heif       # for HEIC support (iPhone photos)
"""
import argparse
import base64
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from PIL.ExifTags import TAGS

from common import front_matter
from config import (
    PHOTO_DESCRIPTION_PROMPT,
    PHOTO_MAX_SIDE,
    PHOTO_VISION_MODEL,
    PHOTO_VISION_OPTIONS,
    PROCESSED_DIR,
    RAW_PHOTOS_DIR,
    ensure_dirs,
)

# ---------------------------------------------------------------------------
# HEIC support — register pillow-heif if available
# ---------------------------------------------------------------------------

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIC_SUPPORTED = True
except ImportError:
    _HEIC_SUPPORTED = False

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# EXIF date extraction
# ---------------------------------------------------------------------------

_EXIF_DATE_TAGS = {"DateTimeOriginal", "DateTimeDigitized", "DateTime"}


def get_exif_date(image_path: Path) -> str | None:
    """Extract a YYYY-MM-DD date from EXIF metadata.

    Tries DateTimeOriginal first (when the shutter fired), then
    DateTimeDigitized, then the generic DateTime tag. EXIF dates are
    formatted as "YYYY:MM:DD HH:MM:SS".
    """
    try:
        img = Image.open(image_path)
        exif_raw = img.getexif()
        if not exif_raw:
            return None
        tagged = {TAGS.get(k, k): v for k, v in exif_raw.items()}
        for tag in _EXIF_DATE_TAGS:
            val = tagged.get(tag)
            if val and isinstance(val, str) and len(val) >= 10:
                # "2015:06:12 14:30:00" → "2015-06-12"
                return val[:10].replace(":", "-")
    except Exception:
        pass
    return None


def get_date_from_filename(image_path: Path) -> str | None:
    """Try to parse a date from the filename (e.g. IMG_20150612_143022.jpg)."""
    import re
    m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", image_path.stem)
    if m:
        y, mo, d = m.group(1), m.group(2), m.group(3)
        try:
            datetime(int(y), int(mo), int(d))
            return f"{y}-{mo}-{d}"
        except ValueError:
            pass
    return None


def resolve_date(image_path: Path) -> str:
    """Return the best available date for a photo, falling back gracefully."""
    return (
        get_exif_date(image_path)
        or get_date_from_filename(image_path)
        or "unknown"
    )


# ---------------------------------------------------------------------------
# Vision description via Ollama
# ---------------------------------------------------------------------------

def load_image_bytes(image_path: Path) -> bytes | None:
    """Read image bytes, converting motion photos (.MP.jpg) to a clean JPEG first.

    Pixel motion photos embed a short video after the JPEG data — most vision
    models choke on the extra bytes. We re-encode via Pillow to strip the video
    and return a clean JPEG buffer, shrunk to PHOTO_MAX_SIDE on the longest side
    (full camera resolution only slows the model down).
    """
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")  # ensure no alpha / palette issues
        img.thumbnail((PHOTO_MAX_SIDE, PHOTO_MAX_SIDE))
        import io
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()
    except Exception as e:
        print(f"    Warning: could not open image {image_path.name}: {e}")
        return None


def sidecar_path(image_path: Path, album_dir: Path) -> Path:
    """Return the description cache path under data/processed/photos/<album>/descriptions/.

    Keeping sidecars in processed/ (not next to the raw image) avoids
    cluttering the photo folder and makes it easy to see which model
    generated each description.
    """
    desc_dir = PROCESSED_DIR / "photos" / album_dir.name / "descriptions"
    desc_dir.mkdir(parents=True, exist_ok=True)
    return desc_dir / f"{image_path.stem}.description.txt"


def read_cache(cache: Path) -> tuple[str, str]:
    """Return (model tag, description) from a cache file, or ("", "") if unusable."""
    if not cache.exists():
        return "", ""
    lines = cache.read_text(encoding="utf-8").splitlines()
    model_tag = lines[0].strip() if lines else ""
    body = "\n".join(lines[2:]).strip() if len(lines) > 2 else ""
    return model_tag, body


def describe_photo(image_path: Path, album_dir: Path, model: str = PHOTO_VISION_MODEL,
                   refresh: bool = False) -> str:
    """Run the vision model on a photo and return a text description.

    Results are cached as sidecar files under
    data/processed/photos/<album>/descriptions/<stem>.description.txt.

    File format (first line = model tag, blank line, then description):
        gemma4:12b
        <blank>
        A group of friends are gathered around a table...

    This lets you see at a glance which model produced each description,
    and makes it safe to swap models and selectively re-generate.

    Empty results are never cached — the next run will retry automatically.
    A cached description from another model is reused unless `refresh` is set.
    """
    cache = sidecar_path(image_path, album_dir)
    model_tag, body = read_cache(cache)
    if body and (model_tag == model or not refresh):
        return body

    try:
        import ollama
    except ImportError:
        raise ImportError(
            "The 'ollama' package is required. Run: pip install ollama"
        )

    image_bytes = load_image_bytes(image_path)
    if not image_bytes:
        return ""

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        response = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": PHOTO_DESCRIPTION_PROMPT,
                "images": [image_b64],
            }],
            options=PHOTO_VISION_OPTIONS,
        )
        description = response["message"]["content"].strip()
    except Exception as e:
        print(f"    Warning: vision model error for {image_path.name}: {e}")
        return ""

    # Only cache if the model returned something useful
    if description:
        cache.write_text(f"{model}\n\n{description}", encoding="utf-8")

    return description


# ---------------------------------------------------------------------------
# Front-matter writer
# ---------------------------------------------------------------------------

def build_processed_txt(
    image_path: Path,
    album_name: str,
    date: str,
    description: str,
) -> str:
    """Build a front-matter .txt for a photo, ready to be ingested."""
    return front_matter(
        {
            "date": date,
            "title": f"Photo — {date}",
            "source": "photo",
            "album": album_name,
            "image_path": str(image_path.resolve()),
        },
        description,
    )


# ---------------------------------------------------------------------------
# Album processing
# ---------------------------------------------------------------------------

def get_album_images(album_dir: Path) -> list[Path]:
    """Return all supported image files in an album folder, sorted."""
    return sorted([
        f for f in album_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ])


def process_album(album_dir: Path, output_dir: Path, dry_run: bool = False,
                  model: str = PHOTO_VISION_MODEL, refresh: bool = False) -> int:
    """Describe all photos in an album and write processed .txt files.

    Returns the number of photos processed.
    """
    if not _HEIC_SUPPORTED:
        heic_files = [f for f in get_album_images(album_dir) if f.suffix.lower() in (".heic", ".heif")]
        if heic_files:
            print("  Warning: HEIC files found but pillow-heif is not installed.")
            print("  Run: pip install pillow-heif   then retry.")

    album_name = album_dir.name
    images = get_album_images(album_dir)

    if not images:
        print(f"  No images found in {album_dir}")
        return 0

    print(f"\nProcessing album: {album_name} ({len(images)} images) — model: {model}"
          f"{' (refreshing other models)' if refresh else ''}")

    if not dry_run:
        out_dir = output_dir / "photos" / album_name
        out_dir.mkdir(parents=True, exist_ok=True)

    date_counts: dict[str, int] = {}
    saved = 0

    for i, img_path in enumerate(images):
        print(f"  [{i+1}/{len(images)}] {img_path.name}", end="")

        date = resolve_date(img_path)

        if dry_run:
            model_tag, body = read_cache(sidecar_path(img_path, album_dir))
            if not body:
                status = "(needs vision)"
            elif model_tag == model or not refresh:
                status = f"(cached: {model_tag})"
            else:
                status = f"(re-describe: was {model_tag})"
            print(f"  date={date}  {status}")
            continue

        cache = sidecar_path(img_path, album_dir)
        before = read_cache(cache)
        description = describe_photo(img_path, album_dir, model, refresh)

        if not description:
            print(f"  → skipped (no description)")
            continue

        # Unique filename for same-date photos
        date_counts[date] = date_counts.get(date, 0) + 1
        count = date_counts[date]
        filename = f"{date}.txt" if count == 1 else f"{date}_{count:02d}.txt"

        txt = build_processed_txt(img_path, album_name, date, description)
        (out_dir / filename).write_text(txt, encoding="utf-8")

        sidecar_status = "→ cached" if read_cache(cache) == before and before[1] else "→ described"
        print(f"  {sidecar_status}  [{date}]")
        saved += 1

    return saved


def inspect_album(album_dir: Path):
    """Print cached descriptions without calling the vision model."""
    images = get_album_images(album_dir)
    print(f"\nInspecting: {album_dir.name} ({len(images)} images)")

    for img_path in images:
        cache = sidecar_path(img_path, album_dir)
        date = resolve_date(img_path)

        if not cache.exists():
            print(f"  {img_path.name}  [{date}]  [no description yet]")
            continue

        lines = cache.read_text(encoding="utf-8").splitlines()
        model_tag = lines[0] if lines else "unknown model"
        body = "\n".join(lines[2:]).strip() if len(lines) > 2 else ""
        preview = body[:100].replace("\n", " ")
        print(f"  {img_path.name}  [{date}]  [model: {model_tag}]")
        print(f"    {preview}{'…' if len(body) > 100 else ''}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Describe photos with a local vision model and prepare them for ChromaDB ingestion."
    )
    source = ap.add_mutually_exclusive_group()
    source.add_argument(
        "--album",
        help="Path to a single album folder (e.g. data/raw/photos/2015-goa)"
    )
    source.add_argument(
        "--all",
        action="store_true",
        help="Process all album folders under data/raw/photos/",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would happen without calling the vision model or writing files",
    )
    ap.add_argument(
        "--inspect",
        action="store_true",
        help="Print cached descriptions without running the vision model",
    )
    ap.add_argument(
        "--model",
        default=PHOTO_VISION_MODEL,
        help=f"Ollama vision model to use (default: {PHOTO_VISION_MODEL}, from config.PHOTO_VISION_MODEL)",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="Re-describe photos whose cached description came from a different model",
    )
    args = ap.parse_args()

    ensure_dirs()

    if args.album:
        album_dir = Path(args.album)
        if not album_dir.exists():
            print(f"Album folder not found: {album_dir}")
            sys.exit(1)
        if args.inspect:
            inspect_album(album_dir)
        else:
            saved = process_album(album_dir, PROCESSED_DIR, dry_run=args.dry_run,
                                  model=args.model, refresh=args.refresh)
            if not args.dry_run:
                print(f"\n  Done. {saved} photo(s) written.")
                print("  Next: python scripts/ingest.py --incremental")

    elif args.all or not args.album:
        album_dirs = sorted([
            d for d in RAW_PHOTOS_DIR.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        if not album_dirs:
            print(f"No album folders found under {RAW_PHOTOS_DIR}")
            print(f"Create a folder like: data/raw/photos/2015-goa/  and drop images in it.")
            sys.exit(0)

        print(f"Found {len(album_dirs)} album(s)")
        total = 0
        for album_dir in album_dirs:
            if args.inspect:
                inspect_album(album_dir)
            else:
                total += process_album(album_dir, PROCESSED_DIR, dry_run=args.dry_run,
                                       model=args.model, refresh=args.refresh)

        if not args.dry_run and not args.inspect:
            print(f"\nTotal: {total} photo(s) processed.")
            print("Next: python scripts/ingest.py --incremental")

    if args.dry_run:
        print("\n[dry-run] No files were written and no vision model was called.")


if __name__ == "__main__":
    main()
