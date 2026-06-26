"""
Convert a scanned journal PDF into individual JPEG images.

Usage:
    python scripts/pdf_to_images.py --pdf data/raw/journal/#16_01_08_2025_15_01_2026.pdf

Parses the filename for journal number and date range, creates the correct
folder structure, and extracts each page as a high-resolution JPEG.

Expected filename format: #<number>_<DD>_<MM>_<YYYY>_<DD>_<MM>_<YYYY>.pdf
Output folder:            data/raw/journal/<number>_<YYYY-MM>_<YYYY-MM>/

After running this, use:
    python scripts/ocr_journals.py --journal data/raw/journal/<folder>
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_JOURNAL_DIR, ensure_dirs


def parse_journal_filename(pdf_path: Path) -> dict:
    """
    Parse journal metadata from filename.
    Expected: #16_01_08_2025_15_01_2026.pdf
    Returns: {number, start_year, start_month, end_year, end_month}
    """
    name = pdf_path.stem  # e.g. #16_01_08_2025_15_01_2026

    m = re.match(
        r"#?(\d+)_(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})_(\d{4})",
        name
    )
    if not m:
        print(f"Warning: could not parse date range from filename '{name}'.")
        print("Expected format: #16_01_08_2025_15_01_2026.pdf")
        # Fall back to just the journal number
        num_match = re.match(r"#?(\d+)", name)
        number = num_match.group(1).zfill(2) if num_match else "00"
        return {
            "number": number,
            "folder_name": f"{number}_unknown",
        }

    number = m.group(1).zfill(2)
    start_day, start_month, start_year = m.group(2), m.group(3), m.group(4)
    end_day, end_month, end_year = m.group(5), m.group(6), m.group(7)

    folder_name = f"{number}_{start_year}-{start_month}_{end_year}-{end_month}"

    return {
        "number": number,
        "start": f"{start_year}-{start_month}-{start_day}",
        "end": f"{end_year}-{end_month}-{end_day}",
        "folder_name": folder_name,
    }


def pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 200) -> int:
    """
    Extract each PDF page as a JPEG into output_dir.
    Returns number of pages extracted.
    DPI 200 is a good balance of quality vs file size for handwriting OCR.
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        print("Error: pymupdf not installed.")
        print("Run: uv pip install pymupdf")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    total = len(doc)
    print(f"Extracting {total} pages at {dpi} DPI...")

    zoom = dpi / 72  # PDF default is 72 DPI
    mat = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(doc):
        page_num = str(i + 1).zfill(3)
        dest = output_dir / f"page_{page_num}.jpg"

        if dest.exists():
            print(f"  Skipping page {page_num} (already extracted)")
            continue

        pix = page.get_pixmap(matrix=mat)
        pix.save(str(dest))

        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"  {i + 1}/{total} pages extracted")

    doc.close()
    return total


def main():
    ap = argparse.ArgumentParser(
        description="Convert a scanned journal PDF to images for OCR."
    )
    ap.add_argument("--pdf", required=True, help="Path to the journal PDF file")
    ap.add_argument("--dpi", type=int, default=200,
                    help="Resolution for image extraction (default: 200)")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    ensure_dirs()

    # Parse journal info from filename
    info = parse_journal_filename(pdf_path)
    print(f"Journal #{info['number']}")
    if "start" in info:
        print(f"Period: {info['start']} → {info['end']}")

    # Create output folder
    output_dir = RAW_JOURNAL_DIR / info["folder_name"]
    print(f"Output folder: {output_dir}")

    # Extract pages
    total = pdf_to_images(pdf_path, output_dir, dpi=args.dpi)

    print(f"\nDone. {total} pages saved to {output_dir}")
    print(f"\nNext step:")
    print(f"  python scripts/ocr_journals.py --journal {output_dir}")


if __name__ == "__main__":
    main()
