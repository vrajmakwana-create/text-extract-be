"""Local storage for detected layout blocks and annotated page images."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Project root / detected
DETECTED_ROOT = Path(__file__).resolve().parent.parent / "detected"

_LABEL_COLORS = {
    "title": (37, 99, 235),
    "plain text": (22, 163, 74),
    "figure": (234, 88, 12),
    "figure_caption": (249, 115, 22),
    "table": (147, 51, 234),
    "table_caption": (168, 85, 247),
    "table_footnote": (192, 132, 252),
    "isolate_formula": (8, 145, 178),
    "formula_caption": (6, 182, 212),
}


def new_detection_run(filename: str = "document") -> Path:
    """Create a timestamped folder for one extraction run."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _safe_name(Path(filename).stem) or "document"
    folder = DETECTED_ROOT / f"{stamp}_{stem}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_page_detections(
    run_dir: Path,
    page_number: int,
    page_image: Image.Image,
    blocks: list,
) -> list[dict]:
    """Save each detected crop and an annotated overview for one page.

    Returns metadata dicts with relative paths under the project root.
    """
    page_dir = run_dir / f"page_{page_number:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)

    annotated = page_image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    saved: list[dict] = []
    for index, block in enumerate(blocks, start=1):
        crop = _crop(page_image, block.bbox)
        if crop is None:
            continue

        label = _safe_name(block.label) or "block"
        crop_name = f"block_{index:02d}_{label}.png"
        crop_path = page_dir / crop_name
        crop.save(crop_path, format="PNG")

        color = _LABEL_COLORS.get(block.label, (220, 38, 38))
        x0, y0, x1, y1 = [int(value) for value in block.bbox]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        caption = f"{index}. {block.label} ({block.score:.2f})"
        draw.rectangle((x0, max(0, y0 - 20), x0 + 8 * len(caption), y0), fill=color)
        draw.text((x0 + 4, max(0, y0 - 18)), caption, fill=(255, 255, 255), font=font)

        saved.append(
            {
                "page": page_number,
                "index": index,
                "label": block.label,
                "score": round(float(block.score), 4),
                "path": _relative(crop_path),
                "bbox": [round(float(value), 1) for value in block.bbox],
            }
        )

    overview_path = page_dir / "annotated.png"
    annotated.save(overview_path, format="PNG")
    saved.insert(
        0,
        {
            "page": page_number,
            "index": 0,
            "label": "annotated",
            "score": 1.0,
            "path": _relative(overview_path),
            "bbox": [0, 0, page_image.width, page_image.height],
        },
    )
    return saved


def _crop(image: Image.Image, bbox) -> Image.Image | None:
    x0, y0, x1, y1 = bbox
    left = max(0, int(x0) - 2)
    top = max(0, int(y0) - 2)
    right = min(image.width, int(x1) + 2)
    bottom = min(image.height, int(y1) + 2)
    if right - left < 8 or bottom - top < 8:
        return None
    return image.crop((left, top, right, bottom))


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", (value or "").strip().lower())
    return cleaned.strip("._-")[:60]


def _relative(path: Path) -> str:
    root = DETECTED_ROOT.parent
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)
