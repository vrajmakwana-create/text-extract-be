import io
import logging
import threading

from PIL import Image, ImageOps

from app.layout import find_image_gutter, join_lines, reading_order

_engine = None
_lock = threading.Lock()

MIN_IMAGE_PX = 16
MAX_IMAGE_PX = 2000
MIN_SCORE = 0.5
# Thin heading / caption strips are often too small for the detector. Upscale
# them before OCR so titles are not dropped.
MIN_OCR_HEIGHT = 64
MIN_OCR_WIDTH = 240
MAX_UPSCALE = 4.0


class OCRUnavailable(Exception):
    """The OCR engine could not be loaded."""


def _get_engine():
    global _engine
    if _engine is None:
        try:
            from rapidocr import RapidOCR

            logging.getLogger("RapidOCR").setLevel(logging.ERROR)
            for handler in logging.getLogger("RapidOCR").handlers:
                handler.setLevel(logging.ERROR)
            _engine = RapidOCR()
        except Exception as exc:
            raise OCRUnavailable("OCR engine could not start") from exc
    return _engine


def _open_image(data: bytes) -> Image.Image | None:
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        image.load()
    except Exception:
        return None

    if image.mode != "RGB":
        image = image.convert("RGB")

    if image.width < MIN_IMAGE_PX or image.height < MIN_IMAGE_PX:
        return None

    if max(image.size) > MAX_IMAGE_PX:
        image.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX), Image.Resampling.LANCZOS)
    return image


def _run_engine(image: Image.Image):
    engine = _get_engine()
    try:
        with _lock:
            return engine(image)
    except Exception:
        return None


def _boxes_from_result(result) -> list[tuple[tuple[float, float, float, float], str]]:
    boxes = getattr(result, "boxes", None)
    scores = getattr(result, "scores", None) or []
    items: list[tuple[tuple[float, float, float, float], str]] = []

    for index, text in enumerate(result.txts):
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        score = float(scores[index]) if index < len(scores) else 1.0
        if score < MIN_SCORE:
            continue
        if boxes is not None and index < len(boxes):
            box = boxes[index]
            x0 = float(min(point[0] for point in box))
            y0 = float(min(point[1] for point in box))
            x1 = float(max(point[0] for point in box))
            y1 = float(max(point[1] for point in box))
        else:
            x0, y0, x1, y1 = 0.0, float(index), 1.0, float(index) + 1.0
        items.append(((x0, y0, x1, y1), cleaned))
    return items


def _text_from_result(result) -> str:
    if result is None or not getattr(result, "txts", None):
        return ""
    items = _boxes_from_result(result)
    if not items:
        return ""
    page_width = max(bbox[2] for bbox, _ in items) - min(bbox[0] for bbox, _ in items)
    ordered = reading_order(items, max(page_width, 1.0))
    return join_lines(ordered)


def _upscale_for_ocr(image: Image.Image) -> Image.Image:
    """Enlarge small crops (especially headings) so OCR can read them."""
    width, height = image.size
    scale = 1.0
    if height < MIN_OCR_HEIGHT:
        scale = max(scale, MIN_OCR_HEIGHT / max(height, 1))
    if width < MIN_OCR_WIDTH and height < MIN_OCR_HEIGHT * 1.5:
        scale = max(scale, MIN_OCR_WIDTH / max(width, 1))
    scale = min(scale, MAX_UPSCALE)
    if scale <= 1.05:
        return image
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def ocr_region(image: Image.Image) -> str:
    """OCR a single region that is already one column / one layout block."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.width < MIN_IMAGE_PX or image.height < MIN_IMAGE_PX:
        return ""
    work = _upscale_for_ocr(image)
    if max(work.size) > MAX_IMAGE_PX:
        work = work.copy()
        work.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX), Image.Resampling.LANCZOS)
    text = _text_from_result(_run_engine(work))
    if text:
        return text
    # Second pass with a stronger upscale for stubborn thin headings.
    if work is image or max(image.size) < 400:
        stronger = image.resize(
            (max(1, image.width * 3), max(1, image.height * 3)),
            Image.Resampling.LANCZOS,
        )
        if max(stronger.size) > MAX_IMAGE_PX:
            stronger.thumbnail((MAX_IMAGE_PX, MAX_IMAGE_PX), Image.Resampling.LANCZOS)
        return _text_from_result(_run_engine(stronger))
    return ""


def ocr_image(data: bytes) -> str:
    """Read text from image bytes. Returns an empty string when none is found.

    Prefers layout-block OCR when the caller crops regions. For raw full-page
    images, split on a detected gutter so two-column pages are not woven.
    """
    image = _open_image(data)
    if image is None:
        return ""

    gutter = find_image_gutter(image)
    if gutter is None:
        return ocr_region(image)

    # Keep a little overlap so characters at the cut are not clipped.
    pad = max(4, int(image.width * 0.01))
    left = image.crop((0, 0, min(image.width, gutter + pad), image.height))
    right = image.crop((max(0, gutter - pad), 0, image.width, image.height))

    parts = []
    left_text = ocr_region(left)
    if left_text:
        parts.append(left_text)
    right_text = ocr_region(right)
    if right_text:
        parts.append(right_text)
    if parts:
        return "\n".join(parts)

    # Gutter looked real but crops failed; fall back to the whole page.
    return ocr_region(image)
