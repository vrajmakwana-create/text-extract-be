import io
import logging
import os
import threading


def _use_cuda() -> bool:
    """OCR_DEVICE=cuda runs OCR on the GPU (needs onnxruntime-gpu)."""
    if os.environ.get("OCR_DEVICE", "cpu").lower() != "cuda":
        return False
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()  # load the CUDA/cuDNN libraries installed by pip
    return "CUDAExecutionProvider" in ort.get_available_providers()

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
            params = None
            if _use_cuda():
                params = {
                    "EngineConfig.onnxruntime.use_cuda": True,
                    # Crops vary in size; EXHAUSTIVE would re-benchmark every new shape
                    "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": "DEFAULT",
                }
            _engine = RapidOCR(params=params)
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


_ocria = None


def _ocr_engine_name() -> str:
    return os.environ.get("OCR_ENGINE", "ocria").lower()


def _ocria_text(image: Image.Image) -> str:
    """Read one region with LlamaIndex OCRia-nano-1 through LiteParse."""
    global _ocria
    try:
        if _ocria is None:
            from liteparse import LiteParse, models

            models.pull("ocria-nano")  # no-op once the models are downloaded
            _ocria = LiteParse(ocr_backend="ocria-nano", output_format="json", quiet=True)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        with _lock:
            result = _ocria.parse(buffer.getvalue())
    except Exception:
        logging.getLogger(__name__).exception("OCRia-nano OCR failed")
        return ""
    # LiteParse reports positions in PDF points; convert to image pixels so the
    # same reading_order/join_lines tolerances as RapidOCR apply.
    items: list[tuple[tuple[float, float, float, float], str]] = []
    for page in result.pages or []:
        scale = image.width / page.width if page.width else 1.0
        for item in page.text_items or []:
            text = (item.text or "").strip()
            confidence = item.confidence if item.confidence is not None else 1.0
            if not text or confidence < MIN_SCORE:
                continue
            x0, y0 = item.x * scale, item.y * scale
            items.append(((x0, y0, x0 + item.width * scale, y0 + item.height * scale), text))
    if not items:
        return ""
    page_width = max(b[2] for b, _ in items) - min(b[0] for b, _ in items)
    return join_lines(reading_order(items, max(page_width, 1.0)))


_trocr = None
TROCR_MODEL = os.environ.get("TROCR_MODEL", "microsoft/trocr-small-handwritten")
TROCR_BATCH = 32
LINE_PAD = 4


def _get_trocr():
    """Load TrOCR once. It reads single text lines; RapidOCR finds the lines."""
    global _trocr
    if _trocr is None:
        # Generated lines vary in length; without this the CUDA cache fragments to several GB.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            device = "cuda" if os.environ.get("OCR_DEVICE", "cpu").lower() == "cuda" and torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            processor = TrOCRProcessor.from_pretrained(TROCR_MODEL)
            model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL, dtype=dtype).to(device).eval()
            _trocr = (processor, model, device, dtype)
        except Exception as exc:
            raise OCRUnavailable("TrOCR could not start") from exc
    return _trocr


def _line_boxes(image: Image.Image) -> list[tuple[float, float, float, float]]:
    engine = _get_engine()
    try:
        with _lock:
            result = engine(image, use_cls=False, use_rec=False)
    except Exception:
        logging.getLogger(__name__).exception("Line detection failed")
        return []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    lines = []
    for box in boxes:
        x0 = max(0.0, float(min(pt[0] for pt in box)) - LINE_PAD)
        y0 = max(0.0, float(min(pt[1] for pt in box)) - LINE_PAD)
        x1 = min(float(image.width), float(max(pt[0] for pt in box)) + LINE_PAD)
        y1 = min(float(image.height), float(max(pt[1] for pt in box)) + LINE_PAD)
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            lines.append((x0, y0, x1, y1))
    return lines


def _trocr_text(image: Image.Image) -> str:
    """Find text lines with RapidOCR detection, then read each line with TrOCR."""
    lines = _line_boxes(image)
    if not lines:
        return ""
    try:
        import torch

        processor, model, device, dtype = _get_trocr()
        texts: list[str] = []
        for start in range(0, len(lines), TROCR_BATCH):
            crops = [image.crop(box) for box in lines[start : start + TROCR_BATCH]]
            pixels = processor(images=crops, return_tensors="pt").pixel_values.to(device, dtype)
            with _lock, torch.inference_mode():
                ids = model.generate(pixels, max_new_tokens=64, no_repeat_ngram_size=3)  # no_repeat stops runaway repeats
            texts.extend(processor.batch_decode(ids, skip_special_tokens=True))
    except OCRUnavailable:
        raise
    except Exception:
        logging.getLogger(__name__).exception("TrOCR OCR failed")
        return ""
    items = [(box, text.strip()) for box, text in zip(lines, texts) if text.strip()]
    if not items:
        return ""
    page_width = max(b[2] for b, _ in items) - min(b[0] for b, _ in items)
    return join_lines(reading_order(items, max(page_width, 1.0)))


def _read(image: Image.Image) -> str:
    engine = _ocr_engine_name()
    if engine == "rapidocr":
        return _text_from_result(_run_engine(image))
    if engine == "trocr":
        return _trocr_text(image)
    return _ocria_text(image)


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
    text = _read(work)
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
        return _read(stronger)
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
