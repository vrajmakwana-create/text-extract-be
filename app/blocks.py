"""Document layout block detection with DocLayout-YOLO (ONNX).

Detects titles, text paragraphs, figures, tables, and related regions, then
returns them in reading order so multi-column pages are not woven together.
"""

from __future__ import annotations

import ast
import logging
import threading
from dataclasses import dataclass
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from app.layout import reading_order

_log = logging.getLogger(__name__)

_MODEL_REPO = "wybxc/DocLayout-YOLO-DocStructBench-onnx"
_MODEL_FILE = "doclayout_yolo_docstructbench_imgsz1024.onnx"
_IMGSZ = 1024
_CONF = 0.25
_PAD_VALUE = 114

# Regions that carry readable text we should keep.
# Include "abandon" (headers/footers/page labels): the user wants every
# detected region in the output, not a filtered subset.
_TEXT_LABELS = {
    "title",
    "plain text",
    "figure_caption",
    "table",
    "table_caption",
    "table_footnote",
    "isolate_formula",
    "formula_caption",
    "abandon",
}
# Do not drop any layout class during detection.
_SKIP_LABELS: set[str] = set()
# Embedded pictures whose OCR should stay marked as image text.
_IMAGE_LABELS = {"figure"}

_session = None
_names: dict[int, str] = {}
_lock = threading.Lock()


@dataclass(frozen=True)
class LayoutBlock:
    bbox: tuple[float, float, float, float]
    label: str
    score: float

    @property
    def is_text(self) -> bool:
        return self.label in _TEXT_LABELS

    @property
    def is_image(self) -> bool:
        return self.label in _IMAGE_LABELS

    @property
    def is_skippable(self) -> bool:
        return self.label in _SKIP_LABELS


def detect_layout_blocks(image: Image.Image | np.ndarray, conf: float = _CONF) -> list[LayoutBlock]:
    """Detect layout blocks on a page image and return them in reading order."""
    array = _to_rgb_array(image)
    if array is None:
        return []

    session, names = _get_session()
    if session is None:
        return []

    tensor, meta = _letterbox(array, _IMGSZ)
    try:
        with _lock:
            preds = session.run(None, {session.get_inputs()[0].name: tensor})[0]
    except Exception:
        _log.exception("Layout model inference failed")
        return []

    preds = np.asarray(preds)
    if preds.ndim == 3:
        preds = preds[0]

    # YOLOv10 export: (N, 6) = x1,y1,x2,y2,conf,cls  or (6, N)
    if preds.ndim != 2:
        return []
    if preds.shape[0] == 6 and preds.shape[1] != 6:
        preds = preds.T
    if preds.shape[1] < 6:
        return []

    height, width = array.shape[:2]
    gain, pad_x, pad_y = meta
    blocks: list[LayoutBlock] = []
    for row in preds:
        score = float(row[4])
        if score < conf:
            continue
        cls_id = int(row[5])
        label = names.get(cls_id, str(cls_id))
        if label in _SKIP_LABELS:
            continue

        x1 = (float(row[0]) - pad_x) / gain
        y1 = (float(row[1]) - pad_y) / gain
        x2 = (float(row[2]) - pad_x) / gain
        y2 = (float(row[3]) - pad_y) / gain
        x1 = float(np.clip(x1, 0, width - 1))
        y1 = float(np.clip(y1, 0, height - 1))
        x2 = float(np.clip(x2, 0, width - 1))
        y2 = float(np.clip(y2, 0, height - 1))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        blocks.append(LayoutBlock((x1, y1, x2, y2), label, score))

    blocks = _nms(blocks, iou_threshold=0.45)
    if not blocks:
        return []

    ordered = reading_order([(block.bbox, block) for block in blocks], float(width))
    return [block for _, block in ordered]


def _get_session():
    global _session, _names
    if _session is not None:
        return _session, _names

    with _lock:
        if _session is not None:
            return _session, _names
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=_MODEL_REPO, filename=_MODEL_FILE)
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            meta = session.get_modelmeta().custom_metadata_map
            names = ast.literal_eval(meta.get("names", "{}"))
            _names = {int(key): str(value) for key, value in names.items()}
            _session = session
        except Exception:
            _log.exception("Could not load DocLayout-YOLO model")
            return None, {}
    return _session, _names


def _to_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray | None:
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        array = np.asarray(image)
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        elif array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    if array.size == 0 or array.shape[0] < 32 or array.shape[1] < 32:
        return None
    return array


def _letterbox(image: np.ndarray, imgsz: int):
    """Resize with unchanged aspect ratio and pad to a square canvas.

    Matches the DocLayout-YOLO ONNX export: BGR pixels, pad value 114, 0..1.
    """
    height, width = image.shape[:2]
    gain = min(imgsz / height, imgsz / width)
    new_w = int(round(width * gain))
    new_h = int(round(height * gain))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((imgsz, imgsz, 3), _PAD_VALUE, dtype=np.uint8)
    pad_x = (imgsz - new_w) // 2
    pad_y = (imgsz - new_h) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    tensor = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = np.expand_dims(tensor, axis=0)
    return tensor, (gain, pad_x, pad_y)


def _nms(blocks: list[LayoutBlock], iou_threshold: float) -> list[LayoutBlock]:
    if len(blocks) < 2:
        return blocks
    ordered = sorted(blocks, key=lambda item: item.score, reverse=True)
    kept: list[LayoutBlock] = []
    for block in ordered:
        if any(_iou(block.bbox, other.bbox) >= iou_threshold for other in kept):
            continue
        kept.append(block)
    return kept


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
