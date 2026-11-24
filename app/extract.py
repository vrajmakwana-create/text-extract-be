import hashlib
import io
from dataclasses import dataclass, field

import pymupdf
from docx import Document
from docx.document import Document as WordDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image, ImageOps

from app.blocks import detect_layout_blocks
from app.layout import reading_order, word_line_chunks
from app.ocr import ocr_image, ocr_region
from app.storage import new_detection_run, save_page_detections

# An image is treated as the page itself when it covers at least this share of
# the page area, or at least this share of both the page width and height.
# Soft margins around a scan often leave the area under 0.7, so both checks
# are needed.
FULL_PAGE_AREA = 0.6
FULL_PAGE_SIDE = 0.8
# A full-page image on a page with at least this much real text is a background
# or a searchable scan, so only the text layer is used.
FULL_PAGE_TEXT_LAYER_CHARS = 80

_COMMON_IMAGE_TYPES = {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"}
_BLIP = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
_IMAGEDATA = "{urn:schemas-microsoft-com:vml}imagedata"
_EMBED = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
_REL_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"


class DocumentError(Exception):
    """The upload could not be read as a document."""


@dataclass
class Piece:
    text: str
    source: str = "text"
    page: int | None = None


@dataclass
class Stats:
    images_detected: int = 0
    images_with_text: int = 0
    scanned_pages: int = 0
    detection_dir: str | None = None
    detected_images: list[dict] = field(default_factory=list)


@dataclass
class Extracted:
    text: str
    segments: list[dict] = field(default_factory=list)
    highlights: list[dict] = field(default_factory=list)
    images_detected: int = 0
    images_with_text: int = 0
    scanned_pages: int = 0
    detection_dir: str | None = None
    detected_images: list[dict] = field(default_factory=list)


def extract_document(filename: str, data: bytes) -> Extracted:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension == "pdf":
        return extract_pdf(data, filename=filename)
    if extension == "docx":
        return extract_docx(data)
    if extension in _COMMON_IMAGE_TYPES:
        return extract_image(data, filename=filename)
    raise DocumentError("Upload a .pdf, .docx, or image file")


def extract_image(data: bytes, filename: str = "image.png") -> Extracted:
    """Extract text from a standalone image (photo, scan, or screenshot)."""
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        image.load()
    except Exception as exc:
        raise DocumentError("Could not open this image") from exc

    if image.mode != "RGB":
        image = image.convert("RGB")
    if image.width < 16 or image.height < 16:
        raise DocumentError("Image is too small to read")

    cache: dict[str, str] = {}
    stats = Stats()
    run_dir = new_detection_run(filename)
    stats.detection_dir = str(run_dir.relative_to(run_dir.parent.parent)).replace("\\", "/")

    blocks = detect_layout_blocks(image)
    if blocks:
        stats.detected_images.extend(save_page_detections(run_dir, 1, image, blocks))
        pieces = _pieces_from_image_blocks(image, blocks, cache, stats)
        if _has_content(pieces):
            stats.scanned_pages = 1
            return _render(pieces, stats)

    # No usable layout: OCR the whole image.
    recognized = _cached_ocr(data, cache)
    pieces: list[Piece] = []
    if recognized:
        stats.images_detected = 1
        stats.images_with_text = 1
        pieces = [Piece(recognized, "image", 1)]
    return _render(pieces, stats)


def _pieces_from_image_blocks(
    image: Image.Image,
    blocks,
    cache: dict[str, str],
    stats: Stats,
) -> list[Piece]:
    """Walk layout blocks on a standalone image and OCR each region."""
    page_area = float(image.width * image.height) or 1.0
    pieces: list[Piece] = []

    for block in blocks:
        if block.is_image:
            crop = _crop_block(image, block.bbox)
            if crop is None:
                continue
            # A figure covering most of the frame is the page itself.
            if _is_full_page_box(block.bbox, image.width, image.height, page_area):
                recognized = _ocr_crop(crop, cache)
                if recognized:
                    _append_piece(pieces, Piece(recognized, "text", 1))
                continue

            stats.images_detected += 1
            recognized = _ocr_crop(crop, cache)
            if recognized:
                stats.images_with_text += 1
                _append_piece(pieces, Piece(recognized, "image", 1))
            continue

        heading = block.label in {
            "title",
            "figure_caption",
            "table_caption",
            "table_footnote",
            "formula_caption",
        }
        crop = _crop_block(image, block.bbox, pad=8 if heading else 3)
        if crop is None:
            continue
        recognized = _ocr_crop(crop, cache)
        if recognized:
            _append_piece(pieces, Piece(recognized, "text", 1))
        elif heading:
            bigger = crop.resize(
                (max(1, crop.width * 4), max(1, crop.height * 4)),
                Image.Resampling.LANCZOS,
            )
            retry = _ocr_crop(bigger, cache)
            if retry:
                _append_piece(pieces, Piece(retry, "text", 1))

    return pieces


def _is_full_page_box(bbox, width: float, height: float, page_area: float) -> bool:
    box_w = bbox[2] - bbox[0]
    box_h = bbox[3] - bbox[1]
    if (box_w * box_h) / page_area >= FULL_PAGE_AREA:
        return True
    return box_w / max(width, 1.0) >= FULL_PAGE_SIDE and box_h / max(height, 1.0) >= FULL_PAGE_SIDE


def extract_pdf(data: bytes, filename: str = "document.pdf") -> Extracted:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentError("Could not open this PDF") from exc

    if document.needs_pass:
        document.close()
        raise DocumentError("This PDF is password protected")

    cache: dict[str, str] = {}
    stats = Stats()
    run_dir = new_detection_run(filename)
    stats.detection_dir = str(run_dir.relative_to(run_dir.parent.parent)).replace("\\", "/")
    try:
        pages = [
            _extract_pdf_page(document, page, cache, stats, number, run_dir)
            for number, page in enumerate(document, start=1)
        ]
    finally:
        document.close()

    pieces: list[Piece] = []
    for page_pieces in pages:
        if not _has_content(page_pieces):
            continue
        if pieces:
            pieces.append(Piece("\n\n"))
        pieces.extend(page_pieces)
    return _render(pieces, stats)


def extract_docx(data: bytes) -> Extracted:
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentError("Could not open this DOCX") from exc

    cache: dict[str, str] = {}
    stats = Stats()
    pieces: list[Piece] = []
    for block in _iter_blocks(document):
        block_pieces = _block_pieces(block, cache, stats)
        if not _has_content(block_pieces):
            continue
        if pieces:
            pieces.append(Piece("\n"))
        pieces.extend(block_pieces)
    return _render(pieces, stats)


def _extract_pdf_page(
    document,
    page,
    cache: dict[str, str],
    stats: Stats,
    page_number: int,
    run_dir,
) -> list[Piece]:
    """Extract one PDF page by detecting layout blocks, then reading in order.

    1. Render the page and run DocLayout-YOLO to find text/title/figure/table
       regions.
    2. Walk those blocks in column-aware reading order.
    3. Prefer the PDF text layer inside each block; OCR the crop when the
       layer is empty (scans) or the block is a figure.
    4. Save detected block crops and an annotated page image locally.
    """
    page_rect = page.rect
    page_area = abs(page_rect.width * page_rect.height) or 1.0
    words = page.get_text("words")
    text_chars = sum(len(str(word[4])) for word in words)

    rendered, scale = _render_page_image(page)
    blocks = detect_layout_blocks(rendered) if rendered is not None else []

    if blocks and rendered is not None:
        stats.detected_images.extend(
            save_page_detections(run_dir, page_number, rendered, blocks)
        )
        pieces = _pieces_from_layout_blocks(
            page=page,
            page_number=page_number,
            blocks=blocks,
            rendered=rendered,
            scale=scale,
            words=words,
            cache=cache,
            stats=stats,
            text_chars=text_chars,
            page_area=page_area,
        )
        if _has_content(pieces):
            return pieces

    # Fallback when the layout model finds nothing useful.
    return _extract_pdf_page_heuristic(document, page, cache, stats, page_number)


def _render_page_image(page, zoom: float = 2.0):
    """Rasterize a PDF page for layout detection. Returns (PIL image, scale)."""
    try:
        matrix = pymupdf.Matrix(zoom, zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        return image, zoom
    except Exception:
        return None, zoom


def _pieces_from_layout_blocks(
    page,
    page_number: int,
    blocks,
    rendered: Image.Image,
    scale: float,
    words,
    cache: dict[str, str],
    stats: Stats,
    text_chars: int,
    page_area: float,
) -> list[Piece]:
    page_rect = page.rect
    used_words: set[int] = set()
    pieces: list[Piece] = []
    scanned = text_chars < FULL_PAGE_TEXT_LAYER_CHARS

    for block in blocks:
        # Layout boxes are in rendered-pixel space; map back to PDF points.
        page_bbox = (
            page_rect.x0 + block.bbox[0] / scale,
            page_rect.y0 + block.bbox[1] / scale,
            page_rect.x0 + block.bbox[2] / scale,
            page_rect.y0 + block.bbox[3] / scale,
        )

        if block.is_image:
            crop = _crop_block(rendered, block.bbox)
            if crop is None:
                continue
            # A figure that fills most of the page on a text-heavy document is
            # a background, not an embedded picture.
            if _is_full_page(page_bbox, page_rect, page_area):
                if text_chars >= FULL_PAGE_TEXT_LAYER_CHARS:
                    continue
                stats.scanned_pages += 1
                scanned = True
                recognized = _ocr_crop(crop, cache)
                if recognized:
                    _append_piece(pieces, Piece(recognized, "text", page_number))
                continue

            stats.images_detected += 1
            recognized = _ocr_crop(crop, cache)
            if recognized:
                stats.images_with_text += 1
                _append_piece(pieces, Piece(recognized, "image", page_number))
            continue

        if not block.is_text:
            # Unknown label: still try OCR so nothing detected is thrown away.
            crop = _crop_block(rendered, block.bbox, pad=4)
            recognized = _ocr_crop(crop, cache) if crop is not None else ""
            if recognized:
                _append_piece(pieces, Piece(recognized, "text", page_number))
            continue

        # Titles/captions are thin strips — use a wider pad and always OCR.
        heading = block.label in {
            "title",
            "figure_caption",
            "table_caption",
            "table_footnote",
            "formula_caption",
        }
        crop = _crop_block(rendered, block.bbox, pad=8 if heading else 3)
        ocr_text = _ocr_crop(crop, cache) if crop is not None else ""

        block_words = _words_in_bbox(words, page_bbox, used_words)
        digital = ""
        if block_words:
            for index, _word in block_words:
                used_words.add(index)
            digital = _lines_from_words(block_words)

        text = _best_block_text(digital, ocr_text, block.label)
        if text:
            _append_piece(pieces, Piece(text, "text", page_number))
        elif crop is not None and heading:
            # Last resort for stubborn headings: stronger upscale OCR.
            bigger = crop.resize(
                (max(1, crop.width * 4), max(1, crop.height * 4)),
                Image.Resampling.LANCZOS,
            )
            retry = _ocr_crop(bigger, cache)
            if retry:
                _append_piece(pieces, Piece(retry, "text", page_number))

    # Keep any PDF words that never landed in a detected box.
    leftover = [(index, word) for index, word in enumerate(words) if index not in used_words]
    if leftover:
        leftover_text = _lines_from_words(leftover)
        if leftover_text and leftover_text not in {piece.text for piece in pieces}:
            _append_piece(pieces, Piece(leftover_text, "text", page_number))

    if scanned and pieces:
        # Count once per scanned page that produced OCR text from blocks.
        if stats.scanned_pages == 0:
            stats.scanned_pages += 1

    return pieces


def _crop_block(image: Image.Image, bbox, pad: int = 2) -> Image.Image | None:
    x0, y0, x1, y1 = bbox
    left = max(0, int(x0) - pad)
    top = max(0, int(y0) - pad)
    right = min(image.width, int(x1) + pad)
    bottom = min(image.height, int(y1) + pad)
    if right - left < 8 or bottom - top < 8:
        return None
    return image.crop((left, top, right, bottom))


def _best_block_text(digital: str, ocr_text: str, label: str) -> str:
    """Pick the fuller reading of a layout block.

    Headings and captions are often missing from the PDF text layer, or only
    readable after upscaling. Never drop a non-empty OCR title in favour of
    an empty/short digital fragment.
    """
    digital = (digital or "").strip()
    ocr_text = (ocr_text or "").strip()

    heading_like = label in {
        "title",
        "figure_caption",
        "table_caption",
        "table_footnote",
        "formula_caption",
    }
    if heading_like:
        # Titles: OCR first, digital only as backup.
        return ocr_text or digital

    if not digital:
        return ocr_text
    if not ocr_text:
        return digital

    # Body text: trust the PDF layer when it has real content.
    if len(digital) >= 12:
        return digital
    return ocr_text if len(ocr_text) > len(digital) else digital


def _ocr_crop(crop: Image.Image, cache: dict[str, str]) -> str:
    buffer = io.BytesIO()
    crop.save(buffer, format="PNG")
    raw = buffer.getvalue()
    key = hashlib.sha1(raw).hexdigest()
    if key not in cache:
        cache[key] = ocr_region(crop)
    return cache[key]


def _words_in_bbox(words, bbox, used: set[int]):
    x0, y0, x1, y1 = bbox
    pad_x = max(1.0, (x1 - x0) * 0.02)
    pad_y = max(1.0, (y1 - y0) * 0.02)
    found = []
    for index, word in enumerate(words):
        if index in used:
            continue
        cx = (float(word[0]) + float(word[2])) / 2
        cy = (float(word[1]) + float(word[3])) / 2
        if x0 - pad_x <= cx <= x1 + pad_x and y0 - pad_y <= cy <= y1 + pad_y:
            found.append((index, word))
    return found


def _lines_from_words(block_words) -> str:
    words = [word for _, word in block_words]
    if not words:
        return ""
    width = max(float(word[2]) for word in words) - min(float(word[0]) for word in words)
    chunks = word_line_chunks(words, page_width=max(width, 1.0))
    # Within one layout block, keep geometric top-to-bottom order only.
    chunks = sorted(chunks, key=lambda item: (round(item[0][1], 1), item[0][0]))
    return "\n".join(text for _, text in chunks)


def _append_piece(pieces: list[Piece], piece: Piece) -> None:
    if pieces:
        pieces.append(Piece("\n"))
    pieces.append(piece)


def _extract_pdf_page_heuristic(document, page, cache: dict[str, str], stats: Stats, page_number: int) -> list[Piece]:
    """Previous word/image heuristic used when layout detection finds nothing."""
    chunks: list[tuple[tuple[float, float, float, float], Piece]] = []
    text_chars = 0
    page_rect = page.rect
    page_area = abs(page_rect.width * page_rect.height) or 1.0
    seen: list[tuple] = []

    for bbox, text in word_line_chunks(page.get_text("words"), page_rect.width):
        chunks.append((bbox, Piece(text)))
        text_chars += len(text)

    def add_image(bbox, raw: bytes | None) -> None:
        if not bbox or not raw:
            return
        clipped = (
            max(float(bbox[0]), float(page_rect.x0)),
            max(float(bbox[1]), float(page_rect.y0)),
            min(float(bbox[2]), float(page_rect.x1)),
            min(float(bbox[3]), float(page_rect.y1)),
        )
        width = clipped[2] - clipped[0]
        height = clipped[3] - clipped[1]
        if width < 15 or height < 15:
            return
        if any(_same_region(clipped, previous) for previous in seen):
            return
        seen.append(clipped)

        if _is_full_page(clipped, page_rect, page_area):
            if text_chars >= FULL_PAGE_TEXT_LAYER_CHARS:
                return
            stats.scanned_pages += 1
            recognized = _cached_ocr(raw, cache)
            if recognized:
                chunks.append((clipped, Piece(recognized, "text", page_number)))
            return

        stats.images_detected += 1
        recognized = _cached_ocr(raw, cache)
        if not recognized:
            return
        stats.images_with_text += 1
        chunks.append((clipped, Piece(recognized, "image", page_number)))

    for info in page.get_image_info(xrefs=True):
        add_image(info.get("bbox"), _pdf_image_bytes(document, info.get("xref") or 0))

    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 1:
            continue
        add_image(block.get("bbox"), block.get("image"))

    pieces: list[Piece] = []
    for _, piece in reading_order(chunks, page.rect.width):
        if pieces:
            pieces.append(Piece("\n"))
        pieces.append(piece)
    return pieces


def _pdf_image_bytes(document, xref: int) -> bytes | None:
    if xref <= 0:
        return None

    extracted = None
    try:
        extracted = document.extract_image(xref)
    except Exception:
        extracted = None

    if extracted:
        raw = extracted.get("image")
        kind = str(extracted.get("ext", "")).lower()
        if raw and kind in _COMMON_IMAGE_TYPES:
            return raw

    try:
        pixmap = pymupdf.Pixmap(document, xref)
        if pixmap.n - pixmap.alpha > 3:
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
        return pixmap.tobytes("png")
    except Exception:
        if extracted and extracted.get("image"):
            return extracted["image"]
        return None


def _iter_blocks(parent):
    if isinstance(parent, WordDocument):
        container = parent.element.body
        owner = parent
    else:
        container = parent._tc
        owner = parent

    for child in container.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, owner)
        elif child.tag == qn("w:tbl"):
            yield Table(child, owner)


def _block_pieces(block, cache: dict[str, str], stats: Stats) -> list[Piece]:
    if isinstance(block, Paragraph):
        return _paragraph_pieces(block, cache, stats)
    if isinstance(block, Table):
        return _table_pieces(block, cache, stats)
    return []


def _paragraph_pieces(paragraph: Paragraph, cache: dict[str, str], stats: Stats) -> list[Piece]:
    pieces: list[Piece] = []
    for run in paragraph.runs:
        if run.text:
            pieces.append(Piece(run.text))
        for text in _text_boxes(run._element):
            pieces.append(Piece(text))
        for blob in _run_images(run):
            stats.images_detected += 1
            recognized = _cached_ocr(blob, cache)
            if not recognized:
                continue
            stats.images_with_text += 1
            if pieces and not pieces[-1].text[-1].isspace():
                pieces.append(Piece(" "))
            pieces.append(Piece(recognized, "image"))
            pieces.append(Piece(" "))

    pieces = _strip_edges(pieces)
    if _has_content(pieces):
        return pieces
    fallback = paragraph.text.strip()
    if fallback:
        return [Piece(fallback)]
    return []


def _table_pieces(table: Table, cache: dict[str, str], stats: Stats) -> list[Piece]:
    rows: list[list[Piece]] = []
    seen_cells = set()
    for row in table.rows:
        cells: list[list[Piece]] = []
        for cell in row.cells:
            # Keep the element itself. id() is unsafe here: lxml can reuse
            # the id of a cell element after the previous wrapper is freed.
            marker = cell._tc
            if marker in seen_cells:
                continue
            seen_cells.add(marker)
            cell_pieces: list[Piece] = []
            for block in _iter_blocks(cell):
                parts = _block_pieces(block, cache, stats)
                if not _has_content(parts):
                    continue
                if cell_pieces:
                    cell_pieces.append(Piece("\n"))
                cell_pieces.extend(parts)
            if _has_content(cell_pieces):
                cells.append(cell_pieces)
        if not cells:
            continue
        row_pieces: list[Piece] = []
        for index, cell_pieces in enumerate(cells):
            if index:
                row_pieces.append(Piece(" | "))
            row_pieces.extend(cell_pieces)
        rows.append(row_pieces)

    pieces: list[Piece] = []
    for row_pieces in rows:
        if pieces:
            pieces.append(Piece("\n"))
        pieces.extend(row_pieces)
    return pieces


def _text_boxes(element) -> list[str]:
    found = []
    for text_box in element.iter(qn("w:txbxContent")):
        chunks = []
        for node in text_box.iter(qn("w:t")):
            if node.text:
                chunks.append(node.text)
        text = "".join(chunks).strip()
        if text:
            found.append(text)
    return found


def _run_images(run) -> list[bytes]:
    blobs = []
    seen = set()
    part = run.part
    for element in run._element.iter():
        relationship_id = None
        if element.tag == _BLIP:
            relationship_id = element.get(_EMBED) or element.get(_REL_ID)
        elif element.tag == _IMAGEDATA:
            relationship_id = element.get(_REL_ID)
        if not relationship_id or relationship_id in seen:
            continue
        seen.add(relationship_id)
        related = part.related_parts.get(relationship_id)
        blob = getattr(related, "blob", None) if related is not None else None
        if blob:
            blobs.append(blob)
    return blobs


def _cached_ocr(data: bytes, cache: dict[str, str]) -> str:
    key = hashlib.sha1(data).hexdigest()
    if key not in cache:
        cache[key] = ocr_image(data)
    return cache[key]


def _is_full_page(bbox, page_rect, page_area: float) -> bool:
    """True when the image is the page itself, not a picture on the page."""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    page_width = abs(page_rect.width) or 1.0
    page_height = abs(page_rect.height) or 1.0
    if (width * height) / page_area >= FULL_PAGE_AREA:
        return True
    return width / page_width >= FULL_PAGE_SIDE and height / page_height >= FULL_PAGE_SIDE


def _same_region(first, second) -> bool:
    """True when two boxes are the same picture drawn twice.

    A small image sitting on a full-page background must not match, so this
    uses intersection-over-union instead of "covers most of the smaller box".
    """
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    intersection = (ix1 - ix0) * (iy1 - iy0)
    area1 = (ax1 - ax0) * (ay1 - ay0)
    area2 = (bx1 - bx0) * (by1 - by0)
    union = area1 + area2 - intersection
    if union <= 0:
        return False
    return intersection / union >= 0.7


def _has_content(pieces: list[Piece]) -> bool:
    return any(piece.source == "image" or piece.text.strip() for piece in pieces)


def _strip_edges(pieces: list[Piece]) -> list[Piece]:
    pieces = [piece for piece in pieces if piece.text]
    while pieces and pieces[0].source == "text" and not pieces[0].text.strip():
        pieces.pop(0)
    while pieces and pieces[-1].source == "text" and not pieces[-1].text.strip():
        pieces.pop()
    if pieces and pieces[0].source == "text":
        pieces[0] = Piece(pieces[0].text.lstrip(), pieces[0].source, pieces[0].page)
    if pieces and pieces[-1].source == "text":
        pieces[-1] = Piece(pieces[-1].text.rstrip(), pieces[-1].source, pieces[-1].page)
    return [piece for piece in pieces if piece.text]


def _render(pieces: list[Piece], stats: Stats) -> Extracted:
    normalized: list[Piece] = []
    for piece in pieces:
        text = piece.text.replace("\r\n", "\n").replace("\r", "\n")
        if piece.source == "text" and "\n" in text:
            text = "\n".join(line.rstrip() for line in text.split("\n"))
        if text:
            normalized.append(Piece(text, piece.source, piece.page))

    normalized = _strip_edges(normalized)
    segments = _merge_text(normalized)

    parts: list[str] = []
    highlights: list[dict] = []
    cursor = 0
    for segment in segments:
        start = cursor
        parts.append(segment.text)
        cursor += len(segment.text)
        if segment.source == "image":
            highlights.append(
                {
                    "start": start,
                    "end": cursor,
                    "text": segment.text,
                    "page": segment.page,
                }
            )

    return Extracted(
        text="".join(parts),
        segments=[
            {"source": segment.source, "text": segment.text, "page": segment.page}
            for segment in segments
        ],
        highlights=highlights,
        images_detected=stats.images_detected,
        images_with_text=stats.images_with_text,
        scanned_pages=stats.scanned_pages,
        detection_dir=stats.detection_dir,
        detected_images=list(stats.detected_images),
    )


def _merge_text(pieces: list[Piece]) -> list[Piece]:
    merged: list[Piece] = []
    for piece in pieces:
        if merged and merged[-1].source == "text" and piece.source == "text":
            previous = merged[-1]
            merged[-1] = Piece(previous.text + piece.text, "text", previous.page)
        else:
            merged.append(piece)
    return merged
