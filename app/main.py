from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.extract import DocumentError, extract_document
from app.ocr import OCRUnavailable
from app.storage import DETECTED_ROOT

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_ALLOWED_EXTENSIONS = {".pdf", ".docx"} | _IMAGE_EXTENSIONS

app = FastAPI(
    title="Text Extraction",
    version="0.5.0",
    description=(
        "Extract text from PDF, DOCX, and image files. "
        "Text found inside images is inserted where the image appeared "
        "and marked so a frontend can highlight it. "
        "Detected layout blocks are also saved under /detected."
    ),
)

MAX_BYTES = 25 * 1024 * 1024
_PAGE = Path(__file__).resolve().parent / "static" / "index.html"
DETECTED_ROOT.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/detected", StaticFiles(directory=str(DETECTED_ROOT)), name="detected")


class Metrics(BaseModel):
    images_detected: int
    images_with_text: int
    scanned_pages: int = 0


class Segment(BaseModel):
    source: Literal["text", "image"]
    text: str
    page: int | None = None


class Highlight(BaseModel):
    start: int
    end: int
    text: str
    page: int | None = None


class DetectedImage(BaseModel):
    page: int
    index: int
    label: str
    score: float
    path: str
    bbox: list[float]


class ExtractResult(BaseModel):
    filename: str
    text: str
    metrics: Metrics
    segments: list[Segment]
    highlights: list[Highlight]
    detection_dir: str | None = None
    detected_images: list[DetectedImage] = []


@app.get("/")
def home():
    return FileResponse(_PAGE)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResult, response_model_exclude_none=True)
def extract(file: UploadFile = File(...)):
    filename = Path(file.filename or "upload").name
    extension = Path(filename).suffix.lower()
    if extension not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Upload a .pdf, .docx, or image file (.png, .jpg, .jpeg, .bmp, .tif, .tiff, .webp)",
        )

    data = file.file.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The file is empty")
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 25 MB")
    if extension == ".pdf" and b"%PDF" not in data[:1024]:
        raise HTTPException(status_code=400, detail="File is not a valid PDF")
    if extension == ".docx" and not data.startswith(b"PK"):
        raise HTTPException(status_code=400, detail="File is not a valid DOCX")
    if extension in _IMAGE_EXTENSIONS and not _looks_like_image(data):
        raise HTTPException(status_code=400, detail="File is not a valid image")

    try:
        extracted = extract_document(filename, data)
    except OCRUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except DocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ExtractResult(
        filename=filename,
        text=extracted.text,
        metrics=Metrics(
            images_detected=extracted.images_detected,
            images_with_text=extracted.images_with_text,
            scanned_pages=extracted.scanned_pages,
        ),
        segments=[Segment(**segment) for segment in extracted.segments],
        highlights=[Highlight(**highlight) for highlight in extracted.highlights],
        detection_dir=extracted.detection_dir,
        detected_images=[DetectedImage(**item) for item in extracted.detected_images],
    )


def _looks_like_image(data: bytes) -> bool:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if data[:3] == b"\xff\xd8\xff":
        return True
    if data[:2] == b"BM":
        return True
    if data[:4] in {b"II*\x00", b"MM\x00*"}:
        return True
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False
