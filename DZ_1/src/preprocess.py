from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import fitz
import numpy as np
import cv2
from PIL import Image

from .utils import normalize_spaces

@dataclass
class PagePreprocessResult:
    page_img: Image.Image          # rendered page
    page_img_pre: Image.Image      # preprocessed image for OCR (may equal page_img)
    page_type: str                 # text/scanned/mixed/unknown
    text_char_count: int

def extract_doc_metadata(doc: fitz.Document) -> Dict:
    md = doc.metadata or {}
    # make it JSON-friendly
    out = {}
    for k, v in md.items():
        try:
            out[k] = v
        except Exception:
            out[k] = str(v)
    return out

def render_page(page: fitz.Page, dpi: int = 350) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def page_text_char_count(page: fitz.Page) -> int:
    try:
        t = page.get_text("text") or ""
        t = normalize_spaces(t)
        return len(t)
    except Exception:
        return 0

def detect_page_type(text_chars: int, has_images_hint: Optional[bool] = None) -> str:
    # простая эвристика
    if text_chars > 400:
        return "text"
    if text_chars > 50:
        return "mixed"
    return "scanned"

def pil_to_cv(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    # PIL RGB -> OpenCV BGR
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv_to_pil(arr_bgr: np.ndarray) -> Image.Image:
    arr_rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(arr_rgb)

def enhance_for_ocr(pil_img: Image.Image) -> Image.Image:
    """
    Мягкое улучшение: grayscale + CLAHE + unsharp mask.
    Не делаем жёсткую бинаризацию по умолчанию (может портить формулы).
    """
    bgr = pil_to_cv(pil_img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE (локальный контраст)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)

    # unsharp mask
    blur = cv2.GaussianBlur(cl, (0, 0), sigmaX=1.2)
    sharp = cv2.addWeighted(cl, 1.6, blur, -0.6, 0)

    # denoise light
    den = cv2.fastNlMeansDenoising(sharp, h=10)

    # back to 3-channel for tesseract stability
    out = cv2.cvtColor(den, cv2.COLOR_GRAY2BGR)
    return cv_to_pil(out)

def preprocess_page(page: fitz.Page, dpi: int = 350) -> PagePreprocessResult:
    text_chars = page_text_char_count(page)
    ptype = detect_page_type(text_chars)

    img = render_page(page, dpi=dpi)
    if ptype in ("scanned", "mixed"):
        img_pre = enhance_for_ocr(img)
    else:
        img_pre = img

    return PagePreprocessResult(
        page_img=img,
        page_img_pre=img_pre,
        page_type=ptype,
        text_char_count=text_chars
    )
