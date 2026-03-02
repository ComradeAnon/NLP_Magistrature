from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PIL import Image

from .schema import FormulaRecord
from .utils import sha1_text, normalize_spaces
from .postprocess import normalize_latex, classify_formula, extract_rhs, rhs_to_float, latex_to_mathml, latex_to_sympy

def crop_with_margin(img: Image.Image, bbox: List[int], margin: int = 12) -> Image.Image:
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(img.width, x1 + margin)
    y1 = min(img.height, y1 + margin)
    return img.crop((x0, y0, x1, y1))

def build_formula_id(doc_id: str, page: int, source: str, payload: str) -> str:
    return sha1_text(f"{doc_id}|{page}|{source}|{payload}")[:24]

def formula_from_text_layer(doc_id: str, pdf_file: str, page: int, latex_raw: str) -> FormulaRecord:
    latex_norm = normalize_latex(latex_raw)
    cls = classify_formula(latex_norm)
    rhs_raw = extract_rhs(latex_norm) if cls == "EQ_NUM" else None
    rhs_float = rhs_to_float(rhs_raw) if rhs_raw else None

    fid = build_formula_id(doc_id, page, "text_layer", latex_norm)
    return FormulaRecord(
        formula_id=fid,
        doc_id=doc_id,
        pdf_file=pdf_file,
        page=page,
        source="text_layer",
        latex_raw=latex_raw,
        latex_norm=latex_norm,
        cls=cls,
        rhs_number_raw=rhs_raw,
        rhs_number_float=rhs_float,
        mathml=latex_to_mathml(latex_norm),
        sympy=latex_to_sympy(latex_norm),
    )

def formula_from_ocr(doc_id: str,
                     pdf_file: str,
                     page: int,
                     page_img: Image.Image,
                     bbox_px: List[int],
                     crop_rel_path: str,
                     line_text_ocr: Optional[str],
                     tesseract_conf: Optional[float],
                     latex_raw: str,
                     ocr_confidence: Optional[float] = None) -> FormulaRecord:

    latex_norm = normalize_latex(latex_raw)
    cls = classify_formula(latex_norm)
    rhs_raw = extract_rhs(latex_norm) if cls == "EQ_NUM" else None
    rhs_float = rhs_to_float(rhs_raw) if rhs_raw else None

    fid = build_formula_id(doc_id, page, "ocr", f"{bbox_px}|{latex_norm}")
    return FormulaRecord(
        formula_id=fid,
        doc_id=doc_id,
        pdf_file=pdf_file,
        page=page,
        source="ocr",
        bbox_px=bbox_px,
        crop_path=crop_rel_path,
        line_text_ocr=line_text_ocr,
        latex_raw=latex_raw,
        latex_norm=latex_norm,
        cls=cls,
        rhs_number_raw=rhs_raw,
        rhs_number_float=rhs_float,
        ocr_confidence=ocr_confidence if ocr_confidence is not None else tesseract_conf,
        mathml=latex_to_mathml(latex_norm),
        sympy=latex_to_sympy(latex_norm),
    )
