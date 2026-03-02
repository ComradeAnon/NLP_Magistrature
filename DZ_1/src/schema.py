from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal

PageType = Literal["text", "scanned", "mixed", "unknown"]
FormulaSource = Literal["ocr", "text_layer"]

@dataclass
class DocumentMeta:
    doc_id: str
    pdf_file: str
    pdf_sha1: str
    page_count: int
    metadata: Dict[str, Any]

@dataclass
class PageInfo:
    page: int
    page_type: PageType
    text_char_count: int

@dataclass
class FormulaRecord:
    formula_id: str
    doc_id: str
    pdf_file: str
    page: int
    source: FormulaSource

    # where it came from
    bbox_px: Optional[List[int]] = None
    crop_path: Optional[str] = None

    # context
    line_text_ocr: Optional[str] = None

    # representations
    latex_raw: Optional[str] = None
    latex_norm: Optional[str] = None
    mathml: Optional[str] = None
    sympy: Optional[str] = None

    # labels / targets
    cls: str = "OTHER"
    rhs_number_raw: Optional[str] = None
    rhs_number_float: Optional[float] = None

    # quality
    ocr_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
