from __future__ import annotations

import re
from typing import List, Dict, Tuple
import fitz

from .utils import normalize_spaces

LATEX_PATTERNS = [
    # \( ... \)
    (re.compile(r"\\\((.+?)\\\)", re.DOTALL), "inline_paren"),
    # \[ ... \]
    (re.compile(r"\\\[(.+?)\\\]", re.DOTALL), "display_bracket"),
    # $$ ... $$
    (re.compile(r"\$\$(.+?)\$\$", re.DOTALL), "display_dollars"),
    # $ ... $
    (re.compile(r"\$(.+?)\$", re.DOTALL), "inline_dollar"),
    # \begin{equation}...\end{equation} and friends (simple)
    (re.compile(r"\\begin\{(equation|align|gather)\}(.+?)\\end\{\1\}", re.DOTALL), "env"),
]

def extract_text_blocks(page: fitz.Page) -> List[Dict]:
    """
    Возвращает текстовые блоки со span/line bbox (из text-layer).
    Это полезно для контекста/порядка чтения.
    """
    try:
        d = page.get_text("dict")
    except Exception:
        return []

    blocks_out = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        blocks_out.append(b)
    return blocks_out

def extract_latex_from_text_layer(page: fitz.Page) -> List[Dict]:
    """
    Ищет LaTeX-формулы, если они реально присутствуют в text layer.
    Возвращает список: {"latex_raw": "...", "pattern": "..."}.
    """
    try:
        text = page.get_text("text") or ""
    except Exception:
        return []

    text = normalize_spaces(text)
    results = []

    # Важно: normalize_spaces "схлопывает" пробелы и может исказить многострочные env.
    # Для env мы попробуем на исходном тексте тоже.
    raw_text = page.get_text("text") or ""

    for rx, kind in LATEX_PATTERNS:
        base = raw_text if kind == "env" else text
        for m in rx.finditer(base):
            if kind == "env":
                # группы: envname, content
                latex = m.group(0)
            else:
                latex = m.group(0)
            results.append({"latex_raw": latex, "pattern": kind})

    return results
