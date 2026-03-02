from __future__ import annotations

import re
from typing import Optional, Tuple

from .utils import normalize_spaces

# Numeric RHS rules (units forbidden; your requirement)
RE_DEC = r"[+\-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+\-]?\d+)?"
RE_POW10 = r"10\^\{\s*[+\-]?\d+\s*\}"
RE_SCI_TEX = rf"{RE_DEC}\s*(?:\\times|×)\s*{RE_POW10}"
NUM_REGEX = re.compile(rf"^\s*(?:{RE_SCI_TEX}|{RE_POW10}|{RE_DEC})\s*$")
FORBIDDEN_UNITS = re.compile(r"([A-Za-zА-Яа-я]|\\mathrm\s*\{{)")

def normalize_latex(latex: str) -> str:
    s = normalize_spaces(latex)
    s = s.replace(r"\,", "")
    s = s.replace("×", r"\times")
    return s

def rhs_is_number(rhs: str) -> bool:
    rhs = normalize_latex(rhs)
    if FORBIDDEN_UNITS.search(rhs):
        return False
    return bool(NUM_REGEX.match(rhs))

def classify_formula(latex_norm: str) -> str:
    s = normalize_latex(latex_norm)

    if ":=" in s or r"\equiv" in s or "≡" in s:
        return "DEF_EQ"
    if any(op in s for op in [r"\le", r"\ge", "≤", "≥", "<", ">"]):
        return "INEQ"
    if "=" in s:
        _, right = s.split("=", 1)
        return "EQ_NUM" if rhs_is_number(right) else "EQ_EXPR"
    return "OTHER"

def extract_rhs(latex_norm: str) -> Optional[str]:
    s = normalize_latex(latex_norm)
    if "=" not in s:
        return None
    return s.split("=", 1)[1].strip()

def rhs_to_float(rhs_raw: str) -> Optional[float]:
    rhs = normalize_latex(rhs_raw)
    if not rhs_is_number(rhs):
        return None

    m = re.fullmatch(r"10\^\{\s*([+\-]?\d+)\s*\}", rhs)
    if m:
        return float(10 ** int(m.group(1)))

    m = re.fullmatch(rf"({RE_DEC})\s*\\times\s*10\^\{{\s*([+\-]?\d+)\s*\}}", rhs)
    if m:
        a = float(m.group(1))
        k = int(m.group(2))
        return float(a * (10 ** k))

    try:
        return float(rhs)
    except Exception:
        return None

def latex_to_mathml(latex: str) -> Optional[str]:
    # optional dependency
    try:
        import latex2mathml.converter
        return latex2mathml.converter.convert(latex)
    except Exception:
        return None

def latex_to_sympy(latex: str) -> Optional[str]:
    # optional dependency (often fails on complex latex)
    try:
        from sympy.parsing.latex import parse_latex
        expr = parse_latex(latex)
        return str(expr)
    except Exception:
        return None
