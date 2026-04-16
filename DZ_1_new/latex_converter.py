"""
Конвертер текстовых формул в LaTeX-нотацию.
"""

import re
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


# ─── Таблицы замен ────────────────────────────────────────────────────────────

UNICODE_TO_LATEX: List[Tuple[str, str]] = [
    ("√",  r"\sqrt"),
    ("∞",  r"\infty"),
    ("∑",  r"\sum"),
    ("Σ",  r"\sum"),
    ("∫",  r"\int"),
    ("∏",  r"\prod"),
    ("π",  r"\pi"),
    ("α",  r"\alpha"),
    ("β",  r"\beta"),
    ("γ",  r"\gamma"),
    ("δ",  r"\delta"),
    ("θ",  r"\theta"),
    ("λ",  r"\lambda"),
    ("μ",  r"\mu"),
    ("σ",  r"\sigma"),
    ("φ",  r"\phi"),
    ("ω",  r"\omega"),
    ("≤",  r"\leq"),
    ("≥",  r"\geq"),
    ("≠",  r"\neq"),
    ("±",  r"\pm"),
    ("×",  r"\times"),
    ("÷",  r"\div"),
    ("·",  r"\cdot"),
    ("°",  r"^{\circ}"),
    ("²",  r"^{2}"),
    ("³",  r"^{3}"),
    ("½",  r"\frac{1}{2}"),
    ("¼",  r"\frac{1}{4}"),
    ("¾",  r"\frac{3}{4}"),
    ("→",  r"\to"),
    ("∈",  r"\in"),
    ("∉",  r"\notin"),
    ("⊂",  r"\subset"),
    ("∩",  r"\cap"),
    ("∪",  r"\cup"),
    ("∅",  r"\emptyset"),
]

# (regex-паттерн, замена)
# Замена хранится как строка — применяем через lambda чтобы
# re.sub не интерпретировал обратные слэши
FUNC_TO_LATEX: List[Tuple[str, str]] = [
    (r"\barcsin\b",  r"\arcsin"),
    (r"\barccos\b",  r"\arccos"),
    (r"\barctg\b",   r"\arctan"),
    (r"\barctan\b",  r"\arctan"),
    (r"\barcctg\b",  r"\text{arcctg}"),
    (r"\bctg\b",     r"\cot"),
    (r"\btg\b",      r"\tan"),
    (r"\bsin\b",     r"\sin"),
    (r"\bcos\b",     r"\cos"),
    (r"\btan\b",     r"\tan"),
    (r"\bcot\b",     r"\cot"),
    (r"\bsec\b",     r"\sec"),
    (r"\bcsc\b",     r"\csc"),
    (r"\bsinh\b",    r"\sinh"),
    (r"\bcosh\b",    r"\cosh"),
    (r"\btanh\b",    r"\tanh"),
    (r"\bln\b",      r"\ln"),
    (r"\blg\b",      r"\lg"),
    (r"\blog\b",     r"\log"),
    (r"\blim\b",     r"\lim"),
    (r"\bexp\b",     r"\exp"),
    (r"\bmax\b",     r"\max"),
    (r"\bmin\b",     r"\min"),
    (r"\bmod\b",     r"\bmod"),
    (r"\bsqrt\b",    r"\sqrt"),
    (r"\bsum\b",     r"\sum"),
    (r"\bprod\b",    r"\prod"),
    (r"\binf\b",     r"\infty"),
]


class LaTeXConverter:
    """Преобразует текстовое представление формулы в LaTeX."""

    def convert(self, text: str) -> str:
        """Основная функция конвертации."""
        result = text

        result = self._replace_unicode(result)
        result = self._replace_functions(result)
        result = self._convert_combinations(result)
        result = self._convert_fractions(result)
        result = self._convert_roots(result)
        result = self._normalize_powers(result)
        result = self._normalize_subscripts(result)
        result = self._convert_limits(result)
        result = self._convert_sums(result)
        result = self._normalize_spaces(result)

        return result.strip()

    # ── Частные методы ────────────────────────────────────────────────────────

    def _replace_unicode(self, text: str) -> str:
        """Простая замена строк — re не нужен, слэши безопасны."""
        for symbol, latex in UNICODE_TO_LATEX:
            text = text.replace(symbol, latex)
        return text

    def _replace_functions(self, text: str) -> str:
        """
        Замена функций через lambda — re.sub не трогает
        обратные слэши в возвращаемой строке.
        """
        for pattern, latex in FUNC_TO_LATEX:
            # lambda игнорирует match и возвращает строку как есть
            text = re.sub(
                pattern,
                lambda m, s=latex: s,
                text,
                flags=re.IGNORECASE,
            )
        return text

    def _convert_combinations(self, text: str) -> str:
        """C_n^k → \\binom{n}{k}, C(n,k) → \\binom{n}{k}"""
        # C_n^k
        text = re.sub(
            r"\bC\s*_\s*(\d+)\s*\^\s*(\d+)",
            lambda m: rf"\binom{{{m.group(1)}}}{{{m.group(2)}}}",
            text,
        )
        # C(n,k)
        text = re.sub(
            r"\bC\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)",
            lambda m: rf"\binom{{{m.group(1)}}}{{{m.group(2)}}}",
            text,
        )
        # A_n^k
        text = re.sub(
            r"\bA\s*_\s*(\d+)\s*\^\s*(\d+)",
            lambda m: rf"A_{{{m.group(1)}}}^{{{m.group(2)}}}",
            text,
        )
        return text

    def _convert_fractions(self, text: str) -> str:
        """3/4 → \\frac{3}{4}"""
        text = re.sub(
            r"(?<![\\{/])\b(\d+)\s*/\s*(\d+)\b(?!/)",
            lambda m: rf"\frac{{{m.group(1)}}}{{{m.group(2)}}}",
            text,
        )
        return text

    def _convert_roots(self, text: str) -> str:
        """sqrt(x) → \\sqrt{x}"""
        # sqrt(expr)
        text = re.sub(
            r"\\?sqrt\s*\(\s*([^)]+)\s*\)",
            lambda m: rf"\sqrt{{{m.group(1)}}}",
            text,
        )
        # \sqrt x без скобок
        text = re.sub(
            r"\\sqrt\s+(\w+)",
            lambda m: rf"\sqrt{{{m.group(1)}}}",
            text,
        )
        return text

    def _normalize_powers(self, text: str) -> str:
        """x^2 → x^{2}"""
        text = re.sub(
            r"\^([0-9a-zA-Z])(?!\{)",
            lambda m: rf"^{{{m.group(1)}}}",
            text,
        )
        return text

    def _normalize_subscripts(self, text: str) -> str:
        """x_1 → x_{1}"""
        text = re.sub(
            r"_([0-9a-zA-Z])(?!\{)",
            lambda m: rf"_{{{m.group(1)}}}",
            text,
        )
        return text

    def _convert_limits(self, text: str) -> str:
        """lim(x->inf) → \\lim_{x \\to \\infty}"""
        def _repl(m: re.Match) -> str:
            var = m.group(1)
            to  = m.group(2).strip()
            if "inf" in to.lower():
                to = r"\infty"
            return rf"\lim_{{{var} \to {to}}}"

        text = re.sub(
            r"\\lim\s*\(\s*([a-zA-Z])\s*->\s*([^)]+)\s*\)",
            _repl,
            text,
        )
        return text

    def _convert_sums(self, text: str) -> str:
        """sum(i=1..n) → \\sum_{i=1}^{n}"""
        text = re.sub(
            r"\\?sum\s*\(\s*([a-zA-Z])\s*=\s*(\d+)\s*\.\.\s*(\w+)\s*\)",
            lambda m: rf"\sum_{{{m.group(1)}={m.group(2)}}}^{{{m.group(3)}}}",
            text,
        )
        return text

    def _normalize_spaces(self, text: str) -> str:
        text = re.sub(r" {2,}", " ", text)
        return text
