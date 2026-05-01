"""
Детектор и классификатор математических формул.

Определяет класс формулы по регулярным выражениям.
Только математические формулы (формула = число).
Фильтрует физические и химические формулы.
"""

import re
import logging
from typing import Dict, Optional, List, Tuple

from config import (
    CLASS_KEYS,
    CLASSIFICATION_CONFIDENCE_THRESHOLD,
    CLASS_THRESHOLDS,
    MATH_DENSITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ─── Паттерны классов ────────────────────────────────────────────────────────
# Структура: (regex_паттерн, вес)
# Положительный вес — признак класса
# Отрицательный вес — штраф (признак другого класса)

_CLASS_PATTERNS_RAW: Dict[str, List[Tuple[str, float]]] = {

    "algebraic": [
        (r"[a-zA-Z]\s*[\^²³]\s*\d",             0.7),
        (r"\d+\s*\^?\s*\d+",                     0.5),
        (r"√\s*\d+",                             0.7),
        (r"\\sqrt\s*[\{\(]",                     0.8),
        (r"\\frac\s*\{",                         0.8),
        (r"\d+\s*/\s*\d+",                       0.6),
        (r"\(\s*[\d+\-a-zA-Z]+\s*\)\s*[\^²³]",  0.7),
        (r"\d+\s*[+\-]\s*\d+",                  0.4),
        (r"\d+\s*\*\s*\d+",                     0.5),
        (r"\\sin|\\cos|\\tan|\\tg",             -0.5),
        (r"\\log|\\ln",                         -0.5),
        (r"\\sum|\\lim|\\int",                  -0.6),
        (r"\bC\s*[\(_]\d|\bA\s*[\(_]\d",        -0.5),
    ],

    "trigonometric": [
        (r"\b(sin|cos|tan|tg|cot|ctg|sec|csc)\s*[\d(°π]", 0.9),
        (r"\b(arcsin|arccos|arctan|arctg)\b",    0.9),
        (r"\b(sinh|cosh|tanh)\b",               0.8),
        (r"\\sin|\\cos|\\tan|\\tg",             0.9),
        (r"\\arcsin|\\arccos|\\arctan",         0.9),
        (r"π\s*/\s*\d+",                        0.6),
        (r"\d+\s*°",                            0.5),
        (r"sin\^2|cos\^2",                      0.8),
    ],

    "logarithmic": [
        (r"\bln\s*[\d(e]",                      0.9),
        (r"\blg\s*[\d(]",                       0.9),
        (r"\blog\s*_?\d",                       0.8),
        (r"\\log|\\ln|\\lg",                    0.9),
        (r"log_\{?\d",                          0.8),
        (r"\blog\s*\(",                         0.7),
        (r"\bexp\s*\(",                         0.6),
        (r"\\sin|\\cos|\\sum|\\lim",            -0.5),
    ],

    "combinatorial": [
        (r"\bC\s*_?\s*\{?\d+\}?\s*[\^_]\s*\{?\d+", 0.9),
        (r"\bC\s*\(\s*\d+\s*,\s*\d+\s*\)",      0.9),
        (r"\bA\s*_\s*\d+\s*\^\s*\d+",           0.8),
        (r"\bP\s*_?\s*\{?\d+\}?",               0.7),
        (r"\d+\s*!",                             0.9),
        (r"[a-zA-Z]\s*!",                        0.7),
        (r"\\binom\s*\{",                        0.9),
        (r"\\choose",                            0.9),
        (r"\\sin|\\cos|\\log|\\sum|\\lim",      -0.5),
    ],

    "series_limit": [
        (r"[∑Σ]",                               0.9),
        (r"\\sum\b",                            0.9),
        (r"\bsum\s*[\(_\{]",                    0.8),
        (r"\blim\b|\\lim\b",                    0.9),
        (r"lim_\s*[\({\\]",                     0.9),
        (r"[∫]|\\int\b",                        0.8),
        (r"\\infty|∞",                          0.6),
        (r"n\s*=\s*[01]",                       0.5),
        (r"i\s*=\s*[01]",                       0.5),
        (r"x\s*->\s*(inf|∞|\\infty)",           0.8),
    ],
}

CLASS_PATTERNS: Dict[str, List[Tuple[re.Pattern, float]]] = {
    cls: [(re.compile(p, re.IGNORECASE), w) for p, w in patterns]
    for cls, patterns in _CLASS_PATTERNS_RAW.items()
}


# ─── Фильтры физика/химия ─────────────────────────────────────────────────────

PHYSICS_PATTERNS = [
    r"\b(F\s*=\s*ma|E\s*=\s*mc|P\s*=\s*UI|W\s*=\s*Fs)\b",
    r"\b(Дж|Па|Вт|Ом|Тл|Гн|Кл)\b",
    r"\bm/s\b|\bJ/mol\b|\bkg\b",
    r"\b(масса|сила|скорость|ускорение|мощность|заряд)\b",
    r"\b(Планк|Больцман|Авогадро|Фарадей)\b",
]

CHEMISTRY_PATTERNS = [
    r"\b[A-Z][a-z]?\d*\s*\+\s*[A-Z][a-z]?\d*\s*[→=]",
    r"\b(H2O|CO2|NaCl|HCl|H2SO4|NH3|CH4)\b",
    r"[→⟶]",
    r"\b(моль|г/моль|атм|pH)\b",
    r"\b(кислота|основание|реакция|катализатор)\b",
]

_PHYSICS_COMPILED = [re.compile(p, re.IGNORECASE) for p in PHYSICS_PATTERNS]
_CHEMISTRY_COMPILED = [re.compile(p, re.IGNORECASE) for p in CHEMISTRY_PATTERNS]


def _is_physics_or_chemistry(text: str) -> bool:
    all_patterns = _PHYSICS_COMPILED + _CHEMISTRY_COMPILED
    for pattern in all_patterns:
        if pattern.search(text):
            return True
    return False


# ─── Плотность математических символов ────────────────────────────────────────

_MATH_SYMBOL_RE = re.compile(
    r"[+\-*/=^√∑∫Π∂∇∞±×÷≠≤≥≈∝∈∉⊂⊃∪∩∧∨∀∃"
    r"₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    r"⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    r"²³"
    r"πθφψωλμσ]"
)

_CYRILLIC_LETTER_RE = re.compile(r"[а-яёА-ЯЁ]")


def math_density(text: str) -> float:
    """
    Доля математических символов в строке (0..1).
    Высокая плотность — вероятная формула, а не prose.
    """
    if not text:
        return 0.0
    math_chars = len(_MATH_SYMBOL_RE.findall(text))
    letters = len(_CYRILLIC_LETTER_RE.findall(text)) + len(re.findall(r"[a-zA-Z]", text))
    total = max(len(text.replace(" ", "")), 1)
    if letters == 0:
        return math_chars / total
    return math_chars / (math_chars + letters)


def _has_numeric_rhs(text: str) -> bool:
    eq_idx = text.rfind("=")
    if eq_idx == -1:
        return False

    rhs = text[eq_idx + 1:].strip()

    if re.fullmatch(r"[-+]?\s*[\d][\d\s.,*/()+\-^√π°]*", rhs):
        return True

    if re.match(r"[-+]?\s*\d", rhs):
        return True

    if rhs in ("0", "1", "-1", "∞", "\\infty"):
        return True

    return False


# ─── Классификатор ────────────────────────────────────────────────────────────

class FormulaClassifier:
    """Классифицирует текстовый фрагмент по классам формул."""

    def classify(self, text: str) -> Optional[str]:
        if _is_physics_or_chemistry(text):
            logger.debug("Отфильтровано (физика/химия): %.50s", text)
            return None

        if not _has_numeric_rhs(text):
            logger.debug("Нет числовой правой части: %.50s", text)
            return None

        scores = {cls: 0.0 for cls in CLASS_KEYS}

        density = math_density(text)
        if density >= MATH_DENSITY_THRESHOLD:
            for cls in CLASS_KEYS:
                scores[cls] += 0.1

        for cls, pattern_list in CLASS_PATTERNS.items():
            for compiled, weight in pattern_list:
                if compiled.search(text):
                    scores[cls] += weight

        best_class = max(scores, key=lambda c: scores[c])
        best_score = scores[best_class]

        logger.debug("Scores: %s | best: %s=%.2f | density=%.2f", scores, best_class, best_score, density)

        threshold = CLASS_THRESHOLDS.get(best_class, CLASSIFICATION_CONFIDENCE_THRESHOLD)

        if best_score < threshold:
            if re.search(r"\d+\s*[+\-*/]\s*\d+", text):
                return "algebraic"
            return None

        return best_class

    def classify_batch(self, candidates: List[Dict]) -> List[Dict]:
        results = []
        skipped = 0

        for item in candidates:
            cls = self.classify(item["text"])
            if cls is not None:
                item_copy         = dict(item)
                item_copy["class"] = cls
                results.append(item_copy)
            else:
                skipped += 1

        logger.info(
            "Классифицировано: %d / %d (пропущено: %d)",
            len(results), len(candidates), skipped
        )
        return results
