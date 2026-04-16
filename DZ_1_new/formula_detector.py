"""
Детектор и классификатор математических формул.

Определяет класс формулы по регулярным выражениям.
Только математические формулы (формула = число).
Фильтрует физические и химические формулы.
"""

import re
import logging
from typing import Dict, Optional, List, Tuple

from config import CLASS_KEYS, CLASSIFICATION_CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


# ─── Паттерны классов ────────────────────────────────────────────────────────
# Структура: (regex_паттерн, вес)
# Положительный вес — признак класса
# Отрицательный вес — штраф (признак другого класса)

CLASS_PATTERNS: Dict[str, List[Tuple[str, float]]] = {

    "algebraic": [
        (r"[a-zA-Z]\s*[\^²³]\s*\d",             0.7),   # x^2, x²
        (r"\d+\s*\^?\s*\d+",                     0.5),   # 2^3
        (r"√\s*\d+",                             0.7),   # √25
        (r"\\sqrt\s*[\{\(]",                     0.8),   # \sqrt{
        (r"\\frac\s*\{",                         0.8),   # \frac{
        (r"\d+\s*/\s*\d+",                       0.6),   # 3/4
        (r"\(\s*[\d+\-a-zA-Z]+\s*\)\s*[\^²³]",  0.7),   # (x+1)^2
        (r"\d+\s*[+\-]\s*\d+",                  0.4),   # 2 + 3
        (r"\d+\s*\*\s*\d+",                     0.5),   # 2 * 3
        # Штрафы — это не алгебра
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
        (r"π\s*/\s*\d+",                        0.6),   # π/6
        (r"\d+\s*°",                            0.5),   # 30°
        (r"sin\^2|cos\^2",                      0.8),   # sin²
    ],

    "logarithmic": [
        (r"\bln\s*[\d(e]",                      0.9),   # ln(e), ln2
        (r"\blg\s*[\d(]",                       0.9),   # lg100
        (r"\blog\s*_?\d",                       0.8),   # log_2, log10
        (r"\\log|\\ln|\\lg",                    0.9),
        (r"log_\{?\d",                          0.8),   # log_{2}
        (r"\blog\s*\(",                         0.7),   # log(
        (r"\bexp\s*\(",                         0.6),   # exp(
        # Штраф — не путаем с алгеброй
        (r"\\sin|\\cos|\\sum|\\lim",            -0.5),
    ],

    "combinatorial": [
        (r"\bC\s*_?\s*\{?\d+\}?\s*[\^_]\s*\{?\d+", 0.9),  # C_n^k
        (r"\bC\s*\(\s*\d+\s*,\s*\d+\s*\)",      0.9),  # C(n,k)
        (r"\bA\s*_\s*\d+\s*\^\s*\d+",           0.8),  # A_n^k
        (r"\bP\s*_?\s*\{?\d+\}?",               0.7),  # P_n
        (r"\d+\s*!",                             0.9),  # n!
        (r"[a-zA-Z]\s*!",                        0.7),  # n!
        (r"\\binom\s*\{",                        0.9),  # \binom{
        (r"\\choose",                            0.9),
        # Штраф
        (r"\\sin|\\cos|\\log|\\sum|\\lim",      -0.5),
    ],

    "series_limit": [
        (r"[∑Σ]",                               0.9),   # Σ
        (r"\\sum\b",                            0.9),   # \sum
        (r"\bsum\s*[\(_\{]",                    0.8),   # sum(
        (r"\blim\b|\\lim\b",                    0.9),   # lim
        (r"lim_\s*[\({\\]",                     0.9),   # lim_{
        (r"[∫]|\\int\b",                        0.8),   # ∫
        (r"\\infty|∞",                          0.6),   # ∞
        (r"n\s*=\s*[01]",                       0.5),   # n=0, n=1
        (r"i\s*=\s*[01]",                       0.5),   # i=0, i=1
        (r"x\s*->\s*(inf|∞|\\infty)",           0.8),   # x→∞
    ],
}


# ─── Фильтры физика/химия ─────────────────────────────────────────────────────

# Паттерны физических формул
PHYSICS_PATTERNS = [
    r"\b(F\s*=\s*ma|E\s*=\s*mc|P\s*=\s*UI|W\s*=\s*Fs)\b",
    r"\b(Дж|Па|Вт|Ом|Тл|Гн|Кл)\b",
    r"\bm/s\b|\bJ/mol\b|\bkg\b",
    r"\b(масса|сила|скорость|ускорение|мощность|заряд)\b",
    r"\b(Планк|Больцман|Авогадро|Фарадей)\b",
]

# Паттерны химических формул
CHEMISTRY_PATTERNS = [
    r"\b[A-Z][a-z]?\d*\s*\+\s*[A-Z][a-z]?\d*\s*[→=]",  # A + B → C
    r"\b(H2O|CO2|NaCl|HCl|H2SO4|NH3|CH4)\b",
    r"[→⟶]",                                              # стрелка реакции
    r"\b(моль|г/моль|атм|pH)\b",
    r"\b(кислота|основание|реакция|катализатор)\b",
]


def _is_physics_or_chemistry(text: str) -> bool:
    """Возвращает True если формула физическая или химическая."""
    all_patterns = PHYSICS_PATTERNS + CHEMISTRY_PATTERNS
    for pattern in all_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _has_numeric_rhs(text: str) -> bool:
    """
    Проверяет что правая часть (после =) является
    числом или числовым выражением.
    """
    eq_idx = text.rfind("=")
    if eq_idx == -1:
        return False

    rhs = text[eq_idx + 1:].strip()

    # Полное числовое выражение
    if re.fullmatch(r"[-+]?\s*[\d][\d\s.,*/()+\-^√π°]*", rhs):
        return True

    # Начинается с числа
    if re.match(r"[-+]?\s*\d", rhs):
        return True

    # Специальные числовые значения
    if rhs in ("0", "1", "-1", "∞", "\\infty"):
        return True

    return False


# ─── Классификатор ────────────────────────────────────────────────────────────

class FormulaClassifier:
    """Классифицирует текстовый фрагмент по классам формул."""

    def classify(self, text: str) -> Optional[str]:
        """
        Определяет класс формулы.

        Returns:
            Ключ класса (algebraic / trigonometric / ...) или None
        """
        # Фильтр: физика и химия
        if _is_physics_or_chemistry(text):
            logger.debug("Отфильтровано (физика/химия): %.50s", text)
            return None

        # Фильтр: числовая правая часть
        if not _has_numeric_rhs(text):
            logger.debug("Нет числовой правой части: %.50s", text)
            return None

        # Считаем очки для каждого класса
        scores = {cls: 0.0 for cls in CLASS_KEYS}

        for cls, pattern_list in CLASS_PATTERNS.items():
            for pattern, weight in pattern_list:
                if re.search(pattern, text, re.IGNORECASE):
                    scores[cls] += weight

        best_class = max(scores, key=lambda c: scores[c])
        best_score = scores[best_class]

        logger.debug("Scores: %s | best: %s=%.2f", scores, best_class, best_score)

        if best_score < CLASSIFICATION_CONFIDENCE_THRESHOLD:
            # Недостаточно уверены — слабая эвристика для алгебры
            if re.search(r"\d+\s*[+\-*/]\s*\d+", text):
                return "algebraic"
            return None

        return best_class

    def classify_batch(self, candidates: List[Dict]) -> List[Dict]:
        """
        Классифицирует список кандидатов.
        Добавляет поле 'class', отбрасывает неклассифицированные.
        """
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
