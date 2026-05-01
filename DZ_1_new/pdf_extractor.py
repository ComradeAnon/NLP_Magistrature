"""
Модуль извлечения текста из PDF-документов.

Использует pymupdf как основной движок,
pdfplumber как запасной.

Автоматически выбирает OCR для сканированных страниц.
Поддерживает вырезание формул из текстовых строк.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz
import pdfplumber

from config import (
    DOCS_DIR,
    MIN_FORMULA_LEN,
    MAX_FORMULA_LEN,
    OCR_TEXT_THRESHOLD,
    FORMULA_MODE,
    MAX_WORKERS,
)

logger = logging.getLogger(__name__)


# ─── Очистка строк ────────────────────────────────────────────────────────────

def _clean_line(line: str) -> str:
    """Базовая очистка строки от артефактов PDF."""
    line = re.sub(r" {2,}", " ", line)
    line = re.sub(
        r"[^\x20-\x7E"
        r"\u0400-\u04FF"
        r"\u00B0-\u00FF"
        r"\u2200-\u22FF"
        r"\u2070-\u209F"
        r"\u2080-\u208F"
        r"\u2100-\u214F"
        r"\u2190-\u21FF"
        r"\u00B2\u00B3\u00B9"
        r"]",
        " ",
        line,
    )
    return line.strip()


# ─── Контекстно-зависимая нормализация кириллицы ────────────────────────────

_CYRILLIC_TO_LATIN = {
    "С": "C",
    "А": "A",
    "Р": "P",
    "с": "c",
    "а": "a",
    "е": "e",
    "о": "o",
    "О": "O",
    "х": "x",
    "Х": "X",
    "у": "y",
}

_CYRILLIC_LOOKALIKE_KEYS = set(_CYRILLIC_TO_LATIN.keys())

_MATH_CONTEXT_RE = re.compile(
    r"[+\-*/^=<>≤≥≠±×÷∑∫Π∂∇√∛∜∞²³⁰¹]"
    r"|[\\](?:frac|sqrt|sum|lim|sin|cos|tan|log|ln|lg|int|binom|prod)"
    r"|[_^{}()]"
    r"|\d+"
)

_CYRILLIC_WORD_RE = re.compile(r"[а-яёА-ЯЁ]{2,}")


def _is_in_math_context(line: str, pos: int) -> bool:
    """Check if character at *pos* is within a math expression context."""
    window_start = max(0, pos - 15)
    window_end = min(len(line), pos + 15)
    window = line[window_start:window_end]
    return bool(_MATH_CONTEXT_RE.search(window))


def _clean_russian_line(line: str) -> str:
    """
    Context-aware normalisation of Russian math texts.

    - Replaces comma with dot in numbers: 2,5 → 2.5
    - Replaces Cyrillic lookalikes with Latin ONLY inside math context
      (preserves genuine Russian words like "Сумма", "Решение")
    """
    line = re.sub(r"(\d),(\d)", r"\1.\2", line)

    result = []
    for i, ch in enumerate(line):
        if ch in _CYRILLIC_LOOKALIKE_KEYS and _is_in_math_context(line, i):
            result.append(_CYRILLIC_TO_LATIN[ch])
        else:
            result.append(ch)

    return "".join(result)


# ─── Проверка строки на формулу ───────────────────────────────────────────────

_MATH_INDICATORS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"[+\-*/^√∑∫π]",
        r"\b(sin|cos|tan|tg|cot|ctg)\s*[\d(°]",
        r"\b(log|ln|lg)\s*[\d_(]",
        r"\b(lim|sum|sqrt)\b",
        r"\d+\s*[+\-*]\s*\d+",
        r"\d+\s*\^\s*\d+",
        r"\d+\s*!",
        r"\bC\s*[\(_]\d",
        r"\bA\s*[\(_]\d",
        r"[a-zA-Z]\s*\^\s*\d",
        r"\d+[a-zA-Z]\s*[+\-]",
        r"\\frac|\\sqrt|\\sum|\\lim",
    ]
]

_MATH_INDICATORS_EXTENDED = _MATH_INDICATORS + [
    re.compile(p, re.IGNORECASE) for p in [
        r"[a-zA-Z]\s*[+\-*/]\s*\d",
        r"\([a-zA-Z\d+\-*/^ ]+\)",
        r"\d+\s*/\s*\d+",
        r"[∑Σ∫∏∂∇∞]",
        r"[₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹]",
        r"[⇒⇐→←↔]",
    ]
]

_STOP_WORDS = [
    "поэтому", "следовательно", "отсюда", "пусть", "дано",
    "условие", "пример", "решение", "ответ", "задача",
    "если",   "тогда",  "так",    "как",   "что",   "это",
    "будем",  "можно",  "нужно",  "должно","вместо",
    "например","итого", "всего",  "остаток","повторя",
    "используем","применим","получим","запишем","имеем",
    "заметим","покажем","докажем","найдем", "вычислим",
]
_STOP_WORDS_SET = set(_STOP_WORDS)


def _is_formula_line(line: str, mode: str = FORMULA_MODE) -> bool:
    """
    Проверка: является ли строка математической формулой.

    strict: requires '=', length ≤ 80, cyrillic ≤ 25%
    extended: also accepts standalone expressions (no '=' needed),
              length ≤ MAX_FORMULA_LEN, cyrillic ≤ 40%
    """
    has_equals = "=" in line

    if mode == "strict":
        max_len = 80
        max_cyrillic_ratio = 0.25
        require_equals = True
        indicators = _MATH_INDICATORS
    else:
        max_len = MAX_FORMULA_LEN
        max_cyrillic_ratio = 0.40
        require_equals = False
        indicators = _MATH_INDICATORS_EXTENDED

    if require_equals and not has_equals:
        return False

    if len(line) > max_len:
        return False

    letters = len(re.findall(r"[а-яёА-ЯЁ]", line))
    total = len(line.replace(" ", ""))
    if total > 0 and letters / total > max_cyrillic_ratio:
        return False

    line_lower = line.lower()
    for word in _STOP_WORDS_SET:
        if word in line_lower:
            return False

    lhs = line
    if has_equals:
        eq_idx = line.rfind("=")
        lhs = line[:eq_idx].strip()

    if len(lhs) < 2:
        return False

    has_math = any(pat.search(lhs) for pat in indicators)
    return has_math


# ─── Вырезание формул из текстовых строк ─────────────────────────────────────

FORMULA_EXTRACTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"\d+\s*!"
            r"\s*=\s*"
            r"[-+]?\d+(?:[.,]\d+)?",
            re.IGNORECASE,
        ),
        "factorial",
    ),
    (
        re.compile(
            r"[CAP]\s*[\(_]\s*\d+\s*[,_^]\s*\d+\s*[\)_]?"
            r"\s*=\s*"
            r"\d+",
            re.IGNORECASE,
        ),
        "combinatorial",
    ),
    (
        re.compile(
            r"(?:sin|cos|tan|tg|cot|ctg)\s*\(?[-+]?\d+(?:[°π/\d\s]*)\)?"
            r"\s*=\s*"
            r"[-+]?\d+(?:[.,]\d+)?",
            re.IGNORECASE,
        ),
        "trig",
    ),
    (
        re.compile(
            r"(?:log_?\d*|ln|lg)\s*\(?[^=\n]{1,20}\)?"
            r"\s*=\s*"
            r"[-+]?\d+(?:[.,]\d+)?",
            re.IGNORECASE,
        ),
        "log",
    ),
    (
        re.compile(
            r"lim\s*[\(_]?[^=\n]{1,30}[\)_]?"
            r"\s*=\s*"
            r"[-+]?\d+(?:[.,]\d+)?",
            re.IGNORECASE,
        ),
        "limit",
    ),
    (
        re.compile(
            r"(?:sum|∑|Σ)\s*[\(_\{]?[^=\n]{1,30}[\)_\}]?"
            r"\s*=\s*"
            r"[-+]?\d+(?:[.,]\d+)?",
            re.IGNORECASE,
        ),
        "series",
    ),
    (
        re.compile(
            r"[-+]?\d*[a-zA-Z]?\s*\^\s*\d+"
            r"\s*=\s*"
            r"[-+]?\d+(?:[.,]\d+)?",
            re.IGNORECASE,
        ),
        "power",
    ),
    (
        re.compile(
            r"\d+\s*/\s*\d+"
            r"\s*=\s*"
            r"[-+]?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?",
            re.IGNORECASE,
        ),
        "fraction",
    ),
    (
        re.compile(
            r"[-+]?\d+(?:[.,]\d+)?"
            r"(?:\s*[+\-*/]\s*[-+]?\d+(?:[.,]\d+)?)+"
            r"\s*=\s*"
            r"[-+]?\d+(?:[.,]\d+)?",
            re.IGNORECASE,
        ),
        "arithmetic",
    ),
    (
        re.compile(
            r"[-+]?\d*\s*[a-zA-Z]\w*"
            r"(?:\s*[+\-*/^]\s*[-+]?\d+(?:[.,]\d+)?)?"
            r"\s*=\s*"
            r"[-+]?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?",
            re.IGNORECASE,
        ),
        "variable",
    ),
]


def _extracted_rhs_is_number(formula: str) -> bool:
    eq_idx = formula.rfind("=")
    if eq_idx == -1:
        return False
    rhs = formula[eq_idx + 1:].strip().replace(",", ".")
    return bool(re.fullmatch(
        r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?",
        rhs,
    ))


def extract_formulas_from_line(line: str) -> List[str]:
    if "=" not in line:
        return []

    found = []
    seen = set()

    for compiled, _name in FORMULA_EXTRACTION_PATTERNS:
        try:
            for match in compiled.finditer(line):
                formula = match.group(0).strip()
                if len(formula) < 5:
                    continue
                if formula in seen:
                    continue
                if not _extracted_rhs_is_number(formula):
                    continue
                seen.add(formula)
                found.append(formula)
        except re.error as exc:
            logger.debug("Regex error: %s", exc)
            continue

    return found


# ─── Склейка перенесённых строк ───────────────────────────────────────────────

def _merge_broken_lines(lines: List[str]) -> List[str]:
    if not lines:
        return []

    merged = []
    buffer = lines[0]

    for current in lines[1:]:
        prev_stripped = buffer.rstrip()
        current_stripped = current.lstrip()

        ends_operator = bool(prev_stripped and prev_stripped[-1] in "+-*/=^,(\\")
        starts_operator = bool(current_stripped and current_stripped[0] in "+-*/^),")
        starts_lower = bool(current_stripped and current_stripped[0].islower())

        if ends_operator or starts_operator or starts_lower:
            buffer = prev_stripped + " " + current_stripped
        else:
            merged.append(buffer)
            buffer = current

    merged.append(buffer)
    return merged


# ─── Шрифтовой экстрактор (font metrics) ─────────────────────────────────────

def _extract_page_fitz_rich(page: fitz.Page) -> List[str]:
    """
    Извлечение страницы с использованием шрифтовых метрик.

    Использует page.get_text("dict") для получения span-level
    информации: флаг is_italic,的大小 шрифта, имя шрифта.
    Это позволяет лучше отделять формулы от prose.
    """
    lines = []
    try:
        data = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    except Exception:
        return _extract_page_fitz_simple(page)

    for block in data.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line_data in block.get("lines", []):
            spans_text = []
            for span in line_data.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    spans_text.append(text)

            if spans_text:
                full_line = " ".join(spans_text)
                cleaned = _clean_line(full_line)
                if cleaned:
                    lines.append(cleaned)

    return lines


def _extract_page_fitz_simple(page: fitz.Page) -> List[str]:
    """Оригинальное извлечение через get_text('blocks')."""
    lines = []
    blocks = page.get_text("blocks")
    blocks.sort(key=lambda b: (round(b[1] / 10), b[0]))

    for block in blocks:
        text = block[4]
        for raw_line in text.split("\n"):
            cleaned = _clean_line(raw_line)
            if cleaned:
                lines.append(cleaned)
    return lines


# ─── Основной экстрактор ──────────────────────────────────────────────────────

class PDFExtractor:
    """Извлекает текстовые строки и формулы из PDF."""

    def __init__(self, pdf_path: Path):
        self.pdf_path = Path(pdf_path)
        self.source_name = pdf_path.stem

    # ── pymupdf ───────────────────────────────────────────────────────────────

    def _extract_page_fitz(self, page: fitz.Page) -> List[str]:
        return _extract_page_fitz_rich(page)

    def _extract_with_fitz(self) -> List[str]:
        lines = []
        try:
            doc = fitz.open(str(self.pdf_path))
            for page in doc:
                page_text = page.get_text().strip()

                if len(page_text) >= OCR_TEXT_THRESHOLD:
                    lines.extend(self._extract_page_fitz(page))
                else:
                    lines.extend(self._try_ocr_page(page))

            doc.close()
        except Exception as exc:
            logger.warning("fitz error for %s: %s", self.pdf_path.name, exc)

        return lines

    def _try_ocr_page(self, page: fitz.Page) -> List[str]:
        try:
            from ocr_extractor import ocr_page
            text = ocr_page(page)
            return [
                _clean_line(line)
                for line in text.split("\n")
                if _clean_line(line)
            ]
        except ImportError:
            logger.debug("OCR недоступен (pytesseract не установлен)")
            return []
        except Exception as exc:
            logger.debug("OCR error: %s", exc)
            return []

    # ── pdfplumber (запасной) ─────────────────────────────────────────────────

    def _extract_with_plumber(self) -> List[str]:
        lines = []
        try:
            with pdfplumber.open(str(self.pdf_path)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        for raw_line in text.split("\n"):
                            cleaned = _clean_line(raw_line)
                            if cleaned:
                                lines.append(cleaned)
        except Exception as exc:
            logger.warning("pdfplumber error for %s: %s", self.pdf_path.name, exc)
        return lines

    # ── Публичный интерфейс ───────────────────────────────────────────────────

    def extract_lines(self) -> List[str]:
        lines = self._extract_with_fitz()

        if not lines:
            logger.info("Fallback to pdfplumber: %s", self.pdf_path.name)
            lines = self._extract_with_plumber()

        merged = _merge_broken_lines(lines)
        logger.info(
            "Extracted %d lines from '%s'",
            len(merged), self.pdf_path.name,
        )
        return merged

    def extract_formula_candidates(self) -> List[Dict]:
        lines = self.extract_lines()
        candidates = []
        seen_texts = set()

        def _add(text: str) -> None:
            text = _clean_russian_line(text).strip()
            if not text:
                return
            if text in seen_texts:
                return
            if not (MIN_FORMULA_LEN <= len(text) <= MAX_FORMULA_LEN):
                return
            seen_texts.add(text)
            candidates.append({
                "text": text,
                "source": self.source_name,
            })

        for line in lines:
            normalised = _clean_russian_line(line)

            if _is_formula_line(normalised):
                _add(normalised)
                continue

            for formula in extract_formulas_from_line(normalised):
                _add(formula)

        logger.info(
            "Found %d candidates in '%s'",
            len(candidates), self.pdf_path.name,
        )
        return candidates


# ─── Обработка всех PDF ───────────────────────────────────────────────────────

def lines_to_candidates(lines: List[str], source: str) -> List[Dict]:
    merged = _merge_broken_lines(lines)
    candidates = []
    seen = set()

    for line in merged:
        normalised = _clean_russian_line(line)

        def _add(text: str) -> None:
            text = text.strip()
            if text in seen:
                return
            if not (MIN_FORMULA_LEN <= len(text) <= MAX_FORMULA_LEN):
                return
            seen.add(text)
            candidates.append({"text": text, "source": source})

        if _is_formula_line(normalised):
            _add(normalised)
            continue

        for formula in extract_formulas_from_line(normalised):
            _add(formula)

    return candidates


def _process_single_pdf(pdf_path: Path) -> List[Dict]:
    """Process a single PDF file — used by both serial and parallel paths."""
    from check_pdf_type import check_pdf_type

    try:
        pdf_info = check_pdf_type(pdf_path)
        logger.info("Processing [%s]: %s", pdf_info["type"], pdf_path.name)

        if pdf_info["type"] == "DIGITAL":
            extractor = PDFExtractor(pdf_path)
            return extractor.extract_formula_candidates()

        elif pdf_info["type"] in ("SCAN", "MIXED"):
            try:
                from ocr_extractor import extract_with_ocr
                lines = extract_with_ocr(pdf_path)
                candidates = lines_to_candidates(lines, pdf_path.stem)
                logger.info(
                    "OCR extracted %d candidates from '%s'",
                    len(candidates), pdf_path.name,
                )
                return candidates
            except ImportError:
                logger.warning(
                    "pytesseract не установлен — fallback: %s",
                    pdf_path.name,
                )
                extractor = PDFExtractor(pdf_path)
                return extractor.extract_formula_candidates()
        else:
            return []

    except Exception as exc:
        logger.error("Ошибка обработки %s: %s", pdf_path.name, exc)
        return []


def extract_all_pdfs(docs_dir: Path = DOCS_DIR) -> List[Dict]:
    """
    Извлекает кандидатов из всех PDF в папке.
    Использует параллельную обработку если MAX_WORKERS > 1.
    """
    from check_pdf_type import check_pdf_type

    pdf_files = sorted(docs_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error("PDF файлы не найдены в %s", docs_dir)
        return []

    if MAX_WORKERS > 1 and len(pdf_files) > 1:
        all_candidates = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(_process_single_pdf, p): p
                for p in pdf_files
            }
            for future in as_completed(futures):
                pdf_path = futures[future]
                try:
                    candidates = future.result()
                    all_candidates.extend(candidates)
                except Exception as exc:
                    logger.error("Parallel error for %s: %s", pdf_path.name, exc)
    else:
        all_candidates = []
        for pdf_path in pdf_files:
            candidates = _process_single_pdf(pdf_path)
            all_candidates.extend(candidates)

    logger.info("Total candidates from all PDFs: %d", len(all_candidates))
    return all_candidates
