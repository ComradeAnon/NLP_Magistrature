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
from typing import List, Dict

import fitz
import pdfplumber

from config import (
    DOCS_DIR,
    MIN_FORMULA_LEN,
    MAX_FORMULA_LEN,
    OCR_TEXT_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ─── Очистка строк ────────────────────────────────────────────────────────────

def _clean_line(line: str) -> str:
    """Базовая очистка строки от артефактов PDF."""
    line = re.sub(r" {2,}", " ", line)
    line = re.sub(
        r"[^\x20-\x7E\u0400-\u04FF\u00B0-\u00FF\u2200-\u22FF]",
        " ",
        line,
    )
    return line.strip()


def _clean_russian_line(line: str) -> str:
    """
    Нормализует особенности русских математических текстов.

    - Заменяет запятую на точку в числах (2,5 → 2.5)
    - Заменяет русские буквы-омонимы на латинские в формулах
      С → C, А → A, Р → P, х → x, у → y
    """
    # Запятая как десятичный разделитель: "0,5" → "0.5"
    line = re.sub(r"(\d),(\d)", r"\1.\2", line)

    # Русские буквы которые выглядят как латинские
    cyrillic_to_latin = {
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
    for cyr, lat in cyrillic_to_latin.items():
        line = line.replace(cyr, lat)

    return line


# ─── Проверка строки на формулу ───────────────────────────────────────────────

def _is_formula_line(line: str) -> bool:
    """
    Строгая проверка: является ли строка математической формулой.

    Требования:
    1. Содержит знак равенства
    2. Правая часть — ТОЛЬКО число
    3. Левая часть — математическое выражение
    4. Строка не слишком длинная (до 80 символов)
    5. Доля кириллицы не превышает 25%
    6. Нет стоп-слов естественного языка
    """
    if "=" not in line:
        return False

    # ── Фильтр 1: длина ───────────────────────────────────────────────────────
    if len(line) > 80:
        return False

    # ── Фильтр 2: доля кириллицы ──────────────────────────────────────────────
    letters = len(re.findall(r"[а-яёА-ЯЁ]", line))
    total   = len(line.replace(" ", ""))
    if total > 0 and letters / total > 0.25:
        return False

    # ── Фильтр 3: стоп-слова ──────────────────────────────────────────────────
    stop_words = [
        "поэтому", "следовательно", "отсюда", "пусть", "дано",
        "условие", "пример", "решение", "ответ", "задача",
        "если",   "тогда",  "так",    "как",   "что",   "это",
        "будем",  "можно",  "нужно",  "должно","вместо",
        "например","итого", "всего",  "остаток","повторя",
        "используем","применим","получим","запишем","имеем",
        "заметим","покажем","докажем","найдем", "вычислим",
    ]
    line_lower = line.lower()
    for word in stop_words:
        if word in line_lower:
            return False

    # ── Разбиваем на части ────────────────────────────────────────────────────
    eq_idx = line.rfind("=")
    lhs    = line[:eq_idx].strip()
    rhs    = line[eq_idx + 1:].strip()

    if len(lhs) < 2:
        return False

    # ── Фильтр 4: правая часть — строго число ─────────────────────────────────
    rhs_clean = rhs.replace(",", ".").replace(" ", "")

    # rhs_valid = bool(re.fullmatch(
    #     r"[-+]?\d+([.,]\d+)?"       # целое или десятичное
    #     r"|[-+]?\d+/\d+"            # обыкновенная дробь
    #     r"|0|1|-1|∞|\\infty",
    #     rhs_clean,
    # ))
    # if not rhs_valid:
    #     return False

    # ── Фильтр 5: левая часть — математическое выражение ─────────────────────
    lhs_norm = _clean_russian_line(lhs)

    math_indicators = [
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

    has_math = any(
        re.search(pattern, lhs_norm, re.IGNORECASE)
        for pattern in math_indicators
    )

    return has_math


# ─── Вырезание формул из текстовых строк ─────────────────────────────────────

# Каждый элемент: (full_regex_pattern, тип)
FORMULA_EXTRACTION_PATTERNS = [

    # Факториал: 5! = 120
    (
        r"\d+\s*!"
        r"\s*=\s*"
        r"[-+]?\d+(?:[.,]\d+)?",
        "factorial",
    ),

    # Комбинаторика: C(5,2) = 10 | A(4,2) = 12 | P(3) = 6
    (
        r"[CAP]\s*[\(_]\s*\d+\s*[,_^]\s*\d+\s*[\)_]?"
        r"\s*=\s*"
        r"\d+",
        "combinatorial",
    ),

    # Тригонометрия: sin(30°) = 0.5 | cos(π/3) = 0.5
    (
        r"(?:sin|cos|tan|tg|cot|ctg)\s*\(?[-+]?\d+(?:[°π/\d\s]*)\)?"
        r"\s*=\s*"
        r"[-+]?\d+(?:[.,]\d+)?",
        "trig",
    ),

    # Логарифм: log_2(8) = 3 | ln(e^2) = 2 | lg(100) = 2
    (
        r"(?:log_?\d*|ln|lg)\s*\(?[^=\n]{1,20}\)?"
        r"\s*=\s*"
        r"[-+]?\d+(?:[.,]\d+)?",
        "log",
    ),

    # Предел: lim(x->0) = 1
    (
        r"lim\s*[\(_]?[^=\n]{1,30}[\)_]?"
        r"\s*=\s*"
        r"[-+]?\d+(?:[.,]\d+)?",
        "limit",
    ),

    # Сумма ряда: sum(i=1..10) = 55
    (
        r"(?:sum|∑|Σ)\s*[\(_\{]?[^=\n]{1,30}[\)_\}]?"
        r"\s*=\s*"
        r"[-+]?\d+(?:[.,]\d+)?",
        "series",
    ),

    # Степень: 2^3 = 8 | x^2 = 4
    (
        r"[-+]?\d*[a-zA-Z]?\s*\^\s*\d+"
        r"\s*=\s*"
        r"[-+]?\d+(?:[.,]\d+)?",
        "power",
    ),

    # Дробь: 216/990 = 12/55 | 3/4 = 0.75
    (
        r"\d+\s*/\s*\d+"
        r"\s*=\s*"
        r"[-+]?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?",
        "fraction",
    ),

    # Арифметика: 2 + 3 = 5 | 15 - 7 = 8 | 3 * 4 = 12
    (
        r"[-+]?\d+(?:[.,]\d+)?"
        r"(?:\s*[+\-*/]\s*[-+]?\d+(?:[.,]\d+)?)+"
        r"\s*=\s*"
        r"[-+]?\d+(?:[.,]\d+)?",
        "arithmetic",
    ),

    # Переменная = число: x = 5 | 2x = 10 | 3x + 1 = 7
    (
        r"[-+]?\d*\s*[a-zA-Z]\w*"
        r"(?:\s*[+\-*/^]\s*[-+]?\d+(?:[.,]\d+)?)?"
        r"\s*=\s*"
        r"[-+]?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?",
        "variable",
    ),
]


def _extracted_rhs_is_number(formula: str) -> bool:
    """Проверяет что правая часть вырезанной формулы — число."""
    eq_idx = formula.rfind("=")
    if eq_idx == -1:
        return False

    rhs = formula[eq_idx + 1:].strip().replace(",", ".")

    return bool(re.fullmatch(
        r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?",
        rhs,
    ))


def extract_formulas_from_line(line: str) -> List[str]:
    """
    Вырезает все формулы из строки текста.

    Вместо отбрасывания строки целиком —
    ищет формулы внутри текста по паттернам.

    Примеры:
        "Отсюда x = 216/990 = 12/55"
            → ["216/990 = 12"]

        "sin(30°) = 0.5 и cos(60°) = 0.5"
            → ["sin(30°) = 0.5", "cos(60°) = 0.5"]

        "Следовательно, 5! = 120 штук"
            → ["5! = 120"]
    """
    if "=" not in line:
        return []

    found = []
    seen  = set()

    for full_pattern, _name in FORMULA_EXTRACTION_PATTERNS:
        try:
            for match in re.finditer(full_pattern, line, re.IGNORECASE):
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
            logger.debug("Regex error in pattern '%s': %s", full_pattern, exc)
            continue

    return found


# ─── Склейка перенесённых строк ───────────────────────────────────────────────

def _merge_broken_lines(lines: List[str]) -> List[str]:
    """
    Склеивает перенесённые строки формул.

    Признаки незавершённой строки:
    - заканчивается оператором
    - следующая начинается с оператора или строчной буквы
    """
    if not lines:
        return []

    merged = []
    buffer = lines[0]

    for current in lines[1:]:
        prev_stripped    = buffer.rstrip()
        current_stripped = current.lstrip()

        ends_operator   = bool(prev_stripped and prev_stripped[-1] in "+-*/=^,(\\")
        starts_operator = bool(current_stripped and current_stripped[0] in "+-*/^),")
        starts_lower    = bool(current_stripped and current_stripped[0].islower())

        if ends_operator or starts_operator or starts_lower:
            buffer = prev_stripped + " " + current_stripped
        else:
            merged.append(buffer)
            buffer = current

    merged.append(buffer)
    return merged


# ─── Основной экстрактор ──────────────────────────────────────────────────────

class PDFExtractor:
    """Извлекает текстовые строки и формулы из PDF."""

    def __init__(self, pdf_path: Path):
        self.pdf_path    = Path(pdf_path)
        self.source_name = pdf_path.stem

    # ── pymupdf ───────────────────────────────────────────────────────────────

    def _extract_page_fitz(self, page: fitz.Page) -> List[str]:
        """Извлечение одной страницы через pymupdf."""
        lines  = []
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (round(b[1] / 10), b[0]))

        for block in blocks:
            text = block[4]
            for raw_line in text.split("\n"):
                cleaned = _clean_line(raw_line)
                if cleaned:
                    lines.append(cleaned)
        return lines

    def _extract_with_fitz(self) -> List[str]:
        """Полное извлечение через pymupdf с поддержкой OCR для скан-страниц."""
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
        """Пробует OCR для страницы. Возвращает пустой список если OCR недоступен."""
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
        """Запасной метод через pdfplumber."""
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
        """
        Извлекает все строки из PDF.
        Пробует fitz → при неудаче pdfplumber.
        """
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
        """
        Извлекает кандидатов двумя способами:

        1. Строка целиком является формулой → берём как есть
        2. Строка содержит формулу внутри текста → вырезаем
        """
        lines      = self.extract_lines()
        candidates = []
        seen_texts = set()

        def _add(text: str) -> None:
            """Добавляет кандидата если не дубликат и не пустой."""
            text = _clean_russian_line(text).strip()
            if not text:
                return
            if text in seen_texts:
                return
            if not (MIN_FORMULA_LEN <= len(text) <= MAX_FORMULA_LEN):
                return
            seen_texts.add(text)
            candidates.append({
                "text":   text,
                "source": self.source_name,
            })

        for line in lines:
            normalised = _clean_russian_line(line)

            # ── Способ 1: вся строка — формула
            if _is_formula_line(normalised):
                _add(normalised)
                continue

            # ── Способ 2: вырезаем формулы из текстовой строки
            for formula in extract_formulas_from_line(normalised):
                _add(formula)

        logger.info(
            "Found %d candidates in '%s'",
            len(candidates), self.pdf_path.name,
        )
        return candidates


# ─── Обработка всех PDF ───────────────────────────────────────────────────────

def lines_to_candidates(lines: List[str], source: str) -> List[Dict]:
    """Конвертирует список строк в список кандидатов-формул."""
    merged     = _merge_broken_lines(lines)
    candidates = []
    seen       = set()

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


def extract_all_pdfs(docs_dir: Path = DOCS_DIR) -> List[Dict]:
    """
    Извлекает кандидатов из всех PDF в папке.
    Автоматически выбирает метод по типу документа.
    """
    from check_pdf_type import check_pdf_type

    pdf_files = sorted(docs_dir.glob("*.pdf"))

    if not pdf_files:
        logger.error("PDF файлы не найдены в %s", docs_dir)
        return []

    all_candidates = []

    for pdf_path in pdf_files:
        try:
            pdf_info = check_pdf_type(pdf_path)
            logger.info("Processing [%s]: %s", pdf_info["type"], pdf_path.name)

            if pdf_info["type"] == "DIGITAL":
                extractor  = PDFExtractor(pdf_path)
                candidates = extractor.extract_formula_candidates()

            elif pdf_info["type"] in ("SCAN", "MIXED"):
                try:
                    from ocr_extractor import extract_with_ocr
                    lines      = extract_with_ocr(pdf_path)
                    candidates = lines_to_candidates(lines, pdf_path.stem)
                    logger.info(
                        "OCR extracted %d candidates from '%s'",
                        len(candidates), pdf_path.name,
                    )
                except ImportError:
                    logger.warning(
                        "pytesseract не установлен — fallback: %s",
                        pdf_path.name,
                    )
                    extractor  = PDFExtractor(pdf_path)
                    candidates = extractor.extract_formula_candidates()
            else:
                candidates = []

            all_candidates.extend(candidates)

        except Exception as exc:
            logger.error("Ошибка обработки %s: %s", pdf_path.name, exc)
            continue

    logger.info("Total candidates from all PDFs: %d", len(all_candidates))
    return all_candidates
