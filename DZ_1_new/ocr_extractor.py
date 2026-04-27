import logging
from pathlib import Path
from typing import List

import fitz
from PIL import Image, ImageFilter, ImageEnhance

from config import OCR_LANG, OCR_DPI, OCR_TEXT_THRESHOLD

logger = logging.getLogger(__name__)

# Путь к tesseract для Windows (раскомментируйте и исправьте если нужно)
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r"D:\Programs\Tesseract-OCR\tesseract.exe"


# ─── Проверка доступности ─────────────────────────────────────────────────────

def _check_tesseract() -> bool:
    """Проверяет доступность tesseract."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


TESSERACT_AVAILABLE = _check_tesseract()

if not TESSERACT_AVAILABLE:
    logger.warning(
        "Tesseract не найден. OCR будет недоступен.\n"
        "Установите: https://github.com/UB-Mannheim/tesseract/wiki"
    )


# ─── Работа с изображениями ───────────────────────────────────────────────────

def pdf_page_to_image(page: fitz.Page, dpi: int = OCR_DPI) -> Image.Image:
    """
    Конвертирует страницу PDF в изображение.
    300 DPI — минимум для нормального OCR.
    """
    zoom   = dpi / 72      # 72 — стандартный DPI в PDF
    matrix = fitz.Matrix(zoom, zoom)
    pix    = page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY)
    img    = Image.frombytes("L", [pix.width, pix.height], pix.samples)
    return img


def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Предобработка изображения для улучшения OCR.

    Применяет:
    - повышение резкости
    - повышение контраста
    - бинаризацию
    """
    # Повышаем резкость
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.SHARPEN)  # двойная резкость

    # Повышаем контраст
    enhancer = ImageEnhance.Contrast(img)
    img      = enhancer.enhance(2.0)

    # Бинаризация — помогает tesseract
    img = img.point(lambda x: 0 if x < 140 else 255, "1")
    img = img.convert("L")

    return img


# ─── OCR функции ─────────────────────────────────────────────────────────────

def ocr_page(page: fitz.Page, lang: str = OCR_LANG) -> str:
    """
    OCR одной страницы PDF.

    Args:
        page : страница pymupdf
        lang : язык(и) для tesseract, например "rus+eng"

    Returns:
        Распознанный текст страницы
    """
    if not TESSERACT_AVAILABLE:
        return ""

    import pytesseract

    img = pdf_page_to_image(page, dpi=OCR_DPI)
    img = preprocess_image(img)

    # Конфигурация tesseract:
    # --psm 6 : блок текста
    # --oem 3 : LSTM + legacy движок
    config = "--psm 6 --oem 3"

    try:
        text = pytesseract.image_to_string(img, lang=lang, config=config)
        return text
    except Exception as exc:
        logger.warning("Tesseract error: %s", exc)
        return ""


def extract_with_ocr(pdf_path: Path, lang: str = OCR_LANG) -> List[str]:
    """
    Полное OCR-извлечение из PDF.

    Стратегия:
    - Если страница содержит текстовый слой → берём текст напрямую
    - Если страница — скан → применяем OCR

    Returns:
        Список строк из всего документа
    """
    if not TESSERACT_AVAILABLE:
        logger.error("Tesseract недоступен, OCR невозможен")
        return []

    lines = []
    doc   = fitz.open(str(pdf_path))

    logger.info("OCR: %s (%d страниц)", pdf_path.name, len(doc))

    for page_num, page in enumerate(doc, 1):
        page_text = page.get_text().strip()

        if len(page_text) >= OCR_TEXT_THRESHOLD:
            # Страница с текстовым слоем
            for line in page_text.split("\n"):
                line = line.strip()
                if line:
                    lines.append(line)
            logger.debug("Страница %d: текстовый слой", page_num)
        else:
            # Скан-страница
            logger.debug("Страница %d: OCR", page_num)
            ocr_text = ocr_page(page, lang=lang)
            for line in ocr_text.split("\n"):
                line = line.strip()
                if line:
                    lines.append(line)

    doc.close()
    logger.info("OCR завершён: %d строк из %s", len(lines), pdf_path.name)
    return lines
