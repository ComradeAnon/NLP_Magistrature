"""
Специализированный OCR для математических формул.

Использует pix2tex — нейросеть которая конвертирует
изображение формулы напрямую в LaTeX.

Точнее tesseract для математических выражений.

Требования:
    pip install pix2tex
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

import fitz
from PIL import Image

logger = logging.getLogger(__name__)


# ─── Проверка доступности pix2tex ────────────────────────────────────────────

def _check_pix2tex() -> bool:
    try:
        from pix2tex.cli import LatexOCR  # noqa: F401
        return True
    except ImportError:
        return False


PIX2TEX_AVAILABLE = _check_pix2tex()

if not PIX2TEX_AVAILABLE:
    logger.warning(
        "pix2tex не установлен. FormulaOCR будет недоступен.\n"
        "Установите: pip install pix2tex"
    )


# ─── Класс FormulaOCR ─────────────────────────────────────────────────────────

class FormulaOCR:
    """
    Извлекает математические формулы из изображений
    страниц PDF и конвертирует их в LaTeX.
    """

    def __init__(self):
        self.model = None
        if PIX2TEX_AVAILABLE:
            self._load_model()

    def _load_model(self):
        """Загрузка модели pix2tex."""
        try:
            from pix2tex.cli import LatexOCR
            self.model = LatexOCR()
            logger.info("pix2tex модель загружена")
        except Exception as exc:
            logger.error("Ошибка загрузки pix2tex: %s", exc)
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    # ── Основные методы ───────────────────────────────────────────────────────

    def image_to_latex(self, img: Image.Image) -> Optional[str]:
        """
        Конвертирует изображение формулы в LaTeX.

        Returns:
            LaTeX строка или None при ошибке
        """
        if not self.available:
            return None
        try:
            latex = self.model(img)
            return latex if latex else None
        except Exception as exc:
            logger.debug("pix2tex error: %s", exc)
            return None

    def extract_from_page(self, page: fitz.Page) -> List[Dict]:
        """
        Ищет формулы на странице и конвертирует в LaTeX.

        Стратегия:
        1. Получаем текстовые блоки страницы
        2. Фильтруем похожие на формулы
        3. Вырезаем изображение каждого блока
        4. Применяем pix2tex

        Returns:
            Список {"text": str, "latex": str}
        """
        if not self.available:
            return []

        results = []
        blocks  = page.get_text("blocks")

        for block in blocks:
            x0, y0, x1, y1, text = block[:5]
            text = text.strip()

            if not text or not self._looks_like_formula(text):
                continue

            # Вырезаем изображение блока
            img = self._render_block(page, x0, y0, x1, y1)
            if img is None:
                continue

            # Конвертируем в LaTeX
            latex = self.image_to_latex(img)
            if latex:
                results.append({
                    "text":  text,
                    "latex": latex,
                })

        return results

    def extract_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Извлекает все формулы из PDF через pix2tex.

        Returns:
            Список {"text": str, "latex": str, "source": str}
        """
        if not self.available:
            logger.warning("pix2tex недоступен")
            return []

        results = []
        source  = pdf_path.stem

        try:
            doc = fitz.open(str(pdf_path))
            for page_num, page in enumerate(doc, 1):
                logger.debug("FormulaOCR страница %d", page_num)
                page_results = self.extract_from_page(page)
                for item in page_results:
                    item["source"] = source
                results.extend(page_results)
            doc.close()
        except Exception as exc:
            logger.error("FormulaOCR error for %s: %s", pdf_path.name, exc)

        logger.info(
            "FormulaOCR: %d формул из %s",
            len(results), pdf_path.name
        )
        return results

    # ── Вспомогательные методы ────────────────────────────────────────────────

    @staticmethod
    def _looks_like_formula(text: str) -> bool:
        """Быстрая эвристика: похоже ли на математическую формулу."""
        import re
        patterns = [
            # r"=\s*[-+]?\d",                          # = число
            r"[+\-*/^]{1,2}",                        # операторы
            r"\b(sin|cos|tan|tg|log|ln|sqrt|lim|sum)\b",
            r"\d+\s*/\s*\d+",                        # дробь
            r"[∑∫√π²³]",                             # спец символы
            r"\d+\s*!",                              # факториал
            r"\bC\s*[\(_]",                          # комбинаторика
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _render_block(
        page: fitz.Page,
        x0: float, y0: float,
        x1: float, y1: float,
        zoom: float = 3.0,
        padding: int = 8,
    ) -> Optional[Image.Image]:
        """Рендерит прямоугольный блок страницы как изображение."""
        try:
            rect   = fitz.Rect(x0 - padding, y0 - padding,
                               x1 + padding, y1 + padding)
            matrix = fitz.Matrix(zoom, zoom)
            pix    = page.get_pixmap(matrix=matrix, clip=rect)
            img    = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return img
        except Exception as exc:
            logger.debug("Block render error: %s", exc)
            return None
