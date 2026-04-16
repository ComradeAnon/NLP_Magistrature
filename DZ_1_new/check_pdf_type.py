"""
Диагностика типа PDF-документа.

Определяет является ли документ:
  - DIGITAL : обычный PDF с текстовым слоем
  - SCAN    : сканированный документ (изображения)
  - MIXED   : часть страниц текст, часть сканы

Запуск для диагностики всех PDF в папке docs/:
    python check_pdf_type.py
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List

import fitz  # pymupdf

from config import (
    DOCS_DIR,
    SCAN_RATIO_THRESHOLD,
    MIXED_RATIO_THRESHOLD,
)

logger = logging.getLogger(__name__)


# ─── Основная функция диагностики ────────────────────────────────────────────

def check_pdf_type(pdf_path: Path) -> Dict:
    """
    Анализирует PDF и определяет его тип.

    Returns:
        Словарь с результатами диагностики:
        {
            "file"       : str   - имя файла
            "type"       : str   - DIGITAL / SCAN / MIXED
            "total_pages": int
            "text_pages" : int   - страницы с текстом
            "image_pages": int   - страницы только с картинками (сканы)
            "mixed_pages": int   - страницы с текстом и картинками
            "empty_pages": int
            "scan_ratio" : float - доля скан-страниц
            "needs_ocr"  : bool  - нужен ли OCR
        }
    """
    doc = fitz.open(str(pdf_path))

    total_pages  = len(doc)
    text_pages   = 0
    image_pages  = 0
    mixed_pages  = 0
    empty_pages  = 0

    for page in doc:
        text        = page.get_text().strip()
        image_list  = page.get_images()

        has_text    = len(text) > 30
        has_images  = len(image_list) > 0

        if has_text and has_images:
            mixed_pages += 1
        elif has_text:
            text_pages += 1
        elif has_images:
            image_pages += 1
        else:
            empty_pages += 1

    doc.close()

    scan_ratio = image_pages / total_pages if total_pages > 0 else 0

    if scan_ratio > SCAN_RATIO_THRESHOLD:
        doc_type = "SCAN"
    elif scan_ratio > MIXED_RATIO_THRESHOLD:
        doc_type = "MIXED"
    else:
        doc_type = "DIGITAL"

    return {
        "file":        pdf_path.name,
        "type":        doc_type,
        "total_pages": total_pages,
        "text_pages":  text_pages,
        "image_pages": image_pages,
        "mixed_pages": mixed_pages,
        "empty_pages": empty_pages,
        "scan_ratio":  round(scan_ratio, 2),
        "needs_ocr":   doc_type in ("SCAN", "MIXED"),
    }


def check_all_pdfs(docs_dir: Path = DOCS_DIR) -> List[Dict]:
    """Диагностирует все PDF в папке."""
    pdf_files = sorted(docs_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("PDF файлы не найдены в %s", docs_dir)
        return []

    results = []
    for pdf_path in pdf_files:
        try:
            info = check_pdf_type(pdf_path)
            results.append(info)
        except Exception as exc:
            logger.error("Ошибка диагностики %s: %s", pdf_path.name, exc)
            results.append({
                "file":      pdf_path.name,
                "type":      "ERROR",
                "needs_ocr": False,
                "error":     str(exc),
            })

    return results


# ─── CLI для ручной диагностики ───────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    docs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DOCS_DIR
    results  = check_all_pdfs(docs_dir)

    if not results:
        print(f"PDF файлы не найдены в папке: {docs_dir}")
        sys.exit(1)

    print("\n" + "=" * 75)
    print(f"  ДИАГНОСТИКА PDF | папка: {docs_dir}")
    print("=" * 75)
    print(f"  {'Файл':<40} {'Тип':<10} {'Стр':<6} {'Скан%':<8} {'OCR?'}")
    print("  " + "-" * 70)

    for r in results:
        if r.get("type") == "ERROR":
            print(f"  ❌ {r['file']:<40} ERROR")
            continue

        icon  = "⚠️ " if r["needs_ocr"] else "✅"
        ocr   = "ДА" if r["needs_ocr"] else "нет"
        print(
            f"  {icon} {r['file']:<40} "
            f"{r['type']:<10} "
            f"{r['total_pages']:<6} "
            f"{r['scan_ratio']*100:.0f}%{'':<4} "
            f"{ocr}"
        )

    needs_ocr = sum(1 for r in results if r.get("needs_ocr"))
    print("  " + "-" * 70)
    print(f"  Итого файлов: {len(results)}, нужен OCR: {needs_ocr}")
    print("=" * 75 + "\n")
