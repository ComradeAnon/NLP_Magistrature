"""
Центральная конфигурация проекта.
Все пути, константы и параметры в одном месте.
"""

from pathlib import Path

# ─── Пути ─────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
DOCS_DIR    = BASE_DIR / "docs"
DATASET_DIR = BASE_DIR / "dataset"
LOGS_DIR    = BASE_DIR / "logs"

DATASET_FILE       = DATASET_DIR / "dataset.json"
DATASET_STATS_FILE = DATASET_DIR / "dataset_stats.json"

# Создаём директории если не существуют
for _dir in (DOCS_DIR, DATASET_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Параметры датасета ───────────────────────────────────────────────────────

# Минимум примеров на класс (5 классов × 40 = 200 минимум)
MIN_EXAMPLES_PER_CLASS = 40

FORMULA_CLASSES = {
    "algebraic":     "Алгебраические выражения (полиномы, дроби, корни)",
    "trigonometric": "Тригонометрические выражения (sin, cos, tg, ctg)",
    "logarithmic":   "Логарифмические выражения (log, ln, lg)",
    "combinatorial": "Комбинаторные выражения (C, P, A, факториал)",
    "series_limit":  "Суммы рядов и пределы (Σ, lim, интеграл = число)",
}

CLASS_KEYS = list(FORMULA_CLASSES.keys())

# ─── Параметры парсинга ───────────────────────────────────────────────────────

MIN_FORMULA_LEN = 3
MAX_FORMULA_LEN = 300

# Порог уверенности классификатора (0..1)
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.5

# ─── Параметры OCR ───────────────────────────────────────────────────────────

# Язык OCR (tesseract)
OCR_LANG = "rus+eng"

# DPI для рендеринга страниц
OCR_DPI = 300

# Минимум символов на странице чтобы не запускать OCR
OCR_TEXT_THRESHOLD = 50

# Порог доли скан-страниц для определения типа документа
SCAN_RATIO_THRESHOLD  = 0.8   # > 80% картинок → SCAN
MIXED_RATIO_THRESHOLD = 0.3   # > 30% картинок → MIXED
