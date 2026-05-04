"""
Конфигурация проекта.
"""
from pathlib import Path

BASE_DIR    = Path(__file__).parent
DOCS_DIR    = BASE_DIR / "docs"
DATASET_DIR = BASE_DIR / "dataset"
LOGS_DIR    = BASE_DIR / "logs"

DPI_ZOOM = 2.125 # Проверено опытным путём

DATASET_FILE       = DATASET_DIR / "dataset.json"
DATASET_STATS_FILE = DATASET_DIR / "dataset_stats.json"
OUTPUT_FILE        = DATASET_DIR / "output.json"

# Создаём директории если не Существуют
for _dir in (DOCS_DIR, DATASET_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ─── Параметры датасета ───────────────────────────────────────────────────────

# Минимум примеров на класс (5 классов × 100 = 500 минимум)
MIN_EXAMPLES_PER_CLASS = 100

# ─── НОВЫЕ КЛАССЫ ПО ОПЕРАТОРАМ ──────────────────────────────────────────────
FORMULA_CLASSES = {
    "integral":         "Интегралы (\\int, \\oint)",
    "summation":        "Суммы и произведения (\\sum, \\prod)",
    "derivative_limit": "Производные и пределы (\\partial, \\nabla, \\lim)",
    "equation":         "Базовые уравнения/неравенства (=, \\le, \\ge)",
    "mixed":            "Смешанные формулы (содержат >= 2 разных главных операторов)",
}

CLASS_KEYS = list(FORMULA_CLASSES.keys())

# ─── Режим детекции формул ─────────────────────────────────────────────────────

FORMULA_MODE = "strict"  # "strict" | "extended"

# ─── Параметры парсинга ───────────────────────────────────────────────────────

MIN_FORMULA_LEN = 3
MAX_FORMULA_LEN = 500

# Порог уверенности классификатора (0..1)
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.5

# Пороги уверенности по классам (если не указан — используется общий)
CLASS_THRESHOLDS = {
    "algebraic":     0.4,
    "trigonometric": 0.5,
    "logarithmic":   0.5,
    "combinatorial": 0.5,
    "series_limit":  0.5,
}

# Порог плотности математических символов (0..1)
MATH_DENSITY_THRESHOLD = 0.25

# ─── Параллельная обработка ────────────────────────────────────────────────────

MAX_WORKERS = 4  # количество процессов для параллельной обработки PDF

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

# ─── Режим сборки датасета ────────────────────────────────────────────────────

BUILD_BALANCED_DATASET = False  # False = сохранить все формулы в output.json
                                # True  = балансировать и сохранить в dataset.json

OUTPUT_FILE = DATASET_DIR / "output.json"