"""
Конфигурация проекта.
"""
from pathlib import Path

BASE_DIR    = Path(__file__).parent
DOCS_DIR    = BASE_DIR / "docs"
DATASET_DIR = BASE_DIR / "dataset"
LOGS_DIR    = BASE_DIR / "logs"

DATASET_FILE       = DATASET_DIR / "dataset.json"
DATASET_STATS_FILE = DATASET_DIR / "dataset_stats.json"

for _dir in (DOCS_DIR, DATASET_DIR, LOGS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

<<<<<<< HEAD
# ─── Параметры датасета ───────────────────────────────────────────────────────

# Минимум примеров на класс (5 классов × 40 = 200 минимум)
MIN_EXAMPLES_PER_CLASS = 40
=======
MIN_EXAMPLES_PER_CLASS = 100
>>>>>>> main

# ─── НОВЫЕ КЛАССЫ ПО ОПЕРАТОРАМ ──────────────────────────────────────────────
FORMULA_CLASSES = {
    "integral":         "Интегралы (\\int, \\oint)",
    "summation":        "Суммы и произведения (\\sum, \\prod)",
    "derivative_limit": "Производные и пределы (\\partial, \\nabla, \\lim)",
    "equation":         "Базовые уравнения/неравенства (=, \\le, \\ge)",
    "mixed":            "Смешанные формулы (содержат >= 2 разных главных операторов)",
}

CLASS_KEYS = list(FORMULA_CLASSES.keys())
