"""
Сборщик датасета.

Полный пайплайн:
  PDF → кандидаты → классификация → LaTeX → балансировка → датасет
"""

import json
import logging
import random
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple

from config import (
    DOCS_DIR,
    DATASET_FILE,
    DATASET_STATS_FILE,
    CLASS_KEYS,
    FORMULA_CLASSES,
    MIN_EXAMPLES_PER_CLASS,
)
from pdf_extractor import extract_all_pdfs
from formula_detector import FormulaClassifier
from latex_converter import LaTeXConverter
from synthetic_generator import SyntheticGenerator

logger = logging.getLogger(__name__)


class DatasetBuilder:

    def __init__(self, seed: int = 42):
        self.classifier = FormulaClassifier()
        self.converter  = LaTeXConverter()
        self.generator  = SyntheticGenerator(seed=seed)
        self.seed       = seed
        random.seed(seed)

    # ── Шаг 1: Извлечение из PDF ──────────────────────────────────────────────

    def collect_from_pdfs(self) -> List[Dict]:
        """Извлекает и классифицирует формулы из всех PDF."""
        logger.info("Шаг 1: Извлечение из PDF")

        candidates = extract_all_pdfs(DOCS_DIR)
        if not candidates:
            logger.warning("Кандидаты из PDF не найдены")
            return []

        classified = self.classifier.classify_batch(candidates)

        # Конвертируем в LaTeX
        for item in classified:
            item["latex"] = self.converter.convert(item["text"])

        # Статистика
        by_class = defaultdict(int)
        for item in classified:
            by_class[item["class"]] += 1

        logger.info("Из PDF получено: %d формул", len(classified))
        for cls, cnt in by_class.items():
            logger.info("  %-20s : %d", cls, cnt)

        return classified

    # ── Шаг 2: Балансировка ──────────────────────────────────────────────────

    def balance_dataset(self, items: List[Dict]) -> List[Dict]:
        """
        Балансирует датасет.

        Если класса не хватает — добавляет синтетические примеры.
        Если класса в избытке — делает случайную выборку.
        """
        logger.info("Шаг 2: Балансировка (цель: %d на класс)", MIN_EXAMPLES_PER_CLASS)

        by_class: Dict[str, List[Dict]] = defaultdict(list)
        for item in items:
            by_class[item["class"]].append(item)

        balanced = []

        for cls in CLASS_KEYS:
            existing = by_class.get(cls, [])
            needed   = MIN_EXAMPLES_PER_CLASS

            if len(existing) >= needed:
                selected = random.sample(existing, needed)
                logger.info(
                    "  %-20s : %d реальных (выбрано %d)",
                    cls, len(existing), needed
                )
            else:
                shortage  = needed - len(existing)
                synthetic = self.generator.generate(cls, count=shortage)
                selected  = list(existing) + synthetic
                logger.info(
                    "  %-20s : %d реальных + %d синтетических",
                    cls, len(existing), shortage
                )

            balanced.extend(selected)

        random.shuffle(balanced)
        logger.info("Итого после балансировки: %d", len(balanced))
        return balanced

    # ── Шаг 3: Валидация ─────────────────────────────────────────────────────

    def validate(self, items: List[Dict]) -> List[Dict]:
        """Проверяет обязательные поля и корректность данных."""
        logger.info("Шаг 3: Валидация")

        required = {"text", "latex", "source", "class"}
        valid    = []
        errors   = 0

        for i, item in enumerate(items):
            # Наличие полей
            if not required.issubset(item.keys()):
                logger.debug("Элемент %d: отсутствуют поля", i)
                errors += 1
                continue

            # Непустые значения
            if not all(str(item[f]).strip() for f in required):
                logger.debug("Элемент %d: пустые поля", i)
                errors += 1
                continue

            # Корректный класс
            if item["class"] not in CLASS_KEYS:
                logger.debug("Элемент %d: неизвестный класс %s", i, item["class"])
                errors += 1
                continue

            valid.append(item)

        logger.info(
            "Валидация: %d прошли / %d отклонено",
            len(valid), errors
        )
        return valid

    # ── Шаг 4: Сохранение ────────────────────────────────────────────────────

    def save_dataset(self, items: List[Dict]) -> None:
        """Сохраняет датасет в JSON."""
        dataset = [
            {
                "text":   item["text"],
                "latex":  item["latex"],
                "source": item["source"],
                "class":  item["class"],
            }
            for item in items
        ]

        with open(DATASET_FILE, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        logger.info("Датасет сохранён: %s (%d элементов)", DATASET_FILE, len(dataset))

    def save_stats(self, items: List[Dict]) -> Dict:
        """Считает и сохраняет статистику."""
        by_class  = defaultdict(int)
        by_source = defaultdict(int)

        for item in items:
            by_class[item["class"]]   += 1
            by_source[item["source"]] += 1

        stats = {
            "generated_at":       datetime.now().isoformat(),
            "total":              len(items),
            "by_class":           dict(by_class),
            "by_source":          dict(by_source),
            "class_descriptions": FORMULA_CLASSES,
        }

        with open(DATASET_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info("Статистика сохранена: %s", DATASET_STATS_FILE)
        return stats

    # ── Основной пайплайн ────────────────────────────────────────────────────

    def build(self) -> Tuple[List[Dict], Dict]:
        """
        Запускает полный пайплайн сборки датасета.

        Returns:
            (dataset_items, stats_dict)
        """
        logger.info("=" * 60)
        logger.info("НАЧАЛО СБОРКИ ДАТАСЕТА")
        logger.info("=" * 60)

        # 1. PDF
        pdf_items = self.collect_from_pdfs()

        # 2. Балансировка
        balanced = self.balance_dataset(pdf_items)

        # 3. Валидация
        valid = self.validate(balanced)

        # 4. Сохранение
        self.save_dataset(valid)
        stats = self.save_stats(valid)

        logger.info("=" * 60)
        logger.info("СБОРКА ЗАВЕРШЕНА | Итого: %d примеров", stats["total"])
        logger.info("=" * 60)

        return valid, stats
