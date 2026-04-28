"""
Сборщик датасета.
Работает через визуальный парсинг. Сохраняет сырой улов в pdf_raw_debug.json.
"""

import json
import logging
import random
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple

from config import (
    DOCS_DIR,
    DATASET_DIR,
    DATASET_FILE,
    DATASET_STATS_FILE,
    CLASS_KEYS,
    FORMULA_CLASSES,
    MIN_EXAMPLES_PER_CLASS,
)

from formula_ocr import FormulaOCR
from formula_detector import FormulaClassifier
from synthetic_generator import SyntheticGenerator

# Отключаем вывод INFO логов в консоль от самого python-logger, 
# оставляем только красивые принты
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class DatasetBuilder:
    def __init__(self, seed: int = 42):
        self.classifier = FormulaClassifier()
        self.generator  = SyntheticGenerator(seed=seed)
        
        print("\n[Инициализация] Загрузка нейросети...")
        self.vision_ocr = FormulaOCR()
        random.seed(seed)

    def collect_from_pdfs(self) -> List[Dict]:
        pdf_files = sorted(DOCS_DIR.glob("*.pdf"))
        all_candidates = []
        
        print("\n[Шаг 1] Визуальный поиск формул в PDF (OpenCV + pix2tex):")
        for pdf_path in pdf_files:
            print(f"  --> Сканирую {pdf_path.name} ... ", end="", flush=True)
            candidates = self.vision_ocr.extract_from_pdf(pdf_path)
            all_candidates.extend(candidates)
            print(f"Найдено {len(candidates)}")

        raw_file = DATASET_DIR / "pdf_raw_debug.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(all_candidates, f, ensure_ascii=False, indent=2)

        if not all_candidates:
            print("  ⚠️ В PDF формул не найдено. Датасет будет синтетическим.")
            return []

        classified = self.classifier.classify_batch(all_candidates)
        return classified

    def balance_dataset(self, items: List[Dict]) -> List[Dict]:
        by_class: Dict[str, List[Dict]] = defaultdict(list)
        for item in items:
            by_class[item["class"]].append(item)

        balanced = []
        print(f"\n[Шаг 2] Балансировка (цель: {MIN_EXAMPLES_PER_CLASS} на класс):")
        
        for cls in CLASS_KEYS:
            existing = by_class.get(cls, [])
            needed   = MIN_EXAMPLES_PER_CLASS

            if len(existing) >= needed:
                selected = random.sample(existing, needed)
                print(f"  {cls:<20} : {len(existing):>4} из PDF (взято {needed})")
            else:
                shortage  = needed - len(existing)
                synthetic = self.generator.generate(cls, count=shortage)
                selected  = list(existing) + synthetic
                print(f"  {cls:<20} : {len(existing):>4} из PDF + {shortage} синтетики")

            balanced.extend(selected)

        random.shuffle(balanced)
        return balanced

    def validate(self, items: List[Dict]) -> List[Dict]:
        required = {"text", "latex", "source", "class"}
        valid = []
        for item in items:
            if not required.issubset(item.keys()) or not all(str(item[f]).strip() for f in required):
                continue
            if item["class"] not in CLASS_KEYS:
                continue
            valid.append(item)
        return valid

    def save_dataset(self, items: List[Dict]) -> None:
        dataset = [{"text": i["text"], "latex": i["latex"], "source": i["source"], "class": i["class"]} for i in items]
        with open(DATASET_FILE, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

    def save_stats(self, items: List[Dict]) -> Dict:
        by_class, by_source = defaultdict(int), defaultdict(int)
        for item in items:
            by_class[item["class"]] += 1
            by_source[item["source"]] += 1

        stats = {
            "generated_at": datetime.now().isoformat(),
            "total": len(items),
            "by_class": dict(by_class),
            "by_source": dict(by_source),
            "class_descriptions": FORMULA_CLASSES,
        }
        with open(DATASET_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        return stats

    def build(self) -> Tuple[List[Dict], Dict]:
        pdf_items = self.collect_from_pdfs()
        balanced = self.balance_dataset(pdf_items)
        valid = self.validate(balanced)
        self.save_dataset(valid)
        stats = self.save_stats(valid)
        return valid, stats
