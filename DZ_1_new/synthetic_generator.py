"""
Синтетический генератор формул по главным операторам.
"""

import random
import logging
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)
SOURCE_SYNTHETIC = "synthetic"

class SyntheticGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._generators: Dict[str, List[Callable]] = {
            "integral":         [self._gen_integral],
            "summation":        [self._gen_summation],
            "derivative_limit": [self._gen_derivative, self._gen_limit],
            "equation":         [self._gen_equation, self._gen_inequality],
            "mixed":            [self._gen_mixed_1, self._gen_mixed_2],
        }

    def generate(self, class_name: str, count: int) -> List[Dict]:
        if class_name not in self._generators:
            return []
        
        gen_list = self._generators[class_name]
        results  = []
        for i in range(count):
            gen_func = gen_list[i % len(gen_list)]
            item = gen_func()
            item["class"] = class_name
            results.append(item)
        return results

    def _gen_integral(self) -> Dict:
        a = random.randint(0, 5)
        b = random.randint(6, 10)
        return {"text": f"Интеграл от {a} до {b}", "latex": rf"\int_{{{a}}}^{{{b}}} f(x) dx = 0", "source": SOURCE_SYNTHETIC}

    def _gen_summation(self) -> Dict:
        n = random.randint(10, 100)
        return {"text": f"Сумма до {n}", "latex": rf"\sum_{{i=1}}^{{{n}}} x_i = Y", "source": SOURCE_SYNTHETIC}

    def _gen_derivative(self) -> Dict:
        return {"text": "Производная функции", "latex": r"\frac{\partial f}{\partial x} + \frac{\partial f}{\partial y} = 0", "source": SOURCE_SYNTHETIC}

    def _gen_limit(self) -> Dict:
        return {"text": "Предел на бесконечности", "latex": r"\lim_{n \to \infty} \left( 1 + \frac{1}{n} \right)^n = e", "source": SOURCE_SYNTHETIC}

    def _gen_equation(self) -> Dict:
        return {"text": "Квадратное уравнение", "latex": r"ax^2 + bx + c = 0", "source": SOURCE_SYNTHETIC}

    def _gen_inequality(self) -> Dict:
        return {"text": "Неравенство", "latex": r"x^2 + y^2 \le R^2", "source": SOURCE_SYNTHETIC}

    def _gen_mixed_1(self) -> Dict:
        return {"text": "Предел интеграла", "latex": r"\lim_{t \to 0} \int_0^t f(x) dx = 0", "source": SOURCE_SYNTHETIC}

    def _gen_mixed_2(self) -> Dict:
        return {"text": "Интеграл суммы", "latex": r"\int_{\Omega} \sum_{i=1}^n x_i d\mu \ge 0", "source": SOURCE_SYNTHETIC}
