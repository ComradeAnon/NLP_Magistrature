"""
Генератор синтетических примеров формул.

Используется для балансировки датасета когда
реальных примеров из PDF недостаточно.

Все генерируемые формулы математически корректны —
значения вычисляются программно.
"""

import random
import logging
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

SOURCE_SYNTHETIC = "synthetic"


class SyntheticGenerator:
    """Генерирует синтетические математические формулы."""

    def __init__(self, seed: int = 42):
        random.seed(seed)

        # Реестр генераторов по классам
        self._generators: Dict[str, List[Callable]] = {
            "algebraic":     [
                self._alg_arithmetic,
                self._alg_polynomial,
                self._alg_fraction,
                self._alg_root,
                self._alg_power,
                self._alg_mixed,
            ],
            "trigonometric": [
                self._trig_sin,
                self._trig_cos,
                self._trig_tan,
                self._trig_identity_pythag,
                self._trig_double_angle,
            ],
            "logarithmic":   [
                self._log_natural,
                self._log_base10,
                self._log_base2,
                self._log_arbitrary,
                self._log_property,
            ],
            "combinatorial": [
                self._comb_binomial,
                self._comb_factorial,
                self._comb_arrangement,
                self._comb_permutation,
                self._comb_binomial_sum,
            ],
            "series_limit":  [
                self._ser_arithmetic_sum,
                self._ser_geometric_sum,
                self._ser_squares_sum,
                self._lim_simple,
                self._lim_over_n,
            ],
        }

    # ── Публичный интерфейс ───────────────────────────────────────────────────

    def generate(self, class_name: str, count: int) -> List[Dict]:
        """
        Генерирует `count` синтетических примеров для класса.

        Returns:
            Список {"text": str, "latex": str, "source": str, "class": str}
        """
        if class_name not in self._generators:
            logger.error("Неизвестный класс: %s", class_name)
            return []

        gen_list = self._generators[class_name]
        results  = []

        for i in range(count):
            # Циклически перебираем генераторы для разнообразия
            gen_func = gen_list[i % len(gen_list)]
            try:
                item          = gen_func()
                item["class"] = class_name
                results.append(item)
            except Exception as exc:
                logger.debug("Генератор %s ошибка: %s", gen_func.__name__, exc)
                # Запасной вариант — простая арифметика
                results.append(self._alg_arithmetic())

        logger.info("Сгенерировано %d примеров для класса '%s'", count, class_name)
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # АЛГЕБРА
    # ─────────────────────────────────────────────────────────────────────────

    def _alg_arithmetic(self) -> Dict:
        """Базовые арифметические операции."""
        a  = random.randint(2, 99)
        b  = random.randint(2, 99)
        op = random.choice(["+", "-", "*"])

        if op == "+":
            result = a + b
            text   = f"{a} + {b} = {result}"
            latex  = f"{a} + {b} = {result}"
        elif op == "-":
            a, b   = max(a, b), min(a, b)
            result = a - b
            text   = f"{a} - {b} = {result}"
            latex  = f"{a} - {b} = {result}"
        else:
            result = a * b
            text   = f"{a} * {b} = {result}"
            latex  = f"{a} \\cdot {b} = {result}"

        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _alg_polynomial(self) -> Dict:
        """Вычисление полинома."""
        a   = random.randint(1, 10)
        b   = random.randint(-10, 10)
        c   = random.randint(-20, 20)
        x   = random.randint(1, 5)
        res = a * x ** 2 + b * x + c

        text  = f"{a}x^2 + ({b})x + ({c}) при x={x} = {res}"
        latex = f"{a}x^{{2}} + ({b})x + ({c}) \\big|_{{x={x}}} = {res}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _alg_fraction(self) -> Dict:
        """Числовая дробь с целым результатом."""
        b   = random.randint(2, 15)
        res = random.randint(1, 20)
        a   = b * res

        text  = f"{a}/{b} = {res}"
        latex = f"\\frac{{{a}}}{{{b}}} = {res}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _alg_root(self) -> Dict:
        """Квадратный корень."""
        n   = random.randint(1, 20)
        res = n * n

        text  = f"sqrt({res}) = {n}"
        latex = f"\\sqrt{{{res}}} = {n}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _alg_power(self) -> Dict:
        """Возведение в степень."""
        base = random.randint(2, 10)
        exp  = random.randint(2, 5)
        res  = base ** exp

        text  = f"{base}^{exp} = {res}"
        latex = f"{base}^{{{exp}}} = {res}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _alg_mixed(self) -> Dict:
        """Смешанное выражение."""
        a   = random.randint(2, 10)
        b   = random.randint(1, 5)
        res = a ** 2 + b

        text  = f"{a}^2 + {b} = {res}"
        latex = f"{a}^{{2}} + {b} = {res}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    # ─────────────────────────────────────────────────────────────────────────
    # ТРИГОНОМЕТРИЯ
    # ─────────────────────────────────────────────────────────────────────────

    # Табличные значения: (градусы, рад_latex, sin, cos, tan|None)
    _TRIG_TABLE = [
        (0,   "0",          "0",              "1",              "0"),
        (30,  "\\pi/6",     "\\frac{1}{2}",   "\\frac{\\sqrt{3}}{2}", "\\frac{1}{\\sqrt{3}}"),
        (45,  "\\pi/4",     "\\frac{\\sqrt{2}}{2}", "\\frac{\\sqrt{2}}{2}", "1"),
        (60,  "\\pi/3",     "\\frac{\\sqrt{3}}{2}", "\\frac{1}{2}",   "\\sqrt{3}"),
        (90,  "\\pi/2",     "1",              "0",              None),
        (120, "\\frac{2\\pi}{3}", "\\frac{\\sqrt{3}}{2}", "-\\frac{1}{2}", "-\\sqrt{3}"),
        (135, "\\frac{3\\pi}{4}", "\\frac{\\sqrt{2}}{2}", "-\\frac{\\sqrt{2}}{2}", "-1"),
        (150, "\\frac{5\\pi}{6}", "\\frac{1}{2}", "-\\frac{\\sqrt{3}}{2}", "-\\frac{1}{\\sqrt{3}}"),
        (180, "\\pi",       "0",              "-1",             "0"),
        (270, "\\frac{3\\pi}{2}", "-1",       "0",              None),
        (360, "2\\pi",      "0",              "1",              "0"),
    ]

    def _trig_sin(self) -> Dict:
        row  = random.choice(self._TRIG_TABLE)
        deg, rad, sin_v = row[0], row[1], row[2]
        text  = f"sin({deg} degrees) = {sin_v}"
        latex = f"\\sin({rad}) = {sin_v}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _trig_cos(self) -> Dict:
        row  = random.choice(self._TRIG_TABLE)
        deg, rad, cos_v = row[0], row[1], row[3]
        text  = f"cos({deg} degrees) = {cos_v}"
        latex = f"\\cos({rad}) = {cos_v}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _trig_tan(self) -> Dict:
        valid = [r for r in self._TRIG_TABLE if r[4] is not None]
        row   = random.choice(valid)
        deg, rad, tan_v = row[0], row[1], row[4]
        text  = f"tg({deg} degrees) = {tan_v}"
        latex = f"\\tan({rad}) = {tan_v}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _trig_identity_pythag(self) -> Dict:
        """sin² + cos² = 1."""
        row   = random.choice(self._TRIG_TABLE)
        deg, rad = row[0], row[1]
        text  = f"sin^2({deg} degrees) + cos^2({deg} degrees) = 1"
        latex = f"\\sin^{{2}}({rad}) + \\cos^{{2}}({rad}) = 1"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _trig_double_angle(self) -> Dict:
        """sin(2x) = 2 sin(x) cos(x) для конкретного x."""
        angles = [(30, "\\pi/6"), (45, "\\pi/4"), (60, "\\pi/3")]
        deg, rad = random.choice(angles)
        # sin(2*30) = sin(60) = √3/2
        double_deg = 2 * deg
        sin_vals   = {60: "\\frac{\\sqrt{3}}{2}", 90: "1", 120: "\\frac{\\sqrt{3}}{2}"}
        sin_v      = sin_vals.get(double_deg, "\\frac{\\sqrt{2}}{2}")
        text  = f"sin(2 * {deg} degrees) = {sin_v}"
        latex = f"\\sin(2 \\cdot {rad}) = {sin_v}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    # ─────────────────────────────────────────────────────────────────────────
    # ЛОГАРИФМЫ
    # ─────────────────────────────────────────────────────────────────────────

    def _log_natural(self) -> Dict:
        """ln(e^n) = n."""
        n     = random.randint(1, 10)
        text  = f"ln(e^{n}) = {n}"
        latex = f"\\ln(e^{{{n}}}) = {n}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _log_base10(self) -> Dict:
        """lg(10^n) = n."""
        n     = random.randint(1, 7)
        arg   = 10 ** n
        text  = f"lg({arg}) = {n}"
        latex = f"\\lg({arg}) = {n}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _log_base2(self) -> Dict:
        """log_2(2^n) = n."""
        n     = random.randint(1, 10)
        arg   = 2 ** n
        text  = f"log2({arg}) = {n}"
        latex = f"\\log_{{2}}({arg}) = {n}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _log_arbitrary(self) -> Dict:
        """log_b(b^n) = n."""
        b     = random.choice([3, 4, 5, 6, 7, 8, 9])
        n     = random.randint(1, 4)
        arg   = b ** n
        text  = f"log_{b}({arg}) = {n}"
        latex = f"\\log_{{{b}}}({arg}) = {n}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _log_property(self) -> Dict:
        """log_b(a*c) = log_b(a) + log_b(c) → числовой результат."""
        b   = random.choice([2, 3, 10])
        n   = random.randint(1, 4)
        m   = random.randint(1, 4)
        res = n + m
        arg = b ** n * b ** m
        text  = f"log_{b}({arg}) = {res}"
        latex = f"\\log_{{{b}}}({arg}) = {res}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    # ─────────────────────────────────────────────────────────────────────────
    # КОМБИНАТОРИКА
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _factorial(n: int) -> int:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    @staticmethod
    def _comb_val(n: int, k: int) -> int:
        if k > n:
            return 0
        num, den = 1, 1
        for i in range(k):
            num *= (n - i)
            den *= (i + 1)
        return num // den

    def _comb_binomial(self) -> Dict:
        """C(n,k) — биномиальный коэффициент."""
        n   = random.randint(2, 12)
        k   = random.randint(0, n)
        res = self._comb_val(n, k)
        text  = f"C({n},{k}) = {res}"
        latex = f"\\binom{{{n}}}{{{k}}} = {res}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _comb_factorial(self) -> Dict:
        """n!."""
        n     = random.randint(0, 10)
        res   = self._factorial(n)
        text  = f"{n}! = {res}"
        latex = f"{n}! = {res}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _comb_arrangement(self) -> Dict:
        """A(n,k) = n!/(n-k)!."""
        n   = random.randint(3, 8)
        k   = random.randint(1, n)
        res = self._factorial(n) // self._factorial(n - k)
        text  = f"A({n},{k}) = {res}"
        latex = f"A_{{{n}}}^{{{k}}} = {res}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _comb_permutation(self) -> Dict:
        """P(n) = n!."""
        n     = random.randint(2, 8)
        res   = self._factorial(n)
        text  = f"P({n}) = {res}"
        latex = f"P_{{{n}}} = {n}! = {res}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _comb_binomial_sum(self) -> Dict:
        """Σ C(n,k) = 2^n."""
        n   = random.randint(2, 8)
        res = 2 ** n
        text  = f"sum C({n},k), k=0..{n} = {res}"
        latex = f"\\sum_{{k=0}}^{{{n}}} \\binom{{{n}}}{{k}} = {res}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    # ─────────────────────────────────────────────────────────────────────────
    # РЯДЫ И ПРЕДЕЛЫ
    # ─────────────────────────────────────────────────────────────────────────

    def _ser_arithmetic_sum(self) -> Dict:
        """Σ(i=1..n) i = n(n+1)/2."""
        n   = random.randint(2, 30)
        res = n * (n + 1) // 2
        text  = f"sum(i=1..{n}) i = {res}"
        latex = f"\\sum_{{i=1}}^{{{n}}} i = {res}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _ser_geometric_sum(self) -> Dict:
        """Σ(i=0..n) r^i = (r^(n+1)-1)/(r-1)."""
        r   = random.choice([2, 3, 4])
        n   = random.randint(2, 7)
        res = (r ** (n + 1) - 1) // (r - 1)
        text  = f"sum(i=0..{n}) {r}^i = {res}"
        latex = f"\\sum_{{i=0}}^{{{n}}} {r}^{{i}} = {res}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _ser_squares_sum(self) -> Dict:
        """Σ(i=1..n) i² = n(n+1)(2n+1)/6."""
        n   = random.randint(2, 15)
        res = n * (n + 1) * (2 * n + 1) // 6
        text  = f"sum(i=1..{n}) i^2 = {res}"
        latex = f"\\sum_{{i=1}}^{{{n}}} i^{{2}} = {res}"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _lim_simple(self) -> Dict:
        """lim(x→∞) c/x = 0."""
        c     = random.randint(1, 100)
        text  = f"lim(x->inf) {c}/x = 0"
        latex = f"\\lim_{{x \\to \\infty}} \\frac{{{c}}}{{x}} = 0"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}

    def _lim_over_n(self) -> Dict:
        """lim(n→∞) n/(n+c) = 1."""
        c     = random.randint(1, 20)
        text  = f"lim(n->inf) n/(n+{c}) = 1"
        latex = f"\\lim_{{n \\to \\infty}} \\frac{{n}}{{n + {c}}} = 1"
        return {"text": text, "latex": latex, "source": SOURCE_SYNTHETIC}
