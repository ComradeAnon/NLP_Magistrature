import re
import logging
import warnings
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sympy.parsing.latex import parse_latex
        from sympy import Eq, Pow, Function, Number, Rational
    _HALF = Rational(1, 2)
    SYMPY_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    SYMPY_AVAILABLE = False
    _IMPORT_ERROR = str(_e)

# Адъюдицируемые операции графа и их класс.
ADJUDICABLE = {
    "equals":      "equals",
    "power":       "power",
    "function_of": "function",
    "applied_to":  "function",
}

# sympy-имя функции → метка-глиф в графе.
_FUNC_GLYPH = {
    "sin": "sin", "cos": "cos", "tan": "tan", "cot": "cot",
    "log": "log", "ln": "ln", "exp": "exp", "sqrt": "√",
}


def _base(name: str) -> str:
    """x_{1} → x (sympy хранит индекс в имени символа)."""
    return re.sub(r"_\{.*?\}", "", str(name))


def _syms(expr) -> set:
    """Множество базовых символов и чисел подвыражения."""
    s = {_base(x) for x in expr.free_symbols}
    s |= {str(x) for x in expr.atoms(Number) if x not in (-1,)}
    return s


def _glyphs(expr) -> set:
    """Глифы функций/операторов, присутствующих в подвыражении (метки графа)."""
    g = set()
    for f in expr.atoms(Function):
        g.add(_FUNC_GLYPH.get(type(f).__name__, type(f).__name__))
    for p in expr.atoms(Pow):
        if p.args[1] == _HALF:
            g.add("√")
    return g


def oracle_relations(latex: str):
    """
    Разбирает формулу sympy и извлекает эталонные отношения.
    Возвращает dict с ключами equals/power/func или None при ошибке разбора.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expr = parse_latex(latex)
    except Exception:
        return None

    rels = {"equals": [], "power": [], "func": []}

    if isinstance(expr, Eq):
        rels["equals"].append((
            _syms(expr.lhs) | _glyphs(expr.lhs),
            _syms(expr.rhs) | _glyphs(expr.rhs),
        ))

    for p in expr.atoms(Pow):
        b, e = p.args
        if e == _HALF:
            rels["func"].append(("√", _syms(b)))            # √ = Pow(_, 1/2)
        else:
            rels["power"].append((_syms(b), _syms(e)))

    for f in expr.atoms(Function):
        rels["func"].append((_FUNC_GLYPH.get(type(f).__name__, type(f).__name__), _syms(f)))

    return rels


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_against_oracle(nodes: list[dict], edges: list[dict], dataset: list[dict]) -> dict:
    """
    Считает истинные precision/recall, сравнивая рёбра графа с отношениями,
    извлечёнными независимым оракулом sympy. Выравнивание формул — по
    formula_id вида f"formula_{i}" (i — индекс записи в датасете).
    """
    if not SYMPY_AVAILABLE:
        return {
            "available": False,
            "reason": f"sympy.parsing.latex недоступен: {_IMPORT_ERROR}",
        }

    label = {n["id"]: n.get("label") for n in nodes}
    edges_by_formula: dict[str, list] = defaultdict(list)
    for e in edges:
        edges_by_formula[e.get("formula_id")].append(
            (label.get(e.get("source")), label.get(e.get("target")), e.get("operation"))
        )

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    parsed = failures = 0
    adjudicated_edges = excluded_edges = 0

    for i, rec in enumerate(dataset):
        rels = oracle_relations(rec.get("latex", ""))
        ge = edges_by_formula.get(f"formula_{i}", [])

        for _, _, op in ge:
            if op in ADJUDICABLE:
                adjudicated_edges += 1
            else:
                excluded_edges += 1

        if rels is None:
            failures += 1
            continue
        parsed += 1

        eq_edges = [(u, v) for u, v, o in ge if o == "equals"]
        pw_edges = [(u, v) for u, v, o in ge if o == "power"]
        fn_edges = [(u, v) for u, v, o in ge if o in ("function_of", "applied_to")]

        # ── equals: рёбра должны соединять разные стороны уравнения ──────────
        for (u, v) in eq_edges:
            ok = any((u in L and v in R) or (u in R and v in L) for (L, R) in rels["equals"])
            tp["equals"] += ok
            fp["equals"] += not ok
        for (L, R) in rels["equals"]:
            ok = any((u in L and v in R) or (u in R and v in L) for (u, v) in eq_edges)
            fn["equals"] += not ok

        # ── power: основание → показатель ────────────────────────────────────
        for (u, v) in pw_edges:
            ok = any(u in B and v in E for (B, E) in rels["power"])
            tp["power"] += ok
            fp["power"] += not ok
        for (B, E) in rels["power"]:
            ok = any(u in B and v in E for (u, v) in pw_edges)
            fn["power"] += not ok

        # ── function: имя функции → аргумент ─────────────────────────────────
        for (u, v) in fn_edges:
            ok = any(u == nm and v in A for (nm, A) in rels["func"])
            tp["function"] += ok
            fp["function"] += not ok
        for (nm, A) in rels["func"]:
            ok = any(u == nm and v in A for (u, v) in fn_edges)
            fn["function"] += not ok

    by_class = {c: _prf(tp[c], fp[c], fn[c]) for c in ("equals", "power", "function")}
    overall = _prf(
        sum(tp.values()), sum(fp.values()), sum(fn.values())
    )

    return {
        "available": True,
        "oracle": "sympy.parsing.latex.parse_latex",
        "total_formulas": len(dataset),
        "parsed_formulas": parsed,
        "parse_failures": failures,
        "adjudicated_edges": adjudicated_edges,
        "excluded_edges": excluded_edges,
        "by_class": by_class,
        "overall": overall,
        "note": (
            "Истинные precision/recall относительно независимого оракула sympy. "
            "Адъюдицируются только equals/power/function; subtraction/division/"
            "multiplication/adjacent/subscript исключены (sympy их канонизирует). "
            "Остаточные расхождения — смесь различий представления и грубости парсера."
        ),
    }
