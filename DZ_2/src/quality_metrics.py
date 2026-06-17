"""
Метрики качества графа знаний.

Считает качество готового графа (math_graph.json), построенного GraphBuilder,
по методологии обзорной статьи Heiko Paulheim «Knowledge graph refinement»
(Semantic Web, 2017) и связанных работ (см. instruction.docx).

Реализован базовый набор метрик — те, что вычисляются напрямую по графу,
без обучения моделей и внешних эталонов:

  - Структурно-статистические (§3.1, внутренняя оценка);
  - Покрытие относительно онтологии предметной области (§3.3);
  - Согласованность / проверка типов (§1.2, внутренняя логическая проверка);
  - Комплексные метрики: F1 и взвешенное качество (§3.3).

«Онтология» для метрики покрытия — это словари ожидаемых математических
элементов и операций из formula_parser (OPERATORS, OP_SYMBOLS, CONSTANTS, GREEK).
"""

import json
import math
import logging
from collections import Counter
from pathlib import Path

import networkx as nx

from .formula_parser import OPERATORS, OP_SYMBOLS, CONSTANTS, GREEK

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Онтология предметной области (эталон для метрики покрытия)
# ═════════════════════════════════════════════════════════════════════════════

# Ожидаемые элементы по типам — берём метки (label) из словарей парсера.
EXPECTED_OPERATORS = {
    label for label, ntype in OPERATORS.values() if ntype == "operator"
}
EXPECTED_FUNCTIONS = {
    label for label, ntype in OPERATORS.values() if ntype == "function"
}
EXPECTED_CONSTANTS = {label for label, _ in CONSTANTS.values()}

# Ожидаемые операции (рёбра): операции-символы из OP_SYMBOLS плюс структурные
# операции, которые порождает построитель графа.
STRUCTURAL_OPERATIONS = {
    "applied_to", "function_of", "parameter", "upper_bound", "adjacent",
}
EXPECTED_OPERATIONS = set(OP_SYMBOLS.values()) | STRUCTURAL_OPERATIONS

# Сигнатуры типов для проверки согласованности (§1.2 «Проверка типов»):
# операция → требуемый node_type вершины-источника.
SIGNATURES = {
    "applied_to":  "operator",
    "parameter":   "operator",
    "upper_bound": "operator",
    "function_of": "function",
}


# ═════════════════════════════════════════════════════════════════════════════
# Загрузка графа
# ═════════════════════════════════════════════════════════════════════════════

def load_graph(graph_json_path: str):
    """
    Загружает math_graph.json.
    Возвращает (nodes, edges, stats, G), где G — networkx.MultiDiGraph.
    """
    path = Path(graph_json_path)
    if not path.exists():
        raise FileNotFoundError(f"Граф не найден: {graph_json_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    stats = data.get("stats", {})

    G = nx.MultiDiGraph()
    for n in nodes:
        G.add_node(n["id"], **{k: v for k, v in n.items() if k != "id"})
    for e in edges:
        G.add_edge(
            e["source"], e["target"],
            operation=e.get("operation"),
            formula_id=e.get("formula_id"),
            weight=e.get("weight", 1.0),
        )

    logger.info(
        f"Загружен граф: {G.number_of_nodes()} вершин, "
        f"{G.number_of_edges()} рёбер"
    )
    return nodes, edges, stats, G


# ═════════════════════════════════════════════════════════════════════════════
# Вспомогательное
# ═════════════════════════════════════════════════════════════════════════════

def _normalized_entropy(counts) -> float:
    """
    Нормированная энтропия Шеннона распределения (0..1).
    1.0 — типы распределены идеально равномерно, 0.0 — всё одного типа.
    """
    total = sum(counts)
    if total <= 0:
        return 0.0
    k = len(counts)
    if k <= 1:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counts if c > 0)
    return h / math.log(k)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# Группа 1. Структурно-статистические метрики (§3.1)
# ═════════════════════════════════════════════════════════════════════════════

def structural_metrics(nodes: list[dict], edges: list[dict], G: nx.MultiDiGraph) -> dict:
    v = G.number_of_nodes()
    e = G.number_of_edges()

    # Плотность ориентированного графа: E / (V·(V−1)).
    density = _safe_div(e, v * (v - 1)) if v > 1 else 0.0

    # Степени по неориентированной проекции (для изоляции и связности).
    UG = G.to_undirected()
    degrees = dict(UG.degree())
    deg_values = list(degrees.values()) or [0]

    isolated = [n for n, d in degrees.items() if d == 0]
    isolated_ratio = _safe_div(len(isolated), v)

    components = list(nx.connected_components(UG)) if v else []
    largest = max((len(c) for c in components), default=0)
    largest_fraction = _safe_div(largest, v)

    node_types = Counter(n.get("node_type") for n in nodes)
    edge_ops = Counter(e_.get("operation") for e_ in edges)

    return {
        "total_nodes":           v,
        "total_edges":           e,
        "density":               round(density, 4),
        "avg_degree":            round(_safe_div(2 * e, v), 3),  # 2E/V
        "max_degree":            max(deg_values),
        "min_degree":            min(deg_values),
        "isolated_nodes":        len(isolated),
        "isolated_ratio":        round(isolated_ratio, 4),
        "connected_components":  len(components),
        "largest_component":     largest,
        "largest_component_fraction": round(largest_fraction, 4),
        "node_type_distribution":    dict(node_types),
        "node_type_entropy":         round(_normalized_entropy(list(node_types.values())), 4),
        "edge_operation_distribution": dict(edge_ops),
        "edge_operation_entropy":      round(_normalized_entropy(list(edge_ops.values())), 4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Группа 2. Покрытие относительно онтологии (§3.3 «Покрытие»)
# ═════════════════════════════════════════════════════════════════════════════

def coverage_metrics(
    nodes: list[dict],
    edges: list[dict],
    stats: dict,
    total_formulas: int | None = None,
) -> dict:
    labels_by_type: dict[str, set] = {}
    for n in nodes:
        labels_by_type.setdefault(n.get("node_type"), set()).add(n.get("label"))

    present_operators = labels_by_type.get("operator", set())
    present_functions = labels_by_type.get("function", set())
    present_constants = labels_by_type.get("constant", set())
    present_operations = {e.get("operation") for e in edges}

    def _cov(present: set, expected: set) -> float:
        return _safe_div(len(present & expected), len(expected))

    # Полнота извлечения: какая доля формул датасета попала в граф хотя бы
    # одним ребром или вершиной (по distinct formula_id). Берём в основном из
    # рёбер: вершины глобально переиспользуются между формулами и хранят лишь
    # formula_id последней коснувшейся формулы, поэтому по ним считать нельзя.
    formula_ids = {e.get("formula_id") for e in edges if e.get("formula_id")}
    formula_ids |= {n.get("formula_id") for n in nodes if n.get("formula_id")}
    denom = total_formulas if total_formulas else stats.get("total_formulas", 0)
    formula_coverage = _safe_div(len(formula_ids), denom)

    return {
        "coverage_operators":  round(_cov(present_operators, EXPECTED_OPERATORS), 4),
        "coverage_functions":  round(_cov(present_functions, EXPECTED_FUNCTIONS), 4),
        "coverage_constants":  round(_cov(present_constants, EXPECTED_CONSTANTS), 4),
        "coverage_operations": round(_cov(present_operations, EXPECTED_OPERATIONS), 4),
        "formula_coverage":    round(formula_coverage, 4),
        "_detail": {
            "operators":  f"{len(present_operators & EXPECTED_OPERATORS)}/{len(EXPECTED_OPERATORS)}",
            "functions":  f"{len(present_functions & EXPECTED_FUNCTIONS)}/{len(EXPECTED_FUNCTIONS)}",
            "constants":  f"{len(present_constants & EXPECTED_CONSTANTS)}/{len(EXPECTED_CONSTANTS)}",
            "operations": f"{len(present_operations & EXPECTED_OPERATIONS)}/{len(EXPECTED_OPERATIONS)}",
            "formulas":   f"{len(formula_ids)}/{denom}",
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Группа 3. Согласованность / проверка типов (§1.2)
# ═════════════════════════════════════════════════════════════════════════════

def consistency_metrics(nodes: list[dict], edges: list[dict]) -> dict:
    node_type = {n["id"]: n.get("node_type") for n in nodes}

    # Проверка сигнатур типов: считаем только рёбра с заданной сигнатурой.
    constrained = 0
    conformant = 0
    for e in edges:
        sig = SIGNATURES.get(e.get("operation"))
        if sig is None:
            continue
        constrained += 1
        if node_type.get(e.get("source")) == sig:
            conformant += 1

    type_consistency = _safe_div(conformant, constrained) if constrained else 1.0

    # Петли (ожидается 0 — построитель их не создаёт).
    self_loops = sum(1 for e in edges if e.get("source") == e.get("target"))

    # Дубликаты рёбер (одинаковые source+target+operation+formula_id).
    seen, duplicates = set(), 0
    for e in edges:
        key = (e.get("source"), e.get("target"), e.get("operation"), e.get("formula_id"))
        if key in seen:
            duplicates += 1
        seen.add(key)

    # «Висячие» операторы/функции — без исходящей операции применения.
    applied_sources = {
        e.get("source") for e in edges
        if e.get("operation") in ("applied_to", "function_of", "parameter", "upper_bound")
    }
    op_fn_nodes = [n for n in nodes if n.get("node_type") in ("operator", "function")]
    dangling = [n for n in op_fn_nodes if n["id"] not in applied_sources]
    dangling_ratio = _safe_div(len(dangling), len(op_fn_nodes))

    return {
        "type_consistency":      round(type_consistency, 4),
        "edges_with_signature":  constrained,
        "conformant_edges":      conformant,
        "self_loops":            self_loops,
        "duplicate_edges":       duplicates,
        "dangling_operators":    len(dangling),
        "dangling_ratio":        round(dangling_ratio, 4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Группа 4. Комплексные метрики (§3.3)
# ═════════════════════════════════════════════════════════════════════════════

def composite_metrics(
    structural: dict,
    coverage: dict,
    consistency: dict,
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> dict:
    # Прокси-показатели (нет gold standard, поэтому это приближения):
    #   precision ≈ согласованность типов рёбер;
    #   recall    ≈ среднее покрытие онтологии;
    #   connectivity ≈ доля крупнейшей связной компоненты.
    precision = consistency["type_consistency"]
    recall = round(
        (coverage["coverage_operators"]
         + coverage["coverage_functions"]
         + coverage["coverage_operations"]
         + coverage["formula_coverage"]) / 4,
        4,
    )
    connectivity = structural["largest_component_fraction"]

    f1 = _safe_div(2 * precision * recall, precision + recall)

    a, b, g = weights
    norm = (a + b + g) or 1.0
    weighted_quality = (a * precision + b * recall + g * connectivity) / norm

    return {
        "precision_proxy":   round(precision, 4),
        "recall_proxy":      round(recall, 4),
        "connectivity":      round(connectivity, 4),
        "f1_score":          round(f1, 4),
        "weighted_quality":  round(weighted_quality, 4),
        "weights":           {"alpha": a, "beta": b, "gamma": g},
        "_note": (
            "precision/recall — прокси-оценки (нет эталонной разметки): "
            "precision ≈ согласованность типов, recall ≈ покрытие онтологии, "
            "connectivity ≈ доля крупнейшей компоненты."
        ),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Группа 5. Обнаружение ошибок (§1.1 статистика типов, §1.2 логические проверки)
# ═════════════════════════════════════════════════════════════════════════════

# Направленные операции, которые по смыслу должны быть ацикличными
# (элемент не может быть степенью/аргументом самого себя через цепочку).
ACYCLIC_OPERATIONS = {
    "power", "subscript", "applied_to", "function_of", "parameter", "upper_bound",
}


def type_anomaly_metrics(
    nodes: list[dict],
    edges: list[dict],
    threshold: float = 0.05,
    max_examples: int = 10,
) -> dict:
    """
    §1.1 «Статистика типов». Для каждого типа источника строим распределение
    шаблонов (операция, тип цели). Шаблоны с условной вероятностью ниже порога
    помечаются как потенциальные ошибки (редкие, аномальные связи).
    """
    ntype = {n["id"]: n.get("node_type") for n in nodes}
    nlabel = {n["id"]: n.get("label") for n in nodes}

    # Счётчики: шаблон (src_type, op, tgt_type) и итог по типу источника.
    pattern_count: Counter = Counter()
    src_total: Counter = Counter()
    pattern_edges: dict[tuple, list] = {}

    for e in edges:
        st = ntype.get(e.get("source"))
        tt = ntype.get(e.get("target"))
        op = e.get("operation")
        key = (st, op, tt)
        pattern_count[key] += 1
        src_total[st] += 1
        pattern_edges.setdefault(key, []).append(
            f"{nlabel.get(e.get('source'))} —[{op}]→ {nlabel.get(e.get('target'))}"
        )

    suspicious = []
    for key, cnt in pattern_count.items():
        st, op, tt = key
        cond_prob = _safe_div(cnt, src_total[st])
        if cond_prob < threshold:
            suspicious.append({
                "pattern": f"{st} —[{op}]→ {tt}",
                "count": cnt,
                "conditional_prob": round(cond_prob, 4),
                "sample": pattern_edges[key][0],
            })

    suspicious.sort(key=lambda x: x["conditional_prob"])
    suspicious_edges = sum(s["count"] for s in suspicious)

    return {
        "distinct_patterns":     len(pattern_count),
        "suspicious_patterns":   len(suspicious),
        "suspicious_edge_ratio": round(_safe_div(suspicious_edges, len(edges)), 4),
        "threshold":             threshold,
        "examples":              suspicious[:max_examples],
    }


def cardinality_metrics(nodes: list[dict], edges: list[dict]) -> dict:
    """
    §1.2 «Кардинальность». Проверяем минимальные ограничения на число связей:
    у оператора должен быть хотя бы один аргумент, у функции — хотя бы один
    операнд (function_of).
    """
    op_args = {
        e.get("source") for e in edges
        if e.get("operation") in ("applied_to", "parameter", "upper_bound")
    }
    fn_args = {
        e.get("source") for e in edges
        if e.get("operation") == "function_of"
    }

    operators = [n for n in nodes if n.get("node_type") == "operator"]
    functions = [n for n in nodes if n.get("node_type") == "function"]

    ops_violations = [n for n in operators if n["id"] not in op_args]
    fns_violations = [n for n in functions if n["id"] not in fn_args]

    total = len(operators) + len(functions)
    violations = len(ops_violations) + len(fns_violations)

    return {
        "operators_without_args": len(ops_violations),
        "functions_without_args": len(fns_violations),
        "violation_ratio":        round(_safe_div(violations, total), 4),
        "examples": [n.get("label") for n in (ops_violations + fns_violations)][:10],
    }


def acyclicity_metrics(edges: list[dict]) -> dict:
    """
    §1.2 «Транзитивность/симметричность». Направленные операции (power, applied_to,
    …) должны быть ацикличными. Для каждой операции строим подграф и ищем циклы
    через сильно связные компоненты размера > 1.
    """
    by_op: dict[str, nx.DiGraph] = {}
    for e in edges:
        op = e.get("operation")
        if op not in ACYCLIC_OPERATIONS:
            continue
        by_op.setdefault(op, nx.DiGraph()).add_edge(e.get("source"), e.get("target"))

    cyclic_operations = []
    for op, g in by_op.items():
        sccs = [c for c in nx.strongly_connected_components(g) if len(c) > 1]
        if sccs:
            cyclic_operations.append({"operation": op, "cyclic_nodes": sum(len(c) for c in sccs)})

    checked = len(by_op)
    with_cycles = len(cyclic_operations)

    return {
        "checked_operations":      checked,
        "operations_with_cycles":  with_cycles,
        "acyclicity_score":        round(1 - _safe_div(with_cycles, checked), 4) if checked else 1.0,
        "cyclic_operations":       cyclic_operations,
    }


def association_rule_metrics(
    edges: list[dict],
    min_support: int = 2,
    min_confidence: float = 0.5,
    top_n: int = 15,
) -> dict:
    """
    §1.1 / §2.2 «Ассоциативные правила». Майним композиционные правила длины 2
    вида r1(X,Y) ∧ r2(Y,Z) → r3(X,Z) с поддержкой (support) и достоверностью
    (confidence), как в AMIE. Поддержка считается по различным парам (X, Z).
    """
    out_by_op: dict[str, dict] = {}   # op -> src -> set(tgt)
    in_by_op:  dict[str, dict] = {}   # op -> tgt -> set(src)
    ops_by_pair: dict[tuple, set] = {}  # (src, tgt) -> set(op)

    for e in edges:
        s, t, op = e.get("source"), e.get("target"), e.get("operation")
        out_by_op.setdefault(op, {}).setdefault(s, set()).add(t)
        in_by_op.setdefault(op, {}).setdefault(t, set()).add(s)
        ops_by_pair.setdefault((s, t), set()).add(op)

    operations = list(out_by_op.keys())
    rules = []

    for r1 in operations:
        for r2 in operations:
            # Тело правила: пары (X, Z), связанные через промежуточный Y.
            body: set = set()
            for y, xs in in_by_op.get(r1, {}).items():
                zs = out_by_op.get(r2, {}).get(y)
                if not zs:
                    continue
                for x in xs:
                    for z in zs:
                        if x != z:
                            body.add((x, z))
            if len(body) < min_support:
                continue

            # Голова: какие r3 связывают X→Z в этих парах.
            head_count: Counter = Counter()
            for (x, z) in body:
                for r3 in ops_by_pair.get((x, z), ()):
                    head_count[r3] += 1

            for r3, support in head_count.items():
                confidence = _safe_div(support, len(body))
                if support >= min_support and confidence >= min_confidence:
                    rules.append({
                        "rule": f"{r1} ∧ {r2} → {r3}",
                        "support": support,
                        "confidence": round(confidence, 3),
                    })

    rules.sort(key=lambda r: (-r["confidence"], -r["support"]))
    return {
        "rules_found": len(rules),
        "min_support": min_support,
        "min_confidence": min_confidence,
        "top_rules": rules[:top_n],
    }


def error_detection_metrics(
    nodes: list[dict],
    edges: list[dict],
    anomaly_threshold: float = 0.05,
) -> dict:
    return {
        "type_anomalies":    type_anomaly_metrics(nodes, edges, anomaly_threshold),
        "cardinality":       cardinality_metrics(nodes, edges),
        "acyclicity":        acyclicity_metrics(edges),
        "association_rules": association_rule_metrics(edges),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Точка входа
# ═════════════════════════════════════════════════════════════════════════════

def compute_all(
    graph_json_path: str,
    dataset_path: str | None = None,
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> dict:
    """
    Считает все группы метрик для графа из graph_json_path.

    dataset_path — опционально; если задан, знаменатель покрытия формул берётся
    из числа записей датасета (иначе из stats графа).
    """
    nodes, edges, stats, G = load_graph(graph_json_path)

    total_formulas = None
    if dataset_path:
        with open(dataset_path, "r", encoding="utf-8") as f:
            total_formulas = len(json.load(f))

    structural = structural_metrics(nodes, edges, G)
    coverage = coverage_metrics(nodes, edges, stats, total_formulas)
    consistency = consistency_metrics(nodes, edges)
    error_detection = error_detection_metrics(nodes, edges)
    composite = composite_metrics(structural, coverage, consistency, weights)

    return {
        "graph":           str(graph_json_path),
        "structural":      structural,
        "coverage":        coverage,
        "consistency":     consistency,
        "error_detection": error_detection,
        "composite":       composite,
    }
