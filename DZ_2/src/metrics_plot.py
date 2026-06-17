"""
Графическая визуализация метрик качества графа знаний.

Строит дашборд из 6 панелей (распределения, покрытие, комплексные метрики,
здоровье/обнаружение ошибок, сводка) и сохраняет PNG.
"""

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Палитра по типам вершин — согласована с graph_builder.visualize_graph.
NODE_COLORS = {
    "variable": "#74b9ff",
    "constant": "#fd79a8",
    "function": "#55efc4",
    "operator": "#a29bfe",
    "number":   "#ffeaa7",
}
ACCENT = "#0984e3"
GOOD   = "#00b894"
WARN   = "#e17055"


def _bar(ax, labels, values, title, colors=None, ylim=None, pct=False):
    bars = ax.bar(range(len(labels)), values, color=colors or ACCENT)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    for b, v in zip(bars, values):
        txt = f"{v:.0%}" if pct else (f"{v:.2f}" if isinstance(v, float) else str(v))
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                txt, ha="center", va="bottom", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_metrics(report: dict, output_path: str):
    s = report["structural"]
    c = report["coverage"]
    k = report["consistency"]
    m = report["composite"]
    ed = report.get("error_detection", {})

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Оценка качества графа знаний", fontsize=16, fontweight="bold")

    # (0,0) Распределение типов вершин
    nt = s["node_type_distribution"]
    _bar(
        axes[0, 0],
        list(nt.keys()), list(nt.values()),
        "Типы вершин",
        colors=[NODE_COLORS.get(t, "#dfe6e9") for t in nt.keys()],
    )

    # (0,1) Топ операций рёбер
    ops = sorted(s["edge_operation_distribution"].items(), key=lambda x: -x[1])[:10]
    _bar(axes[0, 1], [o for o, _ in ops], [v for _, v in ops], "Операции рёбер (топ-10)")

    # (0,2) Покрытие онтологии
    cov_labels = ["операторы", "функции", "константы", "операции", "формулы"]
    cov_values = [c["coverage_operators"], c["coverage_functions"],
                  c["coverage_constants"], c["coverage_operations"], c["formula_coverage"]]
    _bar(axes[0, 2], cov_labels, cov_values, "Покрытие онтологии (§3.3)",
         colors=ACCENT, ylim=(0, 1.1), pct=True)

    # (1,0) Комплексные метрики
    comp_labels = ["точность*", "полнота*", "связность", "F1", "качество"]
    comp_values = [m["precision_proxy"], m["recall_proxy"], m["connectivity"],
                   m["f1_score"], m["weighted_quality"]]
    colors = [GOOD if v >= 0.7 else (WARN if v < 0.4 else ACCENT) for v in comp_values]
    _bar(axes[1, 0], comp_labels, comp_values, "Комплексные метрики (§3.3)",
         colors=colors, ylim=(0, 1.1))

    # (1,1) Здоровье графа / обнаружение ошибок (нормировано 0..1, больше = лучше)
    health_labels, health_values = ["согл. типов"], [k["type_consistency"]]
    if ed:
        health_labels += ["ацикличность", "корр. кардинальность", "норм. связи"]
        health_values += [
            ed["acyclicity"]["acyclicity_score"],
            1 - ed["cardinality"]["violation_ratio"],
            1 - ed["type_anomalies"]["suspicious_edge_ratio"],
        ]
    colors = [GOOD if v >= 0.9 else (WARN if v < 0.6 else ACCENT) for v in health_values]
    _bar(axes[1, 1], health_labels, health_values,
         "Согласованность и обнаружение ошибок (§1.1–1.2)", colors=colors, ylim=(0, 1.1))

    # (1,2) Текстовая сводка
    ax = axes[1, 2]
    ax.axis("off")
    lines = [
        "СВОДКА",
        "",
        f"Вершин / рёбер:        {s['total_nodes']} / {s['total_edges']}",
        f"Плотность:             {s['density']}",
        f"Средняя степень:       {s['avg_degree']}",
        f"Изолированные:         {s['isolated_nodes']} ({s['isolated_ratio']:.0%})",
        f"Компонент:             {s['connected_components']} "
        f"(крупн. {s['largest_component_fraction']:.0%})",
        "",
        f"F1-score:              {m['f1_score']}",
        f"Взвешенное качество:   {m['weighted_quality']}",
    ]
    if ed:
        lines += [
            "",
            f"Аномальных связей:     {ed['type_anomalies']['suspicious_patterns']} шаблонов",
            f"Наруш. кардинальности: {ed['cardinality']['operators_without_args'] + ed['cardinality']['functions_without_args']}",
            f"Циклов в операциях:    {ed['acyclicity']['operations_with_cycles']}/{ed['acyclicity']['checked_operations']}",
            f"Ассоц. правил:         {ed['association_rules']['rules_found']}",
        ]
    gt = report.get("ground_truth")
    if gt and gt.get("available"):
        o = gt["overall"]
        lines += [
            "",
            "Оракул sympy (истинные):",
            f"  precision / recall:  {o['precision']:.2f} / {o['recall']:.2f}",
            f"  F1:                  {o['f1']:.2f}",
        ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
            fontfamily="monospace", fontsize=11, transform=ax.transAxes)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Визуализация метрик сохранена: {output_path}")
