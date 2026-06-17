"""
Анализатор качества графа знаний DZ_2.

Загружает построенный граф (<kb>/math_graph.json) и считает метрики качества
по методологии статьи instruction.docx (Paulheim 2017 и др.). Печатает отчёт
в консоль и сохраняет его в <kb>/quality_report.json.

Примеры:
    python DZ_2/analyze.py --kb knowledge_graph_math --dataset datasets/output.json
    python DZ_2/analyze.py --kb knowledge_graph_math --alpha 0.5 --beta 0.3 --gamma 0.2
"""

import sys
import json
import logging
import argparse
from pathlib import Path

PROJECT_ROOT    = Path(__file__).parent.parent
DZ2_DIR         = Path(__file__).parent
DEFAULT_KB_DIR  = DZ2_DIR / "knowledge_graph"

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from src.quality_metrics import compute_all  # noqa: E402
from src.metrics_plot import plot_metrics      # noqa: E402


# ─── Печать отчёта ───────────────────────────────────────────────────────────

def _line(title: str):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)


def _row(label: str, value, width: int = 36):
    print(f"  {label:<{width}}: {value}")


def _dist(label: str, dist: dict):
    print(f"  {label}:")
    for k, v in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"      {str(k):<24} {v}")


def print_report(report: dict):
    s = report["structural"]
    c = report["coverage"]
    k = report["consistency"]
    m = report["composite"]

    _line("ОЦЕНКА КАЧЕСТВА ГРАФА ЗНАНИЙ")
    print(f"  Граф: {report['graph']}")

    _line("1. Структурно-статистические метрики (§3.1)")
    _row("Вершин (|V|)", s["total_nodes"])
    _row("Рёбер (|E|)", s["total_edges"])
    _row("Плотность графа", s["density"])
    _row("Средняя степень (2E/V)", s["avg_degree"])
    _row("Макс. / мин. степень", f"{s['max_degree']} / {s['min_degree']}")
    _row("Изолированных вершин", f"{s['isolated_nodes']} ({s['isolated_ratio']:.1%})")
    _row("Связных компонент", s["connected_components"])
    _row("Крупнейшая компонента", f"{s['largest_component']} ({s['largest_component_fraction']:.1%})")
    _row("Энтропия типов вершин", s["node_type_entropy"])
    _row("Энтропия операций рёбер", s["edge_operation_entropy"])
    _dist("Распределение типов вершин", s["node_type_distribution"])
    _dist("Распределение операций рёбер", s["edge_operation_distribution"])

    _line("2. Покрытие онтологии (§3.3)")
    _row("Операторы", f"{c['coverage_operators']:.1%}  ({c['_detail']['operators']})")
    _row("Функции", f"{c['coverage_functions']:.1%}  ({c['_detail']['functions']})")
    _row("Константы", f"{c['coverage_constants']:.1%}  ({c['_detail']['constants']})")
    _row("Операции (рёбра)", f"{c['coverage_operations']:.1%}  ({c['_detail']['operations']})")
    _row("Покрытие формул датасета", f"{c['formula_coverage']:.1%}  ({c['_detail']['formulas']})")

    _line("3. Согласованность / проверка типов (§1.2)")
    _row("Согласованность типов рёбер", f"{k['type_consistency']:.1%}  "
                                        f"({k['conformant_edges']}/{k['edges_with_signature']})")
    _row("Петель (self-loops)", k["self_loops"])
    _row("Дубликатов рёбер", k["duplicate_edges"])
    _row("«Висячих» операторов/функций", f"{k['dangling_operators']} ({k['dangling_ratio']:.1%})")

    ed = report.get("error_detection")
    if ed:
        ta, card, acy, ar = (ed["type_anomalies"], ed["cardinality"],
                              ed["acyclicity"], ed["association_rules"])
        _line("4. Обнаружение ошибок (§1.1–1.2)")
        _row("Шаблонов связей (тип–оп–тип)", ta["distinct_patterns"])
        _row(f"Аномальных шаблонов (p<{ta['threshold']})",
             f"{ta['suspicious_patterns']} ({ta['suspicious_edge_ratio']:.1%} рёбер)")
        for ex in ta["examples"][:5]:
            print(f"      ⚠ {ex['pattern']:<34} p={ex['conditional_prob']:.3f}  напр.: {ex['sample']}")
        _row("Наруш. кардинальности (без аргумента)",
             f"операторов {card['operators_without_args']}, функций {card['functions_without_args']}")
        _row("Ацикличность операций",
             f"{acy['acyclicity_score']:.0%}  (циклов в {acy['operations_with_cycles']}/{acy['checked_operations']})")
        _row("Ассоциативных правил найдено",
             f"{ar['rules_found']}  (support≥{ar['min_support']}, conf≥{ar['min_confidence']})")
        for r in ar["top_rules"][:5]:
            print(f"      • {r['rule']:<40} support={r['support']}  conf={r['confidence']}")

    _line("5. Комплексные метрики (§3.3)")
    _row("Точность (прокси)", m["precision_proxy"])
    _row("Полнота (прокси)", m["recall_proxy"])
    _row("Связность", m["connectivity"])
    _row("F1-score", m["f1_score"])
    w = m["weights"]
    _row(f"Взвешенное качество (α={w['alpha']:.2f} β={w['beta']:.2f} γ={w['gamma']:.2f})",
         m["weighted_quality"])
    print(f"\n  Прим.: {m['_note']}")
    print("═" * 60)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Метрики качества графа знаний DZ_2",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--kb", type=Path, default=DEFAULT_KB_DIR, metavar="DIR",
        help=f"Папка базы знаний (читается <kb>/math_graph.json).\nПо умолчанию: {DEFAULT_KB_DIR}",
    )
    parser.add_argument(
        "--dataset", type=Path, default=None, metavar="PATH",
        help="Путь к датасету JSON (для знаменателя покрытия формул).\n"
             "Если не задан — берётся из stats графа.",
    )
    parser.add_argument("--alpha", type=float, default=1 / 3, help="Вес точности в взвешенном качестве.")
    parser.add_argument("--beta",  type=float, default=1 / 3, help="Вес полноты в взвешенном качестве.")
    parser.add_argument("--gamma", type=float, default=1 / 3, help="Вес связности в взвешенном качестве.")
    parser.add_argument(
        "--output", type=Path, default=None, metavar="PATH",
        help="Куда сохранить JSON-отчёт.\nПо умолчанию: <kb>/quality_report.json",
    )
    parser.add_argument(
        "--plot-output", type=Path, default=None, metavar="PATH",
        help="Куда сохранить PNG-визуализацию.\nПо умолчанию: <kb>/quality_metrics.png",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Не строить графическую визуализацию (только JSON и консоль).",
    )

    args = parser.parse_args()

    graph_path = args.kb / "math_graph.json"
    if not graph_path.exists():
        print(f"Ошибка: граф не найден: {graph_path}\n"
              f"Сначала постройте базу знаний:\n"
              f"  python DZ_2/main.py --mode build --kb {args.kb}")
        sys.exit(1)

    if args.dataset is not None and not args.dataset.exists():
        print(f"Ошибка: датасет не найден: {args.dataset}")
        sys.exit(1)

    report = compute_all(
        graph_json_path=str(graph_path),
        dataset_path=str(args.dataset) if args.dataset else None,
        weights=(args.alpha, args.beta, args.gamma),
    )

    print_report(report)

    output = args.output if args.output else args.kb / "quality_report.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт сохранён: {output}")

    if not args.no_plot:
        plot_output = args.plot_output if args.plot_output else args.kb / "quality_metrics.png"
        plot_metrics(report, str(plot_output))
        print(f"Визуализация сохранена: {plot_output}")


if __name__ == "__main__":
    main()
