"""
Построитель графовой базы знаний.
Читает только поле latex из датасета.
Конвертирует формулы в граф и текстовые документы для LightRAG.
"""

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .formula_parser import LaTeXParser, FormulaGraph, MathNode, MathEdge

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Строит граф знаний из датасета математических формул.
    Использует ТОЛЬКО поле latex из каждой записи датасета.
    Вершины = атомарные математические элементы.
    Рёбра   = операции между элементами.
    """

    def __init__(self):
        self.parser        = LaTeXParser()
        self.global_graph  = nx.MultiDiGraph()
        self.formula_graphs: list[FormulaGraph] = []

    # ─── Загрузка датасета ───────────────────────────────────────────────────

    def load_dataset(self, dataset_path: str) -> list[dict]:
        """Загружает датасет из JSON файла."""
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Датасет не найден: {dataset_path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Загружено {len(data)} записей из {dataset_path}")
        return data

    # ─── Построение графа ────────────────────────────────────────────────────

    def build_from_dataset(self, dataset: list[dict]) -> list[FormulaGraph]:
        """
        Строит граф из датасета.
        Читает ТОЛЬКО поле latex из каждой записи.
        """
        self.formula_graphs = []
        self.global_graph   = nx.MultiDiGraph()
        # Сбрасываем реестр узлов парсера для чистого старта
        self.parser._node_registry = {}
        self.parser._counter       = 0

        skipped = 0
        for i, item in enumerate(dataset):
            latex = item.get('latex', '').strip()

            if not latex:
                skipped += 1
                continue

            formula_id = f"formula_{i}"

            graph = self.parser.parse(
                latex=latex,
                formula_id=formula_id,
            )

            self.formula_graphs.append(graph)
            self._add_to_global_graph(graph)

        logger.info(
            f"Построено {len(self.formula_graphs)} графов формул "
            f"(пропущено {skipped} без latex). "
            f"Глобальный граф: "
            f"{self.global_graph.number_of_nodes()} вершин, "
            f"{self.global_graph.number_of_edges()} рёбер"
        )

        return self.formula_graphs

    def _add_to_global_graph(self, fg: FormulaGraph):
        """Добавляет граф формулы в глобальный граф NetworkX."""
        for node in fg.nodes:
            self.global_graph.add_node(
                node.id,
                label=node.label,
                node_type=node.node_type,
                latex=node.latex,
                description=node.description,
                formula_id=fg.formula_id,
            )

        for edge in fg.edges:
            self.global_graph.add_edge(
                edge.source_id,
                edge.target_id,
                operation=edge.operation,
                description=edge.description,
                weight=edge.weight,
                formula_id=edge.formula_id,
            )

    # ─── Конвертация в документы для LightRAG ────────────────────────────────

    def to_lightrag_documents(self) -> list[str]:
        """
        Конвертирует графы формул в текстовые документы для LightRAG.
        Каждый документ описывает одну формулу через её граф.
        """
        documents = []
        for fg in self.formula_graphs:
            doc = self._formula_graph_to_text(fg)
            documents.append(doc)

        logger.info(f"Подготовлено {len(documents)} документов для LightRAG")
        return documents

    def _formula_graph_to_text(self, fg: FormulaGraph) -> str:
        """
        Конвертирует граф формулы в текстовое описание.
        Источник информации — ТОЛЬКО поле latex.
        """
        lines = [
            f"Формула (LaTeX): {fg.latex}",
            f"Идентификатор: {fg.formula_id}",
            "",
        ]

        # Группируем узлы по типу
        by_type: dict[str, list[MathNode]] = {}
        for node in fg.nodes:
            by_type.setdefault(node.node_type, []).append(node)

        if by_type.get('variable'):
            vars_str = ', '.join(n.label for n in by_type['variable'])
            lines.append(f"Переменные: {vars_str}.")

        if by_type.get('number'):
            num_str = ', '.join(n.label for n in by_type['number'])
            lines.append(f"Числа: {num_str}.")

        if by_type.get('constant'):
            const_str = ', '.join(
                f"{n.label} ({n.description})"
                for n in by_type['constant']
            )
            lines.append(f"Константы: {const_str}.")

        if by_type.get('function'):
            fn_str = ', '.join(n.label for n in by_type['function'])
            lines.append(f"Функции: {fn_str}.")

        if by_type.get('operator'):
            op_str = ', '.join(
                f"{n.label} ({n.description})"
                for n in by_type['operator']
            )
            lines.append(f"Операторы: {op_str}.")

        lines.append("")
        lines.append("Математические отношения:")

        for edge in fg.edges:
            src = self._get_node_label(fg, edge.source_id)
            tgt = self._get_node_label(fg, edge.target_id)
            nat = self._edge_to_natural(edge.operation, src, tgt)
            lines.append(f"  - {nat}")

        # Явные утверждения для ключевых отношений
        lines.append("")
        equals_edges = [e for e in fg.edges if e.operation == 'equals']
        for e in equals_edges:
            src = self._get_node_label(fg, e.source_id)
            tgt = self._get_node_label(fg, e.target_id)
            lines.append(
                f"Утверждение: {src} равно {tgt} "
                f"(формула {fg.formula_id}, LaTeX: {fg.latex})."
            )

        return '\n'.join(lines)

    def _get_node_label(self, fg: FormulaGraph, node_id: str) -> str:
        """Возвращает метку узла по ID."""
        for node in fg.nodes:
            if node.id == node_id:
                return node.label
        return node_id

    def _edge_to_natural(self, operation: str, src: str, tgt: str) -> str:
        """Преобразует ребро в естественно-языковое описание."""
        op_map = {
            'equals':           f"{src} равно {tgt}",
            'addition':         f"{src} прибавляется к {tgt}",
            'subtraction':      f"{src} вычитается из {tgt}",
            'multiplication':   f"{src} умножается на {tgt}",
            'division':         f"{src} делится на {tgt}",
            'power':            f"{src} возводится в степень {tgt}",
            'subscript':        f"{src} имеет нижний индекс {tgt}",
            'less_than':        f"{src} меньше {tgt}",
            'greater_than':     f"{src} больше {tgt}",
            'less_or_equal':    f"{src} меньше или равно {tgt}",
            'greater_or_equal': f"{src} больше или равно {tgt}",
            'maps_to':          f"{src} стремится к {tgt}",
            'belongs_to':       f"{src} принадлежит {tgt}",
            'function_of':      f"{src} является функцией от {tgt}",
            'applied_to':       f"оператор {src} применяется к {tgt}",
            'parameter':        f"{tgt} является параметром оператора {src}",
            'upper_bound':      f"{tgt} является верхней границей {src}",
            'adjacent':         f"{src} и {tgt} стоят рядом в выражении",
            'tensor_product':   f"{src} тензорное произведение {tgt}",
            'direct_sum':       f"{src} прямая сумма с {tgt}",
            'union':            f"{src} объединяется с {tgt}",
            'intersection':     f"{src} пересекается с {tgt}",
            'subset':           f"{src} является подмножеством {tgt}",
            'approx_equal':     f"{src} приблизительно равно {tgt}",
            'equivalent':       f"{src} эквивалентно {tgt}",
            'implies':          f"{src} влечёт {tgt}",
            'iff':              f"{src} тогда и только тогда когда {tgt}",
        }
        return op_map.get(operation, f"{src} [{operation}] {tgt}")

    # ─── Сохранение и визуализация ───────────────────────────────────────────

    def save_graph_json(self, output_path: str):
        """Сохраняет граф в JSON формате."""
        graph_data = {
            "nodes": [],
            "edges": [],
            "stats": {
                "total_nodes":    self.global_graph.number_of_nodes(),
                "total_edges":    self.global_graph.number_of_edges(),
                "total_formulas": len(self.formula_graphs),
            }
        }

        for node_id, attrs in self.global_graph.nodes(data=True):
            graph_data["nodes"].append({"id": node_id, **attrs})

        for src, tgt, attrs in self.global_graph.edges(data=True):
            graph_data["edges"].append({"source": src, "target": tgt, **attrs})

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Граф сохранён в {output_path}")

    def visualize_graph(self, output_path: str, max_nodes: int = 50):
        """Визуализирует граф и сохраняет как PNG."""
        if self.global_graph.number_of_nodes() == 0:
            logger.warning("Граф пустой, визуализация невозможна")
            return

        nodes    = list(self.global_graph.nodes())[:max_nodes]
        subgraph = self.global_graph.subgraph(nodes)

        plt.figure(figsize=(20, 15))

        color_map = {
            'variable': '#74b9ff',
            'constant': '#fd79a8',
            'function': '#55efc4',
            'operator': '#a29bfe',
            'number':   '#ffeaa7',
        }

        node_colors = []
        node_labels = {}
        for node_id in subgraph.nodes():
            attrs     = subgraph.nodes[node_id]
            ntype     = attrs.get('node_type', 'variable')
            node_colors.append(color_map.get(ntype, '#dfe6e9'))
            node_labels[node_id] = attrs.get('label', node_id)[:10]

        edge_labels = {}
        for src, tgt, attrs in subgraph.edges(data=True):
            op = attrs.get('operation', '')
            edge_labels[(src, tgt)] = op[:12]

        pos = nx.spring_layout(subgraph, k=3, seed=42)

        nx.draw_networkx_nodes(
            subgraph, pos,
            node_color=node_colors,
            node_size=1200,
            alpha=0.9,
        )
        nx.draw_networkx_labels(
            subgraph, pos,
            labels=node_labels,
            font_size=7,
        )
        nx.draw_networkx_edges(
            subgraph, pos,
            edge_color='#636e72',
            arrows=True,
            arrowsize=20,
            width=1.5,
            alpha=0.7,
        )
        nx.draw_networkx_edge_labels(
            subgraph, pos,
            edge_labels=edge_labels,
            font_size=6,
        )

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color=color, label=ntype)
            for ntype, color in color_map.items()
        ]
        plt.legend(handles=legend_elements, loc='upper left', title="Типы узлов")
        plt.title(
            f"Граф математических знаний\n"
            f"({subgraph.number_of_nodes()} вершин, "
            f"{subgraph.number_of_edges()} рёбер)",
            fontsize=14,
        )
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Граф визуализирован и сохранён в {output_path}")

    def get_stats(self) -> dict[str, Any]:
        """Возвращает статистику по графу."""
        node_types: dict[str, int] = {}
        edge_ops:   dict[str, int] = {}

        for _, attrs in self.global_graph.nodes(data=True):
            ntype = attrs.get('node_type', 'unknown')
            node_types[ntype] = node_types.get(ntype, 0) + 1

        for _, _, attrs in self.global_graph.edges(data=True):
            op = attrs.get('operation', 'unknown')
            edge_ops[op] = edge_ops.get(op, 0) + 1

        return {
            "total_nodes":    self.global_graph.number_of_nodes(),
            "total_edges":    self.global_graph.number_of_edges(),
            "total_formulas": len(self.formula_graphs),
            "node_types":     node_types,
            "edge_operations": edge_ops,
        }
