"""
Быстрый дебаг-запуск для проверки всех компонентов проекта.
Запуск: python debug.py [--step all|embed|llm|parse|graph|rag]
"""

import asyncio
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# ─── Пути ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
DZ2_DIR       = Path(__file__).parent
DATASET_PATH  = PROJECT_ROOT / "datasets" / "output.json"
KB_DIR        = DZ2_DIR / "knowledge_graph"
DEBUG_KB_DIR  = DZ2_DIR / "debug_knowledge_graph"  # отдельная база для дебага

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("debug")

# ─── Вспомогательные утилиты ─────────────────────────────────────────────────

def section(title: str):
    print("\n" + "═" * 60)
    print(f"  {title}")
    print("═" * 60)


def ok(msg: str):
    print(f"  ✅ {msg}")


def fail(msg: str):
    print(f"  ❌ {msg}")


def info(msg: str):
    print(f"  ℹ️  {msg}")


def warn(msg: str):
    print(f"  ⚠️  {msg}")


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 1: Проверка датасета
# ═════════════════════════════════════════════════════════════════════════════

def step_dataset():
    section("ШАГ 1: Проверка датасета")

    if not DATASET_PATH.exists():
        fail(f"Файл не найден: {DATASET_PATH}")
        return None

    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    ok(f"Датасет загружен: {len(data)} записей")

    # Статистика по классам
    classes: dict[str, int] = {}
    empty_latex = 0
    for item in data:
        cls = item.get("class", "unknown")
        classes[cls] = classes.get(cls, 0) + 1
        if not item.get("latex", "").strip():
            empty_latex += 1

    info(f"Записей без LaTeX: {empty_latex}")
    info("Классы:")
    for cls, cnt in sorted(classes.items(), key=lambda x: -x[1])[:10]:
        print(f"      {cls:<30}: {cnt}")

    # Показываем 3 примера
    info("Примеры записей:")
    for i, item in enumerate(data[:3]):
        print(f"    [{i}] text  : {item.get('text', '')[:60]}")
        print(f"         latex : {item.get('latex', '')[:80]}")
        print(f"         class : {item.get('class', '')}")
        print()

    return data


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 2: Проверка LaTeX-парсера
# ═════════════════════════════════════════════════════════════════════════════

def step_parser(data: list | None = None):
    section("ШАГ 2: Проверка LaTeX-парсера")

    from src.formula_parser import LaTeXParser

    parser = LaTeXParser()

    # Тестовые формулы
    test_cases = [
        {
            "text":  "Производная функции",
            "latex": r"\frac{\partial f}{\partial x} + \frac{\partial f}{\partial y} = 0",
            "class": "derivative_limit",
        },
        {
            "text":  "Интеграл",
            "latex": r"\int_{\mathbb{R}^{k}}\mathcal{L}^{*}[\nu]\psi(x)\varphi(x)dx"
                     r"=\int_{\mathbb{R}^{k}}\psi(x)\mathcal{L}[\nu]\varphi(x)dx.",
            "class": "equation",
        },
        {
            "text":  "Предел",
            "latex": r"\lim_{n \to \infty} \left( 1 + \frac{1}{n} \right)^n = e",
            "class": "derivative_limit",
        },
        {
            "text":  "Сумма",
            "latex": r"\sum_{k=0}^{n} x^k = \frac{1 - x^{n+1}}{1 - x}",
            "class": "equation",
        },
    ]

    # Если передан датасет — добавляем несколько реальных примеров
    if data:
        for item in data[10:13]:
            if item.get("latex"):
                test_cases.append(item)

    errors = 0
    for i, case in enumerate(test_cases):
        try:
            t0    = time.perf_counter()
            graph = parser.parse(
                latex=case["latex"],
                formula_id=f"debug_{i}",
                source_text=case.get("text", ""),
                formula_class=case.get("class", "unknown"),
            )
            elapsed = (time.perf_counter() - t0) * 1000

            ok(
                f"[{i}] «{case.get('text','')[:40]}» → "
                f"{len(graph.nodes)} вершин, {len(graph.edges)} рёбер "
                f"({elapsed:.1f} мс)"
            )

            # Детали
            for node in graph.nodes[:4]:
                print(f"       вершина: [{node.node_type}] {node.label}")
            for edge in graph.edges[:3]:
                src = next((n.label for n in graph.nodes if n.id == edge.source_id), "?")
                tgt = next((n.label for n in graph.nodes if n.id == edge.target_id), "?")
                print(f"       ребро  : {src} --[{edge.operation}]--> {tgt}")

        except Exception as e:
            fail(f"[{i}] Ошибка парсинга: {e}")
            errors += 1

    if errors == 0:
        ok("Парсер работает корректно")
    else:
        warn(f"Парсер: {errors} ошибок из {len(test_cases)}")


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 3: Проверка embed-функции
# ═════════════════════════════════════════════════════════════════════════════

async def step_embed():
    section("ШАГ 3: Проверка embed-функции (nomic-embed-text через Ollama)")

    from src.lightrag_manager import embed_func, EMBEDDING_DIM

    test_batches = [
        # (описание, тексты)
        ("Нормальные тексты",    ["частная производная", "интеграл Лебега", "предел последовательности"]),
        ("С пустыми строками",   ["формула", "", "уравнение", "   ", "переменная"]),
        ("Один текст",           ["единственный элемент"]),
        ("Только пустые",        ["", "   ", ""]),
        ("LaTeX-фрагменты",      [
            r"\frac{\partial f}{\partial x}",
            r"\int_{\mathbb{R}} f(x) dx",
            r"\lim_{n\to\infty} a_n = L",
        ]),
    ]

    all_ok = True
    for desc, texts in test_batches:
        try:
            t0     = time.perf_counter()
            result = await embed_func(texts)
            elapsed = (time.perf_counter() - t0) * 1000

            expected_shape = (len(texts), EMBEDDING_DIM)
            shape_ok       = (result.shape == expected_shape)
            dtype_ok       = (result.dtype == np.float32)

            status = "✅" if (shape_ok and dtype_ok) else "❌"
            print(
                f"  {status} {desc:<30} | "
                f"входов={len(texts)} | "
                f"форма={result.shape} | "
                f"dtype={result.dtype} | "
                f"{elapsed:.0f}мс"
            )

            if not shape_ok:
                fail(f"  Ожидалось {expected_shape}, получено {result.shape}")
                all_ok = False

        except Exception as e:
            fail(f"  {desc}: {e}")
            all_ok = False

    if all_ok:
        ok("embed_func работает корректно для всех кейсов")
    else:
        warn("Есть проблемы с embed_func — см. ошибки выше")


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 4: Проверка LLM (qwen2.5 через Ollama)
# ═════════════════════════════════════════════════════════════════════════════

async def step_llm():
    section("ШАГ 4: Проверка LLM (qwen2.5 через Ollama)")

    from src.lightrag_manager import llm_func

    test_prompts = [
        (
            "Короткий вопрос",
            "Что такое производная функции? Ответь в одном предложении.",
            None,
        ),
        (
            "С system prompt",
            "Назови три математические операции.",
            "Ты — математический ассистент. Отвечай кратко.",
        ),
        (
            "LaTeX контекст",
            r"Объясни формулу: \frac{\partial f}{\partial x} + \frac{\partial f}{\partial y} = 0",
            "Объясняй математические формулы просто и кратко.",
        ),
    ]

    all_ok = True
    for desc, prompt, system in test_prompts:
        try:
            t0     = time.perf_counter()
            result = await llm_func(prompt, system_prompt=system)
            elapsed = (time.perf_counter() - t0)

            if result and len(result.strip()) > 5:
                ok(f"{desc:<25} | {elapsed:.1f}с | {len(result)} символов")
                info(f"  Ответ: {result[:120].strip()}...")
            else:
                warn(f"{desc}: пустой или слишком короткий ответ: '{result}'")
                all_ok = False

        except Exception as e:
            fail(f"{desc}: {e}")
            all_ok = False

    if all_ok:
        ok("LLM работает корректно")
    else:
        warn("Есть проблемы с LLM")


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 5: Проверка GraphBuilder
# ═════════════════════════════════════════════════════════════════════════════

def step_graph(data: list | None = None):
    section("ШАГ 5: Проверка GraphBuilder")

    from src.graph_builder import GraphBuilder

    builder = GraphBuilder()

    # Маленький тестовый датасет
    mini_dataset = [
        {
            "text":   "Производная",
            "latex":  r"\frac{\partial f}{\partial x} + \frac{\partial f}{\partial y} = 0",
            "source": "debug",
            "class":  "derivative_limit",
        },
        {
            "text":   "Интеграл",
            "latex":  r"\int_{\mathbb{R}^{k}}\psi(x)\varphi(x)dx = 1",
            "source": "debug",
            "class":  "equation",
        },
        {
            "text":   "Предел",
            "latex":  r"\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e",
            "source": "debug",
            "class":  "derivative_limit",
        },
    ]

    # Добавляем реальные данные если есть
    if data:
        mini_dataset += [
            d for d in data[:7]
            if d.get("latex", "").strip()
        ]

    try:
        t0     = time.perf_counter()
        graphs = builder.build_from_dataset(mini_dataset)
        elapsed = (time.perf_counter() - t0) * 1000

        stats = builder.get_stats()
        ok(f"Граф построен за {elapsed:.1f} мс")
        info(f"Формул обработано : {stats['total_formulas']}")
        info(f"Вершин в графе    : {stats['total_nodes']}")
        info(f"Рёбер в графе     : {stats['total_edges']}")
        info(f"Типы вершин       : {stats['node_types']}")
        info(f"Операции (рёбра)  : {stats['edge_operations']}")

        # Тест конвертации в документы
        docs = builder.to_lightrag_documents()
        ok(f"Подготовлено {len(docs)} документов для LightRAG")
        info(f"Первый документ (первые 300 символов):")
        print()
        print(docs[0][:300] if docs else "(пусто)")
        print()

        # Тест сохранения
        debug_out = DZ2_DIR / "debug_output"
        debug_out.mkdir(exist_ok=True)

        json_path = debug_out / "debug_graph.json"
        builder.save_graph_json(str(json_path))
        ok(f"Граф сохранён: {json_path}")

        png_path = debug_out / "debug_graph.png"
        builder.visualize_graph(str(png_path), max_nodes=30)
        ok(f"Граф визуализирован: {png_path}")

    except Exception as e:
        fail(f"Ошибка GraphBuilder: {e}")
        logger.exception(e)


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 6: Мини-тест LightRAG (3 документа)
# ═════════════════════════════════════════════════════════════════════════════

async def step_rag_mini():
    section("ШАГ 6: Мини-тест LightRAG (3 документа)")

    import shutil
    from src.lightrag_manager import LightRAGManager

    # Чистим дебаг-базу перед тестом
    if DEBUG_KB_DIR.exists():
        shutil.rmtree(DEBUG_KB_DIR)
        info("Старая дебаг-база удалена")

    manager = LightRAGManager(str(DEBUG_KB_DIR))
    manager.initialize()
    await manager.async_initialize()
    ok("LightRAG инициализирован")

    # 3 коротких документа
    mini_docs = [
        (
            "ФОРМУЛА: Производная функции\n"
            "КЛАСС: derivative_limit\n"
            "LATEX: \\frac{\\partial f}{\\partial x} + \\frac{\\partial f}{\\partial y} = 0\n\n"
            "ЭЛЕМЕНТЫ ФОРМУЛЫ:\n"
            "  - [FUNCTION] f: функция двух переменных\n"
            "  - [VARIABLE] x: первая переменная\n"
            "  - [VARIABLE] y: вторая переменная\n"
            "  - [EXPRESSION] ∂f/∂x: частная производная f по x\n"
            "  - [EXPRESSION] ∂f/∂y: частная производная f по y\n\n"
            "ОПЕРАЦИИ:\n"
            "  - f --[partial_derivative]--> ∂f/∂x\n"
            "  - f --[partial_derivative]--> ∂f/∂y\n"
            "  - ∂f/∂x --[addition]--> ∂f/∂y\n"
            "  - ∂f/∂x + ∂f/∂y --[equals]--> 0\n"
        ),
        (
            "ФОРМУЛА: Предел числа e\n"
            "КЛАСС: derivative_limit\n"
            "LATEX: \\lim_{n \\to \\infty} (1 + 1/n)^n = e\n\n"
            "ЭЛЕМЕНТЫ ФОРМУЛЫ:\n"
            "  - [VARIABLE] n: натуральное число\n"
            "  - [CONSTANT] ∞: бесконечность\n"
            "  - [CONSTANT] e: число Эйлера, основание натурального логарифма\n"
            "  - [EXPRESSION] (1+1/n)^n: выражение под пределом\n\n"
            "ОПЕРАЦИИ:\n"
            "  - n --[tends_to]--> ∞\n"
            "  - (1+1/n)^n --[limit_equals]--> e\n"
        ),
        (
            "ФОРМУЛА: Интегральное тождество\n"
            "КЛАСС: equation\n"
            "LATEX: \\int_{R^k} L*[v] psi(x) phi(x) dx = \\int_{R^k} psi(x) L[v] phi(x) dx\n\n"
            "ЭЛЕМЕНТЫ ФОРМУЛЫ:\n"
            "  - [OPERATOR] L*: сопряжённый оператор\n"
            "  - [OPERATOR] L: оператор\n"
            "  - [FUNCTION] ψ: тестовая функция\n"
            "  - [FUNCTION] φ: тестовая функция\n"
            "  - [EXPRESSION] ∫[R^k]: интеграл по R^k\n\n"
            "ОПЕРАЦИИ:\n"
            "  - L* --[adjoint_of]--> L\n"
            "  - ∫[R^k] L*ψφ --[equals]--> ∫[R^k] ψLφ\n"
        ),
    ]

    # Вставляем документы
    try:
        t0 = time.perf_counter()
        await manager.insert_documents(mini_docs)
        elapsed = time.perf_counter() - t0
        ok(f"3 документа вставлены за {elapsed:.1f}с")
    except Exception as e:
        fail(f"Ошибка вставки: {e}")
        logger.exception(e)
        return

    # Тестовые запросы
    test_queries = [
        ("naive",  "Что такое частная производная?"),
        ("local",  "Какие операции применяются к функции f?"),
        ("hybrid", "Чему равен предел (1+1/n)^n при n → ∞?"),
    ]

    info("Тестовые запросы:")
    for mode, question in test_queries:
        try:
            t0     = time.perf_counter()
            answer = await manager.query(question, mode=mode)
            elapsed = time.perf_counter() - t0

            ok(f"[{mode}] {question[:50]} ({elapsed:.1f}с)")
            print(f"       Ответ: {answer[:200].strip()}")
            print()

        except Exception as e:
            fail(f"[{mode}] Запрос упал: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# ШАГ 7: Проверка Ollama (доступность сервера и моделей)
# ═════════════════════════════════════════════════════════════════════════════

async def step_ollama():
    section("ШАГ 0: Проверка Ollama")

    import aiohttp

    base_url = "http://localhost:11434"

    # Проверяем доступность сервера
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    ok(f"Ollama сервер доступен. Моделей: {len(models)}")

                    needed = ["qwen2.5", "nomic-embed-text"]
                    for model in needed:
                        found = any(model in m for m in models)
                        if found:
                            ok(f"Модель найдена: {model}")
                        else:
                            fail(f"Модель НЕ найдена: {model}")
                            warn(f"  Установите: ollama pull {model}")
                else:
                    fail(f"Ollama вернул статус {resp.status}")
    except Exception as e:
        fail(f"Ollama недоступен: {e}")
        warn("  Убедитесь что Ollama запущен: ollama serve")


# ═════════════════════════════════════════════════════════════════════════════
# Главная точка входа
# ═════════════════════════════════════════════════════════════════════════════

STEPS = {
    "ollama":  ("Проверка Ollama",           True),   # (описание, async?)
    "dataset": ("Проверка датасета",         False),
    "parse":   ("Проверка парсера",          False),
    "embed":   ("Проверка embed-функции",    True),
    "llm":     ("Проверка LLM",              True),
    "graph":   ("Проверка GraphBuilder",     False),
    "rag":     ("Мини-тест LightRAG",        True),
}


async def run_all():
    t_start = time.perf_counter()

    await step_ollama()
    data = step_dataset()
    step_parser(data)
    await step_embed()
    await step_llm()
    step_graph(data)
    await step_rag_mini()

    elapsed = time.perf_counter() - t_start
    section(f"ИТОГО: все шаги выполнены за {elapsed:.1f}с")


async def run_step(step: str):
    data = None

    if step == "ollama":
        await step_ollama()

    elif step == "dataset":
        step_dataset()

    elif step == "parse":
        if DATASET_PATH.exists():
            with open(DATASET_PATH, encoding="utf-8") as f:
                data = json.load(f)
        step_parser(data)

    elif step == "embed":
        await step_embed()

    elif step == "llm":
        await step_llm()

    elif step == "graph":
        if DATASET_PATH.exists():
            with open(DATASET_PATH, encoding="utf-8") as f:
                data = json.load(f)
        step_graph(data)

    elif step == "rag":
        await step_rag_mini()

    else:
        fail(f"Неизвестный шаг: {step}")
        info(f"Доступные шаги: {', '.join(STEPS.keys())} | all")


def main():
    parser = argparse.ArgumentParser(
        description="Дебаг-запуск компонентов проекта DZ_2",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="\n".join(
            f"  {k:<10} — {v[0]}"
            for k, v in STEPS.items()
        ) + "\n  all        — все шаги подряд",
    )
    parser.add_argument(
        "--step",
        default="all",
        choices=[*STEPS.keys(), "all"],
        help="Какой шаг запустить (default: all)",
    )
    args = parser.parse_args()

    if args.step == "all":
        asyncio.run(run_all())
    else:
        asyncio.run(run_step(args.step))


if __name__ == "__main__":
    main()
