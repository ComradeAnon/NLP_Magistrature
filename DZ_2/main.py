"""
Главный модуль проекта DZ_2.
"""

import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT       = Path(__file__).parent.parent
DZ2_DIR            = Path(__file__).parent
DEFAULT_DATASET    = PROJECT_ROOT / "datasets" / "output.json"
DEFAULT_KB_DIR     = DZ2_DIR / "knowledge_graph"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DZ2_DIR / "build.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


async def build_knowledge_graph(
    dataset_path: Path,
    kb_dir: Path,
    limit: int | None = None,
):
    from src.graph_builder import GraphBuilder
    from src.lightrag_manager import LightRAGManager

    logger.info("=" * 60)
    logger.info("Запуск построения графовой базы знаний")
    logger.info(f"Датасет     : {dataset_path}")
    logger.info(f"База знаний : {kb_dir}")
    logger.info(f"Записей     : {'первые ' + str(limit) if limit else 'все'}")
    logger.info("=" * 60)

    graph_builder = GraphBuilder()
    rag_manager   = LightRAGManager(str(kb_dir))
    rag_manager.initialize()

    stats = await rag_manager.build_and_index(
        dataset_path=str(dataset_path),
        graph_builder=graph_builder,
        limit=limit,
    )

    logger.info("\n" + "=" * 60)
    logger.info("СТАТИСТИКА:")
    logger.info(f"  Формул обработано : {stats['total_formulas']}")
    logger.info(f"  Вершин в графе    : {stats['total_nodes']}")
    logger.info(f"  Рёбер в графе     : {stats['total_edges']}")
    logger.info("\n  Типы вершин:")
    for ntype, count in stats["node_types"].items():
        logger.info(f"    {ntype:<20}: {count}")
    logger.info("\n  Операции (рёбра):")
    for op, count in sorted(
        stats["edge_operations"].items(), key=lambda x: -x[1]
    ):
        logger.info(f"    {op:<30}: {count}")
    logger.info("=" * 60)

    return rag_manager


async def interactive_query(rag_manager):
    print("\n" + "=" * 60)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print("Введите 'exit' для выхода")
    print("Добавьте --mode=naive|local|global|hybrid (default: hybrid)")
    print("=" * 60)

    example_queries = [
        "Чему равен дискриминант квадратного уравнения?",
        "Как выразить корни уравнения через дискриминант?",
        "Вырази sin(2x) через sin(x) и cos(x)",
        "Как связаны площадь круга и радиус?",
        "Чему равна производная sin(x)?",
        "Вырази длину окружности через диаметр",
        "Как связаны логарифм произведения и сумма логарифмов?",
    ]

    print("\nПримеры (введите номер или свой запрос):")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q}")
    print()

    while True:
        try:
            user_input = input("Запрос> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input or user_input.lower() in ("exit", "quit", "выход"):
            break

        # Парсим --mode=
        mode = "hybrid"
        if "--mode=" in user_input:
            parts      = user_input.rsplit("--mode=", 1)
            user_input = parts[0].strip()
            mode       = parts[1].strip()

        # Номер из примеров
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(example_queries):
                user_input = example_queries[idx]
                print(f"Запрос: {user_input}")

        print(f"\n[{mode}] ...")
        print("-" * 40)
        try:
            result = await rag_manager.query(user_input, mode=mode)
            print(result)
        except Exception as e:
            print(f"Ошибка: {e}")
        print("-" * 40 + "\n")


async def demo_mode(rag_manager):
    queries = [
        ("hybrid", "Чему равен дискриминант квадратного уравнения?"),
        ("local",  "Как выразить корни уравнения через дискриминант?"),
        ("hybrid", "Вырази sin(2x) через sin(x) и cos(x)"),
        ("global", "Как связаны площадь круга и радиус?"),
    ]

    print("\n" + "=" * 60)
    print("ДЕМО-ЗАПРОСЫ")
    print("=" * 60)

    for mode, question in queries:
        print(f"\n[{mode.upper()}] {question}")
        print("-" * 40)
        try:
            result = await rag_manager.query(question, mode=mode)
            print(result[:500] + "..." if len(result) > 500 else result)
        except Exception as e:
            logger.error(f"Ошибка: {e}")


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Графовая база знаний из математических формул",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["build", "query", "demo", "all"],
        default="build",
        help=(
            "build — построить граф из датасета\n"
            "query — интерактивные запросы к существующей базе\n"
            "demo  — демо-запросы к существующей базе\n"
            "all   — построить и запустить демо + интерактив"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        metavar="PATH",
        help=(
            "Путь к файлу датасета JSON.\n"
            f"По умолчанию: {DEFAULT_DATASET}"
        ),
    )
    parser.add_argument(
        "--kb",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Путь к папке базы знаний.\n"
            "При build: куда сохранить базу.\n"
            "При query/demo: откуда загрузить базу.\n"
            f"По умолчанию: {DEFAULT_KB_DIR}"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Обработать только первые N записей датасета.",
    )

    args = parser.parse_args()

    # ── Определяем путь к базе знаний ────────────────────────────────────────
    if args.kb is not None:
        kb_dir = args.kb
    elif args.limit is not None:
        # Автоматическое имя папки при использовании лимита
        kb_dir = DZ2_DIR / f"knowledge_graph_limit_{args.limit}"
    else:
        kb_dir = DEFAULT_KB_DIR

    logger.info(f"База знаний : {kb_dir}")

    try:
        if args.mode in ("build", "all"):
            # Проверяем датасет
            if not args.dataset.exists():
                logger.error(f"Датасет не найден: {args.dataset}")
                sys.exit(1)

            rag_manager = await build_knowledge_graph(
                dataset_path=args.dataset,
                kb_dir=kb_dir,
                limit=args.limit,
            )

        else:
            # Загружаем существующую базу
            from src.lightrag_manager import LightRAGManager

            if not kb_dir.exists():
                logger.error(
                    f"База знаний не найдена: {kb_dir}\n"
                    f"Сначала постройте её:\n"
                    f"  python main.py --mode build --dataset <путь> --kb {kb_dir}"
                )
                sys.exit(1)

            logger.info(f"Загружаем существующую базу: {kb_dir}")
            rag_manager = LightRAGManager(str(kb_dir))
            rag_manager.initialize()
            await rag_manager.async_initialize()
            logger.info("База знаний загружена")

        # ── Запускаем нужный режим ────────────────────────────────────────────
        if args.mode == "query":
            await interactive_query(rag_manager)
        elif args.mode == "demo":
            await demo_mode(rag_manager)
        elif args.mode == "all":
            await demo_mode(rag_manager)
            await interactive_query(rag_manager)
        # build — уже выполнен выше

    except FileNotFoundError as e:
        logger.error(f"Файл не найден: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
