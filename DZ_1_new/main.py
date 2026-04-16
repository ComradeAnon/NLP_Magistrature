"""
Основной скрипт сборки датасета.

Запуск (пересборка по кнопке):
    python main.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from config import (
    DOCS_DIR,
    DATASET_FILE,
    DATASET_STATS_FILE,
    LOGS_DIR,
    CLASS_KEYS,
    FORMULA_CLASSES,
    MIN_EXAMPLES_PER_CLASS,
)
from dataset_builder import DatasetBuilder

# ─── Логирование ──────────────────────────────────────────────────────────────

log_file = LOGS_DIR / f"build_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger  = logging.getLogger(__name__)
console = Console()


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def preflight_check() -> bool:
    """Проверка наличия PDF перед запуском."""
    pdf_files = list(DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        console.print(Panel(
            f"[red]В папке [bold]{DOCS_DIR}[/bold] нет PDF-файлов!\n\n"
            f"Добавьте PDF одним из способов:\n"
            f"  [bold]python add_corpus.py[/bold]        — интерактивно\n"
            f"  [bold]python add_corpus.py --path doc.pdf[/bold]\n"
            f"  [bold]python add_corpus.py --url https://...[/bold]\n\n"
            f"[yellow]Без PDF датасет будет полностью синтетическим.[/yellow][/red]",
            title="⚠️  PDF не найдены"
        ))
        # Не прерываем — датасет будет синтетическим
        return False

    console.print(f"\n[green]Найдено PDF файлов: {len(pdf_files)}[/green]")
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / 1024 / 1024
        console.print(f"  📄 [cyan]{pdf.name}[/cyan] ({size_mb:.1f} MB)")

    return True


def show_stats(stats: dict) -> None:
    """Вывод итоговой статистики."""
    console.print(Panel(
        f"[bold green]Датасет успешно собран![/bold green]\n"
        f"Дата    : {stats['generated_at']}\n"
        f"Всего   : [bold]{stats['total']}[/bold] примеров\n"
        f"Файл    : {DATASET_FILE}",
        title="✅ Результат"
    ))

    # ── Таблица по классам
    table = Table(title="Распределение по классам", show_lines=True)
    table.add_column("Класс",       style="cyan",    no_wrap=True, min_width=15)
    table.add_column("Описание",    style="white",   min_width=35)
    table.add_column("Кол-во",      style="green",   justify="right")
    table.add_column("Статус",      style="yellow",  justify="center")

    for cls in CLASS_KEYS:
        count  = stats["by_class"].get(cls, 0)
        desc   = FORMULA_CLASSES.get(cls, "")
        status = "✅" if count >= MIN_EXAMPLES_PER_CLASS else "⚠️"
        table.add_row(cls, desc, str(count), status)

    console.print(table)

    # ── Таблица по источникам
    if stats.get("by_source"):
        src_table = Table(title="Источники", show_lines=True)
        src_table.add_column("Источник",  style="cyan")
        src_table.add_column("Формул",    style="green", justify="right")
        src_table.add_column("Тип",       style="yellow")

        for src, count in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
            src_type = "🔬 синтетика" if src == "synthetic" else "📄 PDF"
            src_table.add_row(src, str(count), src_type)

        console.print(src_table)


def show_examples(dataset: list) -> None:
    """Показывает по одному примеру из каждого класса."""
    console.print("\n[bold]Примеры из датасета:[/bold]")

    shown = set()
    for item in dataset:
        cls = item["class"]
        if cls in shown:
            continue

        console.print(
            f"\n  [bold cyan][{cls}][/bold cyan]\n"
            f"  text   : [white]{item['text'][:80]}[/white]\n"
            f"  latex  : [yellow]{item['latex'][:80]}[/yellow]\n"
            f"  source : [green]{item['source']}[/green]"
        )
        shown.add(cls)

        if len(shown) >= len(CLASS_KEYS):
            break


# ─── Точка входа ─────────────────────────────────────────────────────────────

def main():
    console.print(Panel(
        "[bold blue]Сборщик датасета математических формул[/bold blue]\n"
        "Группа Гр3 | Задача: формула = число\n"
        "Классы: algebraic | trigonometric | logarithmic | "
        "combinatorial | series_limit",
        title="🔢 Math Formula Dataset Builder"
    ))

    # Проверка PDF (предупреждение, не остановка)
    has_pdfs = preflight_check()
    if not has_pdfs:
        console.print(
            "\n[yellow]Продолжаем без PDF — "
            "датасет будет собран из синтетических данных.[/yellow]\n"
        )

    # ── Запуск сборки
    logger.info("Запуск сборки датасета")
    builder = DatasetBuilder(seed=42)

    try:
        items, stats = builder.build()
    except Exception as exc:
        logger.exception("Критическая ошибка сборки: %s", exc)
        console.print(f"\n[red bold]Ошибка: {exc}[/red bold]")
        sys.exit(1)

    # ── Вывод результатов
    show_stats(stats)

    # ── Примеры
    try:
        with open(DATASET_FILE, encoding="utf-8") as f:
            dataset = json.load(f)
        show_examples(dataset)
    except Exception as exc:
        logger.warning("Не удалось показать примеры: %s", exc)

    # ── Финальное сообщение
    console.print(f"\n[dim]Лог сохранён: {log_file}[/dim]")
    console.print("\n[bold green]Готово![/bold green] "
                  "Для пересборки запустите снова: [bold]python main.py[/bold]\n")


if __name__ == "__main__":
    main()
