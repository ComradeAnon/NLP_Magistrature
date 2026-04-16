"""
╔══════════════════════════════════════════════════════════════════╗
║           ОБОСОБЛЕННЫЙ СКРИПТ ПОПОЛНЕНИЯ КОРПУСА               ║
║                                                                  ║
║  Этот скрипт полностью независим от остального проекта.         ║
║  Единственная зависимость — папка docs/ куда кладутся PDF.      ║
║                                                                  ║
║  Способы запуска:                                               ║
║                                                                  ║
║  python add_corpus.py                    # интерактивный режим  ║
║  python add_corpus.py --path file.pdf    # добавить файл        ║
║  python add_corpus.py --dir /папка/      # добавить папку       ║
║  python add_corpus.py --dir /п/ --recursive  # рекурсивно      ║
║  python add_corpus.py --url https://...  # скачать PDF          ║
║  python add_corpus.py --show             # показать корпус      ║
║  python add_corpus.py --check            # диагностика типов    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# ─── Константы (независимые от config.py) ────────────────────────────────────

BASE_DIR         = Path(__file__).parent
DOCS_DIR         = BASE_DIR / "docs"
LOGS_DIR         = BASE_DIR / "logs"
CORPUS_REGISTRY  = BASE_DIR / "corpus_registry.json"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Логирование ──────────────────────────────────────────────────────────────

log_file = LOGS_DIR / f"corpus_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("add_corpus")


# ─── Реестр файлов корпуса ───────────────────────────────────────────────────

def load_registry() -> dict:
    """Загружает реестр корпуса из JSON."""
    if CORPUS_REGISTRY.exists():
        with open(CORPUS_REGISTRY, encoding="utf-8") as f:
            return json.load(f)
    return {
        "files":      [],
        "hashes":     [],
        "added_at":   [],
        "sizes_mb":   [],
        "urls":       [],
    }


def save_registry(registry: dict) -> None:
    """Сохраняет реестр корпуса."""
    with open(CORPUS_REGISTRY, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def compute_hash(path: Path) -> str:
    """MD5-хэш файла для обнаружения дубликатов."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_registered(file_hash: str, registry: dict) -> bool:
    """Проверяет зарегистрирован ли файл по хэшу."""
    return file_hash in registry.get("hashes", [])


def register_file(
    filename: str,
    file_hash: str,
    size_mb: float,
    url: str,
    registry: dict,
) -> None:
    """Добавляет запись в реестр."""
    registry.setdefault("files",    []).append(filename)
    registry.setdefault("hashes",   []).append(file_hash)
    registry.setdefault("added_at", []).append(datetime.now().isoformat())
    registry.setdefault("sizes_mb", []).append(round(size_mb, 2))
    registry.setdefault("urls",     []).append(url)


# ─── Операции с файлами ───────────────────────────────────────────────────────

def add_file(
    src_path: Path,
    registry: dict,
    force: bool = False,
) -> bool:
    """
    Копирует PDF-файл в корпус (папку docs/).

    Args:
        src_path : исходный путь к PDF
        registry : текущий реестр
        force    : перезаписать если файл уже существует

    Returns:
        True если файл успешно добавлен
    """
    if not src_path.exists():
        logger.error("Файл не найден: %s", src_path)
        return False

    if src_path.suffix.lower() != ".pdf":
        logger.warning("Не PDF, пропуск: %s", src_path.name)
        return False

    # Вычисляем хэш источника
    src_hash = compute_hash(src_path)

    # Проверяем дубликат по содержимому
    if not force and is_registered(src_hash, registry):
        logger.info("Дубликат по содержимому, пропуск: %s", src_path.name)
        return False

    dest_path = DOCS_DIR / src_path.name

    # Файл с таким именем уже есть
    if dest_path.exists() and not force:
        logger.warning(
            "Файл '%s' уже существует. Используйте --force для замены.",
            src_path.name
        )
        return False

    # Копируем
    shutil.copy2(src_path, dest_path)
    size_mb = dest_path.stat().st_size / 1024 / 1024

    register_file(src_path.name, src_hash, size_mb, "", registry)
    logger.info("✅ Добавлен: %s (%.1f MB)", src_path.name, size_mb)
    return True


def download_pdf(url: str, registry: dict, force: bool = False) -> bool:
    """
    Скачивает PDF по URL в папку docs/.

    Returns:
        True если успешно скачан и добавлен
    """
    # Имя файла из URL
    filename = url.split("/")[-1].split("?")[0]
    if not filename.lower().endswith(".pdf"):
        filename = filename.rstrip("/") + ".pdf"

    # Убираем недопустимые символы из имени файла
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    filename   = "".join(c if c in safe_chars else "_" for c in filename)
    dest_path  = DOCS_DIR / filename

    logger.info("Скачивание: %s", url)
    logger.info("Сохранение: %s", dest_path)

    # Прогресс бар
    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest_path, _progress)
        print()  # новая строка
    except urllib.error.URLError as exc:
        logger.error("Ошибка загрузки: %s", exc)
        if dest_path.exists():
            dest_path.unlink()
        return False
    except KeyboardInterrupt:
        print("\nОтменено пользователем")
        if dest_path.exists():
            dest_path.unlink()
        return False

    # Проверяем что это PDF (магические байты)
    with open(dest_path, "rb") as f:
        header = f.read(4)

    if header != b"%PDF":
        logger.error("Скачанный файл не является PDF: %s", url)
        dest_path.unlink()
        return False

    # Проверяем дубликат
    file_hash = compute_hash(dest_path)
    if not force and is_registered(file_hash, registry):
        logger.info("Дубликат по содержимому, удаляем: %s", filename)
        dest_path.unlink()
        return False

    size_mb = dest_path.stat().st_size / 1024 / 1024
    register_file(filename, file_hash, size_mb, url, registry)
    logger.info("✅ Скачан: %s (%.1f MB)", filename, size_mb)
    return True


def add_directory(
    dir_path: Path,
    registry: dict,
    recursive: bool = False,
    force: bool = False,
) -> int:
    """
    Добавляет все PDF из директории.

    Returns:
        Количество добавленных файлов
    """
    if not dir_path.exists() or not dir_path.is_dir():
        logger.error("Директория не найдена: %s", dir_path)
        return 0

    pattern   = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(dir_path.glob(pattern))

    if not pdf_files:
        logger.warning("PDF не найдены в: %s", dir_path)
        return 0

    logger.info("Найдено PDF в '%s': %d", dir_path, len(pdf_files))

    added = 0
    for pdf_path in pdf_files:
        if add_file(pdf_path, registry, force=force):
            added += 1

    logger.info("Добавлено: %d из %d", added, len(pdf_files))
    return added


# ─── Отображение корпуса ──────────────────────────────────────────────────────

def show_corpus() -> None:
    """Показывает текущее содержимое корпуса."""
    pdf_files = sorted(DOCS_DIR.glob("*.pdf"))

    print("\n" + "=" * 65)
    print(f"  КОРПУС ТЕКСТОВ | папка: {DOCS_DIR}")
    print("=" * 65)

    if not pdf_files:
        print("  ⚠️  Корпус пуст. Добавьте PDF-файлы.")
        print("=" * 65 + "\n")
        return

    total_size = 0
    for pdf in pdf_files:
        size_mb     = pdf.stat().st_size / 1024 / 1024
        total_size += size_mb
        print(f"  📄 {pdf.name:<45} {size_mb:>6.1f} MB")

    print("  " + "-" * 60)
    print(f"  Итого: {len(pdf_files)} файлов, {total_size:.1f} MB")
    print("=" * 65 + "\n")


def check_types() -> None:
    """Диагностирует типы PDF в корпусе (DIGITAL/SCAN/MIXED)."""
    try:
        from check_pdf_type import check_all_pdfs
        results = check_all_pdfs(DOCS_DIR)

        if not results:
            print("Корпус пуст.")
            return

        print("\n" + "=" * 70)
        print("  ДИАГНОСТИКА ТИПОВ PDF")
        print("=" * 70)
        print(f"  {'Файл':<40} {'Тип':<10} {'OCR?'}")
        print("  " + "-" * 65)

        for r in results:
            icon = "⚠️ " if r.get("needs_ocr") else "✅"
            ocr  = "требуется" if r.get("needs_ocr") else "не нужен"
            print(f"  {icon} {r['file']:<40} {r.get('type','?'):<10} {ocr}")

        print("=" * 70 + "\n")

    except ImportError:
        print("check_pdf_type.py не найден. Диагностика недоступна.")


# ─── Интерактивный режим ──────────────────────────────────────────────────────

def interactive_mode(registry: dict) -> None:
    """Диалоговый режим для пополнения корпуса."""

    MENU = """
╔════════════════════════════════════════╗
║      ПОПОЛНЕНИЕ КОРПУСА ТЕКСТОВ       ║
╠════════════════════════════════════════╣
║  1. Добавить PDF по пути к файлу      ║
║  2. Добавить все PDF из папки         ║
║  3. Скачать PDF по URL                ║
║  4. Показать текущий корпус           ║
║  5. Диагностика типов PDF             ║
║  6. Выйти и пересобрать датасет       ║
╚════════════════════════════════════════╝"""

    modified = False

    while True:
        print(MENU)
        print(f"  Файлов в корпусе: {len(list(DOCS_DIR.glob('*.pdf')))}")
        choice = input("\n  Ваш выбор [1-6]: ").strip()

        if choice == "1":
            raw = input("  Путь к PDF: ").strip().strip('"').strip("'")
            if add_file(Path(raw), registry):
                save_registry(registry)
                modified = True
                print("  ✅ Добавлен!")
            else:
                print("  ❌ Не добавлен")

        elif choice == "2":
            raw       = input("  Путь к папке: ").strip().strip('"').strip("'")
            rec       = input("  Рекурсивно? [y/N]: ").strip().lower() == "y"
            count     = add_directory(Path(raw), registry, recursive=rec)
            if count:
                save_registry(registry)
                modified = True
                print(f"  ✅ Добавлено файлов: {count}")
            else:
                print("  ❌ Ничего не добавлено")

        elif choice == "3":
            url = input("  URL: ").strip()
            if download_pdf(url, registry):
                save_registry(registry)
                modified = True
                print("  ✅ Скачан и добавлен!")
            else:
                print("  ❌ Не удалось скачать")

        elif choice == "4":
            show_corpus()

        elif choice == "5":
            check_types()

        elif choice == "6":
            break

        else:
            print("  ⚠️  Неверный выбор")

    if modified:
        print("\n" + "=" * 50)
        print("  Корпус обновлён!")
        print("  Для пересборки датасета запустите:")
        print("    python main.py")
        print("=" * 50 + "\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="add_corpus",
        description="Пополнение корпуса PDF для датасета формул Гр3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python add_corpus.py
  python add_corpus.py --path /путь/к/файлу.pdf
  python add_corpus.py --dir /папка/с/pdf/ --recursive
  python add_corpus.py --url https://arxiv.org/pdf/2301.00001.pdf
  python add_corpus.py --show
  python add_corpus.py --check
        """
    )

    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument(
        "--path", type=Path, metavar="FILE",
        help="Путь к PDF-файлу"
    )
    src_group.add_argument(
        "--dir", type=Path, metavar="DIR",
        help="Папка с PDF-файлами"
    )
    src_group.add_argument(
        "--url", type=str, metavar="URL",
        help="URL для скачивания PDF"
    )
    src_group.add_argument(
        "--show", action="store_true",
        help="Показать текущий корпус"
    )
    src_group.add_argument(
        "--check", action="store_true",
        help="Диагностика типов PDF (DIGITAL/SCAN/MIXED)"
    )

    parser.add_argument(
        "--recursive", action="store_true",
        help="Рекурсивный поиск в папке (с --dir)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Перезаписать существующие файлы"
    )

    return parser


# ─── Точка входа ─────────────────────────────────────────────────────────────

def main():
    parser = build_arg_parser()
    args   = parser.parse_args()

    registry = load_registry()

    # Режим: просто показать
    if args.show:
        show_corpus()
        sys.exit(0)

    # Режим: диагностика
    if args.check:
        check_types()
        sys.exit(0)

    # Режим: нет аргументов → интерактивный
    if not any([args.path, args.dir, args.url]):
        interactive_mode(registry)
        sys.exit(0)

    # Режим: CLI
    success = False

    if args.path:
        success = add_file(args.path, registry, force=args.force)

    elif args.dir:
        count   = add_directory(
            args.dir, registry,
            recursive=args.recursive,
            force=args.force,
        )
        success = count > 0

    elif args.url:
        success = download_pdf(args.url, registry, force=args.force)

    if success:
        save_registry(registry)
        logger.info("Реестр обновлён: %s", CORPUS_REGISTRY)
        print("\n  Для пересборки датасета: python main.py\n")
    else:
        logger.error("Операция завершилась неудачей")
        sys.exit(1)


if __name__ == "__main__":
    main()
