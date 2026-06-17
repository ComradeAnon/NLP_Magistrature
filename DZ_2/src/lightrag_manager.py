"""
Менеджер LightRAG v1.5.0.
Использует qwen2.5:0.5b (LLM) и nomic-embed-text (embeddings) через Ollama.
Embeddings вызываются напрямую через HTTP — минуя ollama_embed из LightRAG
у которого hardcoded expected_dim=1024.
"""

import asyncio
import logging
from pathlib import Path

import httpx
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import _ollama_model_if_cache
from lightrag.utils import EmbeddingFunc

from .graph_builder import GraphBuilder

logger = logging.getLogger(__name__)

# ─── Параметры ───────────────────────────────────────────────────────────────
OLLAMA_MODEL       = "qwen2.5:3b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL    = "http://localhost:11434"

EMBEDDING_DIM    = 768
MAX_EMBED_TOKENS = 8192


# ─── Прямой HTTP-клиент для эмбеддингов ─────────────────────────────────────

async def _raw_ollama_embed(texts: list[str]) -> np.ndarray:
    """
    Вызывает Ollama /api/embed напрямую через httpx.
    Обходит ollama_embed из lightrag у которого hardcoded expected_dim=1024.
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={
                "model": OLLAMA_EMBED_MODEL,
                "input": texts,
            },
        )
        response.raise_for_status()
        data = response.json()

    # Ollama возвращает {"embeddings": [[...], [...]]}
    raw = data.get("embeddings") or data.get("embedding") or []
    return np.array(raw, dtype=np.float32)


async def _detect_embedding_dim() -> int:
    """Определяет реальную размерность эмбеддингов через прямой запрос."""
    try:
        arr = await _raw_ollama_embed(["test"])
        if arr.ndim == 2:
            dim = arr.shape[1]
        elif arr.ndim == 1:
            dim = arr.shape[0]
        else:
            dim = EMBEDDING_DIM
        logger.info(f"Определена размерность эмбеддингов: {dim} (shape={arr.shape})")
        return int(dim)
    except Exception as e:
        logger.warning(
            f"Не удалось определить размерность: {e} — используем {EMBEDDING_DIM}"
        )
        return EMBEDDING_DIM


# ─── embed_func для LightRAG ─────────────────────────────────────────────────

async def embed_func(texts: list[str]) -> np.ndarray:
    """
    Функция эмбеддингов для LightRAG.
    Использует прямой HTTP-запрос к Ollama вместо ollama_embed.

    Гарантирует:
      - shape == (len(texts), EMBEDDING_DIM)
      - dtype == float32
      - нулевые векторы для пустых строк
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    PLACEHOLDER = "математическое выражение"

    # Нормализуем тексты: пустые → placeholder
    clean: list[str] = []
    empty_idx: set[int] = set()

    for i, t in enumerate(texts):
        s = (t or "").strip()
        if not s:
            clean.append(PLACEHOLDER)
            empty_idx.add(i)
        else:
            clean.append(s[:MAX_EMBED_TOKENS * 3])

    # Вызываем Ollama напрямую
    try:
        arr = await _raw_ollama_embed(clean)
    except Exception as e:
        logger.error(f"_raw_ollama_embed упал для {len(clean)} текстов: {e}")
        return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)

    # ── Нормализуем форму ────────────────────────────────────────────────────
    if arr.ndim == 1:
        if len(clean) == 1:
            arr = arr.reshape(1, -1)
        elif arr.size % len(clean) == 0:
            arr = arr.reshape(len(clean), -1)
        else:
            logger.error(
                f"Не удалось reshape: size={arr.size}, n_texts={len(clean)}"
            )
            return np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)

    # ── Проверяем размерность ────────────────────────────────────────────────
    actual_dim = arr.shape[1] if arr.ndim == 2 else arr.shape[0]
    if actual_dim != EMBEDDING_DIM:
        logger.warning(
            f"Размерность {actual_dim} ≠ EMBEDDING_DIM {EMBEDDING_DIM}"
        )

    # ── Проверяем количество векторов ────────────────────────────────────────
    if len(arr) < len(texts):
        pad = np.zeros(
            (len(texts) - len(arr), arr.shape[1]),
            dtype=np.float32,
        )
        arr = np.vstack([arr, pad])
    elif len(arr) > len(texts):
        arr = arr[:len(texts)]

    # ── Нулевые векторы для пустых строк ─────────────────────────────────────
    for idx in empty_idx:
        arr[idx] = 0.0

    return arr.astype(np.float32)


# ─── LLM-функция ─────────────────────────────────────────────────────────────

async def llm_func(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot=False,
    keyword_extraction=False,
    entity_extraction=False,
    **kwargs,
) -> str:
    """
    Обёртка для LightRAG 1.5.0.
    Вызываем _ollama_model_if_cache напрямую — минуя ollama_model_complete,
    которая конфликтует с kwargs передаваемыми LightRAG.
    """
    if history_messages is None:
        history_messages = []

    logger.debug(f"llm_func kwargs keys: {list(kwargs.keys())}")

    # Убираем всё что не понимает Ollama API или передаётся дважды
    for key in (
        "model", "host", "llm_model_name", "options",
        "hashing_kv", "llm_response_cache",
    ):
        kwargs.pop(key, None)

    # response_format для JSON-режимов
    response_format = None
    if keyword_extraction or entity_extraction:
        response_format = {"type": "json_object"}

    result = await _ollama_model_if_cache(
        model=OLLAMA_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        host=OLLAMA_BASE_URL,
        options={
            "num_ctx":     8192,    # 0.5b модели хватит меньшего контекста
            "temperature": 0.1,
            "top_p":       0.9,
        },
        **({"response_format": response_format} if response_format else {}),
        **kwargs,
    )

    # Если вернулся AsyncIterator — собираем в строку
    if hasattr(result, "__aiter__"):
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        return "".join(chunks)

    return result


# ─── Фабрика LightRAG ────────────────────────────────────────────────────────

def create_lightrag_instance(working_dir: str, embedding_dim: int) -> LightRAG:
    """Создаёт экземпляр LightRAG 1.5.0."""
    Path(working_dir).mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=working_dir,

        # ── LLM ──────────────────────────────────────────────────────────────
        llm_model_func=llm_func,
        llm_model_name=OLLAMA_MODEL,
        llm_model_max_async=1,
        llm_model_kwargs={},
        default_llm_timeout=180,

        # ── Эмбеддинги ───────────────────────────────────────────────────────
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=MAX_EMBED_TOKENS,
            func=embed_func,
        ),
        embedding_batch_num=8,
        embedding_func_max_async=2,
        default_embedding_timeout=60,

        # ── Чанки ────────────────────────────────────────────────────────────
        chunk_token_size=512,
        chunk_overlap_token_size=50,

        # ── Граф ─────────────────────────────────────────────────────────────
        max_graph_nodes=2000,
        entity_extract_max_gleaning=1,
        cosine_threshold=0.2,

        # ── Кэш ──────────────────────────────────────────────────────────────
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
    )

    logger.info(
        f"LightRAG создан | LLM={OLLAMA_MODEL} | "
        f"Embed={OLLAMA_EMBED_MODEL}({embedding_dim}d) | "
        f"chunk=512tok | Dir={working_dir}"
    )
    return rag


# ─── Менеджер ────────────────────────────────────────────────────────────────

class LightRAGManager:
    """Управляет индексацией и запросами через LightRAG 1.5.0."""

    def __init__(self, working_dir: str):
        self.working_dir    = working_dir
        self.rag: LightRAG | None = None
        self._embedding_dim = EMBEDDING_DIM

    def initialize(self):
        """Синхронно создаёт объект LightRAG с дефолтной размерностью."""
        self.rag = create_lightrag_instance(self.working_dir, self._embedding_dim)
        logger.info("LightRAG объект создан")

    async def async_initialize(self):
        """
        1. Определяет реальную размерность через прямой запрос к Ollama
        2. Пересоздаёт LightRAG если размерность отличается
        3. Инициализирует хранилища
        """
        if not self.rag:
            raise RuntimeError("Сначала вызовите initialize()")

        real_dim = await _detect_embedding_dim()

        if real_dim != self._embedding_dim:
            logger.info(
                f"Пересоздаём LightRAG: "
                f"{self._embedding_dim}d → {real_dim}d"
            )
            self._embedding_dim = real_dim

            # Обновляем глобальную константу для embed_func
            import src.lightrag_manager as _m
            _m.EMBEDDING_DIM = real_dim

            self.rag = create_lightrag_instance(self.working_dir, real_dim)

        await self.rag.initialize_storages()
        logger.info(
            f"Storages инициализированы | embedding_dim={self._embedding_dim}"
        )

    async def insert_documents(self, documents: list[str]):
        """
        Вставляет каждый документ отдельным вызовом ainsert.
        Это гарантирует что timeout одного документа не убивает весь батч.
        """
        if not self.rag:
            raise RuntimeError("LightRAG не инициализирован")

        valid   = [d.strip() for d in documents if d and d.strip()]
        skipped = len(documents) - len(valid)
        if skipped:
            logger.warning(f"Пропущено {skipped} пустых документов")

        total  = len(valid)
        errors = 0
        ok     = 0

        logger.info(f"Вставляем {total} документов по одному...")

        for i, doc in enumerate(valid, 1):
            try:
                await self.rag.ainsert(doc)
                ok += 1
                logger.info(f"  [{i}/{total}] OK ({len(doc)} симв.)")
            except TimeoutError as e:
                errors += 1
                logger.warning(f"  [{i}/{total}] Timeout — пропускаем: {str(e)[:80]}")
            except Exception as e:
                errors += 1
                logger.warning(f"  [{i}/{total}] Ошибка — пропускаем: {str(e)[:80]}")
                if errors > max(5, int(total * 0.5)):
                    raise RuntimeError(
                        f"Слишком много ошибок ({errors}/{total}), останавливаем"
                    )

        logger.info(f"Вставка завершена: {ok}/{total} OK, {errors} ошибок")

    async def query(self, question: str, mode: str = "hybrid") -> str:
        """
        Запрос к базе знаний.
        Режимы: naive | local | global | hybrid
        """
        if not self.rag:
            raise RuntimeError("LightRAG не инициализирован")

        if mode not in {"naive", "local", "global", "hybrid"}:
            logger.warning(f"Неизвестный режим '{mode}', использую 'hybrid'")
            mode = "hybrid"

        logger.info(f"Запрос [{mode}]: {question[:100]}")

        return await self.rag.aquery(
            question,
            param=QueryParam(mode=mode),
        )

    def query_sync(self, question: str, mode: str = "hybrid") -> str:
        """Синхронная обёртка."""
        return asyncio.run(self.query(question, mode))

    async def build_and_index(
        self,
        dataset_path: str,
        graph_builder: GraphBuilder,
        limit: int | None = None,
    ) -> dict:
        """
        Полный пайплайн:
          1. Загрузка датасета (с опциональным лимитом)
          2. Парсинг LaTeX → граф (вершины + рёбра)
          3. Конвертация в текстовые документы
          4. Инициализация хранилищ LightRAG
          5. Индексация документов батчами
          6. Сохранение графа (JSON + PNG)
        """
        # 1. Загрузка
        dataset = graph_builder.load_dataset(dataset_path)

        # 2. Применяем лимит
        if limit is not None:
            total_in_dataset = len(dataset)
            dataset = dataset[:limit]
            logger.info(
                f"Limit: используем первые {limit} из "
                f"{total_in_dataset} записей"
            )

        # 3. Парсинг
        logger.info("Парсим формулы и строим граф...")
        graph_builder.build_from_dataset(dataset)

        # 4. Документы для LightRAG
        documents = graph_builder.to_lightrag_documents()

        # 5. Инициализация хранилищ
        await self.async_initialize()

        # 6. Индексация
        await self.insert_documents(documents)

        # 7. Артефакты
        kb = Path(self.working_dir)
        graph_builder.save_graph_json(str(kb / "math_graph.json"))
        graph_builder.visualize_graph(str(kb / "math_graph.png"))

        stats = graph_builder.get_stats()
        logger.info(f"Готово. Статистика: {stats}")
        return stats
