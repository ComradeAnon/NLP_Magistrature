# DZ_2 — Граф знаний из математических формул

Проект строит **графовую базу знаний** из датасета математических формул и позволяет
делать к ней запросы на естественном языке через [LightRAG](https://github.com/HKUDS/LightRAG),
а также **оценивать качество** построенного графа набором метрик.

Конвейер:

```
датасет (latex) → парсинг формул в граф → индексация в LightRAG → запросы + метрики качества
```

- Построение графа: [main.py](main.py) → [src/graph_builder.py](src/graph_builder.py),
  [src/formula_parser.py](src/formula_parser.py), [src/lightrag_manager.py](src/lightrag_manager.py)
- Метрики качества: [analyze.py](analyze.py) → [src/quality_metrics.py](src/quality_metrics.py),
  [src/metrics_plot.py](src/metrics_plot.py)

> Все команды запускаются **из корня репозитория** (`nlp_mage/`), а не из папки `DZ_2`.
> Так работает `import src...`, а пути к датасету по умолчанию указывают на `datasets/`.

---

## 1. Требования и установка

Используется общий виртуальenv проекта `.venv` (Python 3.10). Зависимости — в
[requirements.txt](requirements.txt): `lightrag-hku`, `networkx`, `matplotlib`,
`latex2sympy2`, `sympy`, `httpx`.

```bash
# из корня nlp_mage
.venv/bin/pip install -r DZ_2/requirements.txt
```

### Ollama (обязательно для построения и запросов)

LightRAG использует локальный сервер **Ollama** на `http://localhost:11434` с двумя моделями
(заданы в [src/lightrag_manager.py](src/lightrag_manager.py)):

```bash
ollama serve                    # запустить сервер (если не запущен)
ollama pull qwen2.5:3b          # LLM
ollama pull nomic-embed-text    # эмбеддинги (768-мерные)
```

> Для **расчёта метрик** (`analyze.py`) Ollama не нужен — метрики считаются по готовому
> файлу графа `math_graph.json`.

---

## 2. Построение графа знаний

```bash
.venv/bin/python DZ_2/main.py --mode build \
    --dataset datasets/output.json \
    --kb knowledge_graph_math
```

Параметры [main.py](main.py):

| Флаг        | Значение по умолчанию          | Описание |
|-------------|--------------------------------|----------|
| `--mode`    | `build`                        | `build` \| `query` \| `demo` \| `all` |
| `--dataset` | `datasets/output.json`         | JSON-датасет с полем `latex` |
| `--kb`      | `DZ_2/knowledge_graph`         | Куда сохранить базу знаний |
| `--limit N` | —                              | Обработать только первые N формул (при отсутствии `--kb` папка назовётся `knowledge_graph_limit_N`) |

Что появляется в папке `--kb` после построения:

- `math_graph.json` — граф (вершины + рёбра + статистика); **используется для метрик**
- `math_graph.png` — визуализация графа (NetworkX)
- хранилища LightRAG: `kv_store_*.json`, `vdb_*.json`, `graph_chunk_entity_relation.graphml`

Прогресс пишется в `DZ_2/build.log`.

---

## 3. Запросы к графу

База знаний должна быть **построена заранее** (см. п. 2). Нужен запущенный Ollama.

```bash
# интерактивный режим
.venv/bin/python DZ_2/main.py --mode query --kb knowledge_graph_math

# демо-запросы (готовый набор вопросов)
.venv/bin/python DZ_2/main.py --mode demo --kb knowledge_graph_math

# построить и сразу спросить
.venv/bin/python DZ_2/main.py --mode all --dataset datasets/output.json --kb knowledge_graph_math
```

В интерактивном режиме можно выбрать режим поиска LightRAG, дописав к вопросу `--mode=`:

```
Запрос> Чему равен дискриминант квадратного уравнения? --mode=hybrid
```

Доступные режимы: `naive` | `local` | `global` | `hybrid` (по умолчанию `hybrid`).
Выход — `exit`, `quit` или `выход`.

---

## 4. Расчёт метрик качества

Считает метрики по готовому графу `<kb>/math_graph.json` (Ollama не требуется).
Метрики основаны на обзорной статье Paulheim 2017 (см. `instruction.docx`).

```bash
.venv/bin/python DZ_2/analyze.py \
    --kb knowledge_graph_math \
    --dataset datasets/output.json
```

Параметры [analyze.py](analyze.py):

| Флаг                       | По умолчанию                     | Описание |
|----------------------------|----------------------------------|----------|
| `--kb DIR`                 | `DZ_2/knowledge_graph`           | Папка базы (читается `<kb>/math_graph.json`) |
| `--dataset PATH`           | —                                | Датасет — для знаменателя «покрытия формул». Без него берётся из статистики графа |
| `--alpha / --beta / --gamma` | `1/3` каждый                   | Веса во взвешенном качестве `Q = α·точность + β·полнота + γ·связность` |
| `--output PATH`            | `<kb>/quality_report.json`       | Куда сохранить JSON-отчёт |
| `--plot-output PATH`       | `<kb>/quality_metrics.png`       | Куда сохранить PNG-дашборд |
| `--no-plot`                | —                                | Не строить визуализацию (только JSON + консоль) |

### Что считается

| Группа (раздел статьи) | Метрики |
|---|---|
| **Структурно-статистические** (§3.1) | плотность, средняя степень, изолированные вершины, связные компоненты, распределение и энтропия типов |
| **Покрытие онтологии** (§3.3) | доля операторов/функций/констант/операций; покрытие формул датасета |
| **Согласованность** (§1.2) | проверка типов рёбер, петли, дубликаты, «висячие» операторы |
| **Обнаружение ошибок** (§1.1–1.2) | аномальные шаблоны связей, ассоциативные правила (support/confidence), кардинальность, ацикличность |
| **Комплексные** (§3.3) | F1-score, взвешенное качество |

### Результат

- консольный отчёт по всем группам;
- `<kb>/quality_report.json` — полный отчёт в JSON;
- `<kb>/quality_metrics.png` — графический дашборд из 6 панелей.

Примеры:

```bash
# другие веса взвешенного качества
.venv/bin/python DZ_2/analyze.py --kb knowledge_graph_math --alpha 0.5 --beta 0.3 --gamma 0.2

# без графика
.venv/bin/python DZ_2/analyze.py --kb knowledge_graph_math --no-plot
```

---

## 5. Полный цикл одной командой (пример)

```bash
# 1. построить граф
.venv/bin/python DZ_2/main.py --mode build --dataset datasets/output.json --kb knowledge_graph_math
# 2. оценить качество
.venv/bin/python DZ_2/analyze.py --kb knowledge_graph_math --dataset datasets/output.json
# 3. задать вопросы
.venv/bin/python DZ_2/main.py --mode query --kb knowledge_graph_math
```
