Для запуска необходимо установить ollama
https://ollama.com/

А также локально развернуть 2 модели
ollama pull qwen2.5:3b
ollama pull nomic-embed-text

Для создания графа нужно в папку datasets радом с DZ_2 добавить данные с формулами в файл outputs.json

Для запуска построения графа:

python main.py --mode build --limit 6 (построение графа по первым 6 элементам датасета)

python main.py --mode query --limit 6 (обращение к графу знаний)
