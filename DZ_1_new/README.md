Инструкция по использованию:
1. Скачать с помощью любых из способов:
  - Архив;
  - `git clone https://github.com/ComradeAnon/NLP_Magistrature.git` в консоли;
  - GitHub Desktop.
2. Зайти в консоли в папку 'DZ_1_new'.
3. Для добавления собственных документов добавьте в папку 'docs' нужные pdf файлы.
4. Запустить команду `docker compose --profile cpu up --build` приложение отсканирует документы и выложит результат создания датасета в папку dataset.
5. Для сканирования новых документов можно использовать команду из 4 пункта несколько раз.
6. По завершении запустить команду `docker compose down`. Папка dataset останется.

Если на устройстве присутствует поддержка графических вычислений то можно выбрать одно из следующих:
- cuda 12.6: `docker compose --profile cuda126 --build`
- cuda 12.8: `docker compose --profile cuda128 --build`
- cuda 13.0: `docker compose --profile cuda130 --build`
- rocm 7.2: `docker compose --profile rocm72 --build`

По умолчанию будет выставлено значение без поддержки графических вычислений
