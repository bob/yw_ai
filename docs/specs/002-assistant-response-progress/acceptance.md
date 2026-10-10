# Фича 002: Приёмочные проверки

Проверено 9 октября 2026 года без вызовов OpenAI, TypeSafe/Jev и настоящего каталога.
Модель и внешние сервисы заменены управляемыми тестовыми объектами; выполняются
реальный агентный цикл, схемы, события, генератор и Streamlit-интерфейс.

## Автоматические проверки

```bash
uv run python -m unittest discover -s tests -q
```

Результат: **103 теста пройдено**.

| Сценарий | Проверка |
| --- | --- |
| Статусы до возврата блокирующей модели/tools, реальные повторные вызовы | [test_response_progress_stream.py](../../../tests/test_response_progress_stream.py) |
| Публичные сообщения, tags → search → повторный search, пустые результаты | [test_response_progress_ui.py](../../../tests/test_response_progress_ui.py) |
| Общий цикл чата, Generate и Regenerate, сворачивание на успехе | AppTest в `test_response_progress_ui.py` и [test_sequence_variants.py](../../../tests/test_sequence_variants.py) |
| Снимки, длительность и ошибки инициализации, replay без повторных вызовов | `test_response_progress_ui.py` |
| Поиск внутри генератора, родительский этап, счётчики и полный commit | `test_response_progress_stream.py`, `test_response_progress_ui.py`, [test_sequence_generator.py](../../../tests/test_sequence_generator.py) |
| Фатальная ошибка до commit, уборка миниатюр и сохранение прежнего варианта | `test_response_progress_ui.py`, `test_sequence_variants.py` |
| Ошибка после commit сохраняет результат и Save | `test_response_progress_ui.py`, `test_sequence_variants.py` |
| Ошибка Regenerate остаётся между успешными попытками и связана с источником | `test_response_progress_ui.py` |
| Восстановимая ошибка tool не делает весь запрос неуспешным; `no_change` без нового варианта | `test_response_progress_ui.py`, `test_sequence_variants.py` |
| Блокировка Generate/chat/Regenerate/Previous/Next/Use и повторное разрешение действий | `test_response_progress_ui.py` |
| Разные request IDs и сессии, повторные сообщения, старые карточки | `test_response_progress_ui.py`, `test_sequence_variants.py` |
| Формат, сериализация, ссылки и ограничения снимков | [test_response_progress_schemas.py](../../../tests/test_response_progress_schemas.py) |

## Проверка в Chrome с задержанными сервисами

Использован [тестовый entrypoint](../../../tests/fixtures/assistant_progress_browser.py),
который запускает настоящий `assistant.py` с fake model и services. Каждый
блокирующий вызов ждёт отдельный файл-разрешение, поэтому статус проверяется
до завершения работы, независимо от скорости компьютера.

Пример запуска с новым пустым каталогом:

```bash
mkdir -p /tmp/assistant-progress-002-check
PROGRESS_CHECK_DIR=/tmp/assistant-progress-002-check uv run streamlit run tests/fixtures/assistant_progress_browser.py --server.headless true --server.address 127.0.0.1 --server.port 8512 --server.enableStaticServing true --browser.gatherUsageStats false
```

Открыть `http://127.0.0.1:8512`, нажать Generate и последовательно разрешать
этапы командой `touch /tmp/assistant-progress-002-check/<имя>`:

| Ещё заблокированная работа | Наблюдаемый результат |
| --- | --- |
| `dependencies` | Уже виден раскрытый `Thinking...`; Generate, количество поз и chat input отключены |
| `model` после разрешения `dependencies` | `Thinking...` остаётся виден до возврата первого ответа модели |
| `search` после разрешения `model` | `Searching catalog items...` появляется до возврата каталога; вручную свёрнутый блок остаётся свёрнутым |
| `selection` после разрешения `search` | `Generating sequence... 1/3 poses`; первая миниатюра вне status, пока генератор ещё работает |
| `reply` после разрешения `selection` | Полная карточка с тремя позами уже видна, запрос ещё активен с `Thinking...` |
| После разрешения `reply` | `Completed in N s` свёрнут, итоговый ответ один раз вне блока; доступны Regenerate, Save и ввод |

Раскрытие завершённого блока показывает реальные этапы, длительности,
счётчики, публичное «I'll find poses for your sequence.» и подтверждение
сохранённого изменения. В тестовой модели намеренно задано три позы независимо
от значения формы; это позволяет быстро проверить счётчик потока.

Эта проверка подтверждает своевременную отрисовку. Поведение реальных провайдеров
и отправка Save в настоящий каталог в фиче 002 не проверялись.
