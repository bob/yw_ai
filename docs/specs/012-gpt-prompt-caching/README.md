# 012 — Стабильный префикс и prompt caching GPT

Статус: **реализована и проверена 14 октября 2026 года**.

Источник задачи — пункт 7 [анализа расхода токенов](../../token-usage-and-optimization.md).
Зависимости: [009](../009-compact-gpt-tool-results/README.md),
[010](../010-bounded-gpt-history/README.md).
[011](../011-focused-gpt-context/README.md) отменена: профили действий,
маршрутизация по грамматике и локальное завершение не используются.
Проверки: [acceptance.md](acceptance.md).

## 1. Проблема и результат

`selected_sequence_context` дописывает меняющийся snapshot к основному
системному сообщению перед историей. После каждой операции или выбора
другого варианта префикс меняется до сообщений, которые могли бы совпасть.

Стабильные общие инструкции и schemas всех семи инструментов остаются
одинаковыми между запросами. История сохраняет порядок и содержание до
очередного сокращения. Текущий snapshot передаётся отдельным последним
контекстным сообщением, вне кэшируемого стабильного начала.

Кэширование сокращает стоимость повторной обработки совпавших входных
токенов. Оно не заменяет ответ модели и не пропускает выполнение tools.
Попадание в provider cache не гарантируется приложением.

Jev и его API/payload/cache settings в эту фичу не входят. Модель GPT,
reasoning, temperature и алгоритм генерации не меняются ради caching.

## 2. Порядок model-visible сообщений

Логический запрос:

```text
Стабильный общий system/developer prompt
Сводка завершённых ходов, если есть (меняется только при сокращении)
Свежая история: user / assistant.tool_calls / tool / assistant
Сообщения текущего хода
Отдельный system/developer context с актуальным selected_sequence
```

Tools передаются отдельным API-параметром в порядке `ASSISTANT_TOOLS`;
они также участвуют в cache prefix провайдера. Schema каждого tool и общий prompt
детерминированы, без random IDs, timestamps и содержимого сессии.

Последнее context message добавляется только в проекцию конкретного model
request. Его не записывать в canonical history, summary или новые tool results.
Старый выбранный snapshot не накапливается в следующих запросах.

Snapshot сохраняет весь контракт текущего `model_context()`: UUID, metadata,
requests, все occurrences, IDs/names/tags/bilateral и mirror flags; image URLs
исключены. Когда нет выбора, явно передаётся `selected_sequence=null`.
Инструкция приоритета snapshot над старой историей сохраняется. Сокращение
самого snapshot в эту фичу не входит.

Не вставлять context между assistant tool call и его tool results. Соблюдать
порядок реальной истории и требования API. Основной prompt всегда сохраняет
роль system/developer; дополнительные данные не подменяют user request.

## 3. Ключ и параметры OpenAI

Текущая конфигурация `.env` и default задают `gpt-4.1`; код допускает другую модель.
Параметры кэша выбираются по поддержке реально настроенной модели и API.
Для `gpt-4.1` используются `prompt_cache_key` и совместимый legacy
`prompt_cache_retention`. Не передавать параметры explicit breakpoints,
доступные только более новым семействам, как универсальный контракт.

Формат стабильного ключа:

```text
langchain-app:<model>:<prefix-version>:<profile>:<opaque-session-key>
```

Session key создаётся один раз для assistant session. Он не содержит request
text, catalog names или UUID текущей секвенции. `request_id`, active UUID,
количество элементов и время не входят в cache key. Для summary calls
использовать отдельный стабильный профиль `history-summary`.
Обычные вызовы имеют namespace `assistant`. Это только разделение ключей и usage,
а не профили действий из отменённой 011: набор инструментов всегда общий.

`prefix-version` определяется ревизией статических инструкций и schemas;
он меняется при изменении этих данных, а не на каждом вызове. Источник версии
детерминирован и одинаков для одного набора prompt/tools: первые 12 hex-символов
SHA-256 от prompt, статических инструкций контекста и упорядоченных schemas.

Начальные настройки:

- `ASSISTANT_PROMPT_CACHE_ENABLED=true` — передавать поддерживаемые настройки.
- Без явного `ASSISTANT_PROMPT_CACHE_RETENTION` используется `in_memory` для
  текущего `gpt-4.1` (включая датированный snapshot).
- `24h` — явная настройка при совместимой модели/API; для моделей с другим
  набором поддерживаемых значений не отправляется неподдержанный default:
  например, `gpt-5.5` получает `24h`; для неизвестных моделей retention опускается.

Приложение не делает дополнительный prewarm request. Кэш не заполняется
вызовами, которые не нужны пользователю. Prompt не удлиняется
пустым текстом ради минимального cacheable prefix.

Для fake, не-OpenAI model или ChatOpenAI с внешним base URL не отправляются
OpenAI-specific kwargs. Если cache settings отключены, порядок сообщений и
поведение диалога сохраняются. Отключение отправки параметров не отключает
автоматический provider cache.
Некорректная явная настройка retention выявляется при инициализации; нельзя
молча повторять ошибочный API-вызов без кэша и скрывать дополнительный расход.

## 4. Взаимодействие с другими оптимизациями

- Короткие tool messages 009 уменьшают raw input независимо от cache hit.
- Суммаризация и удаление старых ходов из проекции 010 могут сбросить общий
  префикс после статического prompt. Это ожидаемая цена ограничения истории.
- Отменённая 011 не является зависимостью. Выбор инструментов и итоговый ответ
  остаются за моделью; routing calls и локальное завершение не добавляются.
- Новый selected_sequence и переключение варианта меняют динамический хвост,
  но не статическое начало, сохранённую историю или cache key `assistant`.

Выбирать меньшую итоговую стоимость при сохранённом поведении, а не максимальный
процент cache hits. Сокращение контекста может быть выгоднее длинного cached input.

## 5. Учёт usage и измерения

На каждый реальный GPT call, включая суммаризацию, сохранять внутреннюю запись:

| Поле | Содержание |
| --- | --- |
| `request_id` | Пользовательская операция для группировки |
| `call_kind` | `assistant` или `history_summary` |
| `model`, `profile`, `prefix_version` | Условия сравнения |
| `input_tokens` | Все входные токены, включая cached input |
| `cached_input_tokens` | Кэшированная часть input, если провайдер сообщил |
| `output_tokens` | Выходные токены, включая reasoning в provider total |
| `reasoning_tokens` | Отдельная reported детализация, если доступна |

Использовать `AIMessage.usage_metadata` и нормализованную детализацию cache read,
либо raw usage SDK. При наличии обоих источников не складывать их повторно.
Отсутствующее поле означает «не сообщено», а не доказанный нулевой расход.
Reasoning не прибавлять к output второй раз.

Внутренние записи не включают полный prompt или snapshot и не требуют нового
публичного progress event. Записи доступны через detached `AssistantAgent.model_usage`
и structured logging (`model_usage`); `request_id` и `message_id` позволяют
сопоставлять их с progress и существующими model traces. Ошибка вызова сохраняется
с неизвестным usage, без повторной отправки ради кэша.

Для стоимости отдельно считать uncached input = input - cached input,
cached input и output по действующему тарифу выбранной модели/API. Условие
`cached <= input` проверяется; неизвестный cache count не заменяется оценкой
токенизатора. Тарифы не фиксировать как вечные константы этой спецификации.

## 6. Приёмка и границы доказательства

Обязательная offline приёмка проверяет фактический порядок сообщений,
детерминированность prompt/tools/key, приоритет текущего snapshot,
правильную передачу поддерживаемых параметров и агрегацию fake usage.
Она доказывает подготовку запросов к кэшированию, но не provider cache hit.

Отдельная live-проверка на текущей GPT-модели сравнивает одинаковые сценарии
до/после, фиксирует reported cache read и общую стоимость всех calls.
Использовать fake catalog/generator, чтобы измерять GPT без Jev API.
Если hit не получен, записать фактические данные; не заявлять гарантированный
процент экономии и не компенсировать это скрытыми дополнительными запросами.

Основные точки: [контекст snapshot](../../../src/langchain_app/services/sequence_context.py),
[инициализация модели](../../../src/langchain_app/services/assistant_agent.py),
[model middleware](../../../src/langchain_app/services/response_progress_middleware.py),
[зависимости](../../../src/langchain_app/dependencies.py).

Источник API-контракта, проверенный 14 октября 2026 года:
[OpenAI Docs — Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).
Перед изменением поддержки моделей сверять параметры установленной версии
LangChain/OpenAI SDK; не переносить настройки новых моделей на `gpt-4.1`.

## 7. Реализация и результаты

`selected_sequence_context` добавляет ровно один system context в конец
model projection. `prompt_cache.py` отвечает за детерминированную ревизию,
ключ сессии, совместимость retention, нормализацию и агрегацию usage.
`BoundedHistoryMiddleware` использует тот же ledger для summary calls;
canonical history, инструменты, progress и Save сохраняют прежний контракт.
Модель по умолчанию и пример окружения приведены к `openai:gpt-4.1`.

Offline-команды:

```bash
uv run python -m unittest discover -s tests -p test_prompt_cache.py -q
uv run python -m unittest discover -s tests -q
git diff --check
```

Проверяются реальные JSON-запросы установленного ChatOpenAI/OpenAI SDK через
`httpx.MockTransport`, а также fake usage и полный agent/tool/UI regression suite.
Результат: **7 проверок фичи и все 303 теста проекта проходят**;
`git diff --check` — без замечаний.
Включены Generate 9 → Add 3, выбор старого варианта, пары tool calls/results,
смена ревизии и сессии, отключение параметров, внешний OpenAI-compatible endpoint,
две суммаризации, ошибки summary/main model и сохранность internal usage.

Live-проверка выполнена 14 октября 2026 года на `gpt-4.1` через Chat Completions.
Все каталоги и генератор — fake; вызовов Jev нет. Сценарии выполнялись один раз
последовательно для baseline 010 и новой реализации, без prewarm, padding,
provider retries и повторов ради hit. Baseline использует snapshot внутри первого
system и provider defaults без ключа; 012 — snapshot в хвосте, новые session keys
и явный `in_memory`. Это разные условия прогрева/маршрутизации и retention;
контролируемой оценки долгосрочной экономии этот запуск не даёт.

| Сценарий / версия | Обычные / summary calls | Input | Cached input | Output | Расчётная стоимость, USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generate 9 → Add 3 → старый вариант / 010 | 5 / 0 | 23 294 | 12 928 | 203 | 0.028820 |
| Generate 9 → Add 3 → старый вариант / 012 | 5 / 0 | 23 288 | 11 776 | 212 | 0.030608 |
| Browsing → addition / 010 | 5 / 0 | 24 757 | 17 664 | 245 | 0.024978 |
| Browsing → addition / 012 | 5 / 0 | 24 831 | 8 832 | 248 | 0.038398 |
| Две суммаризации / 010 | 2 / 2 | 18 351 | 6 144 | 337 | 0.030182 |
| Две суммаризации / 012 | 2 / 2 | 18 337 | 2 944 | 265 | 0.034378 |
| **Всего / 010** | **12 / 2** | **66 402** | **36 736** | **785** | **0.083980** |
| **Всего / 012** | **12 / 2** | **66 456** | **23 552** | **725** | **0.103384** |

Provider cache reads подтверждены в обоих режимах; в этом коротком запуске
экономия не получена. Все записи по отдельным вызовам, включая первые и
последующие, а также ответы и версии библиотек сохранены в
[live-results.json](live-results.json). Input уже включает cached input;
reported reasoning — 0, отдельно к output он не прибавляется.

Стоимость рассчитана по reported usage и проверенным на дату запуска тарифам
[$2 / $0.50 / $8 за миллион input / cached input / output](https://developers.openai.com/api/docs/models/gpt-4.1),
а не по данным invoice. Тарифы передаются fixture аргументами и не зашиты в приложение.
Повторить измерение с актуальными тарифами можно командой:

```bash
uv run --env-file .env python tests/fixtures/prompt_caching.py --live \
  --output docs/specs/012-gpt-prompt-caching/live-results.json \
  --model gpt-4.1 --input-rate 2 --cached-input-rate 0.5 --output-rate 8 \
  --pricing-source https://developers.openai.com/api/docs/models/gpt-4.1
```
