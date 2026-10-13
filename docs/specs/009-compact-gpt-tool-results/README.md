# 009 — Компактные результаты tools для GPT

Статус: **реализовано; offline-приёмка пройдена 13 октября 2026 года**.

Источник задачи — пункт 2 раздела «Где оптимизировать, по приоритету»
в [анализе расхода токенов](../../token-usage-and-optimization.md).
Оптимизация касается контекста GPT. Подбор поз, число вызовов Jev и его
`state`, `questions`, `criteria` сохраняют текущий контракт.

## 1. Проблема и результат

Сейчас sequence tools записывают полный `GeneratedSequence` в `ToolMessage`.
Middleware дополнительно передаёт выбранную секвенцию в системном контексте.
GPT получает две копии результата, а прежние полные snapshots продолжают
увеличивать историю. В tool messages также остаются URL изображений.

После изменения полный результат хранится в `state.sequences`, а tool message
содержит статус и факты операции. Например, Generate на 9 элементов возвращает
`status`, UUID результата, `operation` и `elements_count`, вместо списка девяти
catalog items. Следующий вызов модели получает полный актуальный состав
через существующий `selected_sequence`.

Контракт payloads: [tool-contract.md](tool-contract.md).

## 2. Область изменения

- Компактные сообщения для `generate_sequence`, `replace_sequence_element`,
  `swap_sequence_elements`, `add_sequence_elements`.
- Проекция результатов `search_catalog_items` без image URLs и rigman.
- Общий способ построения компактных результатов, чтобы новые tools
  не возвращали полный snapshot по умолчанию.
- Адаптация progress summaries и проверок, которые читают tool messages.

`get_catalog_tag_names` сохраняет список реальных имён тегов: он нужен для
построения фильтров. `show_catalog_items` уже возвращает короткое сообщение,
а полные данные галереи передаёт в `artifact`; этот принцип сохраняется.

## 3. Состояние и контекст

1. `Command.update.sequences` содержит полный новый snapshot, включая image URLs,
   mirror flags, family/parent UUIDs и исходный запрос генерации.
2. `active_sequence_uuid` и `turn_operation_committed` обновляются как сейчас.
3. Компактное сообщение сохраняет исходный `tool_call_id` и корректный status.
4. Полный snapshot и компактное сообщение фиксируются одной операцией;
   нельзя публиковать успех до доступности snapshot в состоянии.
5. `selected_sequence` продолжает содержать все текущие элементы, occurrence
   IDs, catalog IDs, названия, теги и mirror flags, без image URLs.
6. `latest_catalog_items` сохраняет полную проверенную выборку для replacement
   и gallery. В тексте search result используются только нужные GPT поля.
7. Выбор старого варианта в UI переключает источник `selected_sequence`.
   Старый tool result не переопределяет текущий выбор.

Полные данные не переносятся в текст другого model message. Подключение
`artifact` для внутренних потребителей допустимо, но обязательным источником
состава секвенции остаётся `state.sequences`.

## 4. Совместимость

Карточки, Previous/Next, Use this sequence, Save и Regenerate читают полные
snapshots. Старые snapshots не переписываются. Для Regenerate используется
исходный `generation_request` выбранного варианта.

Сохраняются общий лимит одной операции, cap 45, проверки UUID/occurrences,
атомарность вставки, правила поз, английский каталог и ответ на языке
пользователя. Ошибки и `no_change` не создают новый вариант.

Progress middleware формирует summaries из коротких полей и состояния,
а не ожидает `elements` в tool message. Потоковые `sequence_element`,
`sequence_committed`, catalog display и публичные ошибки сохраняют контракт
[002](../002-assistant-response-progress/event-contract.md).

Уже сохранённые в текущем thread старые tool messages могут иметь прежнюю
форму. Читающий код поддерживает оба формата; новые сообщения создаются
только в компактной форме. Массовая перезапись существующей истории входит
в проекцию [010](../010-bounded-gpt-history/README.md), а не в эту фичу.

При реализации обновить описания model-visible результатов в спецификациях
001, 006 и 007, которые сейчас описывают полный snapshot в tool content.
Новый контракт 009 заменяет эту часть прежних спецификаций; их требования
к полному состоянию и публичному поведению остаются действующими.

## 5. Критерии приёмки

1. Generate, Regenerate, replace, swap и add создают тот же полный результат
   в состоянии, но их новые сообщения не содержат root `elements`, image URLs,
   rigman или вложенную полную секвенцию.
2. Следующий model call видит актуальный полный `selected_sequence` и короткий
   tool result. В успешном простом сценарии состав результата представлен
   один раз, а не двумя полными snapshots.
3. Search message сохраняет все данные для выбора проверенного кандидата;
   галерея и replacement используют исходную полную выборку из состояния.
4. Ответ после replacement называет старую и новую позу; после swap — обе позы
   и позиции; после add — фактический count, позицию блока и итоговый count.
5. Тексты limit, `no_change` и ошибок отражают реальный результат.
6. Проверяются старый выбранный вариант, повтор имён, зеркальность,
   ошибка до commit и ошибка итогового ответа после commit.
7. На одинаковых fixtures Generate 9/36/45, replace, swap и add 3 сокращение
   суммы токенов новых sequence tool results составляет не менее 70%
   относительно прежней сериализации полного snapshot. Считать одной
   версией tokenizer; provider overhead в локальный показатель не включать.
8. Число GPT calls этой фичей не увеличивается. Входы, результаты и число
   вызовов Jev совпадают с baseline при одинаковых fake dependencies.

Основная проверка — offline agent loop со scripted model и перехватом payloads,
плюс существующие AppTest и проверки Save. Платные API для структурной
приёмки не нужны. В README после реализации записать команды, результат
и таблицу размеров baseline/after; статус менять только по фактической проверке.

## 6. Связанные документы

- [Контекст секвенции и история вариантов](../001-create-assistant-agent/sequence-context.md).
- [Добавление элементов](../007-add-sequence-elements/README.md).
- [010 — Ограниченная история GPT](../010-bounded-gpt-history/README.md).
- [011 — Prompt, tools и итоговые ответы](../011-focused-gpt-context/README.md).
- [012 — Prompt caching](../012-gpt-prompt-caching/README.md).

Основные точки реализации: [sequence tools](../../../src/langchain_app/tools/sequence.py),
[catalog tools](../../../src/langchain_app/tools/catalog.py),
[progress middleware](../../../src/langchain_app/services/response_progress_middleware.py)
и [sequence context](../../../src/langchain_app/services/sequence_context.py).

## 7. Реализация и результаты проверки

[Общий helper](../../../src/langchain_app/tools/results.py) формирует компактный
результат и возвращает его одним `Command` вместе с полным snapshot,
активным UUID и флагом commit. Generate/Regenerate возвращают четыре поля;
replace, swap и add добавляют только факты выполненной операции. Search
использует allowlist полей каталога, а `latest_catalog_items` сохраняет
image URLs для replacement и gallery. Prompt обмена читает имена из `swap`;
для старого формата сохраняется сопоставление по occurrence ID в
`selected_sequence`. Progress summaries уже используют поля, общие обоим
форматам; их совместимость проверена без чтения полного состава из content.
Описания результатов в спецификациях 001, 006 и 007 обновлены.

Проверки выполнены без платных API:

```bash
uv run python -m unittest discover -s tests -p 'test_compact_tool_results.py'
uv run python -m unittest discover -s tests
uv run python tests/fixtures/compact_tool_results.py --output docs/specs/009-compact-gpt-tool-results/token-results.json
git diff --check
```

- Новые проверки — **7 тестов, OK**; полный набор — **283 теста, OK**.
- Scripted model перехватывает реальные model messages в цикле агента:
  новые results не содержат root `elements`, URLs или вложенного snapshot,
  а следующий `selected_sequence` равен полной актуальной проекции состояния.
  Итоговые scripted replies строятся из фактических полей результата.
- Baseline запускает тот же цикл, генератор и `TypeSafeService`, восстанавливая
  прежнюю сериализацию tool results. Для Generate 9/36/45, Regenerate 9,
  replace, swap и add 3 совпали число GPT calls, все входы Jev (`state`,
  `questions`, включая `criteria`), его результаты, catalog calls и состав
  секвенций. Новые случайные UUID при сравнении независимых результатов
  исключены; идентичность исходных occurrences проверяется отдельно.
- Проверены legacy history без её переписывания, выбор старого варианта,
  повторяющиеся позы с разными occurrence IDs и mirror flags, внутренняя
  вставка, усечение count и `no_change` при cap 45. Ошибка до commit сохраняет
  источник, ошибка итогового GPT-ответа после commit сохраняет новый полный
  snapshot и компактный успешный result.
- Существующие progress, AppTest, карточки, Previous/Next, Use this sequence,
  Regenerate и Save проверки прошли в полном наборе. Gallery artifact и
  `latest_catalog_items` сохраняют полные items, включая изображения.

### Размеры новых sequence tool results

Конкретный прогон сохранён в [token-results.json](token-results.json).
Обе формы сериализуются с `ensure_ascii=False` и считаются одной версией
**tiktoken 0.14.0**, encoding **o200k_base**. Baseline и after используют один
и тот же полный snapshot с одинаковыми UUID; в baseline replacement содержит
полные старый/новый items, а swap не содержит добавленное в 009 имя.
Измеряется только JSON content sequence tools, без provider overhead,
системного prompt, поиска и всей накопленной истории.

| Сценарий | Baseline, токены | After, токены | Сокращение | GPT calls | Jev calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generate 9 | 1373 | 44 | 96,80% | 2 | 7 |
| Generate 36 | 5092 | 48 | 99,06% | 2 | 34 |
| Generate 45 | 6331 | 46 | 99,27% | 2 | 43 |
| Regenerate 9 | 1414 | 50 | 96,46% | 2 | 7 |
| Replace | 1616 | 130 | 91,96% | 3 | 0 |
| Swap | 1490 | 158 | 89,40% | 2 | 0 |
| Add 3 | 1960 | 192 | 90,20% | 2 | 2 |
| **Сумма** | **19276** | **668** | **96,53%** | | |

Порог 70% выполнен и суммарно, и для каждого сценария. Число GPT/Jev calls
в таблице одинаково для baseline и after. В fixtures первый элемент генерации
выбирается без Jev, а единственный finishing candidate использует существующий
shortcut `TypeSafeService`; поэтому add 3 делает два вызова classifier.
UUID новых вариантов генерируются штатно, поэтому точные размеры при повторном
запуске могут немного различаться. Тест проверяет порог, а не фиксированное
число токенов.
