# Приёмка prompt caching — фича 012

Связанный документ: [README](README.md).

## 1. Prefix и динамический контекст

1. Два model requests одного профиля с разными active snapshots имеют
   одинаковые статические prompt и schemas, порядок tools и cache key.
2. Snapshot отсутствует в основном system prompt и добавлен ровно один раз
   отдельным последним context message. Canonical messages после запроса
   не содержат копии этого сообщения.
3. Поле `selected_sequence` содержит все текущие occurrences без image URLs.
   До первой генерации оно null. Старый вариант, выбранный после нового,
   имеет приоритет над последним tool result и summary.
4. Пары assistant/tool не разделяются context message. Содержимое свежей
   истории не пересериализуется/переписывается ради cache key.
5. После history compaction статический prompt остаётся прежним, а summary
   и suffix меняются только согласно 010. Это не считается cache bug.

## 2. Параметры и совместимость

6. Для `gpt-5.1` переданы поддерживаемые key/retention; ключ не меняется
   с request ID, count, selected UUID, временем или повторным rerun.
7. Другой профиль либо новая ревизия статических schemas/prompts меняет
   соответствующую часть ключа. Отдельные сессии используют разные opaque keys.
8. Summary calls используют `history-summary`; fake/non-OpenAI model не получает
   OpenAI-specific kwargs. Отключение настройки не меняет результат диалога.
9. Unsupported retention/configuration выявляется до запроса, без автоматической
   второй отправки. Prewarm calls и padding отсутствуют.

## 3. Usage и регрессии

10. Fake usages с cache read 0/положительным/отсутствующим корректно агрегируются.
    Provider input включает cache read; reasoning уже входит в output.
    Дублирование normalized/raw usage не удваивает суммы.
11. Шаблонный ответ 011 не учитывается как вызов GPT. Summary входит в число
    calls и общий usage операции. Internal records не попадают в карточки UI.
12. Все обычные операции, выбор старого варианта, cap, ошибки, один commit,
    progress и Save сохраняют поведение. Calls Jev и его payloads не меняются.

## 4. Сценарии сравнения

- Generate 9 → простое добавление 3.
- Повторные операции одного профиля и переключение выбранного варианта.
- Смена профиля, например browsing → addition.
- Длинный диалог с одной и последующей суммаризацией 010.

Сравнивать одну модель, те же inputs и fake dependencies, один tokenizer
для локальных оценок. Для live report привести normal/summary calls,
input/cache-read/output tokens, стоимость и условия retention по каждому
сценарию. Cold и warm requests показывать отдельно; не обещать hit при
истёкшем кэше или неподходящем префиксе.

После реализации записать в README команды offline-проверок, фактические
результаты, проводилась ли live-проверка и какой reported usage она дала.
