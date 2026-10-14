# Приёмка prompt caching — фича 012

Связанный документ: [README](README.md).

## 1. Prefix и динамический контекст

1. Два обычных model requests с разными active snapshots имеют
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

6. Для `gpt-4.1` переданы поддерживаемые key/retention; ключ не меняется
   с request ID, count, selected UUID, временем или повторным rerun.
7. Namespace `history-summary` либо новая ревизия статических schemas/prompts меняет
   соответствующую часть ключа. Отдельные сессии используют разные opaque keys.
8. Summary calls используют `history-summary`; fake/non-OpenAI model не получает
   OpenAI-specific kwargs. Отключение настройки не меняет результат диалога.
9. Unsupported retention/configuration выявляется до запроса, без автоматической
   второй отправки. Prewarm calls и padding отсутствуют.

## 3. Usage и регрессии

10. Fake usages с cache read 0/положительным/отсутствующим корректно агрегируются.
    Provider input включает cache read; reasoning уже входит в output.
    Дублирование normalized/raw usage не удваивает суммы.
11. Отменённая 011 не используется: модель выбирает tools из общего набора и
    готовит итоговый ответ. Summary входит в число calls и общий usage операции.
    Internal records не попадают в карточки UI.
12. Все обычные операции, выбор старого варианта, cap, ошибки, один commit,
    progress и Save сохраняют поведение. Calls Jev и его payloads не меняются.

## 4. Сценарии сравнения

- Generate 9 → простое добавление 3.
- Повторные операции с общим prompt/tools и переключение выбранного варианта.
- Разные действия, например browsing → addition, без переключения профилей.
- Длинный диалог с одной и последующей суммаризацией 010.

Сравнивать одну модель, те же inputs и fake dependencies, один tokenizer
для локальных оценок. Для live report привести normal/summary calls,
input/cache-read/output tokens, стоимость и условия retention по каждому
сценарию. Cold и warm requests показывать отдельно; не обещать hit при
истёкшем кэше или неподходящем префиксе.

После реализации записать в README команды offline-проверок, фактические
результаты, проводилась ли live-проверка и какой reported usage она дала.

## 5. Результат 14 октября 2026 года

Все 7 offline-тестов 012 и 303 теста проекта проходят. Live-сценарии на `gpt-4.1`
завершены для baseline 010 и 012 с fake catalog/generator, без Jev.
Cache reads зарегистрированы; в этом коротком сравнении экономия не получена.
Команды, ограничения измерения и таблица стоимости — в [README](README.md#7-реализация-и-результаты),
полный provider usage каждого вызова — в [live-results.json](live-results.json).
