# Контракт результатов tools — фича 009

Связанный документ: [README](README.md).
Примеры показывают условные UUID и catalog items. Полные предметные данные
сохраняются в состоянии приложения.

## 1. Успешная генерация и Regenerate

```json
{
  "status": "ok",
  "uuid": "<UUID нового варианта>",
  "operation": "generate",
  "elements_count": 9
}
```

У Regenerate `operation` равен `regenerate`. `family_uuid`, `parent_uuid`,
имя/описание, requests, language и весь список элементов доступны в snapshot;
дублировать их в tool result не требуется.

## 2. Replacement

```json
{
  "status": "ok",
  "uuid": "<UUID нового варианта>",
  "operation": "replace",
  "elements_count": 9,
  "replacement": {
    "element_id": "<UUID вхождения>",
    "order": 3,
    "old_catalog_item": {"id": 101, "name": "<прежнее имя>"},
    "new_catalog_item": {"id": 202, "name": "<новое имя>"}
  }
}
```

В diff нужны только факты для ответа; catalog UUIDs, tags, bilateral, URLs
и flags доступны в snapshots. Исходный вариант остаётся в `sequences`.

## 3. Swap

```json
{
  "status": "ok",
  "uuid": "<UUID нового варианта>",
  "operation": "swap",
  "elements_count": 9,
  "swap": [
    {"element_id": "<UUID A>", "name": "<имя A>", "old_order": 3, "new_order": 5},
    {"element_id": "<UUID B>", "name": "<имя B>", "old_order": 5, "new_order": 3}
  ]
}
```

Имена и позиции снимаются с фактически выполненного изменения. Для разных
occurrences одной позы сохраняются разные `element_id`.

## 4. Add

```json
{
  "status": "ok",
  "uuid": "<UUID нового варианта>",
  "operation": "add",
  "elements_count": 12,
  "addition": {
    "requested_count": 3,
    "added_count": 3,
    "limit_applied": false,
    "position": 10,
    "insert_before_element_id": null,
    "elements": [
      {"element_id": "<UUID X>", "order": 10},
      {"element_id": "<UUID Y>", "order": 11},
      {"element_id": "<UUID Z>", "order": 12}
    ]
  }
}
```

`addition.elements` — компактный diff только новых occurrences, как в текущем
контракте. Полных catalog items и списка всех старых элементов в нём нет.
`position` — первая позиция вставленного блока в результате; позиции начинаются
с 1. При усечении count сообщение содержит запрошенное и фактическое количество.

## 5. Search и display

Search сохраняет envelope `status`, `language`, `returned_count`, `limit`,
`items`. Каждый item в model-visible `items` содержит только:

```json
{
  "id": 202,
  "uuid": "<catalog UUID>",
  "name": "<имя из API>",
  "bilateral": false,
  "catalog_tag_names": ["<реальный tag>"]
}
```

В `latest_catalog_items` сохраняются данные, необходимые текущим tools и UI,
включая image URLs. У display короткий результат с IDs/count остаётся в content,
полные элементы галереи — в artifact. `get_catalog_tag_names` возвращает
настоящие tag names без сокращения их значений.

## 6. Ошибки и no_change

Существующие `status`, `code`, `message` и `ToolMessage.status` сохраняют смысл.
Ошибки не содержат полный snapshot. `no_change` содержит UUID исходного
варианта и факты причины, необходимые ответу: для достигнутого лимита —
`requested_count`, `added_count=0`, `limit_applied=true` и прежний code.

Потребители различают `ok`, `no_change` и `error`. Отсутствие полного snapshot
в content не является ошибкой, если state commit выполнен и snapshot доступен.

## 7. Remove — расширение 015

[`remove_sequence_elements`](../015-remove-sequence-elements/README.md) возвращает:

```json
{
  "status": "ok",
  "uuid": "<UUID нового варианта>",
  "operation": "remove",
  "elements_count": 3,
  "removal": {
    "removed_count": 2,
    "elements": [
      {"element_id": "<UUID B>", "name": "<имя B>", "old_order": 2},
      {"element_id": "<UUID D>", "name": "<имя D>", "old_order": 4}
    ]
  }
}
```

`removal.elements` содержит только удалённые вхождения в порядке исходного
списка, независимо от порядка UUID в аргументах. Имена и `old_order` берутся
из родителя; `removed_count` равен длине diff, `elements_count` — числу
оставшихся элементов. Полные удалённые catalog items доступны в родителе,
а результат хранит только оставшиеся вхождения. URLs, tags, catalog UUIDs
и полный состав в content не дублируются. Вложенный компактный список
`removal.elements` не является полным корневым `elements` снимка.

У удаления нет успешного `no_change`: повтор удалённого UUID в новом снимке
даёт `sequence_element_not_found`, попытка удалить всё — `would_empty_sequence`.
Любая ошибка сохраняет исходный снимок и активный UUID. Успешное удаление
расходует одну общую операцию независимо от числа целей. Проекция истории
сохраняет `removal`; прежние результаты и summary не восстанавливают удалённые
вхождения в текущем `selected_sequence` и не ограничивают Regenerate.
