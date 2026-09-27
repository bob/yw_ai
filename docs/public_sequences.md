# Публичная секвенция

## `GET /api/public/sequences/:id`

## `GET /api/public/sequences?name=:name`

Endpoint не требует JWT и возвращает одну секвенцию с `public: true` в JSON.
Переданный JWT не даёт доступа к приватным секвенциям. Язык обязателен и
передаётся query-параметром `language`; допустимы `en`, `uk` и `ru`.

Секвенцию можно получить:

- по числовому ID в path: `/api/public/sequences/42?language=uk`;
- по точному имени через `name`: `/api/public/sequences?language=en&name=Morning%20flow`.

Поиск имени регистронезависимый и проверяет как базовое `sequences.name`, так и
все записи `sequence_translations` — поэтому можно передать перевод на другом
языке. `language` определяет локализованные `name` секвенции и `catalog_item` в
ответе. Имя также можно передать вместо ID в path. Если одинаковому имени
соответствует несколько публичных секвенций, API возвращает `422`; используйте
ID.

Пример:

```http
GET /api/public/sequences?language=uk&name=%D0%A3%D1%82%D1%80%D0%B5%D0%BD%D0%BD%D1%8F%D1%8F%20%D0%BF%D1%80%D0%B0%D0%BA%D1%82%D0%B8%D0%BA%D0%B0
```

Успешный ответ (`200 OK`) — объект без обёртки `data`:

```json
{
  "id": 42,
  "name": "Ранкова практика",
  "slug": "morning-flow",
  "description": "Мягкая утренняя практика",
  "lang": "en",
  "elements_count": 1,
  "elements": [
    {
      "uuid": "7d39dce0-bfce-4dcb-bf7f-b8cbd742d1cf",
      "order": 1,
      "start_mirror": false,
      "end_mirror": true,
      "use_in_breathing_mode": true,
      "catalog_item": {
        "id": 17,
        "uuid": "TREE-POSE",
        "name": "Поза дерева",
        "bilateral": true,
        "image_url": "https://example.com/rails/active_storage/blobs/…/tree-pose.png",
        "catalog_tag_names": ["Баланс", "Стоячи"]
      }
    }
  ]
}
```

`elements` содержит только элементы, одновременно удовлетворяющие
`is_active: true` и `use_in_breathing_mode: true`, в порядке `order`, затем
`id`. Их `start_mirror` и `end_mirror` вычисляются по правилу отображения
**Only static (asanas)**: маркер на исключённом не-статическом элементе
переносится на ближайшую следующую (`start_mirror`) или предыдущую
(`end_mirror`) статическую асану. Это то же правило, которое используется для
миниатюр секвенции. Поле `catalog_item` имеет ровно стандартную форму результата
[`catalog_items/search`](public_catalog_items_search.md) с `with_tags=true`:
`id`, `uuid`, `name`, `bilateral`, `image_url`, `catalog_tag_names`.
`catalog_tag_names` — отсортированный список локализованных имён тегов item.
Для элемента без связанного catalog item поле равно `null`. `elements_count` —
количество объектов в массиве `elements`, то есть
число элементов после этой фильтрации.

Ошибки:

- `404` — публичная секвенция с указанным ID/именем не найдена;
- `422` — `language` отсутствует или не входит в `en`, `uk`, `ru`; не переданы
  ни ID, ни `name`; переданы оба; либо имя неоднозначно.
