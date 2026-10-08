# Публичный поиск catalog items

## `GET /api/public/catalog_folders/:catalog_folder_uuid/catalog_items/search`

Endpoint не требует JWT и возвращает JSON-массив максимум из **10**
items по умолчанию. Он доступен только для папок с `public: true`: отсутствующая или
приватная папка возвращает `404`. Переданный JWT не расширяет эту видимость.

`catalog_folder_uuid` — UUID папки в пути. Язык передаётся обязательным query
параметром `language`; допустимы только `en`, `uk` и `ru`.

| Query-параметр | Тип | Обязателен | Поведение |
| --- | --- | --- | --- |
| `language` | `en \| uk \| ru` | да | Язык, в переводах которого ищутся `name` и `catalog_tag_names`, а также язык возвращаемых названий |
| `name` | строка | нет | Регистронезависимый поиск по части имени item в `catalog_item_translations` для `language` |
| `uuid` | строка | нет | Регистронезависимое точное совпадение UUID item |
| `bilateral` | `true \| false` | нет | Равенство признака bilateral; `false` является полноценным фильтром |
| `catalog_tag_names[]` | массив строк | нет | Регистронезависимое совпадение по части локализованного имени **любого** тега item в `catalog_tag_translations` для `language` |
| `with_rigman` | `true \| false` | нет, по умолчанию `false` | При `true` добавляет `rigman_data` и `second_side_rigman_data` к каждому результату |
| `with_tags` | `true \| false` | нет, по умолчанию `false` | При `true` добавляет `catalog_tag_names` к каждому результату |
| `limit` | целое число `>= 0` | нет, по умолчанию `10` | Максимальное число items; `0` отключает лимит |
| `rand` | `true \| false` | нет, по умолчанию `false` | При `true` возвращает случайный набор; при `false` сортирует по базовому имени по убыванию, затем по `id` по убыванию |

Все указанные фильтры объединяются через «И». Если фильтры не переданы,
возвращаются первые 10 items публичной папки в порядке убывания базового
имени item, затем `id`. Пагинации нет.

Пример:

```http
GET /api/public/catalog_folders/yoga-public-v1/catalog_items/search?language=uk&name=%D0%B4%D0%B5%D1%80%D0%B5%D0%B2&bilateral=false&catalog_tag_names[]=баланс&with_rigman=true&with_tags=true
```

Успешный ответ (`200 OK`) — именно массив, без обёртки `data`:
Поля `rigman_data` и `second_side_rigman_data` присутствуют только при
`with_rigman=true`, а `catalog_tag_names` — только при `with_tags=true`.

```json
[
  {
    "id": 42,
    "uuid": "A2CC4FE0-DAEA-4EDB-9B7B-9E2AF3550B81",
    "name": "Поза дерева",
    "bilateral": false,
    "image_url": "https://example.com/rails/active_storage/blobs/…/tree-pose.png",
    "catalog_tag_names": ["Баланс", "Стоячи"],
    "rigman_data": { "render": { "asana": true } },
    "second_side_rigman_data": null
  }
]
```

Ошибки:

- `404` — папки с данным UUID нет или она не публична;
- `422` — отсутствует/недопустим `language` либо любой переданный булев
  параметр (`bilateral`, `with_rigman`, `with_tags`, `rand`) не равен строго
  `true` или `false`, либо `limit` не является целым неотрицательным числом.

## StrictModel-схемы

Ниже схема аргументов вызова; `catalog_folder_uuid` передаётся в path, остальные
поля — query-параметрами. Для HTTP булево значение кодируется строкой `true` или
`false`, хотя в модели остаётся `bool`.

```python
from typing import Literal
from typing import Any

from pydantic import Field


class CatalogItemsSearchArguments(StrictModel):
    catalog_folder_uuid: str = Field(min_length=1)
    language: Literal["en", "uk", "ru"]
    name: str | None = Field(default=None, min_length=1)
    uuid: str | None = Field(default=None, min_length=1)
    bilateral: bool | None = None
    catalog_tag_names: list[str] = Field(default_factory=list)
    with_rigman: bool = False
    with_tags: bool = False
    limit: int = Field(default=10, ge=0)
    rand: bool = False


class CatalogItemSearchResult(StrictModel):
    id: int
    uuid: str | None
    name: str
    bilateral: bool
    image_url: str | None
    rigman_data: dict[str, Any] | None = None
    second_side_rigman_data: dict[str, Any] | None = None
    catalog_tag_names: list[str] | None = None


CatalogItemsSearchResult = list[CatalogItemSearchResult]
```
