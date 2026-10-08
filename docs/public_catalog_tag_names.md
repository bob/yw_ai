# Публичные имена catalog tags

## `GET /api/public/catalog_folders/:catalog_folder_uuid/catalog_tags`

Endpoint не требует JWT и возвращает уникальные локализованные имена tags из
запрошенной папки. Он доступен только для папок с `public: true`: отсутствующая
или приватная папка возвращает `404`. Переданный JWT не расширяет эту
видимость.

`catalog_folder_uuid` передаётся в path. Обязательный query-параметр `language`
определяет язык имён; допустимые значения: `en`, `uk`, `ru`.

Пример:

```http
GET /api/public/catalog_folders/yoga-public-v1/catalog_tags?language=uk
```

Успешный ответ (`200 OK`) — JSON-массив строк, отсортированный по имени, без
обёртки `data`:

```json
[
  "Баланс",
  "Стоячи"
]
```

При отсутствии папки, либо при UUID приватной папки, endpoint возвращает `404`.
При отсутствующем или недопустимом `language` endpoint возвращает `422
Unprocessable Entity`:

```json
{
  "error": "language must be one of: en, uk, ru"
}
```

## Pydantic-схемы

`catalog_folder_uuid` передаётся в path, `language` — как query-параметр.
Результат — корневая Pydantic-модель, так как HTTP-ответ является массивом, а
не объектом.

```python
from typing import Literal

from pydantic import Field, RootModel


class CatalogTagNamesArguments(StrictModel):
    catalog_folder_uuid: str = Field(min_length=1)
    language: Literal["en", "uk", "ru"]


class CatalogTagNamesResult(RootModel[list[str]]):
    pass
```
