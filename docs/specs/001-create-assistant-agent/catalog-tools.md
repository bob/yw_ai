# Спецификация: поиск элементов и получение тегов

Статус: **первый этап реализован**; HTTP-клиенты, tools, английский язык API
и состояние проверенных кандидатов выполнены. Типизированные UI-события
остаются последующим расширением;
см. [план](README.md).
Источники API: [catalog items search](../../public_catalog_items_search.md),
[catalog tag names](../../public_catalog_tag_names.md).

## 1. Общие правила интеграции

Оба endpoint вызываются на `ASANA_SERVICE_BASE_URL` для папки
`ASANA_CATALOG_FOLDER_UUID`. Это конфигурация приложения, скрытая от модели.
Запросы публичные, без JWT. Отсутствующая или приватная папка даёт `404`;
передача JWT не расширяет доступ.

Во всех запросах фиксируется `language=en`. Аргумент `language` не входит в
tools и новый интерфейс catalog services. Общий тип `Language` для остальных
API можно сохранить: это не изменение сервера или публичного sequence API.

Пользовательский запрос на русском/украинском переводится моделью в английские
поисковые термины. Реальные имена и теги из результата не переводятся и не
подменяются. Для фильтров по тегам предпочтительны значения из tags endpoint.

Клиенты используют `httpx`, timeout по умолчанию 10 секунд и текущий подход
с injectable client. Клиент, созданный сервисом, закрывается сервисом;
переданный извне клиент сервис не закрывает. API-путь собирается из base URL
и URL-encoded UUID папки. Фильтры передаются через `params`, не конкатенацией URL.

## 2. Расширение `CatalogItemService.search`

До реализации этого этапа `search(language=...)` загружал весь случайно отсортированный
каталог с фиксированными `language=en`, `with_tags=true`, `limit=0`, `rand=true`.
Новый `search()` должен принимать фильтры и опции API. Полная выборка генератора
становится отдельным явным способом использования того же клиента.

Endpoint:

```http
GET /api/public/catalog_folders/:catalog_folder_uuid/catalog_items/search
```

| Аргумент клиента | Тип / default | Передача и смысл |
| --- | --- | --- |
| `name` | `str \| None = None` | Поиск по части английского имени, без учёта регистра |
| `uuid` | `str \| None = None` | Точное совпадение UUID, без учёта регистра |
| `bilateral` | `bool \| None = None` | Равенство; `false` обязательно сохраняется |
| `catalog_tag_names` | `list[str] = []` | Повторяющийся query key `catalog_tag_names[]` |
| `with_rigman` | `bool = False` | Добавить два поля rigman |
| `with_tags` | `bool = True` | Добавить `catalog_tag_names`; default клиента намеренно отличается от API |
| `limit` | строго целое `>= 0`, default `10` | `0` означает всю выборку без лимита |
| `rand` | `bool = False` | `true` — случайная выборка |

`language=en` добавляется клиентом всегда. В HTTP все boolean кодируются
строками `true`/`false`; `None` и пустой список tags не передаются.
Непустые строковые фильтры нормализуются по внешним пробелам; строки из одних
пробелов отклоняются. Для `limit` boolean и дробное число не считаются целым.

Семантика API не переопределяется клиентом:

- Разные фильтры сочетаются через AND.
- Несколько tag names означают совпадение с **любым** из них, по части имени,
  без учёта регистра; это OR, а не «элемент имеет все теги».
- При `rand=false` порядок — базовое имя по убыванию, затем ID по убыванию;
  это не ранжирование по релевантности.
- Пагинации нет; количество возвращённых items не равно общему числу совпадений.

Пример сериализации двух тегов и полноценного `false`:

```http
GET /api/public/catalog_folders/yoga-public-v1/catalog_items/search?language=en&bilateral=false&catalog_tag_names[]=Balance&catalog_tag_names[]=Standing&with_tags=true&with_rigman=false&limit=10&rand=false
```

`Balance` и `Standing` здесь иллюстративные значения; рабочие фильтры берутся
из реального ответа tags endpoint.

## 3. Схемы ответа

Успех — корневой JSON-массив, без объекта `data`.
HTTP DTO `CatalogItemSearchResult` описывает:

| Поле | Тип | Обязательность |
| --- | --- | --- |
| `id` | положительное целое | Всегда |
| `uuid` | `str \| None` | Поле всегда, значение может быть null |
| `name` | `str` | Всегда |
| `bilateral` | `bool` | Всегда |
| `image_url` | `HttpUrl \| None` | Поле всегда, значение может быть null |
| `catalog_tag_names` | `list[str] \| None` | При `with_tags=true`; иначе может отсутствовать |
| `rigman_data` | JSON-object или null | При `with_rigman=true`; иначе может отсутствовать |
| `second_side_rigman_data` | JSON-object или null | При `with_rigman=true`; иначе может отсутствовать |

DTO валидируются Pydantic по известным полям. Схема должна принимать опциональные
rigman-поля, которые текущий `CatalogItemSchema(extra="forbid")` не поддерживает.
Отсутствие tags при `with_tags=false` не означает «у элемента нет тегов».
При `with_tags=true` отсутствующий/null `catalog_tag_names` считается нарушением
контракта, а пустой массив — корректным результатом.

Для tools и генератора `with_tags=true`: преобразование DTO в предметный
`CatalogItemSchema` сохраняет обязательный список tags. Весь общий catalog
контракт переносится в `schemas/catalog.py`; public sequence schemas используют
его импортом. Rigman-данные доступны HTTP-клиенту, но первый агент их не запрашивает
и не помещает в контекст.

## 4. Tool `search_catalog_items`

Модели доступны `name`, `uuid`, `bilateral`, `catalog_tag_names`, `limit`, `rand`.
Defaults соответствуют таблице клиента, кроме ограничения `1 <= limit <= 50`.
Лимит 50 — предложение приложения для ревью, а не ограничение Asana API.
Tool всегда задаёт `with_tags=true`, `with_rigman=false`, `language=en`.
`limit=0` остаётся доступным генератору на уровне клиента.

Успешный tool result — сериализованный объект:

```json
{
  "status": "ok",
  "language": "en",
  "returned_count": 1,
  "limit": 10,
  "items": [
    {
      "id": 42,
      "uuid": "A2CC4FE0-DAEA-4EDB-9B7B-9E2AF3550B81",
      "name": "Tree pose",
      "bilateral": false,
      "catalog_tag_names": ["Balance", "Standing"]
    }
  ]
}
```

Результат передаётся модели в `ToolMessage` без image URLs и rigman
по [контракту 009](../009-compact-gpt-tool-results/tool-contract.md).
Полные данные, включая image URLs, остаются в `latest_catalog_items`;
`show_catalog_items` передаёт выбранные элементы галереи в artifact и публичном
событии `catalog_items_displayed` по [спецификации 003](../003-catalog-items-display/README.md).
Последняя успешная выборка хранится в состоянии диалога как проверенные
кандидаты для замены; следующий успешный поиск заменяет эту выборку целиком.
Неуспешный поиск очищает кандидатов, чтобы не использовать старый результат
как результат нового запроса.

Items с null UUID или изображением остаются видимыми в поиске: tool не скрывает
реальные совпадения. При замене применяются отдельные требования к отображению
и сохранению. Пустая выборка возвращает `status=ok`, `returned_count=0`, `items=[]`.
Если получено ровно `limit` items, ассистент не утверждает, что других совпадений нет.

Tool выполняет фильтры API, а не семантический поиск. Запрос «более мягкая
альтернатива» агент превращает в поиск кандидатов и выбор по их реальным
именам/тегам и контексту секвенции; endpoint не обещает оценку сложности.

## 5. Tool `get_catalog_tag_names`

Добавить `CatalogTagService.get_names() -> list[str]`:

```http
GET /api/public/catalog_folders/:catalog_folder_uuid/catalog_tags?language=en
```

Ответ — JSON-массив уникальных строк, отсортированный сервером по имени.
Клиент валидирует корневой массив строк и сохраняет полученные значения и порядок.
Пустой массив — корректный ответ. Tool не имеет аргументов, видимых модели.

Успешный результат для модели:

```json
{"status": "ok", "language": "en", "tag_names": ["Balance", "Standing"]}
```

В UI передаётся `catalog_tags_found`. Из имён не выводятся несуществующие ID,
счётчики или структура групп. Слеш в имени тега остаётся частью строки.
Первый этап делает GET при вызове tool, без загрузки тегов на каждый rerun UI
и без отдельной политики кеширования.

## 6. Ошибки и совместимость генерации

| Ситуация | Поведение клиента / код tool result |
| --- | --- |
| Недопустимые локальные аргументы | Ошибка валидации до HTTP / `invalid_arguments` |
| HTTP 404 | `CatalogFolderNotFoundError` / `catalog_folder_not_found` |
| HTTP 422 | Ошибка query-параметров / `invalid_arguments` |
| Timeout, network error, HTTP 5xx или иной неуспех | Catalog API error / `catalog_api_unavailable` |
| Невалидный JSON, объект вместо массива, неверные поля | Catalog API error / `invalid_api_response` |

Tags service использует тот же принцип ошибок; общие исключения можно вынести
в `services/catalog_errors.py`, сохранив совместимость импортов при миграции.
Ошибка tool result содержит `status=error`, `code`, `message`; пустой каталог
не маскирует ошибку транспорта или валидации.

Генератор обязан явно вызвать:
`search(with_tags=True, with_rigman=False, limit=0, rand=True)`.
В этом запросе также фиксируется `language=en`. Сохраняется текущая фильтрация
items без изображения и выбор стартовых/финальных тегов. Изменение default
`limit` не должно случайно ограничить генератор десятью позами.

## 7. Критерии приёмки

HTTP-контракт проверяется через `httpx.MockTransport` без живого сервера:

- Оба клиента используют настроенные base URL и папку, передают только `en`.
- `bilateral=False` сохраняется; boolean сериализуются строчными строками.
- Два tags передаются двумя ключами `catalog_tag_names[]`, а не CSV/JSON-строкой.
- `None` не превращается в строку `None`; пустые tags не добавляют пустой фильтр.
- Поиск без фильтров использует `limit=10`, `rand=false`, `with_tags=true`.
- Генератор отдельно запрашивает `limit=0`, `rand=true` и полный tags-контракт.
- Принимаются оба optional rigman-поля; корректно различаются отсутствующие tags
  и пустой список tags.
- Пустые items/tags — успех; проверены 404, 422, 5xx, timeout, невалидный JSON
  и неверная форма ответа.
- Видимая модели схема не содержит языка, URL, UUID папки или HTTP-клиента.
- Tool сообщает фактические результаты без выдуманного total count.
- Поиск и получение тегов сами по себе не меняют текущую секвенцию.
