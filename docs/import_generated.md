# Приём секвенции из генератора

Генератор открывает страницу Asana Catalog, передаёт ей JSON, а страница сохраняет
секвенцию от имени **текущего пользователя Asana Catalog** и открывает
`/sequences/:id-:slug`. Пароль и токены Asana Catalog генератору не передаются.

Пример входных данных: [sequence-5703a444aab9440289641e8753923ef3.json](sequence-5703a444aab9440289641e8753923ef3.json).

## Настройка Asana Catalog

Константа `SEQUENCE_GENERATOR_ORIGINS` в
`config/initializers/sequence_generator.rb` читается из одноимённой переменной
окружения при запуске Rails. Для локального генератора задайте:

```sh
export SEQUENCE_GENERATOR_ORIGINS=http://localhost:8502
```

Несколько адресов можно перечислить через запятую. Origin включает протокол,
хост и порт, без пути и завершающего `/`. `localhost` и `127.0.0.1` — разные origin.
Запускайте Rails из окружения с этой переменной; после изменения значения
перезапустите процесс с новым окружением. Адрес не зашит в код: во всех окружениях
отсутствующая или пустая переменная отключает приём сообщений от генераторов.
Настройка передаётся странице сервером; пересборка Vite не требуется.

Сервис конвертации — `GeneratedSequenceImporter`. Схема БД не меняется.

## Кнопка Save в генераторе

1. В обработчике клика откройте новое окно/вкладку
   `http://localhost:4999/sequences/import?origin=http%3A%2F%2Flocalhost%3A8502`.
2. Дождитесь `asana-catalog:sequence-import:ready` от этого окна.
   Если пользователь ещё не вошёл в Asana Catalog, сначала он увидит страницу
   входа; после входа вернётся к импорту с тем же origin.
3. Передайте `{ type: "asana-catalog:sequence-import:save", sequence: json }`.
4. Страница вызывает API со своей авторизацией. Во время запроса она показывает
   сохранение, при ошибке — причину и кнопку повторной попытки.
5. При успехе генератор получает `asana-catalog:sequence-import:saved`, а окно
   Asana Catalog автоматически переходит на страницу секвенции.

Используйте один и тот же адрес Asana Catalog, на котором пользователь обычно
входит в систему. Не передавайте JSON или JWT в query string.

Пример кода на стороне генератора (вызывать непосредственно из клика Save):

```js
function saveToAsanaCatalog(sequence) {
  const catalogOrigin = "http://localhost:4999";
  const importUrl = new URL("/sequences/import", catalogOrigin);
  importUrl.searchParams.set("origin", window.location.origin);

  // Открываем синхронно внутри клика: иначе браузер может заблокировать окно.
  // Не используйте noopener/noreferrer: для обмена нужен window.opener.
  const receiver = window.open(importUrl.href, "_blank");
  if (!receiver) return Promise.reject(new Error("Allow popups for Save"));

  return new Promise((resolve, reject) => {
    const cleanup = () => {
      window.removeEventListener("message", onMessage);
      window.clearInterval(closedTimer);
    };
    const onMessage = (event) => {
      if (event.origin !== catalogOrigin || event.source !== receiver) return;

      switch (event.data?.type) {
        case "asana-catalog:sequence-import:ready":
          receiver.postMessage({
            type: "asana-catalog:sequence-import:save",
            sequence,
          }, catalogOrigin);
          break;
        case "asana-catalog:sequence-import:saved":
          cleanup();
          resolve(event.data);
          break;
        case "asana-catalog:sequence-import:error":
          cleanup();
          reject(new Error(event.data.error));
          break;
      }
    };
    window.addEventListener("message", onMessage);
    const closedTimer = window.setInterval(() => {
      if (!receiver.closed) return;
      cleanup();
      reject(new Error("Import window was closed"));
    }, 1000);
  });
}

// В обработчике кнопки: обработайте resolve/reject в интерфейсе генератора.
// saveToAsanaCatalog(generatedSequence).then(onSaved).catch(showSaveError);
```

Обе стороны проверяют точный origin и `event.source`. Сообщения от посторонних
окон игнорируются. Значение `targetOrigin` всегда конкретное, не `"*"`.
Не закрывайте вкладку генератора до передачи данных. Если используете заголовок
`Cross-Origin-Opener-Policy`, убедитесь, что он не разрывает связь между окнами.
Повторное сообщение `save` в одном окне не создаёт дополнительный запрос.

Ответ `saved` содержит те же `sequence`, `created` и `redirect_url`, что и API,
плюс поле `type`. Ответ `error` содержит `type` и текстовое поле `error`.

## Закрытый POST API

```http
POST /api/sequences/import_generated
Content-Type: application/json
Accept: application/json
Authorization: Bearer <Asana Catalog access token>
```

Используется существующая авторизация Asana Catalog (JWT или текущая сессия).
Без авторизации ответ `401`. Страница импорта сама восстанавливает текущую
авторизацию через существующий `AuthBootstrap` и делает запрос на своём origin.
Генератору не нужно делать cross-origin fetch с cookies или получать токен.
Обычный POST HTML-формы не поддерживается (`415`).

Тело — исходный JSON, **без обёртки `sequence`**:

```json
{
  "info": {
    "uuid": "80625619-08bd-4991-a786-b362768eb6c4",
    "name": "New Sequence",
    "description": null,
    "language": "en",
    "elements_count": 1
  },
  "thumbnails": [
    {
      "uuid": "39D3F4B4-E585-4D72-AC44-889965EC9670",
      "order": 1,
      "start_mirror": false,
      "end_mirror": false
    }
  ]
}
```

| Поле | Правило |
| --- | --- |
| `info.uuid` | Обязательный UUID; сохраняется в `Sequence.uuid` |
| `info.name` | Обязательная непустая строка |
| `info.description` | Необязательная строка или `null` |
| `info.language` | `en`, `uk` или `ru`; при отсутствии — язык пользователя |
| `info.elements_count` | Необязательно; если передано, равно длине `thumbnails` |
| `thumbnails` | От 1 до 75 объектов |
| `thumbnails[].uuid` | UUID доступного пользователю `CatalogItem`, без учёта регистра |
| `thumbnails[].order` | Уникальное неотрицательное целое; порядок сортировки |
| `start_mirror`, `end_mirror` | JSON boolean; при отсутствии `false` |

`info.id`, `image_url`, `alt` и прочие неизвестные поля игнорируются. Поля
`user_id`, `public`, внутренние ID и настройки дыхания из JSON не применяются.
Название и изображение элемента берутся из каталога, внешние изображения не
скачиваются. Асаны должны быть в публичном каталоге или каталоге текущего
пользователя; администратору доступны также остальные каталоги.

## Правила конвертации

- Создаётся приватная `Sequence` с тем же UUID, названием, описанием и языком,
  владельцем становится текущий пользователь. Применяется существующий лимит
  секвенций обычного пользователя (5).
- `thumbnails` сортируются по `order`; внутренний порядок элементов — `0..N-1`.
  Повторение одного UUID асаны допустимо: создаются отдельные элементы.
- Каждый элемент — статическая асана (`element_type: "Asana"`), активный, с новым
  UUID элемента, ссылкой на `CatalogItem`, его переводами и изображением.
  Сохраняются `start_mirror` и `end_mirror`; включено `use_in_breathing_mode`.
- Каждый элемент получает один активный `Breath`: `breathing_cycle_type: "Breath"`,
  вдох 5 пульсов, выдох 5 пульсов, без задержек, один повтор, `say_respiration: true`.
  Длительности берутся из `Sequence.breath_default_pulses` (по умолчанию 5).
- Счётчики элементов и длительности рассчитываются по созданным данным, включая
  зеркальные проходы. Всё сохраняется одной транзакцией.
- Повторный Save с тем же UUID от того же пользователя возвращает существующую
  секвенцию **без перезаписи пользовательских правок**. Это также позволяет
  безопасно повторить запрос после обрыва соединения. Для новой версии с новым
  содержимым генератор должен выпустить новый UUID.
- UUID другого владельца даёт `409`, даже если секвенция публичная. Неизвестные,
  недоступные или неоднозначные UUID каталога дают `422`; ничего не сохраняется.

## Ответы

При создании — `201 Created`; при повторном Save — `200 OK` с `created: false`:

```json
{
  "sequence": {
    "id": 200,
    "uuid": "80625619-08bd-4991-a786-b362768eb6c4",
    "name": "New Sequence",
    "slug": "new-sequence"
  },
  "created": true,
  "redirect_url": "/sequences/200-new-sequence"
}
```

Заголовок `Location` также содержит этот относительный URL. API возвращает JSON,
а переход выполняет страница импорта. Это существующая страница просмотра
секвенции с возможностью дальнейшего редактирования и воспроизведения.

| HTTP | Причина |
| --- | --- |
| `400` | Синтаксически некорректный JSON |
| `401` | Нет действующей авторизации |
| `409` | `uuid_conflict`: UUID уже занят другим владельцем |
| `415` | Content-Type не `application/json` |
| `422` | `invalid_sequence`, `catalog_items_not_found`, `ambiguous_catalog_items` или `validation_failed` |

Пример ошибки конвертации:

```json
{
  "error": "Catalog items were not found or are not accessible",
  "code": "catalog_items_not_found",
  "details": { "uuids": ["39d3f4b4-e585-4d72-ac44-889965ec9670"] }
}
```

Проверка API отдельно от браузерного сценария:

```sh
curl -i http://localhost:4999/api/sequences/import_generated \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json' \
  -H "Authorization: Bearer $ASANA_ACCESS_TOKEN" \
  --data-binary @docs/import_sequence/sequence-5703a444aab9440289641e8753923ef3.json
```

Команда создаёт реальную секвенцию для владельца токена; UUID асан из примера
должны существовать и быть доступны в целевой базе.
