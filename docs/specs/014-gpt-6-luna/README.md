# 014 — Переход на GPT-6 Luna

Общий default — `gpt-6-luna` (`openai:gpt-6-luna` для LangChain).
Он используется ассистентом, суммаризацией, учебными приложениями и defaults
тестовых fixtures. `.env` и `.env.example` выбирают ту же модель.
Старые модели остаются только в проверках совместимости и исторических отчётах.

В [model_config.py](../../../src/langchain_app/model_config.py) задаются
`reasoning_effort="none"` и `use_responses_api=False`: Luna поддерживает tools
в текущем Chat Completions при отключённом reasoning. Явная настройка API
предотвращает автоматический переход установленного LangChain на Responses.
Кэш использует `prompt_cache_options={"mode":"implicit","ttl":"30m"}`;
legacy `prompt_cache_retention` для Luna не отправляется.

Основание: [OpenAI Docs — GPT-6 Luna](https://developers.openai.com/api/docs/models/gpt-6-luna)
и [Prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).

Проверка — только локальные mocks и `httpx.MockTransport`: модель, API endpoint,
reasoning, кэш, суммаризация и сохранность истории. Платных API-проверок миграции нет.
Текущий `tiktoken` не знает Luna; счётчик истории использует предусмотренный
fallback `o200k_base` с запасом 25%, без выдачи оценки за provider usage.

Локально прошли 42 теста ассистента/истории/настроек и 7 SDK/cache-тестов.
`compileall` для `src` и fixtures и `git diff --check` — без замечаний.
Уже запущенное приложение нужно перезапустить: `.env` загружается при старте,
а экземпляр агента сохраняется в Streamlit session state.
