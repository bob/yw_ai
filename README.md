# Yoga Sequence Generator

A conversational assistant for building yoga sequences from an external Asana
Catalog. Generate a routine, browse poses, and refine the result through chat.

[Try the app at ai.yogaworkout.app](https://ai.yogaworkout.app)

https://github.com/user-attachments/assets/7244722f-2907-4ef7-b183-09c2c00aaacd

## Features

- Generate sequences by pose count or duration, with an optional starting pose.
- Search the catalog and view poses as thumbnails.
- Replace, swap, add, or remove poses using natural-language requests.
- Follow live progress, compare sequence variants, and regenerate from the original prompt.
- Save the selected sequence to your Asana Catalog account.

Generated sequences contain up to 45 poses. Conversation history and sequence
variants stay in the current browser session; they do not survive a server restart.

## Technology and models

| Technology | Role |
| --- | --- |
| Python 3.14+ and uv | Application runtime and dependency management. |
| Streamlit | Chat interface, pose galleries, sequence controls, and live progress. |
| LangChain and LangGraph | One tool-calling assistant with conversation state and in-memory checkpoints. |
| OpenAI GPT-6 Luna (`openai:gpt-6-luna`) | Default model for conversation, tool calls, and history summarization; configured through `ASSISTANT_AGENT_MODEL`. |
| TypeSafe Jev (`langchain-typesafe`) | Typed decisions that select poses from catalog candidates during generation and insertion. |
| Pydantic and HTTPX | Structured data validation and HTTP integration with Asana Catalog. |
| LangSmith (optional) | Agent, model, and tool tracing; disabled by default. |

Python services handle sequence limits, validation, and variant history. Pose
names, tags, and images come from Asana Catalog.

## Run locally

Requires Python 3.14+, [uv](https://docs.astral.sh/uv/), OpenAI and TypeSafe API
keys, and access to a running Asana Catalog service.

```bash
uv sync --locked
cp .env.example .env
```

Set these values in `.env`:

```dotenv
OPENAI_API_KEY=your-openai-api-key
TYPESAFE_API_KEY=your-typesafe-api-key
ASANA_SERVICE_BASE_URL=http://localhost:4999
ASANA_CATALOG_FOLDER_UUID=yoga-public-v1
```

Use the catalog URL where you normally sign in. To enable **Save**, configure
`SEQUENCE_GENERATOR_ORIGINS` on the catalog server with this app's origin; see
[the import protocol](docs/import_generated.md).

```bash
uv run --env-file .env streamlit run src/langchain_app/assistant.py
```

Try: “Generate a 10-minute sequence starting with Tree pose”, then “Swap the
first and last poses” or “Add two seated poses at the end”.

## Development

Application code lives in [`src/langchain_app`](src/langchain_app), with agent
tools in `tools/`, business logic in `services/`, and Streamlit views in `ui/`.
The `*-training/` directories contain standalone learning examples.

Tests use fake models and mocked HTTP. The offline runner is
[`scripts/test_offline.py`](scripts/test_offline.py); it requires cached tokenizer
data and blocks network calls. GitHub Actions runs offline tests and deploys
successful pushes to `master`.

See [feature specifications](docs/specs) for implementation details and
[security and deployment controls](docs/specs/017-application-security-hardening/README.md)
for production setup. Set `LANGSMITH_TRACING=true` and supply `LANGSMITH_API_KEY`
only when you want to send conversation and tool data to LangSmith.
