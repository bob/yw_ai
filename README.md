# LangChain app

The Sequence Generator keeps the pose count and **Generate** at the top and a
chat prompt at the bottom. **Generate** fills and submits a text request such as
`Generate a yoga sequence with 12 poses.` You can also type a request directly.
Both paths send the same kind of prompt to a LangChain agent, which infers the
pose count and calls the existing generator as a tool, streaming poses into the
UI. You can specify a duration in minutes instead of a pose count: each pose
takes 40 seconds, so 10 minutes means 15 poses. Fractional counts round to the
nearest integer (halves round up), with a minimum of one pose. An explicit pose
count takes precedence over duration. If neither is specified, the agent uses
9 poses. **Regenerate** resends the original prompt. Set `SEQUENCE_AGENT_MODEL`
to change the model (default:
`openai:gpt-5.5`). The agent requires `OPENAI_API_KEY` in addition to the existing
catalog and TypeSafe settings.

```bash
uv run --env-file .env streamlit run src/langchain_app/generate_sequence.py
```

Sequence Generator's **Save** opens Asana Catalog in a new tab and transfers the
selected sequence to its authenticated import page. Set `ASANA_SERVICE_BASE_URL`
to the address where you sign in to the catalog (for example `http://localhost:4999`).
On the catalog server, allow the generator's origin with
`SEQUENCE_GENERATOR_ORIGINS=http://localhost:8502`; see
[the import protocol](docs/import_generated.md). Save no longer writes JSON to `docs`.

This sample agent answers a weather question with a LangChain tool. LangSmith
automatically traces the agent, model, and tool calls when tracing is enabled.

Add your OpenAI and LangSmith API keys to `.env`, using `.env.example` as a guide.
The local `.env` already enables tracing and uses the `langchain-app` project.
Then run:

```bash
uv run --env-file .env langchain-app
```

To trace the second example, run:

```bash
uv run --env-file .env src/langchain_app/second.py
```

The runs appear in the `langchain-app` project at
https://smith.langchain.com. The API endpoint in `.env` must use the same
region as the LangSmith website and API key. To turn tracing off, set
`LANGSMITH_TRACING=false` in `.env`.
