# LangChain app

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
