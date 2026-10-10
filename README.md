# LangChain app

The Sequence Generator keeps the pose count and **Generate** at the top and a
chat prompt at the bottom. **Generate** fills and submits a text request such as
`Generate a yoga sequence with 12 poses.` You can also type a request directly.
Both paths use `AssistantAgent`, a single LangChain agent with tools in
`langchain_app.tools`: `generate_sequence`, `search_catalog_items`,
`get_catalog_tag_names`, and `replace_sequence_element`. Generation still uses
the `SequenceGenerator` service and streams poses into the UI. Catalog searches,
tag queries, and text replies can run without generation. Both catalog endpoints
always receive `language=en`; the assistant replies in the user's language and
keeps item names and tags exactly as returned by the API.

You can specify a duration in minutes instead of a pose count: each pose
takes 40 seconds, so 10 minutes means 15 poses. Fractional counts round to the
nearest integer (halves round up), with a minimum of one pose. An explicit pose
count takes precedence over duration. If neither is specified, the agent uses
9 poses. **Regenerate** resends the original prompt. Set `ASSISTANT_AGENT_MODEL`
to change the model. A nonempty `SEQUENCE_AGENT_MODEL` remains a fallback for
existing `.env` files; the default is `openai:gpt-5.5`.
The agent requires `OPENAI_API_KEY` in addition to the existing
catalog and TypeSafe settings.

The agent stores complete sequence variants, including catalog IDs/UUIDs, names,
tags, order, occurrence IDs, and mirror flags. Every model call receives the
currently selected variant, including calls after tools and subsequent chat turns.
**Previous Sequence**, **Next Sequence**, and **Use this sequence** select the
context without calling the model. The active card is marked **Active sequence**.
**Regenerate** creates a new UUID in the source variant's family and retains its
original request. UI thumbnails and Save data are projections of the stored snapshot.
An incomplete generation is discarded; a completed variant survives failure of
the model's subsequent text reply.

Ask the assistant to replace one pose in the selected variant. It verifies the
candidate against the latest catalog search and creates a new UUID in the same
family, with the selected variant as its parent. The original remains in history;
the replacement card includes the family's variants for Previous/Next navigation.
Other poses and occurrence IDs stay unchanged, and the replacement's mirror flags
follow its bilateral setting. Save uses the selected variant's UUID and catalog
UUIDs. Replacing a pose with itself returns `no_change` without creating a variant.
Stale variants, unverified candidates, unusable items, and duplicate poses are
rejected. Failed searches clear previous candidates. Tools run sequentially, with
at most one sequence change per request.

The first release supports replacing one element per request. For multiple
replacements, the assistant asks which element to change first. Adding, deleting,
moving elements, and changing mirroring are planned extensions; their exact scope
can change during review. The assistant explains that these edits are unavailable.

Conversation history and variants live in one assistant's `InMemorySaver` in the
current Streamlit session. Sessions do not share conversation state, and ordinary
reruns retain it. Recovery after a process restart or a new browser session is not
implemented. Legacy cards with thumbnails alone remain viewable without supplying
invented catalog data to the model.

Chat, **Generate**, and **Regenerate** show a live execution block immediately.
It follows real model and catalog work, streams provisional poses, and collapses
when the response finishes. Public intermediate messages, completed steps, elapsed
time, and errors remain available after reruns. Each regeneration keeps its earlier
attempts. Request and variant controls are disabled during execution; a failed
attempt keeps any already committed sequence available for Save. See
[feature 002: assistant response progress](docs/specs/002-assistant-response-progress/README.md).

This completes steps 1–7 of the first release of [feature 001: create-assistant-agent](docs/specs/001-create-assistant-agent/README.md).

```bash
uv run --env-file .env streamlit run src/langchain_app/assistant.py
```

For a detailed walkthrough from `assistant.py` through the agent, tools, services,
state, and Save, see [how the assistant server works](docs/assistant-server-process.md)
(in Russian).

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
