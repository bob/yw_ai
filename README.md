# LangChain app

The Sequence Generator keeps the pose count and **Generate** at the top and a
chat prompt at the bottom. **Generate** submits a text request such as
`Generate a yoga sequence with 12 poses.` without filling the input. You can also
type a request directly. After each response, the input is cleared and focused
so you can immediately type the next request.
Both paths use `AssistantAgent`, a single LangChain agent with tools in
`langchain_app.tools`: `generate_sequence`, `search_catalog_items`,
`show_catalog_items`, `get_catalog_tag_names`, `replace_sequence_element`,
`swap_sequence_elements`, and `add_sequence_elements`. Generation still uses
the `SequenceGenerator` service and streams poses into the UI. Catalog searches,
tag queries, and text replies can run without generation. Both catalog endpoints
always receive `language=en`; the assistant replies in the user's language and
keeps item names and tags exactly as returned by the API.

Ask for a specific first pose when generating a sequence, for example
“Generate 12 poses, starting with Tree pose.” The assistant resolves the real
catalog item and passes `starting_catalog_item_id`; it overrides the random start
and does not require `Other/Starting`. Without an explicit first pose, starting
positions are chosen with probabilities of 10% lying, 60% sitting, and 30% standing,
then a pose is chosen randomly within that position. After a lying start, the
second pose must be lying on the back. After a standing start, the second pose
must not be lying, with standing preferred.
See [feature 004: starting sequence element](docs/specs/004-starting-sequence-element/README.md).

Ask to browse catalog poses, ask which poses are available, or find replacement
candidates without changing a sequence. After searching, the assistant selects
verified IDs with `show_catalog_items`; the selection appears as thumbnails above
any reply, outside the progress panel. Its `catalog_items_displayed` event can
arrive before the final text. Searching alone does not show a gallery.
Generation, Regenerate, replacement, swapping, addition, and technique answers keep their existing
output unless catalog browsing is separately requested. Generate and Regenerate
commands reject catalog display. Catalog thumbnails use the same size and styles
as sequence thumbnails, with real names as alt text and a name fallback for
missing images. They have no sequence controls, metadata, or mirror brackets.
Each message stores a detached `catalog_items` snapshot for session reruns,
including when the model's final text fails. New searches and later selections
leave earlier messages unchanged.
See [feature 003: catalog item display](docs/specs/003-catalog-items-display/README.md).

Successful sequence operations and catalog displays need no text when the
thumbnails fully answer the request. The assistant omits completion announcements,
action recaps and generic offers of further help. Useful explanations, applied
limits, errors and clarifying questions remain visible below the thumbnails,
both during the response and in chat history, so arriving text does not move
the gallery. A successful regeneration with no text clears the previous variant's
reply.

You can specify a duration in minutes instead of a pose count: an ordinary pose
takes 40 seconds, while a mirrored (bilateral) pose takes 80 seconds for both
sides and remains one sequence element. The generator accounts for selected
mirrored poses while building the sequence, so 10 minutes allows up to 15
ordinary poses, fewer with mirrored poses. The time budget rounds to the nearest
40 seconds (halves round up), with at least one pose; the last mirrored pose can
exceed the rounded budget by 40 seconds. An explicit pose count takes precedence
over duration. **Every generated sequence is limited to 45 elements**, even for
requests with a higher count or longer duration. If neither is specified, the
agent uses 9 poses. Size calculation, explicit-count priority, and capping run in
Python; the model supplies the requested count or duration. See
[feature 005: sequence duration and limit](docs/specs/005-sequence-duration-and-limit/README.md)
and [feature 013: compact system instructions](docs/specs/013-compact-system-instructions/README.md).
**Regenerate** resends the original prompt. Set `ASSISTANT_AGENT_MODEL`
to change the model. A nonempty `SEQUENCE_AGENT_MODEL` remains a fallback for
existing `.env` files; the default is `openai:gpt-6-luna`.
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

Ask the assistant to swap two poses in the selected variant, for example
“Swap the third and fifth poses” or “Swap the first and last poses.” Unique pose
names also work; repeated names or an incomplete pair require clarification.
The swap creates a new UUID in the same family and moves the complete occurrences,
including their IDs and mirror flags. Other positions stay unchanged. Catalog
search, generation, and TypeSafe are not needed. The original remains in history;
Save sends the new variant, and Regenerate repeats its original generation request.
Swapping an occurrence with itself returns `no_change`. See
[feature 006: swap sequence elements](docs/specs/006-swap-sequence-elements/README.md).

Ask to add poses to the selected variant, for example “Add two poses before the
third”, “Add a pose at the beginning”, or “Fill to 20 elements”. The default is
one new pose at the end. “Another 20” adds 20; “to 20” adds the difference.
The same `SequenceGenerator` and TypeSafe select the new block with the existing
prefix and suffix in context. The original occurrences, order and mirror flags
stay intact. A block creates one new variant and appears as a whole after commit;
selection progress shows the number of new poses without a provisional card.
Additions fill only the available space within 45 elements; a full sequence
returns `no_change`. A selection failure leaves the selected variant intact.
History, Save and Regenerate use the existing controls, and Regenerate repeats
the original generation request. See
[feature 007: add sequence elements](docs/specs/007-add-sequence-elements/README.md).

Each request supports one successful generation, one replacement, one pair
swap, or one addition block. For multiple edits or combined operations, the assistant asks which to do
first. Deleting, moving one element while shifting others, and changing
mirroring independently are unavailable.

Conversation history and variants live in one assistant's `InMemorySaver` in the
current Streamlit session. Sessions do not share conversation state, and ordinary
reruns retain it. Recovery after a process restart or a new browser session is not
implemented. Legacy cards with thumbnails alone remain viewable without supplying
invented catalog data to the model.

GPT receives recent whole turns plus a cumulative summary when completed history
exceeds its token budget. The full transcript and sequence variants remain in
session state, and the current request and tool loop are always retained.
Set `ASSISTANT_HISTORY_MAX_TOKENS` (8192), `ASSISTANT_HISTORY_TARGET_TOKENS`
(4096), and `ASSISTANT_HISTORY_SUMMARY_MAX_TOKENS` (768), with
`0 < summary_max < target < max`. These budgets include the summary; system
instructions, tools, the selected snapshot and the current turn are separate.
Summarization uses the same GPT model without tools, at most once per request.
Failure preserves the previous summary and uses a bounded recent suffix.
See [feature 010: bounded GPT history](docs/specs/010-bounded-gpt-history/README.md).

Model requests keep instructions and the ordered tool schemas stable, then send
the current selected sequence as a separate final system message. This snapshot
is never appended to the stored transcript. OpenAI calls use a stable per-session
`prompt_cache_key`, with separate keys for assistant and history-summary calls.
`ASSISTANT_PROMPT_CACHE_ENABLED` defaults to `true`; setting it to `false` stops
sending cache settings (OpenAI's automatic caching can still apply).
For `gpt-6-luna`, requests use Chat Completions with `reasoning_effort="none"`
and `prompt_cache_options={"mode":"implicit","ttl":"30m"}`. The optional
`ASSISTANT_PROMPT_CACHE_RETENTION=30m` matches this default. Explicit
unsupported settings fail at initialization. Internal `AssistantAgent.model_usage`
records reported input, cached input, output, and reasoning tokens per request,
including summarization; missing usage remains unknown. Cache hits and savings
depend on the provider. The canceled feature 011's routing and local completion
are not used. See [feature 012: GPT prompt caching](docs/specs/012-gpt-prompt-caching/README.md).
The app, summary model, training examples, and fixture defaults share
[the Luna configuration](src/langchain_app/model_config.py); see
[014: GPT-6 Luna migration](docs/specs/014-gpt-6-luna/README.md).

Chat, **Generate**, and **Regenerate** show a live execution block immediately.
It follows real model and catalog work, streams provisional poses, and collapses
when the response finishes. Public intermediate messages, completed steps, elapsed
time, and errors remain available after reruns. Each regeneration keeps its earlier
attempts. Request and variant controls are disabled during execution; a failed
attempt keeps any already committed sequence available for Save. See
[feature 002: assistant response progress](docs/specs/002-assistant-response-progress/README.md).

When OpenAI reports exhausted API credits, the error explains that credits have
run out and links to billing settings. A general quota error asks you to check
credits and usage limits. These messages remain visible in request history; see
[feature 008: API credit errors](docs/specs/008-api-credit-errors/README.md).

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
