# 017: Application security hardening

The application remains anonymous behind the deployment's existing Basic Auth
gate. This work preserves that access model, generation, all sequence edits,
variant navigation, progress history, and authenticated catalog Save.

See [the security audit](audit.md) for findings, evidence, scope, and remaining
operational work. No production changes, history rewrites, or live model calls
are part of this change.

## Implemented controls

- Validate chat requests on the server before queuing/storing them and before
  model initialization. The UI and generation/addition tool schemas also enforce
  the 16,000-character limit. Requests below the limit retain their full text;
  history summarization does not truncate the current request or tool loop.
- Limit paid work to four simultaneous requests per process by default. Set
  `ASSISTANT_MAX_CONCURRENT_REQUESTS` to an integer from 1 to 32 and restart the
  process to change it. Capacity is released on completion, failure, and iterator
  cancellation. Excess requests fail immediately with a retry message.
- Bound retained decoded catalog and public-sequence response bodies to 16 MiB.
  Request identity encoding, reject oversized declared lengths, and stop reading
  oversized streamed responses. Explicitly disable automatic redirects, including
  on injected clients. Keep existing timeouts and client ownership behavior.
- Restrict the literary training tool to HTTPS `www.gutenberg.org` and
  `gutenberg.org`, without credentials, alternate ports, or other URL schemes.
  Validate every redirect, stop after four fetches, and limit documents to 2 MiB.
  Importing that example no longer initializes or calls a model; running it as a
  script retains the Gatsby example.
- Keep unexpected exception details in server logs; retain CORS and XSRF
  protection and disable static directory serving. Disable Streamlit usage
  telemetry. Existing handled public errors remain visible.
- Disable LangSmith tracing in the example environment by default. Existing
  private environment files retain their settings; operators can explicitly opt
  in to tracing when its data handling is appropriate.
- Prefer GitHub environment secrets for deployment configuration while preserving
  existing variable fallbacks. This is migration support: values still supplied
  through variables remain unmasked until migrated.
- Replace the current sample export's signed image URLs with synthetic URLs.
  Catalog UUIDs, metadata, sequence order, and mirror flags remain unchanged.

## Verification

```bash
# Fetch static vocabulary files separately if they are not already cached.
# This downloads tokenizer data; it does not invoke a model.
uv run --locked python -c 'import tiktoken; tiktoken.get_encoding("o200k_base"); tiktoken.get_encoding("cl100k_base")'

# Dummy credentials, tracing off, dotenv loading disabled, network/process
# egress blocked with a Python audit hook. Presentation sleeps are skipped.
uv run --locked python scripts/test_offline.py

# Built-in Node tests execute the actual inline Save component against doubles.
node --test tests/js/sequence_import.test.mjs
```

The dedicated pacing test supplies a fake clock and sleep, so presentation timing
logic is still tested. Linux CI additionally runs Python tests in an OS network
namespace; the local audit hook is a guard for trusted Python tests rather than
a sandbox for arbitrary malicious code. The runner does not discover or execute
live acceptance scripts or training applications.

Security regressions cover local-file and metadata-address rejection, deceptive
URL hosts, redirect escapes, oversized and compressed responses, response stream
closure, request rejection before model initialization, concurrency slot cleanup,
and forged browser messages. Existing tests cover the complete sequence flow.
