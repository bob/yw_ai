# Application security audit — 20 October 2026

## Scope and conclusion

This is a repository security review with offline regression testing, public
dependency-advisory lookups, and read-only GitHub metadata/log inspection. It
includes the production Streamlit entrypoint, agent and tools, schemas, shared
clients, browser components, training examples, dependency lockfile, publication
artifacts, and deployment/CI workflows. The reviewed branch initially pointed at
`2c22545`; the working tree was clean before the audit.

The highest-impact confirmed code issue was a model-controlled arbitrary URL
fetcher in a training example. A harmless temporary marker demonstrated a local
file read. The production yoga agent does not register that tool. That issue is
fixed, alongside availability and disclosure hardening described below.

No live models, production application requests, exploit probes against the
deployed site, or catalog writes were used. The external Rails catalog, proxy,
host firewall, SSH configuration, and systemd service were not independently
audited on the running server. Basic Auth at the proxy is an operator-confirmed
deployment control; its live configuration was not verified here. These limits
mean this review is not a production penetration test or a guarantee of safety.

## Findings and disposition

| ID | Severity and scope | Evidence / risk | Disposition |
| --- | --- | --- | --- |
| APP-01 | High when the literary demo is run | Its `urllib` tool accepted model-selected URLs, including `file://` and internal HTTP addresses, followed redirects, and read unrestricted bodies. Local-file access was reproduced using a harmless marker. | Fixed: explicit Gutenberg HTTPS origin allowlist, validation of redirects, fetch-count/body limits, sanitized errors, and a main guard. The intended Gatsby example remains available. |
| APP-02 | Medium, production availability and provider costs | Chat had no request-length cap; separate browser sessions could initiate unlimited simultaneous agent work. The existing 45-pose and 20 graph-step limits did not bound either entry point. | Mitigated: server and UI enforce 16,000 characters; generation/addition tool schemas use the same bound; a process-wide concurrency limit defaults to four. This is not a spending ceiling or distributed rate limiter. |
| APP-03 | Medium, upstream resource consumption | Catalog and legacy public-sequence clients eagerly buffered entire responses. | Mitigated: 16 MiB retained decoded-body cap, early Content-Length checks, chunked/compressed response checks, deterministic stream closure, and explicit redirect refusal. Existing per-client timeouts remain. |
| APP-04 | Low, unexpected exception disclosure | Streamlit's default full browser exception details could expose paths, provider/configuration error text, and stack traces outside the application's handled-error paths. | Hardened: tracked configuration sets `client.showErrorDetails="none"`. Server logs retain diagnostics. Handled billing, quota, and validation errors still render normally. |
| APP-05 | Low, optional telemetry | The example environment enabled LangSmith tracing, which can export conversation/tool data; Streamlit telemetry was not explicitly disabled. | Safer defaults added. Existing private environment values and third-party provider data policies are unchanged; operators must review tracing separately. |
| PUB-01 | Low, infrastructure metadata | Both successful deployment jobs exposed the host/IP, SSH port, app/shared paths, and service name in their environment listings. SSH user/key/known-host values were masked. | Partly prepared: workflow now prefers masked secrets, retaining existing variables for compatibility. Migrating values and removing historical logs remain operational work. Infrastructure paths also occur in the workflow source. |
| PUB-02 | Conditional data disclosure | The sample sequence export contained nine signed Rails image URLs pointing at a local catalog origin. Their validity against production was not tested. | Current file sanitized. Old versions remain in Git history. If those assets must remain private, resolve the historical signed URLs before publishing. |
| OPS-01 | Medium, deployment governance | GitHub reported `master` unprotected and no repository rulesets. Successful pushes to `master` automatically deploy. The production environment allows only `master`, without a reviewer rule. | Open: configure required CI checks and branch protections appropriate to the repository. Public contributors cannot directly deploy through the current workflow guards, but authorized pushes and merged changes are production changes. |
| OPS-02 | Operational boundary | The app itself has no per-user login or durable per-user budget. Existing Basic Auth deliberately gates otherwise anonymous access. | Retained per operator instruction. Verify proxy coverage, backend reachability, and abuse controls as described below. |

## Controls reviewed without a confirmed vulnerability

- The production agent has a fixed tool list: public catalog queries and edits to
  the current session's sequences. It exposes no shell, filesystem, arbitrary URL
  fetcher, database query executor, or direct catalog write operation.
- Each assistant creates its own checkpointer and random thread ID. Conversation
  and sequence state are not stored in a globally cached assistant. Returned
  snapshots and sequence variants are detached copies. Shared cached clients and
  generators do not hold conversation history.
- Tool context and clients are injected by the application, not provided as
  model arguments. Replacement/display candidates must come from verified
  catalog results. Edits validate the active variant and occurrence IDs. The
  operation guard prevents multiple successful sequence mutations per turn.
- Catalog requests use a configured origin/folder, encoded folder IDs, structured
  query parameters, forced English, TLS verification, and timeouts. Public
  catalog data is not authorization for private catalog operations.
- Thumbnail attribute values are HTML-escaped. Main model replies use Markdown
  without unsafe HTML. The focus script interpolates a server-generated UUID;
  untrusted model/catalog text is not passed to JavaScript execution.
- Save verifies both `event.origin` and the exact receiving popup window. It
  sends the payload only to the explicit catalog origin. Status/error content
  uses `textContent`. No catalog access token or password is sent to the generator.
  The opener relationship is part of this protocol; adding `noopener` would
  break the current handoff. Four Node tests exercise these boundaries and cleanup.
- Expected failures use sanitized progress/error messages. Provider usage
  records contain counts/IDs rather than API keys or raw exception bodies.
- CI has read-only permissions, pinned action revisions, no persisted checkout
  credentials, dummy provider keys, disabled tracing, and OS network isolation
  during Python tests. Production deployment checks the triggering event, source
  repository, successful push CI, branch, exact tested SHA, clean checkout,
  remote identity, and SSH host key. It does not consume untrusted PR artifacts.

Prompt injection can still influence generated text or the choice of an allowed
pose/tool. System prompts are behavioral guidance, not authorization controls.
The deterministic tool and session boundaries are the relevant protection. The
external catalog import endpoint must independently authenticate and authorize
every save; its actual implementation is outside this repository review.

## Dependency and static analysis results

- Baseline: **360 offline Python tests passed** before integration. Final
  regression suite: **377 offline Python tests passed**; **four browser-component
  Node tests passed**. Network/process egress was denied during Python tests.
  Live acceptance/model tests were not run. One history stress fixture was sized
  within the new request cap while retaining its above-budget current-turn and
  tool-loop assertions.
- `pip-audit` checked all **75 exact registry package versions** exported from
  `uv.lock`: **zero known vulnerabilities**, **zero skipped packages** on the audit
  date. No application dependencies were upgraded or added.
- Bandit initially reported one medium finding for the arbitrary `urllib` loader
  and five low findings for `random.choice`, `random.choices`, and `random.sample`.
  After the fix, the medium/high scan is clean. The five low findings are reviewed
  nonissues: they choose poses or demo greetings, not security tokens or secrets.
- The earlier publication review checked 225 tracked files and 32 local commits,
  including comparison with the three current local provider API-key values.
  No matching API keys, private keys, or common credential formats were found.
  `.env`, Streamlit secrets, and private deployment docs were ignored and absent
  from the checked branch history. Four saved screenshots showed test data.
- GitHub inspection found one branch matching the local revision, no tags,
  issues/PRs, or stored Actions artifacts. Available test/deployment logs were
  inspected for common credential patterns; the infrastructure metadata above
  was the confirmed disclosure. The enabled wiki was not independently inspected.

Advisory lookups and pattern scans cannot detect every unknown, encoded, or
application-specific secret or vulnerability. They do not replace patch review,
dependency maintenance, or production controls.

## Operational follow-up before publication

1. Copy deployment host/port and any configuration intended to remain masked into
   identically named **production environment secrets**. The workflow uses secrets
   first and existing variables second, so deployment remains compatible during
   migration. Remove old configuration variables after migration if appropriate.
   Values in variable fallbacks are still logged in plain text.
2. Remove the logs of the two successful deployment runs if production metadata
   must remain private: run IDs `39168464035` and `39301248954`. Also review future
   logs after migration. Neither logs nor remote settings were changed here.
3. Decide whether the old sample image URLs and raw test transcripts should be
   public. Removing or sanitizing a current file does not remove earlier commits.
   Do not rewrite production history casually: coordinate any rewrite with the
   deployed checkout, collaborators, and stored deployment/rollback SHAs.
4. Enable branch protection and required CI. Consider approval for production
   deployments according to the project's release process. Recheck applicable
   rules after changing repository visibility.
5. Verify Basic Auth over HTTPS on the proxy's application and WebSocket routes;
   avoid logging its Authorization header. Verify the backend binds privately
   and its application port is not publicly reachable. Authenticated catalog Save
   should retain its own authentication and exact-origin allowlist.
6. Set suitable proxy rate/concurrency limits and provider budget controls. Four
   app slots are **per process**, and input/body limits do not limit total use,
   model output length, all SDK retries, or the number of distinct sessions.
   Sequence variants and canonical transcripts remain in session memory to
   preserve navigation; disconnected session cleanup and host memory limits matter.
7. Review existing `.env` tracing settings and LangSmith retention/access. Test
   JSON contains synthetic dialogues, usage/cost metrics, and provider request IDs;
   these were not identified as API credentials but need publication judgment.

The audit deliberately did not rotate credentials, remove remote logs, alter
Basic Auth, change repository visibility, rewrite history, push, or deploy.

## References

- [OWASP SSRF prevention](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html)
- [Streamlit chat input limits](https://docs.streamlit.io/develop/api-reference/chat/st.chat_input)
- [GitHub repository visibility consequences](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility)
- [GitHub variables are unmasked by default](https://docs.github.com/en/actions/concepts/workflows-and-actions/variables)
- [Rails Active Storage authenticated controllers](https://guides.rubyonrails.org/active_storage_overview.html#authenticated-controllers)
- [PyPA pip-audit](https://github.com/pypa/pip-audit)
