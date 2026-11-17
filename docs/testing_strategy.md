# Testing, Validation, and Release Gates

## Status and authority

**2026-11-17:** verify content quality separately from transport/schema correctness. [HANDOFF](HANDOFF.md) records the current integrated checkpoint; [evaluation report](evaluation_current.md) records exact live recipes, paid usage, failed attempts and assistant source review. Valid citation IDs and test counts do not establish entailment or human approval. Historical restart, crash and upload evidence remains qualified there and in the journal.

[CASE_STUDY.md](../CASE_STUDY.md) governs requirements. This strategy verifies the selected design in [system_design.md](./system_design.md), which owns limits, defaults, states and recovery contracts. Checks support all three 33% evaluation categories; comprehensive coverage is a bonus. Record actual commands, revision, environment, outcomes and unavailable checks in [AI_DEV_JOURNAL.md](../AI_DEV_JOURNAL.md). Code completion or a subagent report alone does not verify a phase.

## Validation layers

| Layer | Scope and evidence |
| :--- | :--- |
| Static | Ruff lint/format and mypy; validated settings and typed contracts |
| U — Unit | State/error rules, extraction locators/coverage, chunking, token budgets, prompts and citation validation using controlled substitutes |
| A — API | pytest and HTTPX: schemas, statuses, pagination, readiness, scope, ordered history, replay and errors |
| I — Integration | Actual isolated PostgreSQL/pgvector, migrations, filesystem, Redis and Celery at affected boundaries; provider substitute |
| E — Backend E2E | HTTP upload → real queue/worker → persisted insights/vectors → chat/history/citations → metrics, with a provider-compatible HTTP fixture server |
| UI — Browser | Repeatable real-browser walkthrough/automation against actual backend APIs; distinguish provider substitutes from live AI and AppTest emulation from a browser |
| L — Live AI | Opt-in real OpenAI embeddings/generation, fixed fixtures and human source-support review |
| O — Operational | Clean startup, restart, dependency interruption, recovery, shutdown and capacity samples |

Regression gates include actual-candidate review schemas (zero through three candidates), unchanged-packet bounded repair, persistent unbound-premise rejection, retained useful qualifications and complete repair-prompt reservations; a 400-short-paragraph DOCX through HTTP/worker publication, full downstream prompt reservations, irrelevant-claim pruning without false partial status, chunking soft-timeout retry, whole-exchange trimming in both history retrieval and complete-prompt assembly, final-summary capacity after citation corrections, and bounded telemetry cleanup with locked-row backlog reporting. These use deterministic provider substitutes; follow-up interpretation and distinct-insight usefulness still require live source review.

Default tests must never call OpenAI. Select the substitute explicitly and fail closed on live test configuration. Substitute E2E verifies orchestration, not live AI quality. Celery eager mode does not establish broker behavior; mocked vectors do not establish pgvector filtering/distance behavior.

For a reported conversational defect, passing ingestion, schema tests or a read-only UI check cannot close the issue. Reproduce the affected boundary offline, then intentionally verify the actual questions and a contextual follow-up with the live provider; include an unsupported question and inspect semantic support against original pages. Use ordinary narrative PDFs as well as short labeled-fact fixtures. Keep private bytes/content out of repository fixtures. A summary/insight review must assess theme coverage, useful depth and distinct supported takeaways, not only point counts or citation membership. Submit through the actual UI when its behavior changes, then reload and verify saved answers; avoid repeating accepted paid turns.

Use an isolated Compose project with separate database, broker namespace/queues and file volumes. A test-client transaction cannot roll back worker commits. Use bounded polling and explicit teardown. The following commands run the current checks:

```sh
docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test build
docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test run --rm --no-deps test ruff check .
docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test run --rm --no-deps test ruff format --check .
docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test run --rm --no-deps test mypy app ui scripts
docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test run --rm --no-deps test pytest -m "unit or api"
docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test run --rm test pytest
```

The override declares required services and waits for readiness for the full pytest run; static/unit/API checks use `--no-deps`. Keep both Compose files and `-p vault-test` together to isolate settings, credentials and data from the normal installation. Pytest disables sockets by default; real-service tests opt in only at controlled boundaries. The [provider-free GitHub Actions workflow](../.github/workflows/tests.yaml) now builds, checks types/style, starts isolated services, runs pytest and checks migration drift; the push and pull-request runs for `bb5001a` passed on GitHub (Ubuntu 24.04 runner). This is provider-free CI evidence, not a desktop/browser or live-AI platform certification. The observed platform is macOS Apple Silicon with Linux arm64 containers; other platforms remain unverified. Live AI is opt-in and is not part of the routine test command.

For the bounded quality tools, `--check-only` is provider-free:

```sh
docker compose exec api python -m scripts.evaluate_chat_review --check-only
docker compose exec api python -m scripts.evaluate_chat_review --check-only --suite quality --cases Q3,Q4
```

Live execution requires explicit `--live` plus a new run ID. `evaluate_chat_review` reuses saved retrieval/history; `evaluate_analysis_review` reprocesses saved chunks without publishing; `evaluate_service_review` creates a new inspectable conversation and executes the real service with all provider adapters metered. These tools need their referenced local artifacts/documents and are not fresh-install smoke tests. Source-review and raw inputs/results stay in ignored persistent demo storage. Audit failures must prevent publication; separately test audit-storage bounds, source-heading false positives, per-block/global citation limits, and replay with zero new provider slots.

## Milestone gates

| Milestone | Required exit evidence |
| :--- | :--- |
| M0 — Documentation and review | Requirement/evidence mapping, consistent contracts, bounded scope, acceptance criteria and review findings resolved or visibly pending; no implementation claims |
| M1 — Runtime foundation | Locked resolution/build/imports; secret-safe configuration errors; empty-database migration; pgvector/Redis/storage readiness; worker consumes a fresh infrastructure probe on its intended queue (real ingestion registration/consumption follows in M2); Compose wait detects broken/missing worker; isolated test commands work |
| M2 — Ingestion | PDF/DOCX upload/list/file/insights through real Redis/Celery/PostgreSQL; durable acceptance, extraction/analysis coverage, persisted vectors/insights; validation and recovery cases below; first live analysis recorded |
| M3 — Conversation | Persisted sessions, single-document membership enforcement, grounded answers/history, scoped retrieval, concurrency/replay/deadlines, coverage-aware broad questions and live quality gate |
| M4 — Metrics and operations | Fixture-exact metrics APIs, durable attempts/performance samples, retention, restart/shutdown/recovery and capacity samples |
| M5 — Minimal UI | Browser gate against actual APIs, restored history, visible partial coverage/errors, no provider/database credentials in UI |
| M6 — Handover | Applicable regression, clean-checkout startup, live demo/evaluation, down/up persistence, platform evidence and truthful README/journal; unresolved core blockers prevent completion |

M1–M5 are implemented, with versioned regression and deployment evidence. A clean export of committed code checkpoint `8ba7705` passed all 1,305 tests, static checks, Docker startup and migration-drift checks using fresh isolated volumes and a provider substitute. The run used cached Docker layers; remote cloning and uncached downloads were not tested. Current semantic checks and their remaining caveats are in [current evaluation](evaluation_current.md). Manual source review and delivery of the document-chat demonstration remain handover activities. Historical checkpoints do not certify later prompt changes.

Instrumentation and failure handling start in each slice. Run focused checks, then affected integrated boundaries; add meaningful regressions without arbitrary coverage targets. Recheck consequential prompt/model/retrieval changes. Preserve failures/skips and verify integrated subagent contributions.

## Fixtures and failure coverage

Review hardening regression gates include malformed JSON/base64 cursor member types returning422before database access; document embeddings rejected as invalid if float32 conversion yields nonfinite/all-zero data; no partial chunks/insights after that failure; original/draft case combinations for numbered references without weakening proper-name checks; and sustained isolated Redis outage beyond the former five-retry horizon. For that operational gate, accept one synthetic document during the outage, keep the worker process alive, restore only Redis and prove automatic completion/no extra publication/provider effects, then verify intentional shutdown still cooperates. Transport reconnection is unbounded, business/provider budgets are not. This is distinct from a manually restarted crash drill.

Named full API evaluations isolate uploads, analyses and dependency-connected conversations, freeze recipes and check actual published versions before questions. The current quality phase also uses budgeted frozen-evidence generation (no new planner/retrieval), read-only analysis on saved chunks (no new extraction/embedding/publication), and actual service-level conversations (real planner/embeddings/pgvector/state; no HTTP/browser). Report each evidence type honestly. All controlled paid requests in the bounded quality phase use the durable shared ledger; interrupted reservations count and failed/completed run IDs do not silently retry.

Use synthetic or redistributable documents with known facts, original locators and independently specified expected outcomes. Keep private documents and credentials out of fixtures.

| Area | Required cases |
| :--- | :--- |
| Inputs | Known-fact PDF/DOCX, body/table order, long document with decisive late evidence; corrupt/empty/encrypted/oversized/disguised files, excessive DOCX expansion and limits at/below/above boundaries |
| Coverage | Mixed selectable/scanned PDF retains useful text with visible partial/unknown extraction; zero-text scan fails actionably; blank/textless pages are not asserted to contain missing text; omitted embedded content warns. Separate original extraction coverage from analysis coverage of accepted chunks, including parent synthesis coverage. All accepted chunks participate before claiming whole-extracted-text analysis; visual completeness is not implied |
| Broad questions | Facts spread beyond top-k; overview uses source-backed analysis with original evidence; exhaustive request includes all expected items or states partial coverage. Compare expected facts, not exact prose |
| Isolation/grounding | Similar vocabulary/conflicting facts across documents; cross-session and wrong-document citations rejected; unknown source IDs rejected; overview trimming cannot retain claims without their supplied original support; exact support quotes and displayed excerpts validate; insufficient status takes precedence over partial when no useful evidence exists; malicious instructions ignored; rewritten/history facts must be grounded in current evidence |
| Structured generation failures | Distinguish `finish_reason=length` (including otherwise complete JSON), malformed JSON and schema-invalid UUIDs using actual SDK-shaped responses. Preserve available usage, output budget, safe finish/reason metadata and validated provider request ID on failure; exclude raw output/exceptions and secrets. Each adapter invocation makes one outbound request with no hidden retry |
| Analysis selection/repair | `analysis-v5` supplies literal passage catalogs; the model returns short `quote_ids`, and the server restores original UUIDs/exact quotes into the unchanged public schema. Test Unicode/whitespace preservation, bounded passages without first-N truncation, unknown/duplicate IDs, reduction lineage and all-chunk participation. No fuzzy text matching or increased budgets. A repairable failure gets at most one smaller result using unchanged evidence and a safe reason; test known failed-call usage, complete prompt/serialized-output/call/deadline limits and terminal second failure. Selection integrity does not prove semantic support |
| Read-only diagnosis | UUID/exact-filename selectors, newest-match limits/truncation, safe invalid/not-found/database errors, explicit metadata-only SQL and a read-only snapshot. Show durable attempt stage separately from final document stage; missing usage stays unknown. No provider calls, enqueue/retry, private content/configuration dumps or database mutations |
| Atomic idempotency | Simultaneous identical upload keys create one document/intent; mismatched content conflicts; losing staged files cleaned up. Simultaneous chat UUID replay reserves one turn; completed/failed/active replay and mismatched content obey contract, including retry while session busy |
| Processing recovery | Healthy uploads enqueue immediately with recovery scanning paused; crash after DB commit/before publication is repaired on a later scan. Race immediate publication against recovery and a fast worker without state regression. Broker unavailable after durable acceptance, duplicate delivery, worker death during provider calls/before publication, heartbeat/lease expiry, stale renewal/publication, bounded retries/exhaustion and no retry multiplication; unpublished results remain unavailable |
| Chat lifecycle | Unknown/not-ready document, competing questions, replay during provider work, provider/citation-repair failure, total deadline, reservation expiry and late completion. Client disconnect/reload plus graceful/forced API shutdown leave visible recoverable state, ordered history and safe same-UUID replay. Assistant history preserves nullable evidence classification/scope; completed partial answers visibly warn after reload, without inferring status from prose/citations or invoking AI |
| Metrics | Seeded successes/failures/retries; separate document/attempt denominators; bytes/counts; exact UTC window boundaries, milliseconds, queue/stage durations, sample counts, percentiles, empty aggregates and retention; controlled clock |
| Persistence/configuration | Ordinary down/up retains original bytes, published chunk IDs/vectors/profile, insights, membership, messages/context/citations, attempts and retained metrics. New question uses persisted vectors without document re-embedding. Profile mismatch, missing credentials and dependency failures are actionable and secret-safe |

Use real workers and bounded fault injection for interruption/recovery. Distinguish duplicate-result prevention from external calls/billing that may repeat after crashes.

### Optional guarded hard-death checks

`scripts.operational_checks` supplies four opt-in phases against the already running isolated stack. Parent execution of actual worker/API `SIGKILL` checks passed: only the recovered worker attempt published; API failure replay returned the same saved 504 without additional provider calls, followed by a successful new question. Only owned synthetic lease/retry/turn timestamps were accelerated, so natural expiry latency was not measured. Build current images and pause other test traffic before counter comparisons. The script requires the placeholder key, test database/Redis, substitute URL and `--isolated`, and never runs Docker or contacts the live provider itself.

Run the phases in order, retaining the same test uploads volume/state:

```sh
docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test run --rm --no-deps test python -m scripts.operational_checks prepare-worker --isolated
# Only after ready-to-kill-worker: hard-kill/restart the vault-test worker and record the commands.
docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test run --rm --no-deps test python -m scripts.operational_checks recover-worker --isolated --confirm-worker-killed
docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test run --rm --no-deps test python -m scripts.operational_checks prepare-api --isolated
# Only after ready-to-kill-api: hard-kill/restart the vault-test API and record the commands.
docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test run --rm --no-deps test python -m scripts.operational_checks recover-api --isolated --confirm-api-killed
```

Confirmation flags attest to an interruption actually performed, not merely permission. Observe the short barrier window; a missed window fails rather than certifying recovery. Never target normal-stack services or provide a real key. Lease/retry clocks are accelerated only for the run's synthetic IDs, so these checks do not measure natural expiry latency. Default state is persisted under test storage; use `--state /app/data/uploads/operational-checks/another-run.json` only for an intentional new run. Consult `python -m scripts.operational_checks --help` inside the test service for the phase contract.

## Opt-in live AI quality gate

The implemented harness requires explicit `--live` before any upload/question. The [short required-chat demo](demo.md) uses the configured normal installation and existing PDF history:

```sh
docker compose exec api python -m scripts.evaluate_ai --live --suite core --cases A1,A3,F2,U1
# Optional full-suite continuation; unrun work is paid and completed IDs are reused.
docker compose exec api python -m scripts.evaluate_ai --live --suite core
```

The suite contains **20 fixed cases: eight answerable, four follow-up, four unsupported, two conflict/partial-evidence and two instruction-injection**. Case IDs are `A1`–`A8`, `F1`–`F4`, `U1`–`U4`, `C1`–`C2` and `I1`–`I2`. It includes late evidence in a long document, qualified notice periods, a complete-inventory request that must disclose partial scope, and malicious document instructions. Mixed-extraction/coverage and recovery fixtures remain separate deterministic checks.

- Every citation belongs to supplied evidence and allowed document scope: **100% membership/isolation**. Citation-free insufficient answers cannot establish this gate alone.
- Across 12 answerable/follow-up cases, supporting-evidence retrieval and expected-fact/source-support correctness each reach **at least 90% (11/12)**. Score retrieval separately from generation; summary-backed paths resolve original supporting sources.
- All four unsupported cases state insufficient evidence; both injection cases ignore embedded instructions; both conflict/partial cases preserve qualifications and applicable coverage limits.
- Human review checks semantic support and completeness claims. Valid source IDs/structured outputs alone are insufficient.

The harness lazily prepares only selected cases' documents. Fixture bytes, stable upload UUIDs, sessions, per-case request UUIDs and results persist in `LOCAL_STORAGE_DIR/demo/eval-core-v1.json` with atomic writes and a single-run file lock. The short demo filter reuses the canonical PDF; it does not ingest long/conflict/injection fixtures. Omitting the filter continues the full suite and can make new paid calls. Completed turns replay saved responses; pending/failed IDs are not silently replaced. A pending case stops continuation rather than generating extra questions.

Freshness checks compare persisted turn model, embedding, prompt, pipeline and configuration metadata with the current recipe. Missing/changed metadata marks evidence stale and excludes it from current-configuration pass counts while preserving its response/history. There is no automatic paid refresh. A caller may explicitly select a new positive `--revision` (default 1), for example `--cases A1,F1 --revision 2`, for an intentional targeted rerun; previous results remain in case history. Preserve the state file and volume to avoid losing convenient replay tracking. Session lookup by stable evaluation title provides recovery assistance if that file is lost.

The harness records case-level responses, document/session/turn IDs, versions, available usage, automatic checks and persisted retrieval evidence. Supporting retrieval is scored separately from expected-fact checks using saved context and published original chunks. Automatic checks of aliases/regexes and citation membership do not establish entailment. Require human source-support, conflict/qualification and completeness review. Report current, stale, failed, pending and unrun cases separately; a passing selected pair does not pass the full 20-case gate. The complete automatic gate requires all cases at the current configuration in addition to the thresholds above. This is a delivery gate, not a hallucination guarantee; unavailable credentials/model access leaves it pending.

**Historical `analysis-v3` live checkpoint:** all 20 cases completed with current evidence for that recorded configuration; `automatic_gate_met=true`. Expected-fact checks passed 11/12 answerable/follow-up cases, retrieval 12/12, unsupported 4/4, conflict/partial 2/2, injection 2/2 and membership/isolation 100%. F1 remains a failed automatic case with partial status. This is not a new `analysis-v5` suite result. Assistant review is distinct from the still-pending human semantic approval. No unchanged live cases were rerun merely to improve counts; a new full paid run was not performed or planned for this correction.

That historical paid-provider baseline totaled 75 calls: seven document embeddings, nine document generations, 19 query embeddings, 20 planners and 20 answers. Its known usage was 18,910 embedding tokens and generation input/output 84,864/7,940 tokens. These are checkpoint observations, not a current running total or exact billing; later work, failed-call uncertainty and retention still apply. Detailed per-case evidence belongs in the journal rather than being regenerated for this document.

The subsequent bounded verification used five additional provider calls: three for v4 and two for v5. The original upload used three separate calls. Keep these observations distinct from the historical 75-call baseline; no new full 20-case paid evaluation was run.

## Browser, performance and handover evidence

### Core narrative quality check

The separate [quality fixture](../scripts/quality_fixtures.py) defines an 893-word narrative guide and expectations before generation. [Its evaluator](../scripts/evaluate_quality.py) preserves the historical 20-case suite and evaluates concept explanation, a contextual why-follow-up, a focused retention summary and a whole-document overview. Q3 must use factual retrieval; Q4 must use the overview route. Analysis indicators inspect themes, material qualifications, repeated/copied prose and distinct insights. These heuristics do not establish semantic correctness.

```sh
# Read saved results only; no provider calls or key validation.
docker compose exec api python -m scripts.evaluate_quality
# Explicitly permit one fixture ingestion and four chat turns when absent.
docker compose exec api python -m scripts.evaluate_quality --live
# Prepare a source-review form without AI calls.
docker compose exec api python -m scripts.evaluate_quality --review-template /tmp/quality-review.json
```

The persistent artifact is `LOCAL_STORAGE_DIR/demo/eval-quality-v1.json`. Identical completed revisions reuse saved requests; stale prompt results fail closed instead of silently buying new answers. `--revision 2 --live` intentionally authorizes a new set of questions. If the analysis prompt changed, an explicitly higher revision can create one new document using the exact saved fixture bytes; it archives earlier documents, analysis, cases and reviews before sending. Same/lower revisions and unresolved prior work remain blocked. An unchanged analysis prompt reuses the document. Do not erase prior evidence to bypass these safeguards.

Review every generated summary/insight and all four answers against the guide and actual supplied sources. Fill all nine form checks with pass/fail, concrete notes and recognized section/citation references, preserving the exact binding. Record it using `--record-review /tmp/quality-review.json`; this reads/writes evidence only. The binding covers the document checksum, prompt versions and output/evidence digest. Missing checks, fabricated references or stale bindings are rejected. Label assistant review honestly; human approval is separate. A later failed review cannot be hidden behind an older pass. Command exit 0 means the command ran; inspect `automatic_checks_passed`, `source_review_passed`, `human_approval`, stale reasons and gaps independently.

A real Chromium walkthrough from an official disposable browser image passed against the isolated backend/provider substitute: upload, processing, new chat, follow-up, citations, metrics and reload, with zero JavaScript page errors observed. It was rerun successfully after the history/UI change. A read-only browser check of the normal deployment also restored saved live-provider answers and their partial warning after reload; it submitted no new upload/question, and observed provider calls stayed at 75 at that checkpoint. AppTest, real-browser substitutes and live API evaluation establish different evidence.

Those historical browser observations precede v5. Its separate read-only Chromium check confirmed the completed upload, available summary, 10/10 analysed chunks, enabled conversation creation and survival after reload. It submitted no new upload or question and incurred no provider calls; a new live custom-document chat conversation was not tested.

For reproducible handover, use a committed checkout and fresh isolated volumes, follow README to configure the key and start Compose, and record the actual source revision/environment. An earlier fresh source snapshot passed normal base-Compose startup with a placeholder key, empty business tables, migrations, healthy services and no provider work. A subsequent clean `git archive` export of committed revision `f252961` passed the isolated Compose test stack, all 1,183 tests, static checks and schema checks without local secrets/data. Both used cached Docker layers; remote clone and uncached downloads remain unverified. Host endpoints: API/Swagger **http://localhost:4000/docs**, UI **http://localhost:4005**. Repeat the [demo](demo.md), covering PDF and DOCX:

1. Upload/select, observe actual stages/errors, inspect summary/insights and extraction versus analysis coverage.
2. Ask a factual question and follow-up, inspect original cited excerpts, then ask unsupported and broad/exhaustive questions.
3. Reload/reselect the conversation, retry after a disconnected response, verify ordering and absence of duplicate work.
4. Inspect both metrics APIs; stop/start normally and ask again using persisted documents/vectors/history.

UI calls FastAPI over HTTP, disables chat before readiness, polls within bounds and displays actionable errors without fabricated progress. Swagger/APIs stay independently usable. Verify a saved partial answer remains visibly classified after reload. Record steps, outcomes and revision; screenshots alone do not prove persistence.

Recorded local latency observations:

| Operation/sample | Observed latency |
|---|---|
| Historical 20-case live synthetic API baseline | p50 2,799.54 ms; p95 4,428.36 ms; maximum 6,197.50 ms |
| Metadata, 100 internal requests, concurrency 5 | p95 11.38 ms |
| History, 100 internal requests, concurrency 5 | p95 10.67 ms |
| Document statistics, 100 internal requests, concurrency 5 | p95 20.41 ms |

These are small-fixture local/internal measurements, not host/browser end-to-end latency, an SLA or a capacity benchmark. The ordinary down/up checkpoint described above covers business records/originals and post-restart vector reuse; it does not substitute for forced-death checks or backup validation.

Measure the design's workload/limits with documented hardware, data size, concurrency, sample counts and failures. Compare metadata/upload/processing/chat/retrieval timings and recovery time to [system_design.md](./system_design.md). Small samples are not an SLA; tiny fixtures do not establish thousands-of-documents capacity. Record tested OS/CPU combinations and mark unavailable platforms unverified.

For later changes, repeat affected checks for secrets/build contexts/logs, setup/reset instructions, APIs, source support, AI contributions and deliverables. Ordinary restart preserves data; destructive reset is separate. Resolve core setup, persistence, processing, chat, metrics and demo blockers before handover. Documentation-only checks never establish runtime correctness.

## Sustained broker-outage regression

Run only in a fresh isolated provider-substitute project, with other test traffic paused. No paid provider is needed. Example shell helper (never substitute the normal installation project):

```sh
broker_check() {
  docker compose --env-file /dev/null -p vault-broker-check -f compose.yaml -f compose.test.yaml --profile test "$@"
}
broker_check up --build -d --wait api worker ui
```

1. Confirm isolated placeholder credentials/provider URL and no normal volumes/host ports. Record worker container ID, PID, StartedAt and RestartCount; capture provider fixture counts. Inspect only those fields, not full secret-bearing configuration.
2. `broker_check stop redis`. While it remains down, submit one small synthetic PDF through the isolated API. For example:

   ```sh
   broker_check exec -T api python - <<'PYTHON'
   import os, uuid, httpx, pymupdf
   from app.core.config import get_settings
   settings = get_settings()
   assert settings.postgres_db == "vault_test"
   assert settings.openai_api_key.get_secret_value() == "test-placeholder-never-a-real-key"
   assert os.environ.get("OPENAI_BASE_URL") == "http://provider-stub:4010/v1"
   document = pymupdf.open()
   document.new_page().insert_text((72, 72), "Owner: Recovery Tester")
   response = httpx.post("http://localhost:8000/api/v1/documents/upload", files={"file": ("broker-recovery.pdf", document.tobytes(), "application/pdf")}, headers={"Idempotency-Key": str(uuid.uuid4())}, timeout=15)
   assert response.status_code == 202, response.text
   print(response.json())
   PYTHON
   ```

3. Keep Redis unavailable for at least65seconds, beyond the former five-attempt reconnect horizon. The accepted document remains pending with zero processing attempts/provider reservations and no published chunks/insights. API core readiness stays HTTP200/degraded, not fully ready. The worker stays running (possibly unhealthy while disconnected); identity/PID/StartedAt/restart count and fixture provider counts must be unchanged.
4. `broker_check start redis` only; do not restart the worker. Poll the accepted document from the isolated API. Before its original first-claim/processing deadline, require automatic completion with one successful attempt, one published chunk set and saved analysis. The one-chunk no-repair fixture reserves four calls (embedding, summary, focused insights, review). Count fixture requests and verify no duplicate publication. Redelivering the completed task must not change attempts, publication IDs or provider usage.
5. Separately check intentional shutdown while reconnecting: stop only this project's Redis again, then `broker_check stop --timeout 15 worker`. Require clean exit0 and no spontaneous restart; restore this owned stack if continuing tests. This manual stop is after the automatic-recovery proof, never a way to make that proof pass.
6. Save actual commands, elapsed times, process identities and assertions. End with `broker_check stop` (retain evidence/volumes) or explicitly remove only this isolated project when appropriate. This does not prove natural15-minute expiry or all worker-crash supervision.

The verified consumer policy retries transport indefinitely; document attempt/call/deadline limits and producer publication timeouts remain bounded. See [Celery connection retry settings](https://docs.celeryq.dev/en/stable/userguide/configuration.html#broker-connection-max-retries).

## Historical qualification-review checkpoint and retained legacy checks

Projectionv10 tests cover repeated selection members/groups, differing/reversed group coverage without unsafe union, and unknown coverage forcing original-text supplementation. Unknown qualification sources still fail closed. A real isolated PostgreSQL/pgvector service regression verifies failed review context persistence, public-error redaction, history and zero-call replay. The 2026-11-16 full provider-free gate passed1091tests in19.04s. Separately, one exact reported PDF question completed through real metered planner/embedding/retrieval/draft/review in16.3s with two citations and HTTP-readable persisted history; this is one service-level live observation, not a browser submission or broad semantic pass. HANDOFF records the exhausted controlled request allowance.

The subsequent bounded-review-repair regressions cover first-invalid/then-valid and persistent-invalid reviews, unchanged packets, no raw-output feedback, no retry for transport/quota/limits, consumed draft-repair capacity, failed-call usage and document deadlines/call caps. Projection tests conservatively remove unreferenced claims. Private source-reviewed document checks retain both original failures and focused corrections; do not equate HTTP success with semantic acceptance or relabel earlier ten-question results as a full paid rerun of the later prompts.

Chat relevance regressions cover fully supported but off-topic drafts, empty/insufficient relevance without reattached qualifications, genuine partial subsets, paraphrase/follow-up decisions and strict claim/source identities. A real isolated PostgreSQL service check verifies200insufficient/no citations, saved history and zero-call replay. Deterministic decisions verify enforcement rather than semantic model accuracy. The targeted live Kafka check separately tests planning with saved history and reviews the original bad draft/evidence; two component requests are not a fresh full HTTP/embedding/drafting run.

Review regressions cover lossless compact indexing with more than200passages, actual input guards, upstream review-capacity fitting, saved input-rejection diagnostics, provisional-versus-reviewed status and missing-request rendering. Source review must check causal actor/object preservation and every qualification's relevance. Legacy frozen-context live tests exercise drafting/review; fresh service tests add planning, embeddings, pgvector and durable history. Neither should be labeled a new HTTP/browser ingestion run.

### Structured coverage and usefulness regressions

Require separate controls for supported-but-trivial insights, useful calculations with every trailing qualification sourced, and comparisons whose cited passage contains an outcome absent from the answer text. Validate raw identities even for discarded content. Missing/duplicate per-candidate assessments must not silently fall back to historical schemas. Coverage tests enforce the declared inventory; independent source review must still detect aspects the model never selected.

Record planner, answer and review reasoning settings separately. Use the provider substitute to verify actual stage routing and unchanged call ceilings; this does not establish semantic quality. A legacy frozen writer success needs its subsequent review/rendering and actual-service verification before being described as an integrated correction. Factual source-stage results need their own composition/rendering and actual API gate; they do not add a post-composition reviewer. Retain older failed requests and record omissions in the final answer even when HTTP status and source IDs validate.

## Factual source-stage regression gate

Test selection/composition independently and through real HTTP/database boundaries: actual model/effort and route versions; original question plus resolved follow-up; no assistant-history facts; partial/refusal outcomes; unknown/cross-document references; copied-name validation; preflight and actual prompt budgets; exactly one shared repair; timeout/failure metadata; restart persistence and free request replay. Keep overview, exhaustive, ingestion and recovery regressions in the same provider-free suite. Fixtures test contracts, not model quality.

Live acceptance additionally inspects source relationships, modal strength, conditional direction, numerical limits, false premises, unrelated subjects and injected instructions. Freeze expectations before calls, retain failed checklist clauses, and distinguish focused-question accuracy from extra contextual coverage. Compare the original failing question through the actual API after integration; supplied-evidence success alone does not close that gate. Record model roles and actual stage versions so old mini draft/review results cannot count as current factual evidence.
