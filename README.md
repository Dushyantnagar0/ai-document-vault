# AI Document Vault

A self-hosted service for document summaries, insights, and natural-language conversations with source citations. Each installation keeps its own data and uses its owner's OpenAI API key.

**Status: core backend and Streamlit client implemented.** Upload a document, inspect its summary/insights, and ask follow-up questions with source citations. Analysis and chat use evidence review; model judgments remain fallible. See [current evaluation](./docs/evaluation_current.md) for results and limitations.

## Review guide

Start with local setup below, then follow the [document-chat walkthrough](./docs/demo.md). For design, read the [architecture overview](./docs/architecture.md) and [detailed HLD/LLD](./docs/system_design.md). [AI_USAGE.md](./AI_USAGE.md) explains the coding workflow and prompt decisions; [current evaluation](./docs/evaluation_current.md) distinguishes automated tests from live-AI evidence.

## Run locally

Install and launch Docker with Compose. The container workflow uses Python 3.12; no host Python, PostgreSQL, Redis or manual database setup is needed. Use Docker Engine 27+ and Compose 2.24.4+; the latter supports the test override configuration. Image downloads/builds need network access. New uploads automatically trigger chargeable OpenAI embedding and analysis calls, sending extracted text to the provider. This is local hosting with external AI, not offline processing.

Verified environment: **macOS Apple Silicon with Linux arm64 containers**, Docker Engine **29.9.2**, Compose **5.6.1**, Python **3.12.16** in the image and approximately **8 GB** assigned to Docker. Other OS/CPU combinations remain unverified. Planning targets include Linux, macOS Intel and Windows/WSL2 with Docker. Suggested resources are 4 CPU cores, 8 GB RAM and 20 GiB free disk; larger collections need more persistent storage.

From the repository root, copy the example only if you do not already have `.env`; otherwise skip that first command and preserve your configuration:

```sh
cp .env.example .env
# Edit .env: set OPENAI_API_KEY to your own key.
docker compose up --build -d --wait
curl http://localhost:4000/health/live
curl http://localhost:4000/health/ready
```

Copy the example only when creating a new configuration; preserve an existing `.env`. Keep your key in that ignored local file, never in committed examples or support output. Defaults supply local database credentials and model/budget settings. Health checks do not call OpenAI or verify account/model access. A placeholder key permits infrastructure inspection, but real document processing needs a working key and quota. After changing container configuration, run `docker compose up --build -d --wait` again.

Startup builds the image, initializes shared-volume permissions, applies migrations and waits for healthy API/worker services. Open [Swagger](http://localhost:4000/docs) to inspect and exercise the current APIs. Readiness reports core database/schema/storage status and broker/worker degradation; saved reads remain available during broker outages when core dependencies are healthy. Startup waiting also checks the worker.

The API and UI are published only on localhost ports 4000 and 4005; PostgreSQL and Redis stay internal. The UI calls the API over HTTP and receives no OpenAI key or database credentials. This is a trusted personal installation without authentication. Shared accounts and public hosting are outside scope. Optional settings and compatibility constraints are in [.env.example](./.env.example) and [system design](./docs/system_design.md); changing an embedding model/dimension is not a substitute for a compatible profile/migration.

## Upload, inspect and retrieve a document

Use a PDF with selectable native text or a DOCX containing body paragraphs/tables. The defaults accept at most 20 MiB per file, 500 PDF pages, 100 MiB expanded DOCX content/2,000 ZIP entries, and 100,000 extracted tokens/2 MiB text/500 chunks. Excessive inputs fail visibly rather than being silently truncated. OCR and image interpretation are not supported yet. The replaceable extraction boundary provides an extension point for future OCR/vision adapters; these would still need source metadata, coverage reporting, configured dependencies/budgets and validation. Textless pages may be blank or visual. English fixtures have been evaluated; other Unicode text is best effort.

The upload below starts automatic paid analysis. Replace both placeholders; choose a fresh UUID for each intentional new upload and reuse it for retries of the identical file, filename and media type. Without the optional idempotency key, another upload creates another document and can incur new charges.

```sh
curl -H 'Idempotency-Key: REPLACE_WITH_UPLOAD_REQUEST_UUID' \
  -F 'file=@sample.pdf;type=application/pdf' \
  http://localhost:4000/api/v1/documents/upload
```

A 202 response returns `document_id`, `status`, `stage` and `status_url`. It means the original and processing intent are saved; analysis may still be pending. Normal acceptance immediately attempts queue publication, with periodic recovery for missed dispatches and interrupted work. Reusing an upload UUID with different content or metadata returns 409.

Copy the returned document UUID, then inspect its progress:

```sh
DOCUMENT_ID=REPLACE_WITH_DOCUMENT_UUID
curl "http://localhost:4000/api/v1/documents/$DOCUMENT_ID"
curl 'http://localhost:4000/api/v1/documents?limit=20'
```

Status progresses through `PENDING`/`PROCESSING` to `COMPLETED` or `FAILED`. Inspect the actual stage, extraction warnings and safe error details. Processing/retries are bounded; a terminal failure remains visible. Correct its cause before intentionally uploading again with a new request UUID.

Once completed:

```sh
curl "http://localhost:4000/api/v1/documents/$DOCUMENT_ID/insights"
curl -o downloaded-original.pdf \
  "http://localhost:4000/api/v1/documents/$DOCUMENT_ID/file"
```

The summary synthesizes the document's purpose, themes and material facts. Insights identify distinct supported relationships, conditions, tradeoffs or limitations, with exact supporting quotes available for verification. Their depth scales to substantive content within configured budgets; an overview is not an exhaustive report on every fact. Empty insights are valid when no distinct takeaway is supported. Requesting insights before completion returns 409. Extraction warnings describe original-file limitations; analysis coverage counts participating extracted chunks. Separate analysis-review warnings disclose narrowed supported scope. None certifies visual completeness or semantic correctness.

In the [UI](http://localhost:4005), choose or upload a file in **Your library**, wait for **Ready to chat**, then type a question. The first empty conversation opens automatically; use **New conversation** for an additional chat. Continue in that conversation for follow-ups. **Document overview** contains the summary and insights; each answer's **Source** expander shows original evidence. Saved conversations and diagnostics are expandable. Refreshing saved views does not call AI. There is no API question-count quota per conversation; one question may be active at a time. By default, up to four recent completed turns, within 2,000 tokens, inform the next question. Older messages stay persisted. The current UI bounds history loading at 5,000 messages (about 2,500 question/answer turns); larger histories need improved pagination or a new conversation.

For DOCX, use a `.docx` filename and media type `application/vnd.openxmlformats-officedocument.wordprocessingml.document`. Source locations use paragraph/table paths rather than invented Word page numbers.

## Small live ingestion check

These commands explicitly opt into paid processing of tiny generated fixtures using the key already configured locally. They keep stable fixture bytes and upload UUIDs in persistent storage: rerunning an unchanged completed fixture replays its saved document without re-embedding or re-analysis. A new fixture version is a new upload and can incur charges.

```sh
docker compose exec api python -m scripts.live_ingestion --live --format pdf
docker compose exec api python -m scripts.live_ingestion --live --format docx --fixture-version 2
```

The explicit DOCX version selects the corrected evaluation fixture. Script output contains status, summary/insights, coverage and available usage. It never needs the key pasted into a command argument. Saved documents are immutable; updating prompts does not automatically regenerate earlier analyses. New small uploads normally use one embedding request plus summary, insight and evidence-review requests; larger documents and a bounded repair can use more. Ordinary factual chat normally uses four requests (planner, query embedding, answer and review), with at most five sharing one draft repair or one review repair. Empty insufficient answers and literal partial inventories skip review. Configurable review completion limits include reasoning tokens; visible insight content has a separate limit.

## Verification and limits

The required [document-chat demo](./docs/demo.md) gives a step-by-step UI walkthrough for a fresh installation. Saved demo artifacts/history are local data, not a bundled video or data shipped with a clone.

Core content quality has a separate narrative check. Its source and expected ideas are authored before generation; it assesses summaries/insights and concept, contextual why, focused-summary and overview questions. The default command only reads saved results. `--live` explicitly permits one fixture ingestion and four new chat turns when absent:

```sh
docker compose exec api python -m scripts.evaluate_quality
docker compose exec api python -m scripts.evaluate_quality --live
```

Inspect automatic checks and source-review results separately. The [testing strategy](./docs/testing_strategy.md#core-narrative-quality-check) explains bound review forms, replay, stale versions and costs. Assistant review is not human approval; a valid source ID alone does not prove a useful or correct answer.

Provider-free checks cover source/citation validation, budgets, replay, history and real PostgreSQL/pgvector/Redis/Celery/file boundaries. Ruff, formatting, mypy and schema-drift checks run through the isolated image. A provider substitute establishes orchestration, not live AI quality. [Current evaluation](./docs/evaluation_current.md) records versions, counts, source-review results and limitations; the [journal](./AI_DEV_JOURNAL.md) preserves historical AI-assisted iterations.

A clean export of committed revision `f252961` passed Docker build/startup, all 1,183 tests, lint, formatting, typing and migration-drift checks with fresh isolated volumes and a local provider substitute. It contained no real environment file or local data and used the existing Docker cache. This verifies the exported committed source, not remote cloning, uncached downloads or live-provider quality. Historical API/browser/restart results and current prompt checks have separate scopes in [current evaluation](./docs/evaluation_current.md). Remote CI, other platforms and thousands-of-documents capacity remain unverified; manual semantic review is still required.

Routine tests use a separate Compose project, isolated database/volumes, a placeholder key and a local provider substitute. They do not need a real OpenAI key. Run from the repository root:

```sh
docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test build
docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test run --rm --no-deps test ruff check .
docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test run --rm --no-deps test ruff format --check .
docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test run --rm --no-deps test mypy app ui scripts
docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test run --rm --no-deps test pytest -m "unit or api"
docker compose -p vault-test -f compose.yaml -f compose.test.yaml --profile test run --rm test pytest
```

`--no-deps` keeps static/unit/API checks independent of running infrastructure. The full pytest run starts/waits for its actual test services; retain those dependencies. Always keep `-p vault-test` and both Compose files together. The test override removes host ports and isolates credentials/settings from the personal installation. See [testing strategy](./docs/testing_strategy.md) for gates and explicit live evaluation procedures.

## Storage and lifecycle

Named volumes retain PostgreSQL data, Redis state and application files. Completed document retrieval/insights reuse persisted data; ordinary restart does not request document embeddings or analysis again. New chat questions require provider access for planning, query embedding, answering and applicable evidence review; replaying an already completed request uses its saved response. Sessions, ordered messages/context/citations, attempts and metrics are stored in PostgreSQL. Hourly bounded maintenance targets seven-day raw-performance retention; document/conversation/attempt history remains until reset. The recorded down/up check included conversation data; it is separate from forced-death recovery and final release verification.

```sh
docker compose stop             # Stop services; retain containers and data.
docker compose up -d --wait     # Restart using the same persistent data.
docker compose down            # Remove containers/network; retain named volumes.
```

**Destructive reset:** `docker compose down -v` deletes the installation's named volumes and their data. It is not a backup. To remove only isolated test resources/data, use `docker compose -p vault-test -f compose.yaml -f compose.test.yaml down -v`. Retaining volumes through restart also does not provide a backup.

## Chat and monitoring

Create a conversation for a completed document. Each session has one immutable document; the same document can have several sessions. Copy the returned session `id` into `SESSION_ID`:

```sh
curl -H 'Content-Type: application/json' \
  -d '{"document_id":"REPLACE_WITH_COMPLETED_DOCUMENT_UUID","title":"Renewal questions"}' \
  http://localhost:4000/api/v1/chat/sessions
SESSION_ID=REPLACE_WITH_SESSION_UUID
curl -H 'Content-Type: application/json' \
  -d '{"question":"What does this document say about renewal?","client_request_id":"REPLACE_WITH_NEW_REQUEST_UUID"}' \
  "http://localhost:4000/api/v1/chat/sessions/$SESSION_ID/messages"
curl "http://localhost:4000/api/v1/chat/sessions/$SESSION_ID/messages?limit=50&cursor=0"
curl "http://localhost:4000/api/v1/chat/sessions?document_id=$DOCUMENT_ID&limit=20"
```

A supported answer must address the current question. `insufficient` returns an explicit abstention with no citations; unrelated background is not a partial answer. A genuine `partial` answer includes supported content and separately names the requested information the supplied evidence cannot establish.

Answers return as complete HTTP JSON, with `answer`, `evidence_status` (`answered`, `partial` or `insufficient`), citations, extraction coverage/warnings and model/prompt versions. There is no SSE or token streaming. Ask follow-ups in the same session using a new request UUID. History returns ordered messages and `next_cursor`; pass that cursor to continue. Assistant history retains nullable `evidence_status`/`evidence_scope`; the UI visibly warns on completed partial answers, including after reload, using the saved classification rather than guessing from text or citations.

Reuse the same `client_request_id` and question after a transport failure: completed turns replay their saved response without more AI calls. Reusing it for a different question conflicts. An active turn returns 409; wait and retry the same request, or reload history. Failed requests remain saved failures; a deliberate new attempt requires a new UUID. Only one question runs per session at a time. Inspect insufficient/partial outcomes and original supporting excerpts rather than treating citations as proof of semantic correctness.

```sh
curl http://localhost:4000/api/v1/metrics/documents
curl 'http://localhost:4000/api/v1/metrics/processing?limit=20'
```

Document statistics describe the current installation. Processing metrics default to the last 24 hours; optional timezone-aware `start`/`end` select a window of at most seven days, with an inclusive start and exclusive end. Responses distinguish document outcomes from processing attempts and include paginated attempt history, operation latency/usage and retention information. Durations use milliseconds; small samples are flagged and empty percentiles are null. Performance telemetry is best effort, not a complete billing ledger. Swagger documents the full response/query schemas.

The UI provides document selection/upload, progress, insights, conversations/history, source excerpts and metrics through these APIs. Its component checks, real-browser substitute-backed walkthrough and separate live API checks establish different evidence. Streaming, multi-document sessions and additional document formats are deferred.

## Opt-in live chat evaluation

For a fresh full API evaluation, use a named isolated run (`--run-id`) so documents, processing and conversation chains have independent identities. The [evaluation report](./docs/evaluation_current.md) distinguishes historical full API runs, current frozen-evidence generation checks, saved-chunk analysis and real-service conversation checks. Replaying saved evidence saves provider work but does not establish a new upload-to-chat pass.

These commands make paid requests for unrun cases; existing completed IDs replay saved responses. Start with the [small demo](./docs/demo.md), inspect its results, then optionally continue the full 20-case suite:

```sh
docker compose exec api python -m scripts.evaluate_ai --live --suite core --run-id reviewer-check --cases A1,A3,F2,U1
docker compose exec api python -m scripts.evaluate_ai --live --suite core --run-id reviewer-check
```

The filter prepares only required documents. These named-run commands preserve stable fixture bytes, upload/request UUIDs, sessions and results under `LOCAL_STORAGE_DIR/demo/eval-core-v1-reviewer-check.json`; completed cases replay without new calls. The legacy unnamed artifact is `eval-core-v1.json`. Full-suite continuation can upload additional fixtures and ask the remaining questions. Preserve this state and its persistent volume. A single-run lock prevents concurrent harness runs.

A different `--run-id YOUR_RUN_NAME` uses `eval-core-v1-YOUR_RUN_NAME.json`, fresh document/request identities and conversations shared only by intentional follow-up chains. The recipe is frozen before HTTP work; a changed recipe requires a new name. Named runs use revision 1 and stop after failed/stale/unresolved cases for inspection. Reusing the same name with unchanged settings resumes saved work. New names can incur new ingestion and chat charges.

The harness compares stored model/prompt/pipeline/configuration versions with the current recipe. Stale results remain available but do not count toward current-configuration gates; it never silently pays to refresh them. Only an intentional targeted rerun should use a new positive `--revision`, for example adding `--cases A1,F1 --revision 2`; prior results remain in history. Failed/pending request IDs are not automatically replaced. Automated expected-fact/retrieval/membership checks require human review of semantic support, qualifications and completeness before declaring the live quality gate passed. See [testing strategy](./docs/testing_strategy.md).

## Troubleshooting

Start with saved metadata rather than repeating paid processing:

```sh
docker compose exec api python -m scripts.diagnose --document-id REPLACE_WITH_DOCUMENT_UUID
docker compose exec api python -m scripts.diagnose --filename 'example.pdf'
```

This read-only report includes durable attempt stages, safe errors, call reservations and known usage; exact-filename lookup reports up to five latest matches rather than silently choosing one. It never uploads, retries or invokes AI. Reports were verified against the isolated database and the original failed record in the normal installation. See the [debugging runbook](./docs/debugging.md) for metadata-first investigation, safe log correlation and why Celery task success can coexist with document failure.

| Symptom | What to check |
| :--- | :--- |
| Docker command unavailable | Launch Docker Desktop. On macOS, add `/Applications/Docker.app/Contents/Resources/bin` and `$HOME/.docker/bin` to the shell PATH if needed. |
| Build fails | Network/image access and the actual build error; dependencies install inside Docker, not host Python. |
| API/startup unhealthy | `docker compose ps` and service logs; inspect database/migration/storage and worker health. Confirm ports 4000/4005 are free and settings are valid. |
| Upload rejected | Check filename, PDF/DOCX media/signature, configured limits and the returned error code. |
| Processing failed or remains pending | Read document metadata/stage/error; verify worker/broker and provider access/quota. A 202 response does not mean analysis completed. |
| Provider key/model error | Supply your key only in local `.env`, retain supported model/profile settings and recreate affected services. Health checks do not verify provider access. |
| Insights return 409 | Inspect document status; wait for completion or diagnose its terminal error. |
| UI unavailable or chat fails | Check `docker compose ps`, UI/API logs and the completed document/session; inspect safe error codes and retry an interrupted question with its original UUID. |
| Embedding configuration rejected | Restore a compatible profile; dimension/model changes require migration/re-embedding work. |

## Architecture and project records

Selected stack: Python/FastAPI, PostgreSQL + pgvector, Celery + Redis, OpenAI chat/embeddings, PyMuPDF, python-docx and local files, with a minimal Streamlit HTTP client. FastAPI owns document APIs and durable acceptance. Workers extract, chunk, embed and analyse documents; PostgreSQL owns durable state and Redis transports tasks. Focused LangChain provider/splitting utilities sit behind application-owned orchestration, budgets and source validation.

[CASE_STUDY.md](./CASE_STUDY.md) governs requirements, including the three 33% evaluation categories and required backend/source/documentation/chat-demo deliverables. Recorded live API, browser, lifecycle and clean committed-source checks have distinct scopes in [current evaluation](./docs/evaluation_current.md). The required document-chat demonstration and manual semantic review remain separate handover activities.

- [Product scope](./PRODUCT.md): requirements, evaluation traceability, selected additions and deferred scope.
- [Delivery plan](./PLAN.md): milestones and dependencies.
- [System design](./docs/system_design.md): baseline contracts and HLD/LLD, updated with consequential changes.
- [Architecture](./docs/architecture.md): concise component and operating overview.
- [Testing strategy](./docs/testing_strategy.md): procedures, fixtures and release gates.
- [Debugging runbook](./docs/debugging.md): read-only diagnosis, safe log correlation and economical verification.
- [Document-chat demo](./docs/demo.md): fresh-install walkthrough, optional generated fixture and evidence limits.
- [AI workflow](./AI_USAGE.md) and [development journal](./AI_DEV_JOURNAL.md): guardrails and actual contributions/results.
- [Session handoff](./docs/HANDOFF.md): development continuity and next work; optional for reviewers.
