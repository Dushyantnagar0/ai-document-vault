# Coding Standards

**2026-11-15: implementation conventions for the ingestion, conversation, monitoring and UI slices.** Check HANDOFF for current verification and pending release gates.

[CASE_STUDY.md](../CASE_STUDY.md) governs requirements; [system_design.md](system_design.md) owns contracts, defaults and state machines. Follow [PRODUCT.md](../PRODUCT.md), [AI_USAGE.md](../AI_USAGE.md) and [testing_strategy.md](testing_strategy.md). Proposed behavior is not evidence of implementation.

## Structure and I/O

- Use thin typed FastAPI routers, services for orchestration and application-owned protocols at storage, extraction, AI, vector and dispatch boundaries. Keep SDK/SQL/parser types inside adapters.
- Implement the selected adapter per capability. Add formats via a small explicit extractor mapping returning blocks, locators, coverage and warnings; avoid a speculative plugin platform.
- Use Pydantic v2 request/response/settings models, SQLAlchemy 2 declarative mappings and Alembic. File upload is multipart, not a JSON model.
- Interactive database/provider I/O is async. Offload blocking file work; parsing, document chunking, embedding and analysis stay in Celery. FastAPI background ingestion is not the durable queue.
- Give each API request/worker operation its own session; heartbeat threads use independent sessions. Never hold a transaction or session row lock during provider calls.
- Use focused LangChain OpenAI and token-aware splitter utilities. Application services own retrieval, context, history, budgets and citation checks.

## Configuration and local runtime

- One `pyproject.toml`/`uv.lock` workflow owns dependencies, including tests/UI. Use locked container installation; no independently maintained requirements lists. Candidate versions require resolver/build/import verification.
- Compose provides the app and infrastructure; no host Python/database installation or manual schema setup in the default flow. Migrations and upload-volume permissions initialize before non-root API/worker processes.
- Explicitly pass environment values to the services that need them. Compose interpolation alone is not runtime injection. UI gets HTTP URLs only, never database/provider credentials.
- Central settings construct DB URLs and validate positive bounds, token relationships and model/profile/schema compatibility. Reject unsupported adapters; do not silently fall back.
- Keep real environment files/data/caches out of Git and build contexts. Redact credentials and validation inputs; no secrets in images, build arguments, prompts or logs. Hosted tracing defaults off.
- Respect separate internal versus host ports; localhost exposure defaults to API4000/UI4005. Worker health must establish registration/queue readiness; diagnostics should distinguish unavailable ingestion from accessible saved data.

## APIs and persistent state

- Use versioned routes, declared schemas, UUIDs, UTC times, stable cursor pagination and the design's error envelope. Return sanitized errors with stable codes/request IDs.
- 202 upload means original plus metadata and durable processing intent are committed; broker publication and analysis may still be pending. Polling GETs never re-enqueue work.
- Store upload UUID/fingerprint under a unique constraint. Resolve concurrent insert conflicts using the winning record; compare fingerprints and clean up losing staged files. Do not implement check-then-insert idempotency.
- Resolve existing chat UUIDs before checking session busy state. Reserve ordered user/assistant messages in a short transaction; guard finalization and active-turn release by the same claim token.
- Enforce FKs, uniqueness and nonnegative bounds in the database. Queries are bound, never interpolated. Membership restricts retrieval/citations; v1 session creation inserts exactly one document under lock.
- Persist failed turns and processing attempts. Retain originals, compatible vectors, insights and history across normal restart. Do not make broker result state the durable source of truth.
- Publish chunks/insights/completed state atomically after validation. Unpublished or obsolete attempt outputs never become chat evidence. No blind cascades that erase history/citations.

## Processing, AI and grounding

- Stream/limit uploads, check content/container structure, generate opaque keys and preserve source metadata. Original filenames are display values, never paths.
- Differentiate expected recoverable extraction gaps, zero text, corrupt files and infrastructure/provider failures. Never silently skip failed embedding/analysis chunks.
- Extraction coverage describes observed original-content limitations; analysis coverage counts accepted extracted chunks. Surface both in APIs/UI. Do not infer missing words merely from a blank page or claim complete visual understanding.
- Tokenize actual chunks with the appropriate model encoding, prefer structural boundaries, preserve Unicode/offsets and verify emitted size. Bound embedding batches by items and total tokens.
- Use bounded section summaries and recursive reduction, maintaining original source lineage. Budget complete prompts, including metadata/instructions/output reservation.
- One application-owned retry policy; disable SDK/Celery retry multiplication. Enforce attempt, call and overall time budgets. Document claims/leases/heartbeats use conditional writes; lost owners cannot publish.
- Keep consumer transport reconnection separate from bounded document attempts/deadlines/provider reservations. Broker recovery must not require manual consumer restart; deliberate shutdown must still work.
- Reject vectors unusable after float32 conversion before publication; validate every chunk rather than silently dropping invalid data.
- Retry only transient failures; distinguish rate limiting from exhausted quota. Record safe errors and interruptions. External calls may repeat after crashes despite duplicate-result prevention.
- Propagate Celery soft-time-limit interruptions through parser/provider/telemetry adapters to the worker's bounded retry policy. Classify transient database failures separately from integrity/programming errors; if failure persistence is unavailable, retain the claim for lease recovery rather than inventing a terminal outcome.
- Treat document instructions as untrusted. History resolves follow-ups; only supplied evidence supports facts. Preserve the original question and planned query/intent.
- Persist selected audit bindings only; independently bound original prompt snapshots and review metadata. Never feed audit metadata back as chat history or source evidence.
- Validate structured output and source membership with one bounded repair. Source IDs alone do not prove claim support; live semantic evaluation is required.
- Ordinary overviews use validated saved analysis plus original sources. Exhaustive requests must disclose incomplete evidence scope. Unsupported questions return an explicit successful insufficient-evidence response.
- UI reruns preserve submission IDs; reload recovers backend history. Bound polling, stop at terminal states and render untrusted text without executable HTML.

## Verification and evidence

Use Ruff format/check, mypy and pytest/HTTPX through the documented isolated test image. Write behavioral tests for boundaries/failure modes, not mirrors of implementation. Tests must not access live OpenAI by default.

Real PostgreSQL/pgvector establishes vector behavior; SQLite does not. Real Celery/Redis establishes delivery; eager mode does not. Isolate queues/databases/volumes because worker commits survive a client rollback. Check actual clean startup, worker readiness, concurrent replay, crash recovery, partial extraction and restart/new-question persistence.

Record revision/configuration, commands, outcomes and unavailable checks. Re-run affected integrations after relevant changes; a subagent's report is not system verification. Keep consequential contract/design changes synchronized. Record actual AI work, corrections and limitations without invented gains or guarantees.
