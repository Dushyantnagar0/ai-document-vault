# Architecture Overview

**Implemented architecture, reconciled 2026-11-17.** [Current evaluation](evaluation_current.md) records the verified source/prompt versions, operational checks and remaining limitations. [HANDOFF.md](HANDOFF.md) supports development continuity.

[CASE_STUDY.md](../CASE_STUDY.md) governs requirements. [system_design.md](system_design.md) owns detailed HLD/LLD, diagrams, APIs, schemas, settings, recovery and capacity assumptions. [PRODUCT.md](../PRODUCT.md) maps scope; [PLAN.md](../PLAN.md) orders delivery; [testing_strategy.md](testing_strategy.md) defines evidence. This overview does not duplicate those contracts.

## Components and flows

| Component | Responsibility |
|---|---|
| FastAPI | Typed upload/retrieval/insights, interactive document chat, history and monitoring APIs; Swagger |
| Celery worker | PDF/DOCX extraction, source-preserving chunks, batched embeddings and whole-extracted-text analysis |
| PostgreSQL + pgvector | Metadata/status, compatible vectors, insights, conversations/evidence, attempts and performance history |
| Redis | Task transport; database state remains durable authority |
| Local file storage | Opaque original-file keys on a persistent volume shared by API/worker |
| OpenAI | External embeddings and generation, through focused LangChain integrations |
| Streamlit | Thin HTTP client for upload/status/insights/chat/citations and small metrics view |

Upload commits the original, metadata and processing intent before returning 202. The upload service immediately attempts bounded Celery publication after commit, then returns the acknowledgement. A separate 15-second API recovery scan repairs missed dispatch, due retries and expired claims; successful uploads do not wait for that scan. Workers claim leases, process outside long transactions and atomically publish validated results. Duplicate delivery is safe; stale workers cannot publish. Partial extraction is visible, while analysis covers all accepted extracted text. Failure never silently drops embedding/analysis chunks.

A separate startup/hourly maintenance task expires seven-day raw telemetry in small committed batches within a bounded sweep and reports remaining backlog and aged unreferenced files. It uses database checks and cooperative inode locks, including upload locks held through commit; it is not part of the 15-second work-recovery cadence. Exact limits and response contracts are maintained in the detailed design.

Analysis processes all accepted extracted chunks, freezes its final summary, generates focused additional insights, then audits claim support and repetition against the bounded original passages supplied to final synthesis. Deterministic projection preserves accepted prose and literal source references. Generation permits eight summary blocks; final analysis permits sixteen to accommodate corrected-citation splits and literal qualifications within unchanged per-block limits. Chat still accepts eight saved summary blocks for overview context and discloses partial scope when it omits more; its final answer cap remains eight. Review scope/warnings are separate from chunk participation and extraction capability.

Chat reserves one ordered turn per session, resolves follow-ups, retrieves only member-document evidence, validates its draft, and audits supported synthesized prose before publication. Normal factual turns use four provider calls; overview uses three. A five-call ceiling permits one draft repair or one review repair, sharing the same allowance. Valid insufficient answers and incomplete exhaustive literal-excerpt replies skip review. Model review can still miss support or qualifications; selected qualifications receive literal supplements unless retained covering claims also select their supporting passages. This checks binding consistency, not semantic coverage; source review remains necessary. Saved source-backed analysis serves ordinary overviews; top-k answers cannot claim exhaustive coverage without checking the necessary content. Citations resolve original excerpts and an array of actual PDF/DOCX locators. Follow-ups are allowed after each answer/failure. History is persistent, including failed turns; numeric embeddings select text rather than serve as answer content. HTTP replies contain complete saved answers/failures; no SSE or token streaming is implemented.

## Operating model and tradeoffs

Anyone can run an independent copy with their own OpenAI key. Compose runs API, worker, database, broker, migrations, volume initialization and UI. Publish localhost API 4000/UI 4005; infrastructure remains internal. Named volumes preserve originals/vectors/history through ordinary restart, with no document re-embedding. New AI calls still require OpenAI. Health checks distinguish core reads from degraded ingestion, and startup verifies worker readiness.

Keep exact cosine retrieval with a document B-tree initially; bounded single-document scope does not justify ANN tuning or a separate vector service. Use LangChain OpenAI utilities and token-aware splitting without hiding application transactions/history in framework memory. Local files, one API process and bounded workers serve the initial workload; capacity estimates and performance targets require separate measured qualification.

## Extension boundaries

Storage, vector retrieval, extraction, generation/embeddings and dispatch have focused application-owned boundaries and one selected implementation. Native async provider adapters serve interactive requests; synchronous ones serve ingestion. OCR/HTML/Markdown and additional format extractors must return the same ordered blocks, locators and coverage contract before being enabled. Session-document membership supports future multi-document behavior while enforcing exactly one member in v1.

S3 requires object copy/checksum verification; a dedicated vector store requires reindexing and cross-store recovery while retaining PostgreSQL. Embedding/chunking changes need compatible profile migration; a generation-model change needs quality evaluation. These are migration plans, not implemented integrations.

## Limits and verification

Initial supported content is native PDF text and DOCX body paragraphs/table cells. Images/OCR, complex layouts and other formats are deferred. English is evaluated; other Unicode text is best effort. Extraction coverage, chunk participation and review scope are distinct; neither prompts, valid citation IDs nor model review guarantee semantic support. Current recipes are `analysis-v14`, `focused-insights-v2`, `evidence-review-v8`, `evidence-projection-v14`, `literal-sentences-v2` (300-character passages), `source-aliases-v1` and `grounded-chat-v16` (`answer-v13`). Prompt/model/configuration and review selections persist with results. Additional generation/review calls share existing deadlines and persisted reservations; answer/review reasoning and completion/visible budgets are configurable.

The UI uses API URLs only, restores saved state, polls GET endpoints within bounds and preserves request UUIDs for uncertain submissions. Viewing saved data does not call AI. Both monitoring APIs distinguish document outcomes, attempt history and best-effort performance samples; null/missing telemetry is not displayed as a fabricated zero.

No SSE, streaming, public shared hosting or authentication in v1. Default tests incur no AI charges; real broker/database/worker tests, AppTest emulation, browser interaction and explicit live AI checks establish different evidence. See the detailed design and testing strategy for limits, locked dependencies and remaining measurable gates.

Chat-specific `chat-evidence-review-v7` records a bounded requested-aspect inventory and separately judges answerability and claim relevance within the existing audit call. Valid citations to unrelated material do not justify a partial answer: no relevant supported prose yields an insufficient response with no citations. Planner-v3 preserves explicit new subjects and uses history only for real references; no topic blacklist or extra routine provider request is added.

Current prompts ask for comparisons grounded first in each option’s source conditions and consequences, preserve requirements versus recommendations and conditional outcomes, and require all material premises for a useful distinct insight. Empty insights remain valid. These instructions guide fallible model judgments; known live failures and current evidence remain recorded in the evaluation report.

Analysis review independently classifies each candidate insight for useful additional meaning and validates its factual support. Trivial or repeated candidates may be removed without invalidating a supported overview. Both review contracts persist their decisions in existing audit metadata and retain the current provider-call limits. Their semantic judgments remain fallible; current verification is tracked in [evaluation_current.md](evaluation_current.md).
