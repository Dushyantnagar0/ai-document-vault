# Product Scope and Requirements Traceability

**Status: implemented scope reconciled, 2026-11-15.** Ingestion, conversation, monitoring and the minimal UI are implemented on local macOS/arm64. The live automatic gate, long-document repair, restart/browser/crash checks and final verification status are recorded in [HANDOFF.md](./docs/HANDOFF.md). Human semantic approval remains distinct from automatic scoring.

[CASE_STUDY.md](./CASE_STUDY.md) is the absolute source of truth. This document consolidates scope, selected additions, and evidence mapping. Detailed contracts, limits, defaults, schemas, and recovery rules belong in [docs/system_design.md](./docs/system_design.md); verification procedures belong in [docs/testing_strategy.md](./docs/testing_strategy.md). Implementation and a planned check do not establish that every quality or release gate passed.

## Product and operating model

Users upload documents, receive automatic summaries and insights, and ask natural-language questions with supporting sources. Follow-up questions retain conversational context. Operators inspect processing status, failures, document statistics, and performance history.

A summary synthesizes the document's purpose, themes and material facts. Insights explain distinct supported relationships, dependencies, conditions, contrasts or limitations beyond that overview; copied excerpts and repeated summary facts are not sufficient quality evidence. Depth follows substantive content within configured budgets, not a fixed two- or three-point rule. A bounded overview is not an exhaustive report on every fact in a long document. Supporting quotations verify provenance and accompany generated explanations; they do not replace those explanations or prove entailment.

Preserve all five brief objectives: an AI-first RAG backend; effective AI coding-tool use; intelligent processing and conversations; asynchronous pipelines and monitoring APIs; practical product thinking and system design. Record actual assistant contributions, corrections, and validation in [AI_DEV_JOURNAL.md](./AI_DEV_JOURNAL.md), following [AI_USAGE.md](./AI_USAGE.md).

Anyone should be able to self-host an independent instance with Docker Compose and their own OpenAI API key. The initial operating model is one trusted local installation, without shared hosted accounts, tenant management, or authentication screens. Documents and conversations belong to that installation. Public multi-user hosting requires additional access controls and is outside this release. Local hosting still requires external OpenAI access and initial image/dependency downloads.

The implemented local setup uses the environment example, the installation owner's key, Compose and the UI or Swagger. Validated defaults, migrations, persistent/shared volumes and locked dependencies avoid separate host installations of application services. Clean-checkout handover and platform-support claims must still follow recorded verification.

## Required capabilities and evidence mapping

These requirements cover the brief's core backend. Specific formats, stack, error contracts, and acceptance details are project choices. Evidence families are **U** unit, **A** API/contract, **I** real-service integration, **E** backend end-to-end, **L** live AI, **O** operational/performance, and **UI** recorded browser walkthrough. AppTest emulation is labeled separately from a real browser. Cases, commands, and pass criteria live in the testing strategy.

| ID | Capability and acceptance intent | Milestone / evidence |
| :--- | :--- | :--- |
| R1 | Intelligent upload/ingestion: validate content and limits, store the file and metadata safely, promptly acknowledge asynchronous work, and expose a stable document ID and truthful processing status. | M2 · U/A/I/E/O |
| R2 | Smart analysis/async pipeline: extract, chunk, embed, and generate summaries/basic insights; persist attempts, failures, and bounded retry/recovery outcomes. Analyze all extracted text within accepted limits. | M2 · U/I/E/L/O |
| R3 | Document management APIs: upload/ingestion, document retrieval with metadata/status and source access, and AI summary/insight retrieval. Make readiness and missing/failed documents clear. | M2 · A/I/E |
| R4 | Chat APIs: start a document-scoped session, ask questions, retrieve ordered history, and preserve multi-turn context. Use history to interpret follow-ups and retrieved evidence for facts. Return identifiable citations, validate source scope, and state insufficient evidence explicitly. | M3 · U/A/I/E/L |
| R5 | Document statistics API: document counts/status breakdown and storage statistics with clear definitions. | M4 · A/I/E |
| R6 | Processing metrics API: outcomes, durations, attempt history, and system performance data; define units, windows, sample counts, and empty-data behavior. Preserve failed attempts honestly. | M4 · U/A/I/E/O |
| R7 | Persistent storage: metadata/status; summaries/insights; conversations/history; messages with context/citations; processing metrics/history; system performance data. Verify session/document isolation and normal-restart persistence. | M1–M4 · U/I/E/O |
| R8 | Usability/reliability: actionable validation/dependency errors, unsupported/empty/corrupt inputs, safe file/source handling, bounded timeouts/retries, duplicate/interrupted processing recovery. Explain scaling to thousands of documents through queues, indexes, and capacity limits. | M1–M6 · U/A/I/E/O |

### Content coverage and language

Initial formats are PDF and DOCX with native text extraction. OCR, visual/table-layout understanding, and additional formats are deferred. Preserve source locations: PDF pages when available and stable DOCX/chunk locators without invented page numbers.

Extraction coverage and analysis coverage are separate. Some content may be scanned, embedded, or unsupported, yielding only partial text. Surface detectable extraction limitations without claiming every omission can be detected. Empty extraction needs an actionable outcome. Summarization must process all extracted text within accepted limits rather than silently selecting its first portion. Analysis of all extracted text can still be partial relative to the original file; disclose that distinction in insights and chat. Oversized input follows explicit limits/failure behavior rather than silent truncation.

English is the evaluated language for extraction, summaries, retrieval, and chat. Preserve Unicode and provide best-effort behavior for other languages; multilingual quality is not a verified support claim. Fixtures cover both formats, source support, follow-ups, missing/partial/conflicting evidence, and document instructions treated as untrusted content. Prompts alone cannot guarantee grounding.

## Selected stack and bounded extensions

| Area | Source brief | Selected project approach |
| :--- | :--- | :--- |
| Backend / API docs | Python required; FastAPI and API docs recommended | Python, FastAPI, generated OpenAPI/Swagger |
| Database / retrieval | PostgreSQL required; vector integration is bonus | PostgreSQL + pgvector |
| Background work | Celery + Redis, RQ, or similar required | Celery + Redis; durable status in PostgreSQL |
| Parsing / storage | Suitable parsers; local or S3-compatible storage | PyMuPDF for PDF, python-docx for DOCX, local files initially |
| AI | OpenAI chat/embeddings or the brief's allowed alternative | OpenAI chat/embeddings with validated model/dimension configuration |
| RAG libraries | LangChain or LlamaIndex highly recommended | Focused `langchain-openai` and `langchain-text-splitters`; explicit application orchestration and persisted conversation state |
| Verification / workflow | pytest and AI assistants recommended | pytest, reviewed assistant contributions, fixtures, separately labeled live checks |
| UI / local runtime | Frontend optional; deployment tooling is a project choice | Minimal Streamlit HTTP client and Docker Compose |

Keep focused adapters for extraction, storage, embeddings/generation, vector retrieval, and task dispatch. Implement one selected adapter per capability. Future OCR/formats enter through extraction/source-locator contracts; future storage/vector providers preserve application contracts. PostgreSQL remains required if vector storage changes. These are extension boundaries, not commitments to alternate implementations now.

The implemented `chat_session_documents` association enforces exactly one immutable document per session through database uniqueness and transactional creation. Retrieval and citation validation enforce membership. Multi-document selection, retrieval, and comparative answers remain deferred and require explicit API, quality, and isolation checks before expansion.

## Minimal Streamlit client

The wide screen places the document library/upload in the left sidebar, chat in the center, and overview/insights plus document details/original on the right. The message transcript and right pane scroll independently; the question box stays below the transcript, and reading chat keeps the document overview in place. The overview starts open. Overview/insights, saved conversations, source excerpts, original-file details and diagnostics are expandable; ordinary labels use names and readiness rather than UUIDs. Extraction and partial-answer warnings remain visible. Bounded polling, actionable errors and pagination preserve access to later documents/conversations/messages. Chat is disabled before readiness and while an assistant outcome is pending; reload/session reselection restores backend history.

Streamlit calls FastAPI over HTTP only, reads only its two API URLs, and runs as a Compose service; processing, provider access and persistence stay in the backend. Uncertain uploads/questions keep their request UUID for manual replay; a new attempt after a known upload failure requires explicit consent for potentially chargeable work. Saved views/polling do not invoke AI. Show extraction limitations separately from analysis coverage, render original source excerpts/locators safely, and display unavailable metrics honestly. Swagger/APIs remain independently usable. Elaborate dashboards, authentication screens, streaming and custom design systems are outside scope. The UI is a selected addition; the brief requires the document-chat demo, not a frontend.

## Evaluation traceability

The brief assigns **33% to each category**, retaining its rounding. Useful analysis, grounded explanations and multi-turn chat are core acceptance work. Their quality is not established by valid JSON, citation membership, test counts or the presence of a prompt. The table is a review checklist, not a prediction of an evaluator's score. Historical results do not certify a changed prompt.

| Category | Subcriterion | Acceptance evidence and current qualification |
|---|---|---|
| AI-First 33% | Effective use of AI coding tools | Journal records bounded tasks, parent review, corrections and actual commands; identify only tools used, with no invented time savings. |
| AI-First 33% | AI integration quality | Actual upload produces a substantive synthesis and distinct useful supported insights; inspect original evidence and omissions. Content-quality review is a core gate, alongside pipeline completion. |
| AI-First 33% | Prompt engineering | Versioned planner/analysis/answer rationale, representative concept/why/follow-up/focused-summary/overview cases, observed failures and before/after source review. Writing a prompt is insufficient evidence of its effectiveness. |
| AI-First 33% | AI-driven automation | Upload-to-analysis and automatic follow-up resolution work without manual orchestration. Completion must preserve qualifications and useful content; refusal/failure cannot count as successful understanding. |
| AI-First 33% | Innovation | Demonstrate practical value from literal evidence selection, inspectable support and coverage-aware synthesis. Extra features cannot compensate for weak core answers. |
| Product 33% | User experience | Actual browser upload/select → ready → conversation → follow-up → source → reload, with understandable statuses and recognizable saved conversations. Keep technical controls expandable. |
| Product 33% | Feature prioritization | Close processing, analysis/chat quality, persistence and monitoring gates before deferred bonuses. Basic insights and contextual chat remain mandatory. |
| Product 33% | Error handling | Preserve failed attempts/turns and replay IDs; distinguish missing evidence from citation-display or provider failures. Partial answers must explain their relevant limitation. |
| Product 33% | Real-world usability | Own-key Compose startup, useful narrative PDF/DOCX responses, restart persistence, visible extraction limits and readable sources. Representative source review supplements synthetic fact fixtures. |
| Product 33% | Solution architecture | Bounded queue-driven processing, document membership, transactions/recovery and replaceable adapters solve the present scope; no enterprise infrastructure is required to close semantic gaps. |
| Technical 33% | Code quality | Typed modules/adapters, safe diagnostics and reviewed regressions. Tests must challenge design assumptions; a passing test that accepts an unhelpful outcome is corrected. |
| Technical 33% | System performance | Report recorded latency/usage with sample/configuration and failure counts. Historical small-fixture timings are not current-version or large-collection guarantees. |
| Technical 33% | API design | Typed upload/retrieval/insights/chat/history/metrics contracts, stable errors/pagination/replay and Swagger examples. Verify the actual insights payload matches its published schema. |
| Technical 33% | Asynchronous processing | Real Redis/Celery/PostgreSQL flow, immediate dispatch plus recovery, lease fencing, bounded retries and duplicate delivery/crash checks. Mocks alone cannot establish this. |
| Technical 33% | Documentation | Setup, design assumptions, prompt rationale/evidence, testing/debugging and required real chat demo stay aligned with actual versions. Reopen acceptance when real use exposes a core gap. |

The separate narrative quality evaluation adds source-based review of summary themes/qualifications, insight distinctness/support, conceptual explanation, contextual why-answers, focused-summary routing and whole-document synthesis. Automated idea checks are indicators, not semantic proof. Review records identify the reviewer, source references and observed versions; assistant review must not be presented as human approval. Commands/results belong in the testing strategy and handoff. The historical 20-case suite remains separately versioned evidence.

## Milestones and deliverables

This is the approved dependency sequence, open to evidence-based iteration. M1–M5 source is implemented; recorded checkpoints do not alone close live-quality, operational or handover gates. Metrics events are captured with ingestion/chat so monitoring exposes retained work history. Procedures/evidence definitions remain in the testing strategy.

| Milestone | Exit intent | Planned evidence |
| :--- | :--- | :--- |
| M0 — Documentation review | Review scope, HLD/LLD, defaults/contracts, acceptance checks, and unresolved decisions; record outcome before implementation. | Brief traceability; documentation/link/consistency checks |
| M1 — Foundation | Reproducible dependencies, validated settings, Compose, migrations, shared/persistent storage, API/worker readiness. | U/A/I/O |
| M2 — Ingestion | PDF/DOCX upload through real workers to persisted text, embeddings and analysis; retrieval and failure/coverage state exposed. | R1–R3/R7/R8 · U/A/I/E/L/O |
| M3 — Chat | Single-document sessions, grounded questions, follow-ups/history, citations and insufficient-evidence handling. | R4/R7/R8 · U/A/I/E/L/O |
| M4 — Metrics | Both metrics APIs reflect known processing/performance events accurately. | R5–R7 · U/A/I/E/O |
| M5 — UI | Minimal client works through backend APIs with history, citations and useful errors. | E/UI; repeatable manual browser walkthrough |
| M6 — Handover | Clean-checkout setup, regression evidence, real-provider chat demo, timings, README and limitations aligned with release revision. | Applicable U/A/I/E/L/O/UI; documentation review |

Required brief deliverables: working backend; source repository; README setup/architecture with assumptions; AI_USAGE or equivalent actual workflow documentation; and a **document-chat demo**. Demonstrate upload/processing, insights, a grounded question with inspectable citations, follow-up context, history, and an unanswerable question. Cover both selected formats in acceptance evidence and verify both monitoring APIs. Selected additions are Streamlit, pgvector, and reproducible local Compose operation.

Provider substitutes establish deterministic behavior, not live AI quality. Default automated checks incur no provider charges; live checks are explicit. Record commands, revision/configuration, passes/failures/skips, and platform limitations. Pending required checks leave handover unverified.

## Deferred scope

Core work takes priority over bonuses. Deferred AI features: multi-document chat/comparison, follow-up suggestions, summary customization, categorization/tags, sentiment and advanced insight extraction. Basic insights remain core.

Also deferred: OCR/additional formats, real-time updates, streaming, embedding/response caching, comprehensive coverage, advanced cost optimization, quotas/rate limiting, document versioning, batch optimizations, and extensive observability. Basic health/readiness and diagnostics for local operation do not replace required monitoring APIs. Shared hosted accounts, public deployment, cloud infrastructure, and elaborate UI are outside the initial operating model. Extend only with bounded requirements/verification; document scale assumptions without claiming tested capacity.
