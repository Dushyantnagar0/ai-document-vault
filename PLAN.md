# Delivery Plan

**2026-11-17: factual source-selection/composition is integrated and deployed locally after the targeted verification gates.** [HANDOFF](docs/HANDOFF.md) records deployment and validation status; [current evaluation](docs/evaluation_current.md) separates provider-free checks, live source review and remaining qualifications. Earlier committed-source/CI results are historical baselines, not certification of later changes.

[CASE_STUDY.md](CASE_STUDY.md) governs requirements. [PRODUCT.md](PRODUCT.md) maps scope and evidence. [System design](docs/system_design.md) is the sole detailed HLD/LLD/contracts record; [testing strategy](docs/testing_strategy.md) owns verification commands and thresholds. This roadmap replaces the earlier duplicated design proposal.

## Milestones and acceptance

| ID | Tasks and dependencies | Exit gate |
|---|---|---|
| M0 — Documentation review | Consolidate scope/design/defaults, diagrams, standards and validation; reconcile guides/environment example | Documentation checks and user review; record outcome before implementation |
| M1 — Foundation | Resolve/lock dependencies and image digests; central settings; Compose, volume initialization, migrations; API/worker health; isolated test image | Clean build/import/start from empty data; correct configuration without secrets; registered consuming worker, pgvector and shared persistent files; planned checks runnable |
| M2 — Ingestion | Safe idempotent uploads and retrieval; worker extraction/chunks/batched embeddings/staged analysis; durable states/attempts and recovery | Both formats through real services with provider substitute; live analysis; truthful partial coverage; duplicate/concurrent/interrupted work handled within budgets |
| M3 — Conversation | Sessions/membership/history, turn idempotency, query planning, scoped retrieval, validated answers/citations | Supported/follow-up/unsupported/conflicting questions; broad-query scope; concurrency, disconnect/replay and live quality gate |
| M4 — Metrics and operations | Complete statistics/processing-history APIs and retained performance data; operational/performance scripts | Exact fixture metrics, bounded retention, dependency/crash recovery, recorded timing samples and restart/new-question persistence |
| M5 — Minimal UI | HTTP-only upload/select/status/insights/session/chat/citations/metrics | Repeatable browser flow with backend history, partial warnings, real errors and stable submission IDs; Swagger remains usable |
| M6 — Handover | Clean checkout, regression, live demo, supported-platform evidence and final README/journal | Required APIs/storage/demo pass on recorded revision; failures/skips/platform limitations visible; no unresolved core-flow blocker |

M1 precedes M2, M2 precedes M3, M3 precedes M5; M4 instrumentation starts in M2/M3 and finishes before M6. The first working backend flow takes priority over UI polish and bonuses. No arbitrary deadlines or unmeasured productivity claims.

## Current evidence and remaining exits

Current work replaces factual drafting/review with source selection followed by composition, preserving the normal four-request flow, shared fifth repair and unchanged legacy/analysis routes. The original comparison and additional controls have source-reviewed supplied-evidence results; strict contextual checklist omissions remain recorded. Four current actual API cases and restart/replay checks satisfy the targeted integrated gate; documented strict-checklist and semantic limits remain.

| Area | Recorded checkpoint | Remaining qualification |
|---|---|---|
| Backend/static | Clean code export `3cf1bb6`: 1,406 provider-free tests, static/startup/schema checks passed | Current hosted status belongs to PR2; historical CI remains scoped to its commit |
| Live AI | Frozen controls and four actual API/source checks completed | Strict contextual omissions and internal per-component citation qualifications remain explicit |
| Persistence/recovery | Integrated suite plus current isolated API/worker restart and exact zero-call replay | Ordinary restart evidence is not a new hard-kill drill |
| UI/demo | Existing UI is unchanged; AppTest is part of the provider-free suite | Human walkthrough/recording delivery remains distinct from tests |
| Setup/performance | Isolated local builds/startup and earlier clean-export evidence | No uncached-download, broad-platform or thousand-document benchmark claim |

[HANDOFF](docs/HANDOFF.md) owns the current integrated checkpoint and remaining review steps. Reuse completed paid requests; new revisions must be intentional and retain prior failures.

## Review decisions captured

- Anyone runs their own installation/key; OpenAI is external. API4000/UI4005, localhost only, no accounts.
- Native PDF/DOCX extraction, all accepted extracted text processed asynchronously in bounded chunks/batches; extraction limitations differ from analysis coverage.
- Focused LangChain utilities; source-aware retrieval/prompts with valid citations and explicit insufficient/partial evidence.
- Persist originals/vectors/insights/chat/attempts across normal restart; new questions reuse document vectors.
- Exactly one session member initially; association and adapter contracts support future multi-document/OCR/formats without implementing them.
- Configurable, jointly validated input/context/call/time budgets; atomic upload/chat replay and fenced worker/turn publication.
- Normal uploads enqueue immediately after commit; the 15-second recovery scan repairs missed dispatch, due retries and expired claims.
- Worker startup verification, safe client-disconnect recovery and honest whole-document/exhaustive-answer scope.
- English evaluated, other Unicode best effort; no claims of offline AI, universal portability or production-scale verification.

Defaults, fields, diagrams and error policies are in system_design rather than repeated here. Dependencies and images are locked and built locally; model access was exercised with the operator's key, which does not establish another account's access or quota.

## Parallel work and integration

Parent owns shared contracts, migrations, lockfiles/composition and integrated verification. After interfaces stabilize, independent tasks can own extractor modules/fixtures, prompt/evaluation files, metrics service/tests or UI files. Give bounded briefs, exclusive paths and evidence-based handoffs; avoid overlapping writers. Parent reviews outputs and runs affected integrated checks before claiming a milestone passed.

Record actual prompt/contribution, corrections, command/results and limitations in the journal. Prompts and mock results do not prove live grounding.

## Documentation responsibilities

| Document | Owns |
|---|---|
| system_design | Detailed HLD/LLD, Mermaid diagrams, capacity, entities/contracts, settings, reliability and extensions |
| architecture | Concise component/operating overview linking to detailed design |
| PRODUCT | Core/selected/deferred scope, requirement/deliverable/evaluation traceability |
| testing_strategy | Test layers, fixtures, commands, acceptance thresholds and release procedures |
| demo | Reproducible document-chat walkthrough and optional generated fixture; demonstration delivered separately |
| coding_standards | Implementation and verification conventions |
| README | Status, prerequisites, configuration, setup, examples/demo, tests, lifecycle, troubleshooting and limitations |
| AI_USAGE / AI_DEV_JOURNAL | Workflow/prompt rationale versus actual contributions/results |
| AGENTS / HANDOFF | Agent navigation/rules versus current phase, review state and next action |
| .env.example | Placeholder key, agreed defaults, units and optional tuning |

Keep CASE_STUDY unchanged. Keep detailed contracts in one place. Update design alongside consequential implementation changes after review; do not rewrite historical evidence to imply unexecuted work.

## Current prerequisites and pending evidence

Docker Desktop and the operator-supplied ignored key are configured locally. Host Python was left unchanged; application checks use container Python 3.12. Routine tests and crash/upload-browser checks use a provider substitute. Historical live API evaluation met its recorded automatic gate. Current frozen-evidence, saved-chunk analysis and real-service checks retain their own failures, partial outcomes and assistant reviews. Consult HANDOFF for the latest integrated tests and deployed recipe; earlier regression/deployment/browser checks do not certify later source changes. Human semantic approval remains open. A clean export of committed code checkpoint `8ba7705` passed the isolated build/startup/regression/schema checks using cached Docker layers. GitHub provider-free checks passed for `bb5001a`; other platforms, large-collection capacity, a manual remote-clone setup and uncached-download checks remain unverified; see [current evaluation](docs/evaluation_current.md).
