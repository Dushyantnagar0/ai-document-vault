# Project handoff

[CASE_STUDY.md](../CASE_STUDY.md) governs requirements. This record separates current source, deployed behavior and verified evidence.

## Reading order

1. [AGENTS.md](../AGENTS.md), the case study, [PRODUCT.md](../PRODUCT.md), [AI_USAGE.md](../AI_USAGE.md), and [PLAN.md](../PLAN.md).
2. [README.md](../README.md), [system_design.md](system_design.md), and the concise [architecture.md](architecture.md).
3. [coding_standards.md](coding_standards.md), [testing_strategy.md](testing_strategy.md), [evaluation_current.md](evaluation_current.md), and [debugging.md](debugging.md) for the affected task.
4. Confirm Git state and running versions. Ignored `.private/current-task.md`, when present, records local execution controls. Paid validation requires applicable explicit authorization and the durable ledger.

## Current source and deployment — 2026-11-17

- Core FastAPI/pgvector/Celery/Redis/OpenAI/PDF/DOCX functionality and the Streamlit HTTP client are implemented. Originals, vectors, analyses, conversations, failures and metrics remain durable across ordinary restart. The trusted personal installation uses localhost API4000/UI4005 and external OpenAI.
- The source integrates factual chat pipeline `grounded-chat-v17`: mini planner → query embedding/scoped retrieval → `source-selection-v1` → `source-composition-v1`. Both source stages use `gpt-6.1-sol` at fixed low effort. Four normal requests and one shared structural/source repair retain the five-call ceiling. Unsupported selection skips composition. No post-composition semantic certification is claimed.
- Source stages preserve original questions/resolved references, validate source/component identity, reuse copied-name/reference safeguards and persist separate bounded `source_answer` metadata before finalization. History informs the planner, never the source stages. Replies and stored recipes identify the actual model/stage version.
- Analysis remains `analysis-v14`/`focused-insights-v2`/reviewv8; overview/exhaustive routes retain their prior mini drafting/review behavior (`answer-v13`, chat reviewv7, projectionv14). Embeddings remain `text-embedding-3-small`/1536. Model changes do not regenerate historical results or require re-embedding.
- **Normal installation now runs pipeline v17 following user-authorized deployment.** API/UI/worker health passed; all nine durable business-table hashes, stored-original hashes and real `.env` match before/after deployment. API/worker application hashes match all 52 source files. No new provider requests ran during deployment. Git packaging/publication is recorded in the delivery checkpoint below.

## Verification

- Integrated current-source provider-free gate: **1,406 tests passed in 22.19s**, including real PostgreSQL/pgvector/Redis/Celery/HTTP, recovery, persistence, UI AppTest, legacy routes and new shared-repair/partial/refusal cases. Ruff/format checked137 application/test files; mypy checked74 source files. Build/startup and Alembic no-drift passed. This is local working-tree evidence with cached images, not a new committed export or live semantic proof.
- Supplied-evidence Sol controls include the original comparison, independent policy comparison, partial/refusal, false premise, injected instructions, fee limits and genuine planner follow-up. The strict modal/follow-up checklists each retain an omitted contextual clause although their direct answers are accurate. Displayed citations support the checked answers; some internal per-component selections omit premises selected elsewhere. See [current evaluation](evaluation_current.md) for limits and failures.
- Four actual isolated API turns passed their frozen source/status criteria: original comparison, genuine follow-up, partial answer and unrelated topic switch after history. Exact replay spent no additional calls. An API/worker restart preserved all nine business-table hashes and both stored-original hashes; post-restart responses/history/PDF stayed exact. The targeted comparison gate is satisfied with the documented internal-citation/checklist qualifications; this is not universal quality certification.
- Historical clean export of code checkpoint `8ba7705` passed1,305tests/static/startup/schema checks; baseline `bb5001a` passed GitHub push/PR CI on Ubuntu24.04. Those historical results do not certify later changes. Observed local platform is macOS Apple Silicon/Linuxarm64; other platforms, uncached downloads and thousand-document load remain unverified.

## Delivery checkpoint

Code checkpoint `3cf1bb6` packages the source after separate provider/configuration (`59e3fa1`), source-helper (`2bf8734`) and factual-integration commits. Its exact 170-file export, without `.env`, `.git`, private artifacts or local data, passed cached Docker build/startup, **1,406 tests in 21.71s**, Ruff/format on 142 files, mypy on 74 source files and Alembic no-drift with fresh isolated volumes and the provider substitute. Documentation reconciliation follows that code checkpoint; see Git history and [PR2](https://github.com/Dushyantnagar0/ai-document-vault/pull/2) for publication/hosted checks.

The user-authorized normal deployment passed UI/API/worker health and matches all 52 application Python files. All nine durable business-table hashes, original-file hashes and `.env` matched before/after. No paid requests ran during deployment or packaging. Existing replies remain historical; submit a new question to exercise the new recipe.

## Next actions and limits

The targeted fix, integrated checks and documentation reconciliation are complete locally. Normal deployment and coherent source packaging are complete. Preserve historical failures. The implementation validation campaign used 29/35 requests, leaving 6 unused, separate from earlier 25/90-call rounds; any independently approved future campaign uses its own authorization/ledger. No unchanged rerun is automatic. The four live API samples span 7.735–27.312seconds; the proposed 20-second percentile target is not established.

Native PDF text/DOCX paragraphs and table cells are supported; OCR/images, extra formats, multi-document chat, SSE, caching, cloud storage and public hosting remain deferred behind documented boundaries. English is evaluated; other Unicode text is best effort. Correct provenance and model judgments do not guarantee semantic completeness. Human walkthrough/recording delivery remains distinct from automated validation; [demo.md](demo.md) is a runnable guide, not a recording.

Preserve the working tree, real environment and retained data. Follow AGENTS for explicit commit/push authorization and coherent commits; do not reset volumes or rewrite saved analyses/history to conceal old failures.
