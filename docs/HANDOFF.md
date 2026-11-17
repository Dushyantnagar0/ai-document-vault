# Project handoff

[CASE_STUDY.md](../CASE_STUDY.md) governs requirements. This handoff describes the implementation and verification scope; it does not certify every generated answer.

## Reading order

1. [AGENTS.md](../AGENTS.md), the case study, [PRODUCT.md](../PRODUCT.md), [AI_USAGE.md](../AI_USAGE.md), and [PLAN.md](../PLAN.md).
2. [README.md](../README.md) for setup/commands and [system_design.md](system_design.md) for HLD/LLD contracts; [architecture.md](architecture.md) is the concise overview.
3. [coding_standards.md](coding_standards.md), [testing_strategy.md](testing_strategy.md), [evaluation_current.md](evaluation_current.md), and [debugging.md](debugging.md) for the affected work.
4. Confirm actual Git state and running versions. Local private session controls, when present, are in ignored `.private/current-task.md`; paid validation always needs applicable authorization.

## Implemented baseline

- FastAPI; PostgreSQL/pgvector; Celery/Redis; external OpenAI generation/embeddings; PyMuPDF/PDF and python-docx/DOCX; persistent local originals, vectors, analysis, chat and metrics; Streamlit over HTTP.
- Compose initializes permissions and migrations. API/Swagger: localhost4000; UI: localhost4005. No host Python/database/Redis setup is required. The API/UI are intended for trusted personal self-hosting, without public authentication.
- Uploads durably record originals and processing intent. Normal dispatch uses the queue; reconciliation recovers missed dispatch/interrupted work. Processing attempts, deadlines and provider reservations remain bounded; worker broker reconnection does not exhaust a fixed reconnect count.
- Analysis processes every accepted extracted chunk, freezes its overview, generates distinct supported insights and audits source selections. Participation is separate from extraction capability and semantic/visual completeness; an empty distinct-insight list is valid.
- Chat persists ordered turns and citations, resolves genuine follow-ups and preserves explicit topic changes. Source support, relevance and requested scope are separate checks. Unrelated supported facts yield insufficient evidence. Partial replies separate factual content from requested information not established by the excerpts.
- Source aliases compact review input without dropping passages or changing server provenance. Context assembly reserves draft/claim/review capacity before generation; the actual complete review prompt still passes a final input guard. Safe failure context retains counts, limits, claims and a packet digest for offline diagnosis.
- Normal factual chat uses four provider requests. A fifth is shared by one draft repair or one invalid structured/source review repair. There are no hidden SDK retries. Scope/relevance judgments remain fallible.
- The first empty conversation opens automatically after processing; subsequent chats use New conversation. Composer visibility and readiness are independently checked. Default model context includes up to four recent completed turns within2,000tokens; older history persists. The UI bounds history loading at5,000messages.

## Verification and limits

[Current evaluation](evaluation_current.md) separates provider-free tests, live-provider observations, frozen-context checks and operational evidence. Run the documented checks after consequential changes; a written design or mock result is not live semantic evidence. Preserve failed turns and historical results rather than silently rewriting them after a fix.

Clean committed-source checkpoint: a `git archive` export of `f252961` (160 tracked files, no `.env`, `.git`, local data or private artifacts) passed Docker build, API/worker/UI startup, all 1,183 tests, Ruff lint/format, mypy and Alembic drift checks. The run used fresh isolated volumes, no host ports and a local provider substitute. Its test project was removed afterward; the normal application/data were preserved. Docker layers were cached; this was not a remote clone or uncached-download check. Later documentation-only polish does not change the verified application source.

Observed platform: macOS Apple Silicon with Linuxarm64 containers; Docker Engine29.9.2, Compose5.6.1 and image Python3.12.16/uv0.13.8. Other platforms are targets only. Restart persistence and isolated transport/recovery checks have local evidence; uncached downloads, remote CI execution, thousand-document load and broader platform compatibility are not certified.

Generation is pinned to `gpt-5.4-mini-2026-03-17`; embeddings use `text-embedding-3-small`/1536dimensions. Current source versions and exact evaluation recipes belong in [current evaluation](evaluation_current.md). Existing analyses/history retain their original versions; changing prompts does not regenerate them or require re-embedding.

## Next work and handover

- Manual source review and delivery of the required document-chat demonstration remain separate from automated checks; see [demo.md](demo.md). The brief does not prescribe a video format. The walkthrough is not a recording.
- Continue held-out relevance, causal-support, qualification and partial-scope evaluation. Neither valid source IDs nor model review proves entailment or prevents all hallucinations.
- OCR/images, additional formats, multi-document chat, SSE, caching, cloud storage and public hosting remain deferred. Their adapter boundaries are documented; do not implement them merely because they are extensible.
- Preserve real environment files and retained data. Never reset volumes, buy unchanged paid reruns or publish Git changes without applicable authorization. Follow the logical-commit rule in AGENTS.md and keep private artifacts/secrets out of source delivery.
