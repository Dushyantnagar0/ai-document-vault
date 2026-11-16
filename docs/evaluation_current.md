# Current AI evaluation

This record distinguishes prompt/model behavior from deterministic contracts. [CASE_STUDY.md](../CASE_STUDY.md) governs requirements; [testing strategy](testing_strategy.md) owns checks. Valid citations, a model's self-assessment and HTTP200 do not by themselves establish correctness.

## Current recipe

- Analysis: `analysis-v13`, `focused-insights-v1`, `evidence-review-v4`.
- Chat: `planner-v3`, `answer-v11`, `grounded-chat-v12`, standalone `chat-evidence-review-v4`.
- Shared deterministic projection: `evidence-projection-v11`; source passages: `literal-sentences-v2`.
- Generation: `gpt-5.4-mini-2026-03-17`; embeddings: `text-embedding-3-small`,1536dimensions.

Chat review requires answerability, relevant claim IDs and bounded missing-request labels. It checks source relationships and question relevance separately, including causal actors/objects and each qualification passage. Compact aliases preserve full source provenance; context assembly reserves review space before drafting and retains the final actual-input guard. Normal factual chat still uses four provider requests, with a fifth shared by a draft or review repair.

## Local evidence

| Evidence | Observed scope and result |
|---|---|
| Clean committed-source validation | A clean `git archive` export of revision `f252961` contained exactly 160 tracked files and no real environment file, local data or private artifacts. With fresh isolated volumes, no host ports and a local provider substitute, Docker build/startup, **1,183 tests in 18.75s**, Ruff lint/format, mypy and Alembic drift checks passed. The project was removed afterward. Existing Docker cache was used; this was not a remote-clone, uncached-download or live-provider test. Subsequent documentation polish does not change the tested application source. |
| Provider-free regression | **1,183 tests passed locally in 19.43s**, with Ruff lint/format on 127 files and mypy on 71 source files. The suite covers real isolated PostgreSQL/pgvector, Redis, Celery, files, HTTP APIs, Streamlit emulation and provider substitutes. Static checks, schema drift and final tracked-source checks are separate gates; [HANDOFF](HANDOFF.md) and runnable commands identify the operating baseline. Substituted model outputs do not establish semantic quality. |
| Initial mixed live HTTP run | Ten questions in one persisted conversation, with supported explanations, a follow-up, arithmetic, genuine partials, unrelated topics, false-premise correction and forced fabrication. All returned HTTP200; eight matched the expected status and six passed the initial strict source/content review. This run exposed provisional-status stickiness, missing partial explanations and causal/relevance drift; its failures were preserved. |
| Focused corrections | Later HTTP checks verified explicit index/benchmark and WAL/Kafka missing-part labels. An exact saved follow-up draft/verdict was reprocessed offline to verify status correction. These are narrower than a new ten-question live chain. |
| Current frozen-context live regression | All ten fixed cases matched their original expected statuses and material source/content expectations: five answered, two partial, three insufficient. Seventeen provider requests were used because empty insufficient drafts skipped review. Retrieval/history came from the earlier run; this did not rerun planning, query embeddings, ingestion or a new conversation chain. |
| Current actual RAG service | Two additional checks used real planning, query embeddings, pgvector retrieval, generation, review and durable TurnStore history. The backup decision/mechanism returned answered; WAL plus missing Kafka configuration returned partial with explicit missing parts. Both completed, persisted and replayed without additional provider calls. This was service-level execution, not a browser/HTTP submission. |
| Review-input incident | The failing catalog had223passages over6sources. With a representative bounded draft, full-UUID wire encoding required14483tokens against a14224limit; compact encoding required9991. The production input guard and upstream reservation were exercised offline without a provider call. The default limit was not raised, and passages were not silently removed after drafting. |
| Extraction and operational history | Native PDF/DOCX, mixed text/images, image-only, corrupt, encrypted, unsupported and over-page-limit inputs were exercised. Restart checks preserved document/vector/analysis/chat records and original hashes. A separate real-clock broker outage of66.37s preserved the worker process; restoring only Redis resumed one attempt/publication with bounded business/provider work. These observations are not universal crash or platform guarantees. |

The ten-case expected outcomes were fixed before execution. Later corrections are recorded separately; no failing case was silently replaced or relabeled. Local raw document/provider artifacts are ignored rather than distributed as part of the repository. Public tests and evaluation tools provide reproducible contract checks and explicit live opt-in controls.

## Remaining qualifications

- Current source review still notes a **secondary causal-framing caveat** in the actual backup response: its main replicated-deletion explanation is correct, but an extra paragraph introduces backup-usability obstacles using broader failure wording than its selected passages establish. Do not claim every sentence is independently proven or every future causal question will pass.
- Frozen-context and two-service observations do not establish a fresh full ten-question end-to-end run after every prompt change. They are also not a new paid upload with every final recipe component.
- Bounded overviews may be uneven or partial. Distinct insights can be empty when supported candidates repeat the summary; a nonempty list is not a quality guarantee.
- Model review can omit a necessary condition, accept an unsupported relation, remove useful text or select an irrelevant qualification. Exact source membership and copied-identifier guards are not entailment proofs. Manual source review remains required.
- English is the evaluated language. OCR/visual interpretation, other languages, larger collections, remote CI execution, uncached setup and other operating systems are not certified by these local results.
- Delivery of the required document-chat demonstration remains a separate handover activity; see [demo.md](demo.md). The brief does not require a particular recording format. Saved artifacts/history and written instructions do not themselves supply a video or human sign-off.

Earlier analysis/review versions and their measured failures remain historical evidence in [AI_DEV_JOURNAL.md](../AI_DEV_JOURNAL.md). Do not apply their automatic pass counts to this recipe or infer an older failure's precise cause from newer diagnostics.
