# Current AI evaluation

[CASE_STUDY.md](../CASE_STUDY.md) governs requirements; [testing strategy](testing_strategy.md) owns checks. This record separates deterministic correctness, live observations and assistant source review. Valid citations, model review and HTTP 200 do not by themselves establish correctness or human approval.

## Current recipe

- Analysis: `analysis-v14`, `focused-insights-v2`, `evidence-review-v8`; insight/review wire format `source-aliases-v1`.
- Chat: `planner-v3`, `answer-v13`, `grounded-chat-v16`, `chat-evidence-review-v7`.
- Projection: `evidence-projection-v14`; source passages: `literal-sentences-v2`.
- Generation: `gpt-5.4-mini-2026-03-17`; embeddings: `text-embedding-3-small`, 1536 dimensions. Answer and review effort default to `low`; planning and analysis drafting retain `none`. Both effort settings accept `none|low`. Medium-review, larger-model and additional planning-protocol experiments were not adopted.

Analysis reserves downstream prompt space before final synthesis. Compact aliases preserve original provenance. Generation still permits eight summary blocks; final analysis allows sixteen to accommodate citation corrections and literal qualifications without discarding later supported sections. Per-block support, visible-text budgets and chat limits remain unchanged. History trimming preserves complete exchanges both in storage retrieval and complete-prompt assembly.

Prompts distinguish modal strength and conditional triggers, ask comparisons to preserve linked consequences, and require every premise of an insight to be supported. These instructions remain fallible. Normal factual chat uses four provider requests; one draft or review repair shares the fifth-call ceiling.

## Structured-contract verification

Analysis now records explicit whole-candidate usefulness assessments; chat records requested source-based aspects. The final working-tree provider-free checkpoint passed **1,305 tests in 19.80s**, Ruff lint/format, mypy (73 source files), Docker startup and Alembic schema checks. Container application/script sources match the working tree. HTTP tests verify actual planner=`none`, answer=`low`, review=`low` routing, a persisted explicit `none` override and free replay. A later Compose pass-through correction was separately checked with default and explicit effort overrides. The final live comparison failed semantic review. The user subsequently requested this candidate be deployed for manual testing; the normal application now runs the current source. Deployment does not close the quality finding.

A live review correctly rejected two trivial inventory candidates. It initially failed an overly strict empty-reference requirement; exact offline replay after validation correction preserves six summary blocks and discards both candidates. A positive conditional calculation first failed missing-claim validation, then retained an uncited trailing caveat. The clause-audit revision retained it with all four required sources and unchanged summary claims, confirmed by independent assistant source review. These are frozen-review and offline-projection results, not new ingestion or browser runs.

The comparison omission is **not closed**: required aspects alone and a stronger-model diagnostic still missed it. Private source-first planning identified the relationship, but the provisional span-based coverage protocol produced invalid metadata and repetition. Those architecture/model changes were not adopted. Low answer reasoning on the existing mini model produced one better frozen draft; the actual HTTP check did not reproduce a complete supported answer. Its first draft contained the grace-period link but combined two quote IDs into one invalid string and strengthened a recommendation into a requirement. The permitted draft retry restored valid references but omitted that link. The reviewer then omitted the same aspect and incorrectly accepted full coverage. Source review also found a separate borrowing-eligibility rule presented as a cause of the reservation check without support for that causal relationship. Correct retrieval, valid references and HTTP 200 therefore did not satisfy the content-quality gate.

The final HTTP request returned in **14.02 seconds**, using all five allowed requests: planning, query embedding, draft, one draft repair and review. Persisted versions/settings matched the current recipe, including low answer/review effort. Replay returned the identical answer/history without further calls. An ordinary restart of the isolated API/worker preserved counts and hashes of all nine durable business tables and the original PDF checksum. This is operational evidence for the candidate, not a semantic pass or a normal-installation deployment.

The subsequent user-authorized normal deployment passed API/UI/worker health checks and API/worker source-hash comparison. Before/after counts and hashes of all nine durable business tables and all stored originals matched. No provider calls, data reset or regeneration were performed during deployment; new manual questions remain separate from preserved historical responses.

**Paid usage: 90/90 authorized outbound requests used**, all durably recorded, including unsuccessful experiments. The last five used 14,526 input tokens (including the query embedding) and 1,330 generation output tokens. No requests remain authorized; no additional paid rerun is scheduled. Earlier failures below remain historical evidence.

## Committed-source check

A clean export of `8ba7705` contains all 167 tracked files needed for the code/test checkpoint and excludes real environment files, private artifacts and local data. With fresh isolated volumes and the provider substitute, Docker build/startup, **1,305 tests in 21.32 seconds**, Ruff lint/format, mypy on 73 source files and Alembic no-drift passed. Checks of the exact staged slices also passed before their separate commits. The first slice-level database run lacked foundation migrations and stopped at fixture setup; after applying the migrations, all 65 history checks passed. These are provider-free checks with cached Docker layers on the existing local platform. No semantic failure was relabelled and no paid call was added during packaging.

## Earlier evidence for this work

| Check | Observed result and boundary |
|---|---|
| Integrated working-tree regression, 2026-11-17 | **1,220 tests passed in 19.26s** against isolated PostgreSQL/pgvector, Redis, Celery, files and HTTP services with provider substitutes. Docker build/startup, network-disabled Ruff lint/format, mypy and schema drift checks passed. This is working-tree validation, not a new committed export, remote clone, uncached install or live semantic test. |
| Ingestion and reliability regressions | A 400-short-paragraph DOCX completes through HTTP and a real worker; all chunks publish in one attempt. Other checks cover downstream prompt boundaries, relevance pruning without false partial status, chunking soft-timeout retry, complete-exchange trimming at both budget boundaries, and bounded retention with locked-row backlog reporting. |
| Initial fresh live HTTP run | A new three-page narrative PDF and a 400-paragraph DOCX completed. Eight questions in document-scoped conversations returned HTTP 200 with the expected status: supported answers, a pronoun-dependent follow-up, a genuine partial, unrelated questions and a false premise. Source review found a missing comparison detail, analysis modal/condition drift and low-value insights; this was **not an eight-case semantic pass**. |
| Refined analysis and exact offline projection | Three live generation calls reused the narrative document's saved original chunks; no new extraction or embeddings. The draft preserved the targeted conditions and modalities, but citation corrections exposed a rendering-capacity defect. With the bounded projection correction, exact offline replay preserves all **37 summary claims in 11 blocks**, with unchanged text/provenance and 1,059/2,100 visible tokens. This validates projection of saved judgments, not a fresh final-recipe upload. |
| Latest selected live HTTP regression | Five questions used a new conversation and actual planning, query embeddings, retrieval, generation, review and persisted history. All returned HTTP 200 and the expected status. Direct deposit, contextual scuff follow-up, deposit-plus-unknown-price partial and unrelated Kafka refusal passed assistant source review. The comparison improved but still omitted one explicit detail required by its frozen test; no five-case semantic pass is claimed. |
| Focused counterexamples | Frozen low- and medium-effort reviews still approved an earlier draft's recommendation-to-requirement change and missed a linked consequence. On repetitive inventory text, the revised insight generator and reviewer still accepted trivial counts/range restatements. These remain failures, not successes inferred from schema validity. |
| Paid usage | **72 outbound requests from a 75-request authorized allowance**, including embeddings, generation, reviews and repairs. Durable gateway reservations enforced the caps; no hidden SDK retries. No further paid requests were needed for deterministic replay or final checks. |

Expected source facts and initial question order were frozen before execution. Selected rerun order and evidence scope were recorded separately. Original failures were preserved; no case was silently relabeled after observing output. Raw synthetic documents/provider records remain private. Public regressions and evaluation tools supply reproducible contract checks and explicit live opt-in controls.

## Remaining qualifications

- The final current-recipe HTTP comparison still omits the revised-deadline/grace link, advance-request recommendation and late-fee cap; request-alone insufficiency is only implicit. It also adds an unsupported causal connection. A prior better frozen draft does not override this failed integrated result. Coverage and relation-support judgments need further work before this candidate meets the quality gate.
- The two targeted trivial inventory insights are now rejected in the saved-review replay, and the useful-calculation control is retained with complete selected support. This does not certify usefulness across other documents or guarantee the model always chooses an empty list when appropriate.
- The refined narrative overview preserves major later themes after projection, but still omits some details, including bank-posting delay and aspects of disputed-balance refunds. A bounded overview is not exhaustive. Chunk participation, extraction coverage and semantic completeness are distinct.
- Model review can approve a strengthened obligation, omit a condition or accept weak novelty. Exact source membership and copied-identifier guards do not prove entailment. Independent source review remains necessary; no universal hallucination or grading guarantee is made.
- This is not a fresh complete eight-question run after every change. Final analysis was checked by saved-chunk generation plus exact offline projection; final chat used a selected five-question HTTP run. No new browser interaction is claimed because the UI was unchanged.
- English and native PDF/DOCX are the evaluated scope. OCR/visual interpretation, other languages, thousand-document load, remote CI, uncached setup and other platforms remain unverified or deferred as documented.

## Historical evidence

The clean committed export at `f252961` passed 1,183 tests, static/schema checks and isolated startup. The later `cd22361` review independently reproduced that baseline and ordinary restart persistence. Earlier frozen ten-case generation checks and two actual RAG service checks used older recipes and retained their own qualifications; they do not certify the current recipe. Historical observations remain in version history and [AI_DEV_JOURNAL.md](../AI_DEV_JOURNAL.md).

Delivery of the required document-chat demonstration remains a separate handover activity; see [demo.md](demo.md). Saved artifacts or written instructions are not a recording or human sign-off.
