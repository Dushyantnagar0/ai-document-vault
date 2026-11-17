# Diagnose processing without repeating paid work

Start with durable metadata, attempts and known telemetry. Reading these records does not invoke AI. A terminal document status of `FAILED` describes its outcome; the last processing attempt records the stage that actually failed.

## Find the exact document

From the repository root, check the local API and list failed documents:

```sh
curl http://localhost:4000/health/ready
curl 'http://localhost:4000/api/v1/documents?status=FAILED&limit=20'
```

Health checks do not test provider credentials/quota. Inspect the returned document UUID, timestamps and stage; do not assume that the newest document or a repeated filename identifies the intended upload. Follow `next_cursor` to continue the list. Then read its metadata:

```sh
curl http://localhost:4000/api/v1/documents/REPLACE_WITH_DOCUMENT_UUID
docker compose exec api python -m scripts.diagnose --document-id REPLACE_WITH_DOCUMENT_UUID
```

Alternatively, use an exact display filename, quoted for your shell:

```sh
docker compose exec api python -m scripts.diagnose --filename 'example.pdf'
```

Filename lookup reports every match within the newest five, total matches and whether older matches were omitted; it never chooses one silently. `--limit 1` through `--limit 5` adjusts that cap. Use a returned UUID for subsequent investigation. The report deliberately omits filenames and document contents, although your shell history can retain a filename you type.

The CLI uses the API container's existing settings and a bounded, repeatable-read **read-only PostgreSQL transaction**. It selects metadata columns, attempt columns and grouped performance observations; it does not load originals, extracted text, embeddings, insights, prompts or chat messages. It never uploads, enqueues, retries, calls a provider or changes database records. It requires database access, not a working AI account; normal central configuration validation still applies. Do not paste keys or connection URLs into arguments.

Read-only reports have been exercised against the isolated database and an existing failed record in the normal installation. This verifies saved-state inspection, not regeneration or successful analysis of every document.

## Read the report before opening logs

The JSON includes document status/stage, a safe error code, extracted counts, current/published attempt IDs, timestamps and the configured document call budget. A call reservation is not proof of a completed provider request or a billable response. A changed configuration can differ from the budget in force when an older attempt ran.

`attempts` preserves the newest 100 attempts, with an explicit truncation flag, durable stage/outcome, broker task UUID, stage timestamps/durations and safe error metadata. For example, a document's final stage may be `FAILED` while its attempt stage is `ANALYSING`; investigate analysis rather than assuming extraction failed. Null counts/times mean unavailable data, not zero. Non-UUID broker IDs are omitted with a presence flag rather than exposing arbitrary strings.

`performance` groups retained document-associated ingestion/chat observations by operation, outcome and safe error code. Compare `embedding.document` and `generation.document` sample counts with the document call reservations. Token sums are null where no usage was observed; corresponding sample counts explain partial availability. Failed structured-output calls can lack generation usage even when an embedding succeeded. Best-effort telemetry, seven-day target retention and uncertain failed calls prevent exact billing claims. Missing observations do not prove that a stage or outbound request never happened.

Errors expose recognized application codes only. When present, `error.details` is filtered to approved reason/finish labels, numeric counters and a validated provider request ID; arbitrary messages, nested data and provider exceptions are excluded. An unknown code/label is withheld rather than copied into support output. `ai_invalid_output` alone does not identify a schema field or prove truncation, refusal or a citation defect. Use a recorded safe reason if present; if absent, state that the precise cause is unknown.

New diagnostic records distinguish `ai_output_truncated`, `ai_invalid_json` and `ai_schema_invalid`, with approved reason, finish reason, known usage/output budget and a validated provider request ID when available. A length finish reason can support a truncation diagnosis for that recorded call. It does not retroactively diagnose an older `ai_invalid_output` failure whose raw response/usage was not retained; truncation remains only a hypothesis in that case.

In the current analysis pipeline, AI output selects supplied `quote_ids` from a literal-passage catalog. The backend resolves those IDs to exact original quotes and source UUIDs before storing the unchanged public `{text,source_ids,support_quotes}` blocks. The model does not copy supporting quotes or source UUIDs into its analysis response. No fuzzy matching, whitespace normalization or larger budget is used to accept a reference. Unknown/duplicate selections remain `analysis_citation_invalid` failures with safe reasons such as `unknown_quote_id` and `duplicate_quote_id`; these labels are retained by the diagnostic CLI. Exact reference restoration still cannot establish that a generated factual claim follows from its selected passage.

A recorded `quote_not_in_original` on a pre-v5 attempt identifies a failed quote check for that attempt, even when its JSON was valid and within budget. It does not establish the cause of a separate older generic failure. Preserve both observations rather than rewriting the earlier record after a successful replacement.

The CLI returns exit 0 with a report, or exit 1 with a safe JSON error such as `INVALID_DOCUMENT_ID`, `INVALID_FILENAME`, `INVALID_LIMIT`, `DOCUMENT_NOT_FOUND`, `CONFIGURATION_INVALID` or `DATABASE_UNAVAILABLE`. Invalid selectors are rejected before connecting. Database failures do not print DSNs, SQL parameters or exception text. See `docker compose exec api python -m scripts.diagnose --help` for arguments.

## Correlate only the necessary logs

### When the document completed but chat failed

Successful ingestion does not establish that a question succeeded. Inspect the saved turns separately:

```sh
docker compose exec api python -m scripts.diagnose_chat --document-id REPLACE_WITH_DOCUMENT_UUID
```

This read-only snapshot reports the newest 20 turns (configurable with `--limit 1` through `50`), total/truncation, session/turn IDs, outcome, HTTP status, reserved call count, safe error details and prompt versions. It selects no question, answer, retrieval query, context snapshot or source content. No turns yields an explicit empty report. Use the turn ID and timestamp to narrow API logs. The document diagnostic's performance aggregates supply known usage without fetching conversations.

`CHAT_INVALID_OUTPUT` on an older turn means the final answer failed validation, not that extraction or embeddings failed. If its precise reason was not retained, do not invent one. A current factual turn normally reserves four calls for planner, query embedding, source selection and composition; one shared source-stage repair is bounded by five total calls; reservation counts alone are not billing evidence. Saved failed requests replay their original failure: a code deployment does not rewrite history, and a deliberate new question uses a new request identity.

The answer path selects literal passage IDs; the server restores quotes/source UUIDs before existing grounding validation. See the [current recipe](evaluation_current.md#current-recipe) for active prompt/pipeline versions and compare them with the saved turn's own versions; historical records are not upgraded by deployment. Ordered passages preserve source text and repetition without sending the full excerpt twice. Unknown/duplicate IDs remain failures; terminal safe reason/usage details survive persisted replay. Logs correlate turn/document IDs, repair number and static reason without raw model output. Legacy answer-path INFO events also record generated/final evidence status and block/selection/citation counts, enabling attribution without retaining raw generations. Multiple exact excerpts may share one source ID; eight total entries remain the cap. A display-limit overflow is not proof of absent evidence; wholly undisplayable supported output and inconsistent insufficient-with-blocks statuses request bounded repair. Exact quotation attachment verifies reference integrity, not whether the explanation is supported; inspect the cited original for semantic review.

Current factual turns save a separately bounded `source_answer` record: actual model, stage versions, input counts, request digests, validated selections/source bindings, stage outcomes and repair details. `SOURCE_ANSWER_INPUT_LIMIT` is a preflight budget failure; `SOURCE_ANSWER_INVALID` identifies a source/component contract rejection. Inspect this record privately before paying for another attempt. Selection success is not composition success, and composition success does not prove semantic completeness. The metadata CLI shows stage versions without loading this content. Overview and historical factual turns retain their `evidence_review` records.

An `EVIDENCE_REVIEW_INVALID` / `invalid_qualification_selection` on a legacy turn does not distinguish duplicate selections from unknown references. Do not invent the subcause. Projectionv10 normalizes repeated qualification selections and treats unknown claimed prose coverage as absent, supplementing with selected original text; unknown source IDs still fail. Legacy review-path contexts retain the bounded original review, claims and selected known source bindings even on projection failure, with `projection_status` and safe `projection_error`. Inspect this private persisted context locally for offline reproduction; never put source/model bodies in logs or public diagnostics. Older failures without this record cannot be reconstructed exactly.

After identifying the document/attempt/broker IDs and time window, inspect bounded service logs locally:

```sh
docker compose logs --since 15m --tail 100 api worker
```

The worker emits JSON business events such as `document_processing_started`, `document_processing_stage`, `document_processing_failed`, `document_processing_retry` and `document_processing_completed`. Correlate their `document_id`/`attempt_id`, stage, outcome, generation-call number and safe `error_code`/`details` with the report and available broker/provider request IDs. Inspect only the relevant window; metadata often already identifies the failed boundary. Share a small sanitized event excerpt or the content-free diagnostic report, not full logs/configuration. Never enable prompt/body, filename, credential/header or raw provider-exception logging to diagnose a schema failure. Historical events may not contain newly added safe fields; missing fields are not evidence of a particular failure cause.

Celery's task `succeeded` log means the transport task returned normally. The worker can persist an expected processing failure and then return, leaving the document correctly marked `FAILED`. Read the durable document/attempt outcome and structured business event; do not interpret Celery transport success as completed analysis or usable insights.

## Fix and verify economically

Reproduce the identified boundary with deterministic fixtures and the isolated test stack first; see [testing strategy](testing_strategy.md). Check schema, source correspondence, budgets and failure classification against the actual safe reason. Keep unknown causes explicit instead of guessing from a generic public error code.

Preserve the failed record, exact fixture bytes, request IDs and observed counters. Replaying a completed request reads its saved result; changing prompts does not regenerate an immutable document. Re-uploading with a new identity or asking a new question can incur charges. Do not automatically repeat unchanged paid cases, compare models or create a new request merely because a response was uncertain. An intentional live verification should be explicitly authorized, target only the affected small/failing fixture and record versions, result and known usage. The [demo](demo.md) explains saved-result replay and the distinction between automatic checks and semantic approval.

The later projectionv11 also drops claims with no supporting selections rather than accepting a reviewer's unsupported `supported=true` assertion. This covers isolated discourse such as “No.” without guessing source IDs. Reviewv4 checks relation direction/negation and separate catalog IDs explicitly. One bounded structured/source review repair uses unchanged evidence and static instructions; legacy chat shares its five-call ceiling with draft repair, and analysis shares existing document limits. Persistent invalid sources remain failures. A prompt instruction does not prove semantic correctness; retain source-reviewed reversed-relation counterexamples.

For a cited but off-topic reply, inspect the original question, rewritten query and private saved review separately. Vector similarity and exact citations establish neither answerability nor relevance. Historically, pipelinev9/plannerv3 introduced chat-evidence-review-v1 with saved answerability/relevant-claim selections; those version labels describe that earlier implementation, not the current recipe. An unrelated draft must become insufficient with no citations. A broad historical query can expose topic contamination from earlier turns. Preserve the original turn; prompt/schema deployment does not rewrite saved replies. Replay supports both historical source-only and current legacy-review relevance-aware verdicts, explicitly labeling the latter.

For `EVIDENCE_REVIEW_INPUT_LIMIT`, inspect saved review `input_tokens`, `input_limit`, passage count, claims, wire version and digest before another generation. Repeated UUID metadata can overflow a catalog even when raw text fits. Aliases preserve every passage; evidence fitting happens before drafting. The CLI retains safe input-budget counts/reasons and review operations. Offline replay restores strict alias bindings, identifies historical gapless schemas and never invents missing historical gap labels.
