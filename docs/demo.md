# Document-chat demo

The required demo is upload → automatic analysis → grounded question → contextual follow-up → saved history. **Saved demo means persisted documents, conversations and evaluation artifacts, not a video.** Local saved data is ignored and is not included in a fresh clone. [HANDOFF](HANDOFF.md) records running versions and completed checks; [current evaluation](evaluation_current.md) separates their evidence and remaining limitations.

## Run it with your own document

1. Complete [local setup](../README.md#run-locally): Docker/Compose running, your OpenAI key in local `.env`, then `docker compose up --build -d --wait`. Open [the UI](http://localhost:4005); [Swagger](http://localhost:4000/docs) is independently usable.
2. Upload a native-text PDF or DOCX under **Your library**. Upload starts paid embedding/analysis work; wait for **Ready to chat**. A failed document remains visible with its error. Inspect diagnostics before intentionally paying for another upload.
3. Read **Document overview** on the right. Compare the synthesis and any additional insights with their supporting quotations; open the original file from document details to inspect the source itself. Empty insights are valid when no useful distinct addition is supported. Extraction limitations, extracted-chunk participation and analysis review scope mean different things.
4. The first empty conversation opens automatically when processing completes. Use **New conversation** for an additional chat. Ask a concrete question whose answer you can verify in the file, such as “What does this document say about renewal?” Open its **Source** expander and check the actual excerpt/location.
5. Ask a contextual follow-up in that conversation, such as “What year is that?” Then ask a question the file cannot answer. Confirm useful supported answers and an explicit insufficient-evidence outcome; citations alone do not prove correctness.
6. Ask for a focused topic summary and a whole-document overview. Check qualifications and scope. A bounded overview is a synthesis, and partial source excerpts are not a complete inventory of every item in a long file.
7. Reload, reselect the document/conversation, and inspect saved history. Reading existing analysis/history does not call AI. New questions use paid provider work. After an uncertain submission, recover saved history or replay the same request identity rather than creating another chargeable request.
8. Open **Advanced: diagnostics and metrics**, or inspect both metrics APIs in Swagger. Optionally stop/restart with `docker compose stop` then `docker compose up -d --wait`; retained originals, compatible vectors, analysis and history should remain available. New chat still needs OpenAI access.

DOCX sources use paragraph/table paths, not invented page numbers. PDF sources use available page/block locations. English native text is evaluated; OCR/image interpretation, extra formats, multi-document chat and streaming are deferred behind extension boundaries.

## Optional reproducible narrative fixture

The redistributable request guide and expected ideas are authored in [quality_fixtures.py](../scripts/quality_fixtures.py). On your installation, this command reads saved results without provider work:

```sh
docker compose exec api python -m scripts.evaluate_quality
```

For a fresh installation, explicitly create/process the guide and ask its four scripted questions:

```sh
docker compose exec api python -m scripts.evaluate_quality --live
```

`--live` permits paid ingestion and questions. The harness preserves exact fixture bytes, upload/request identities, versions, results and reviews under `LOCAL_STORAGE_DIR/demo/eval-quality-v1.json`. Unchanged completed requests replay saved responses; changed recipes are stale evidence, not silently refreshed successes. On an installation with historical results, inspect the report and testing strategy before intentionally selecting a new revision. Preserve uncertain/failed identities and old results.

In the UI select the ready `eval-quality-v1-request-guide.docx` and its saved walkthrough. Duplicate names include creation times. Review these expected meanings against actual answers:

| Question | What to verify |
|---|---|
| What does an accepted request mean? | Durable receipt/queueing for later work, distinct from worker start, completion and external delivery. |
| Why doesn't that mean work is complete? | Context resolves the previous reference and explains separate asynchronous worker/completion events. |
| Summarize retention rules and exceptions. | Ordinary/dispute clocks, active exclusion, holds and the ordinary/dispute rule after release; no invented universal new clock. |
| Give a whole-document overview. | Intake, workers/recovery, qualified planning capacity, review conditions, retention and staffed-hours acknowledgement versus resolution. |

## What the retained evidence establishes

Historical full API runs, current frozen-evidence generation, saved-chunk analysis and real-service conversations are separate checks. The current frozen core/narrative outputs and assistant source reviews are indexed in [current evaluation](evaluation_current.md). Frozen checks reuse earlier retrieval/history and do not establish new ingestion, planner/retrieval execution or browser behavior. Saved-chunk analysis buys no new embeddings and does not publish a replacement document. Earlier narrative revision 3 is historical evidence, not a current-recipe pass.

Earlier real-service conversations titled **Quality check: databases …** were archived during the owner-requested scratch reset; they are historical evidence, not conversations guaranteed to exist in the active installation. [HANDOFF](HANDOFF.md) records the archive and subsequent verification state. Follow the fresh walkthrough above using your own current documents and conversations. A fresh clone never contains ignored local originals, chat history or evaluation artifacts.

Preserve the ambiguous owner-follow-up mismatch, partial long-document scope, failed attempts and source-review caveats. An assistant review is not human approval, and a regex/valid-citation check is not semantic proof. No blanket current-version quality, video recording, universal compatibility or production-capacity claim is made here.
