# Document-chat demo

The required document-chat demo can be presented live or as a recording. This walkthrough covers upload, automatic analysis, grounded answers, contextual follow-ups and saved history. **This document is a walkthrough; no video recording is provided.** A fresh clone contains no local documents or chat history.

## Run it with your own document

1. Complete [local setup](../README.md#run-locally) with Docker/Compose and your own OpenAI key in local `.env`. Run `docker compose up --build -d --wait` and open [the UI](http://localhost:4005). Uploads and new questions use paid provider work; viewing saved results does not.
2. Upload a native-text PDF or DOCX under **Your library**. Watch processing until **Ready to chat**. If it fails, inspect the displayed error and diagnostics before retrying.
3. Read **Document overview** and any additional insights. Compare their supporting quotations with the original file available in document details. Empty insights are valid when no distinct addition is supported; read any extraction or analysis warnings.
4. In the automatically opened conversation, ask a question you can verify in the document, such as “What does this document say about renewal?” Open **Source** to inspect the cited excerpt and location. Use **New conversation** for a separate chat.
5. Ask a follow-up that refers to the answer, such as “What year is that?” Choose wording that fits your document. Confirm the response resolves the reference within the same conversation.
6. Exercise both evidence limits: combine a supported topic with a detail absent from the file and check that a partial answer identifies the missing part; then ask a wholly unsupported question and check for an explicit insufficient-evidence response. Also try a focused summary and a whole-document overview, checking qualifications against the source.
7. Reload, reselect the document and conversation, and confirm saved history and citations remain available. If a submission's outcome is uncertain, check history before submitting it again.

Optionally open **Advanced: diagnostics and metrics** or both metrics APIs in [Swagger](http://localhost:4000/docs). To demonstrate persistence, run `docker compose stop` followed by `docker compose up -d --wait`, then reopen the document and history. Originals, vectors, analysis and conversations persist across this restart; new questions still need OpenAI access.

PDF citations use available page/block locations; DOCX citations use paragraph/table paths. English native text is evaluated; OCR and image interpretation are outside the current scope. Citations and automated checks can miss semantic errors, and a bounded overview may omit details. See [current evaluation](evaluation_current.md) for observed results and limitations.

## Optional reproducible narrative fixture

The redistributable request guide and expected ideas are authored in [quality_fixtures.py](../scripts/quality_fixtures.py). On your installation, this command reads saved results without provider work:

```sh
docker compose exec api python -m scripts.evaluate_quality
```

For a fresh installation, explicitly create/process the guide and ask its four scripted questions:

```sh
docker compose exec api python -m scripts.evaluate_quality --live
```

`--live` permits paid ingestion and questions. Results and request identities are retained under `LOCAL_STORAGE_DIR/demo/eval-quality-v1.json`; unchanged completed requests replay saved responses. If an existing report has a stale recipe, consult the [testing strategy](testing_strategy.md) before selecting a new revision.

In the UI select the ready `eval-quality-v1-request-guide.docx` and its saved walkthrough. Duplicate names include creation times. Review these expected meanings against actual answers:

| Question | What to verify |
|---|---|
| What does an accepted request mean? | Durable receipt/queueing for later work, distinct from worker start, completion and external delivery. |
| Why doesn't that mean work is complete? | Context resolves the previous reference and explains separate asynchronous worker/completion events. |
| Summarize retention rules and exceptions. | Ordinary/dispute clocks, active exclusion, holds and the ordinary/dispute rule after release; no invented universal new clock. |
| Give a whole-document overview. | Intake, workers/recovery, qualified planning capacity, review conditions, retention and staffed-hours acknowledgement versus resolution. |
