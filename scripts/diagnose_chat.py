"""Read bounded chat metadata; never read questions, responses or evidence."""

import json
import re
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import Connection, Engine, create_engine, func, select
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import ConfigurationError, get_settings
from app.db.models import ChatSessionDocument, ChatTurn
from scripts.diagnose import DiagnosticError, Parser, _error, _id, _label, _time


def diagnose_chat(connection: Connection, *, document_id: UUID, limit: int = 20) -> dict[str, Any]:
    """Read explicit metadata columns in newest-first order with an honest cap."""
    if not 1 <= limit <= 50:
        raise DiagnosticError("INVALID_LIMIT")
    condition = ChatTurn.session_id.in_(
        select(ChatSessionDocument.session_id).where(ChatSessionDocument.document_id == document_id)
    )
    total = connection.scalar(select(func.count()).select_from(ChatTurn).where(condition)) or 0
    columns = [
        getattr(ChatTurn, key)
        for key in (
            "id",
            "session_id",
            "turn_number",
            "status",
            "response_status",
            "ai_calls_reserved",
            "planner_intent",
            "evidence_status",
            "versions",
            "error",
            "created_at",
            "ended_at",
        )
    ]
    rows = (
        connection.execute(
            select(*columns)
            .where(condition)
            .order_by(ChatTurn.created_at.desc(), ChatTurn.id.desc())
            .limit(limit)
        )
        .mappings()
        .all()
    )
    turns = []
    for row in rows:
        versions = row["versions"] if isinstance(row["versions"], dict) else {}
        turns.append(
            {
                "turn_id": _id(row["id"]),
                "session_id": _id(row["session_id"]),
                "turn_number": row["turn_number"],
                "status": _label(row["status"], {"PENDING", "COMPLETED", "FAILED"}),
                "response_status": row["response_status"],
                "ai_calls_reserved": row["ai_calls_reserved"],
                "planner_intent": _label(
                    row["planner_intent"], {"factual", "overview", "exhaustive"}
                ),
                "evidence_status": _label(
                    row["evidence_status"], {"answered", "partial", "insufficient"}
                ),
                "versions": {
                    key: value
                    for key, value in versions.items()
                    if key in {"answer_prompt", "planner_prompt", "answer", "planner", "pipeline"}
                    and isinstance(value, str)
                    and re.fullmatch(r"(?:answer|planner|grounded-chat)-v[0-9]{1,4}", value)
                },
                "error": _error(row["error"]),
                "created_at": _time(row["created_at"]),
                "ended_at": _time(row["ended_at"]),
            }
        )
    return {
        "read_only": True,
        "document_id": str(document_id),
        "total_matches": total,
        "returned": len(turns),
        "truncated": total > len(turns),
        "order": "newest_first",
        "turns": turns,
    }


def main(argv: Sequence[str] | None = None) -> int:
    engine: Engine | None = None
    try:
        parser = Parser(description=__doc__)
        parser.add_argument("--document-id", required=True)
        parser.add_argument("--limit", type=int, default=20)
        args = parser.parse_args(argv)
        if not 1 <= args.limit <= 50:
            raise DiagnosticError("INVALID_LIMIT")
        try:
            identifier = UUID(args.document_id)
        except ValueError:
            raise DiagnosticError("INVALID_DOCUMENT_ID") from None
        engine = create_engine(
            get_settings().database_url,
            hide_parameters=True,
            connect_args={
                "connect_timeout": 3,
                "options": "-c statement_timeout=5000 -c lock_timeout=2000",
            },
        )
        with engine.connect() as raw:
            connection = raw.execution_options(
                isolation_level="REPEATABLE READ", postgresql_readonly=True
            )
            with connection.begin():
                report = diagnose_chat(connection, document_id=identifier, limit=args.limit)
        print(json.dumps(report, allow_nan=False))
        return 0
    except DiagnosticError as error:
        code = str(error)
    except ConfigurationError:
        code = "CONFIGURATION_INVALID"
    except SQLAlchemyError:
        code = "DATABASE_UNAVAILABLE"
    except Exception:
        code = "DIAGNOSTIC_FAILED"
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass
    print(json.dumps({"read_only": True, "error": {"code": code}}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
