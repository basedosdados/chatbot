import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage

from app.agent.context import AgentContext
from app.agent.tools import dataviz as dataviz_module
from app.agent.tools.dataviz import (
    chart_query_result,
    export_query_result,
    list_query_results,
)
from app.charts import ChartResultTooLarge
from app.db.models import QueryHandle
from app.exports import ExportedFile, ResultTableExpired, ResultTooLarge


def _exported(size_bytes: int = 2048) -> ExportedFile:
    return ExportedFile(
        bucket="b",
        object_key="query_results/m/qr_1.parquet",
        filename="resultado.parquet",
        mime_type="application/vnd.apache.parquet",
        size_bytes=size_bytes,
    )


def _runtime(thread_id: str = "test-thread") -> ToolRuntime[AgentContext]:
    return ToolRuntime(
        state={},
        context=AgentContext(thread_id=thread_id, user_id="test-user", language="pt"),
        config={},
        stream_writer=None,
        tool_call_id="test-tool-call",
        store=None,
    )


async def _ainvoke(tool, args: dict) -> ToolMessage:
    """Invoke an async tool the way the ToolNode does, returning the ToolMessage."""
    return await tool.ainvoke(
        {
            "type": "tool_call",
            "id": "1",
            "name": tool.name,
            "args": {**args, "runtime": _runtime()},
        }
    )


def _patch_db(monkeypatch, db: MagicMock) -> None:
    @asynccontextmanager
    async def mock_sessionmaker():
        yield None  # AsyncDatabase is mocked, so the session is never used

    monkeypatch.setattr("app.agent.tools.dataviz.sessionmaker", mock_sessionmaker)
    monkeypatch.setattr("app.agent.tools.dataviz.AsyncDatabase", lambda session: db)


def _handle(
    query_ref: str = "qr_1",
    slug: str = "resultado",
    age: timedelta = timedelta(0),
    message_id: uuid.UUID | None = None,
) -> QueryHandle:
    return QueryHandle(
        query_ref=query_ref,
        message_id=message_id or uuid.uuid4(),
        slug=slug,
        destination_table={"projectId": "p", "datasetId": "d", "tableId": "t"},
        created_at=datetime.now(timezone.utc) - age,
    )


class TestListQueryResults:
    async def test_lists_thread_handles_with_expiry_flag(self, monkeypatch):
        """Each thread handle is listed oldest-first with an expiry flag from its age."""
        fresh = _handle("qr_fresh", "recent", age=timedelta(hours=1))
        stale = _handle("qr_stale", "old", age=timedelta(hours=48))

        db = MagicMock()
        db.get_query_handles_by_thread = AsyncMock(return_value=[fresh, stale])
        _patch_db(monkeypatch, db)

        message = await _ainvoke(list_query_results, {})
        parsed = json.loads(message.content)

        assert [r["query_ref"] for r in parsed] == ["qr_fresh", "qr_stale"]
        assert parsed[0]["description"] == "recent"
        assert parsed[0]["expired"] is False
        assert parsed[1]["expired"] is True
        db.get_query_handles_by_thread.assert_awaited_once_with("test-thread")

    async def test_empty_when_no_results(self, monkeypatch):
        db = MagicMock()
        db.get_query_handles_by_thread = AsyncMock(return_value=[])
        _patch_db(monkeypatch, db)

        message = await _ainvoke(list_query_results, {})

        assert json.loads(message.content) == []


class TestExportQueryResult:
    async def test_returns_export_affordance(self, monkeypatch):
        """A valid, fresh handle yields an `export` artifact for the interface."""
        message_id = uuid.uuid4()
        handle = _handle(
            "qr_1", "resultado", age=timedelta(hours=1), message_id=message_id
        )

        db = MagicMock()
        db.get_query_handle_from_thread = AsyncMock(return_value=handle)
        _patch_db(monkeypatch, db)
        materialize = MagicMock(return_value=_exported(size_bytes=2048))
        monkeypatch.setattr(dataviz_module, "materialize_export", materialize)

        message = await _ainvoke(
            export_query_result, {"query_ref": "qr_1", "file_format": "parquet"}
        )

        # Case-insensitive format, thread-scoped resolution, client-facing artifact.
        db.get_query_handle_from_thread.assert_awaited_once_with("qr_1", "test-thread")
        # The file is materialized eagerly (so the card can show its size and filename),
        # with the slug sanitized into the base filename the endpoint would also produce.
        assert materialize.call_args.kwargs["file_format"] == "PARQUET"
        assert materialize.call_args.kwargs["message_id"] == str(message_id)
        assert materialize.call_args.kwargs["filename"] == "resultado"
        assert message.artifact == {
            "type": "export",
            "query_ref": "qr_1",
            "format": "PARQUET",
            "message_id": str(message_id),
            "filename": "resultado.parquet",
            "size_bytes": 2048,
        }
        content = json.loads(message.content)
        assert content["status"] == "ready"
        assert content["format"] == "PARQUET"
        assert content["filename"] == "resultado.parquet"
        assert content["size_bytes"] == 2048

    async def test_unsupported_format_is_rejected(self, monkeypatch):
        db = MagicMock()
        db.get_query_handle_from_thread = AsyncMock(return_value=_handle())
        _patch_db(monkeypatch, db)

        message = await _ainvoke(
            export_query_result, {"query_ref": "qr_1", "file_format": "XLSX"}
        )
        content = json.loads(message.content)

        assert content["status"] == "error"
        assert "Unsupported format" in content["message"]
        # The handle is never even looked up for an unsupported format.
        db.get_query_handle_from_thread.assert_not_awaited()

    async def test_missing_ref_is_rejected(self, monkeypatch):
        db = MagicMock()
        db.get_query_handle_from_thread = AsyncMock(return_value=None)
        _patch_db(monkeypatch, db)

        message = await _ainvoke(
            export_query_result, {"query_ref": "qr_missing", "file_format": "CSV"}
        )
        content = json.loads(message.content)

        assert content["status"] == "error"
        assert "No query result found" in content["message"]

    async def test_expired_ref_is_rejected(self, monkeypatch):
        handle = _handle("qr_old", "old", age=timedelta(hours=48))
        db = MagicMock()
        db.get_query_handle_from_thread = AsyncMock(return_value=handle)
        _patch_db(monkeypatch, db)

        message = await _ainvoke(
            export_query_result, {"query_ref": "qr_old", "file_format": "CSV"}
        )
        content = json.loads(message.content)

        assert content["status"] == "error"
        assert "expired" in content["message"]

    async def test_too_large_result_is_reported(self, monkeypatch):
        """An over-limit result fails at tool time, not as a click that 400s later."""
        db = MagicMock()
        db.get_query_handle_from_thread = AsyncMock(return_value=_handle())
        _patch_db(monkeypatch, db)
        monkeypatch.setattr(
            dataviz_module,
            "materialize_export",
            MagicMock(side_effect=ResultTooLarge("over the export limit")),
        )

        message = await _ainvoke(
            export_query_result, {"query_ref": "qr_1", "file_format": "CSV"}
        )
        content = json.loads(message.content)

        assert content["status"] == "error"
        assert "too large" in content["message"]

    async def test_expired_during_materialize_is_reported(self, monkeypatch):
        """The table can vanish between the age check and the extract; report it cleanly."""
        db = MagicMock()
        db.get_query_handle_from_thread = AsyncMock(return_value=_handle())
        _patch_db(monkeypatch, db)
        monkeypatch.setattr(
            dataviz_module,
            "materialize_export",
            MagicMock(side_effect=ResultTableExpired("gone")),
        )

        message = await _ainvoke(
            export_query_result, {"query_ref": "qr_1", "file_format": "CSV"}
        )
        content = json.loads(message.content)

        assert content["status"] == "error"
        assert "expired" in content["message"]


class TestChartQueryResult:
    async def test_renders_chart_artifact(self, monkeypatch):
        """The tool describes the chart; the generated spec is bound to the exact rows."""
        handle = _handle("qr_1", "vendas")
        rows = [{"ano": 2025, "total": 10}]
        monkeypatch.setattr(
            dataviz_module,
            "load_chart_source",
            AsyncMock(return_value=(handle, ["ano", "total"], rows)),
        )
        # The spec generator (validate-and-repair loop) is exercised in test_charts;
        # here it stands in for the returned, already-validated spec.
        monkeypatch.setattr(
            dataviz_module,
            "generate_chart_spec",
            AsyncMock(
                return_value={"mark": "bar", "encoding": {"x": {"field": "ano"}}}
            ),
        )

        message = await _ainvoke(
            chart_query_result,
            {"query_ref": "qr_1", "instructions": "a bar chart of total by year"},
        )

        assert message.artifact["type"] == "chart"
        assert message.artifact["query_ref"] == "qr_1"
        assert message.artifact["spec"]["mark"] == "bar"
        # The card renders the spec; the slug is not part of the artifact.
        assert "slug" not in message.artifact
        # The server binds the exact rows; the model never supplied them.
        assert message.artifact["spec"]["data"] == {"values": rows}
        content = json.loads(message.content)
        assert content["status"] == "rendered"
        assert content["row_count"] == 1
        # generate_chart_spec receives the columns/rows/instructions, not a spec.
        dataviz_module.generate_chart_spec.assert_awaited_once_with(
            ["ano", "total"], rows, "a bar chart of total by year"
        )

    async def test_too_large_result_is_reported(self, monkeypatch):
        monkeypatch.setattr(
            dataviz_module,
            "load_chart_source",
            AsyncMock(side_effect=ChartResultTooLarge("2500 rows, over the limit")),
        )

        message = await _ainvoke(
            chart_query_result,
            {"query_ref": "qr_big", "instructions": "a bar chart"},
        )
        content = json.loads(message.content)

        assert content["status"] == "error"
        assert "over the limit" in content["message"]
