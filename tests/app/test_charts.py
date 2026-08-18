import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.api_core.exceptions import NotFound
from langchain_core.messages import SystemMessage

from app import charts
from app.charts import (
    VEGA_LITE_SCHEMA,
    ChartHandleNotFound,
    ChartResultTooLarge,
    ChartSpec,
    ChartSpecInvalid,
    _validate_chart_spec,
    build_chart_spec,
    generate_chart_spec,
    load_chart_source,
    read_chart_data,
)
from app.db.models import QueryHandle
from app.exports import ResultTableExpired

DESTINATION = {"projectId": "p", "datasetId": "d", "tableId": "t"}


def _handle(query_ref="qr_1", slug="resultado", age=timedelta(0)):
    return QueryHandle(
        query_ref=query_ref,
        message_id=uuid.uuid4(),
        slug=slug,
        destination_table=DESTINATION,
        created_at=datetime.now(timezone.utc) - age,
    )


class TestStripUntrusted:
    def test_strips_data_datasets_and_urls_recursively(self):
        raw = {
            "data": {"url": "https://evil.example/top.json"},
            "datasets": {"foo": [1, 2, 3]},
            "layer": [
                {
                    "mark": "line",
                    "encoding": {"x": {"field": "ano"}},
                    "data": {"url": "https://evil.example/layer.json"},
                }
            ],
        }

        clean = charts._strip_untrusted(raw)

        assert "data" not in clean
        assert "datasets" not in clean
        assert "data" not in clean["layer"][0]
        # No URL survives anywhere (external-fetch / SSRF vector).
        assert "url" not in json.dumps(clean)
        # The presentational parts are untouched.
        assert clean["layer"][0]["encoding"] == {"x": {"field": "ano"}}


class TestReadChartData:
    def _client(self, rows, columns=("col1",)):
        client = MagicMock()
        client.get_table.return_value = SimpleNamespace(
            schema=[SimpleNamespace(name=name) for name in columns],
        )
        client.list_rows.return_value = iter(rows)
        return client

    def test_reads_columns_and_rows(self, mocker):
        client = self._client([{"col1": "a"}, {"col1": "b"}])
        mocker.patch("app.charts._bq_client", return_value=client)

        columns, rows = read_chart_data(DESTINATION)

        assert columns == ["col1"]
        assert rows == [{"col1": "a"}, {"col1": "b"}]

    def test_allows_a_dense_result_within_budget(self, mocker):
        """A high-cardinality result (e.g. a municipal choropleth ~5.5k rows) is allowed."""
        rows = [{"id": i} for i in range(6000)]
        client = self._client(rows, columns=("id",))
        mocker.patch("app.charts._bq_client", return_value=client)

        _, got = read_chart_data(DESTINATION)

        assert len(got) == 6000

    def test_raises_when_over_byte_budget(self, mocker, monkeypatch):
        """The bound data is rejected once it would exceed the payload budget."""
        monkeypatch.setattr(
            charts,
            "settings",
            charts.settings.model_copy(update={"CHART_MAX_BYTES": 32}),
        )
        client = self._client([{"col1": "x" * 100}, {"col1": "y" * 100}])
        mocker.patch("app.charts._bq_client", return_value=client)

        with pytest.raises(ChartResultTooLarge):
            read_chart_data(DESTINATION)

    def test_missing_table_maps_to_expired(self, mocker):
        client = MagicMock()
        client.get_table.side_effect = NotFound("gone")
        mocker.patch("app.charts._bq_client", return_value=client)

        with pytest.raises(ResultTableExpired):
            read_chart_data(DESTINATION)


class TestBuildChartSpec:
    def test_binds_rows_and_strips_untrusted(self):
        spec = {
            "mark": "bar",
            "encoding": {"x": {"field": "ano"}},
            "data": {"url": "https://evil.example/x.json"},
        }
        rows = [{"ano": 2025, "total": 10}]

        chart = build_chart_spec(spec, rows)

        assert chart["$schema"] == VEGA_LITE_SCHEMA
        # Model-supplied data is stripped; the server binds the real rows.
        assert chart["data"] == {"values": rows}
        assert chart["mark"] == "bar"


class TestLoadChartSource:
    def _patch_db(self, monkeypatch, handle):
        db = MagicMock()
        db.get_query_handle_from_thread = AsyncMock(return_value=handle)

        @asynccontextmanager
        async def mock_sessionmaker():
            yield None

        monkeypatch.setattr(charts, "sessionmaker", mock_sessionmaker)
        monkeypatch.setattr(charts, "AsyncDatabase", lambda session: db)
        return db

    async def test_returns_handle_columns_and_rows(self, monkeypatch):
        handle = _handle(age=timedelta(hours=1))
        self._patch_db(monkeypatch, handle)
        monkeypatch.setattr(
            charts, "read_chart_data", lambda dest: (["ano"], [{"ano": 2025}])
        )

        got_handle, columns, rows = await load_chart_source("qr_1", "test-thread")

        assert got_handle is handle
        assert columns == ["ano"]
        assert rows == [{"ano": 2025}]

    async def test_missing_handle_raises(self, monkeypatch):
        self._patch_db(monkeypatch, None)

        with pytest.raises(ChartHandleNotFound):
            await load_chart_source("qr_missing", "test-thread")

    async def test_expired_handle_raises(self, monkeypatch):
        self._patch_db(monkeypatch, _handle(age=timedelta(hours=48)))

        with pytest.raises(ResultTableExpired):
            await load_chart_source("qr_old", "test-thread")


class TestValidateChartSpec:
    """Uses the real vl-convert compiler (in-process)."""

    def test_valid_spec_has_no_errors(self):
        spec = {
            "mark": "bar",
            "encoding": {"x": {"field": "ano"}, "y": {"field": "total"}},
        }
        assert _validate_chart_spec(spec, ["ano", "total"]) == []

    def test_layered_dual_axis_combo_is_valid(self):
        """A layer + resolve combo — the case the old allowlist could not express."""
        spec = {
            "layer": [
                {
                    "mark": "line",
                    "encoding": {"y": {"field": "media", "type": "quantitative"}},
                },
                {
                    "mark": "bar",
                    "encoding": {"y": {"field": "var", "type": "quantitative"}},
                },
            ],
            "encoding": {"x": {"field": "ano", "type": "ordinal"}},
            "resolve": {"scale": {"y": "independent"}},
        }
        assert _validate_chart_spec(spec, ["ano", "media", "var"]) == []

    def test_missing_column_flagged_across_layers(self):
        """A field not in the result, even nested in a layer — else it renders empty."""
        spec = {"layer": [{"mark": "bar", "encoding": {"x": {"field": "vendas"}}}]}

        errors = _validate_chart_spec(spec, ["ano", "total"])

        assert any("vendas" in error for error in errors)

    def test_transform_derived_field_is_allowed(self):
        """A field created by a transform's `as` is not flagged as missing."""
        spec = {
            "transform": [{"calculate": "datum.total * 2", "as": "dobro"}],
            "mark": "bar",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "dobro", "type": "quantitative"},
            },
        }
        assert _validate_chart_spec(spec, ["ano", "total"]) == []

    def test_invalid_mark_fails_to_compile(self):
        """A structurally invalid spec is caught by the vl-convert compile step."""
        spec = {"mark": "notamark", "encoding": {"x": {"field": "ano"}}}

        errors = _validate_chart_spec(spec, ["ano"])

        assert any("compile" in error for error in errors)


def _structured_reply(spec: dict | None) -> dict:
    """A `with_structured_output(include_raw=True)` result: a parsed spec, or a parse miss."""
    parsed = ChartSpec(spec=spec) if spec is not None else None
    return {"raw": MagicMock(), "parsed": parsed, "parsing_error": None}


class TestGenerateChartSpec:
    def _model(self, monkeypatch, *messages):
        model = MagicMock()
        model.ainvoke = AsyncMock(side_effect=list(messages))
        monkeypatch.setattr(charts, "_chart_spec_model", lambda: model)
        return model

    async def test_retries_until_valid(self, monkeypatch):
        """A spec that fails validation is regenerated with the errors fed back."""
        bad = {"mark": "bar", "encoding": {"x": {"field": "nope"}}}
        good = {"mark": "bar", "encoding": {"x": {"field": "ano"}}}
        model = self._model(
            monkeypatch, _structured_reply(bad), _structured_reply(good)
        )
        monkeypatch.setattr(
            charts,
            "_validate_chart_spec",
            lambda spec, columns: []
            if spec["encoding"] == good["encoding"]
            else ["bad column"],
        )

        result = await generate_chart_spec(["ano"], [{"ano": 2025}], "a bar chart")

        assert result["encoding"] == good["encoding"]
        assert model.ainvoke.await_count == 2

    async def test_rejected_spec_is_fed_back_on_retry(self, monkeypatch):
        """The retry prompt carries the model's own rejected spec plus the errors."""
        bad = {"mark": "bar", "encoding": {"x": {"field": "nope"}}}
        good = {"mark": "bar", "encoding": {"x": {"field": "ano"}}}
        model = self._model(
            monkeypatch, _structured_reply(bad), _structured_reply(good)
        )
        monkeypatch.setattr(
            charts,
            "_validate_chart_spec",
            lambda spec, columns: []
            if spec["encoding"] == good["encoding"]
            else ["column 'nope' is not in the result"],
        )

        await generate_chart_spec(["ano"], [{"ano": 2025}], "a bar chart")

        system_message, user_message = model.ainvoke.await_args_list[1].args[0]
        # The durable how-to is a system message; the retry data rides the user message.
        assert isinstance(system_message, SystemMessage)
        assert system_message.content == charts._CHART_SPEC_INSTRUCTIONS
        assert '"nope"' in user_message.content  # its own rejected spec, echoed back
        assert "not in the result" in user_message.content  # the validator's reason

    async def test_model_supplied_data_is_stripped(self, monkeypatch):
        """A model spec carrying data/url is sanitized (not rejected) before returning."""
        spec = {
            "mark": "bar",
            "encoding": {"x": {"field": "ano"}},
            "data": {"url": "https://evil.example/x.json"},
        }
        model = self._model(monkeypatch, _structured_reply(spec))
        monkeypatch.setattr(charts, "_validate_chart_spec", lambda spec, columns: [])

        result = await generate_chart_spec(["ano"], [{"ano": 2025}], "a bar chart")

        assert "data" not in result
        assert "url" not in json.dumps(result)
        assert model.ainvoke.await_count == 1

    async def test_missing_spec_is_retried(self, monkeypatch):
        """A reply the parser could not turn into a spec is treated as a failure and retried."""
        empty = _structured_reply(None)
        good = _structured_reply({"mark": "bar", "encoding": {"x": {"field": "ano"}}})
        model = self._model(monkeypatch, empty, good)
        monkeypatch.setattr(charts, "_validate_chart_spec", lambda spec, columns: [])

        result = await generate_chart_spec(["ano"], [{"ano": 2025}], "a bar chart")

        assert result["mark"] == "bar"
        assert model.ainvoke.await_count == 2

    async def test_raises_after_max_attempts(self, monkeypatch):
        spec = {"mark": "bar", "encoding": {}}
        messages = [
            _structured_reply(spec) for _ in range(charts.MAX_CHART_SPEC_ATTEMPTS)
        ]
        model = self._model(monkeypatch, *messages)
        monkeypatch.setattr(
            charts, "_validate_chart_spec", lambda spec, columns: ["always bad"]
        )

        with pytest.raises(ChartSpecInvalid):
            await generate_chart_spec(["ano"], [{"ano": 2025}], "a bar chart")

        assert model.ainvoke.await_count == charts.MAX_CHART_SPEC_ATTEMPTS
