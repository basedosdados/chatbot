import json
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.api_core.exceptions import NotFound
from langchain_core.messages import SystemMessage

from app import charts
from app.charts import (
    _CHART_SPEC_INSTRUCTIONS,
    MAX_CHART_SPEC_ATTEMPTS,
    VEGA_LITE_SCHEMA,
    ChartHandleNotFound,
    ChartResultTooLarge,
    ChartSpec,
    ChartSpecInvalid,
    _chart_spec_user_prompt,
    _collect,
    _fetch_rows,
    _geo_stub_node,
    _geo_url_node,
    _resolve_named_data,
    _sanitize_chart_spec,
    _validate_chart_spec,
    fetch_chart_data,
    generate_chart_spec,
    inject_chart_data,
)
from app.db.models import QueryHandle
from app.exports import ResultTableExpired
from app.settings import settings

DESTINATION = {"projectId": "p", "datasetId": "d", "tableId": "t"}


def _handle(query_ref="qr_1", slug="resultado", age=timedelta(0)):
    return QueryHandle(
        query_ref=query_ref,
        message_id=uuid.uuid4(),
        slug=slug,
        destination_table=DESTINATION,
        created_at=datetime.now(timezone.utc) - age,
    )


def _choropleth_spec() -> dict:
    """A states choropleth as the model would emit it (named sources, lookup join)."""
    return {
        "data": {"name": "brazil_states"},
        "transform": [
            {
                "lookup": "id",
                "from": {
                    "data": {"name": "query_result"},
                    "key": "sigla_uf",
                    "fields": ["valor"],
                },
            }
        ],
        "projection": {"type": "mercator"},
        "mark": "geoshape",
        "encoding": {
            "color": {"field": "valor", "type": "quantitative"},
            "tooltip": [{"field": "properties.name", "type": "nominal"}],
        },
    }


class TestSanitizeChartSpec:
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

        clean = _sanitize_chart_spec(raw)

        assert "data" not in clean
        assert "datasets" not in clean
        assert "data" not in clean["layer"][0]
        # No URL survives anywhere (external-fetch / SSRF vector).
        assert "url" not in json.dumps(clean)
        # The presentational parts are untouched.
        assert clean["layer"][0]["encoding"] == {"x": {"field": "ano"}}

    def test_keeps_allowlisted_named_sources(self):
        clean = _sanitize_chart_spec(_choropleth_spec())

        assert clean["data"] == {"name": "brazil_states"}
        assert clean["transform"][0]["from"]["data"] == {"name": "query_result"}

    def test_drops_unknown_named_source_inline_values_and_url(self):
        assert "data" not in _sanitize_chart_spec({"data": {"name": "secret"}})
        assert "data" not in _sanitize_chart_spec({"data": {"values": [{"x": 1}]}})
        assert "data" not in _sanitize_chart_spec({"data": {"url": "http://x"}})

    def test_strips_href_click_link_but_keeps_a_field_named_href(self):
        raw = {
            "mark": {"type": "point", "href": "javascript:alert(1)"},
            "encoding": {
                "href": {"field": "link"},  # the href channel is dropped
                "x": {"field": "href"},  # a column named "href" is a value, not a key
            },
        }

        clean = _sanitize_chart_spec(raw)

        assert "href" not in clean["mark"]
        assert "href" not in clean["encoding"]
        assert clean["encoding"]["x"] == {"field": "href"}
        # No clickable-link href survives anywhere (click-XSS vector).
        assert "javascript:" not in json.dumps(clean)


class TestFetchRows:
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

        columns, rows = _fetch_rows(DESTINATION)

        assert columns == ["col1"]
        assert rows == [{"col1": "a"}, {"col1": "b"}]

    def test_allows_a_dense_result_within_budget(self, mocker):
        """A high-cardinality result (e.g. a municipal choropleth ~5.5k rows) is allowed."""
        rows = [{"id": i} for i in range(6000)]
        client = self._client(rows, columns=("id",))
        mocker.patch("app.charts._bq_client", return_value=client)

        _, got = _fetch_rows(DESTINATION)

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
            _fetch_rows(DESTINATION)

    def test_coerces_non_json_types_to_json_native(self, mocker):
        """BigQuery date/datetime/Decimal values become JSON-serializable rows.

        A Decimal (NUMERIC/BIGNUMERIC) must become a float, not a string — a quantitative
        axis or colour scale binds these values, and a string breaks the numeric encoding.
        """
        rows = [
            {"data": date(2025, 12, 31), "temperatura_media": Decimal("24.79")},
        ]
        client = self._client(rows, columns=("data", "temperatura_media"))
        mocker.patch("app.charts._bq_client", return_value=client)

        _, got = _fetch_rows(DESTINATION)

        assert got == [{"data": "2025-12-31", "temperatura_media": 24.79}]
        assert isinstance(got[0]["temperatura_media"], float)
        # The bound rows must be plain JSON values, or spec serialization fails.
        json.dumps(got)

    def test_missing_table_maps_to_expired(self, mocker):
        client = MagicMock()
        client.get_table.side_effect = NotFound("gone")
        mocker.patch("app.charts._bq_client", return_value=client)

        with pytest.raises(ResultTableExpired):
            _fetch_rows(DESTINATION)


class TestGeoUrlNode:
    def test_points_at_the_static_geo_url(self):
        node = _geo_url_node("brazil_states")

        assert node == {
            "url": f"{settings.GEO_ASSET_URL_BASE}/brazil_states.topojson",
            "format": {"type": "topojson", "feature": "uf"},
        }

    def test_carries_no_inline_geometry(self):
        # The whole point: geometry travels by URL, never inline in the spec.
        node = _geo_url_node("brazil_municipalities")

        assert "values" not in node
        assert node["url"].endswith("/brazil_municipalities.topojson")


class TestGeoStubNode:
    def test_is_a_minimal_single_feature_topojson(self):
        node = _geo_stub_node("brazil_states")
        geometries = node["values"]["objects"]["uf"]["geometries"]

        assert node["format"] == {"type": "topojson", "feature": "uf"}
        assert node["values"]["type"] == "Topology"
        assert len(geometries) == 1

    def test_keys_the_stub_by_the_assets_feature(self):
        node = _geo_stub_node("brazil_municipalities")

        assert node["format"]["feature"] == "Munic"
        assert set(node["values"]["objects"]) == {"Munic"}


class TestResolveNamedData:
    def test_resolves_query_result_to_rows(self):
        rows = [{"sigla_uf": "SP", "valor": 1}]

        assert _resolve_named_data({"name": "query_result"}, rows, _geo_url_node) == {
            "values": rows
        }

    def test_resolves_a_geo_name_via_the_given_resolver(self):
        node = _resolve_named_data({"name": "brazil_states"}, [], _geo_url_node)

        assert node == _geo_url_node("brazil_states")

    def test_resolves_references_nested_anywhere(self):
        rows = [{"sigla_uf": "SP", "valor": 1}]
        spec = {"transform": [{"from": {"data": {"name": "query_result"}}}]}

        resolved = _resolve_named_data(spec, rows, _geo_url_node)

        assert resolved["transform"][0]["from"]["data"] == {"values": rows}

    def test_leaves_unknown_names_and_plain_nodes_untouched(self):
        # An unknown name is not resolved here (sanitize drops it earlier).
        assert _resolve_named_data({"name": "secret"}, [], _geo_url_node) == {
            "name": "secret"
        }
        # A node with no named source passes through unchanged.
        assert _resolve_named_data({"mark": "bar"}, [], _geo_url_node) == {
            "mark": "bar"
        }


class TestInjectChartData:
    def test_binds_rows_for_a_plain_chart(self):
        # inject_chart_data trusts an already-sanitized spec (see generate_chart_spec);
        # it does not strip — it binds the rows a plain chart declared no data for.
        spec = {"mark": "bar", "encoding": {"x": {"field": "ano"}}}
        rows = [{"ano": 2025, "total": 10}]

        chart = inject_chart_data(spec, rows)

        assert chart["$schema"] == VEGA_LITE_SCHEMA
        assert chart["data"] == {"values": rows}
        assert chart["mark"] == "bar"

    def test_resolves_named_sources_in_a_choropleth(self):
        rows = [{"sigla_uf": "SP", "valor": 1}]

        chart = inject_chart_data(_choropleth_spec(), rows)

        # Top-level geometry becomes a URL node (fetched client-side); the rows fill the
        # lookup source inline.
        assert chart["data"] == _geo_url_node("brazil_states")
        assert chart["transform"][0]["from"]["data"] == {"values": rows}


class TestFetchChartData:
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
            charts, "_fetch_rows", lambda dest: (["ano"], [{"ano": 2025}])
        )

        got_handle, columns, rows = await fetch_chart_data("qr_1", "test-thread")

        assert got_handle is handle
        assert columns == ["ano"]
        assert rows == [{"ano": 2025}]

    async def test_missing_handle_raises(self, monkeypatch):
        self._patch_db(monkeypatch, None)

        with pytest.raises(ChartHandleNotFound):
            await fetch_chart_data("qr_missing", "test-thread")

    async def test_expired_handle_raises(self, monkeypatch):
        self._patch_db(monkeypatch, _handle(age=timedelta(hours=48)))

        with pytest.raises(ResultTableExpired):
            await fetch_chart_data("qr_old", "test-thread")


class TestCollect:
    def test_collects_string_values_under_key_recursively(self):
        node = {
            "encoding": {"x": {"field": "ano"}, "y": {"field": "total"}},
            "layer": [{"encoding": {"color": {"field": "uf"}}}],
        }

        assert _collect(node, "field") == {"ano", "total", "uf"}

    def test_ignores_non_string_values_under_the_key(self):
        assert _collect({"field": 5, "x": {"field": "ano"}}, "field") == {"ano"}


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

    def test_unknown_color_scheme_fails_to_compile(self):
        """An invalid scheme (d3's RdBu) is rejected by the compile step, like any bad value."""
        spec = {
            "mark": "rect",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "mes", "type": "ordinal"},
                "color": {
                    "field": "temp",
                    "type": "quantitative",
                    "scale": {"scheme": "RdBu"},
                },
            },
        }

        errors = _validate_chart_spec(spec, ["ano", "mes", "temp"])

        assert any("compile" in error for error in errors)

    def test_choropleth_spec_is_valid(self):
        """A choropleth compiles (real geometry) and its geo properties are allowed."""
        assert _validate_chart_spec(_choropleth_spec(), ["sigla_uf", "valor"]) == []


class TestChartSpecUserPrompt:
    def test_carries_task_data_on_the_first_attempt(self):
        prompt = _chart_spec_user_prompt(
            ["ano", "total"], [{"ano": 2025, "total": 10}], "a bar chart", None, []
        )

        assert "a bar chart" in prompt
        assert "ano" in prompt and "total" in prompt
        assert "rejected" not in prompt  # no retry feedback on the first attempt

    def test_echoes_the_rejected_spec_and_errors_on_retry(self):
        prompt = _chart_spec_user_prompt(
            ["ano"], [{"ano": 2025}], "a bar chart", {"mark": "nope"}, ["bad column"]
        )

        assert '"nope"' in prompt  # the model's own rejected spec, echoed back
        assert "bad column" in prompt  # the validator's reason
        assert "rejected" in prompt


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
        assert system_message.content == _CHART_SPEC_INSTRUCTIONS
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
        messages = [_structured_reply(spec) for _ in range(MAX_CHART_SPEC_ATTEMPTS)]
        model = self._model(monkeypatch, *messages)
        monkeypatch.setattr(
            charts, "_validate_chart_spec", lambda spec, columns: ["always bad"]
        )

        with pytest.raises(ChartSpecInvalid):
            await generate_chart_spec(["ano"], [{"ano": 2025}], "a bar chart")

        assert model.ainvoke.await_count == MAX_CHART_SPEC_ATTEMPTS
