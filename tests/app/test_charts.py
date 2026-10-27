import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import vl_convert as vlc
from google.api_core.exceptions import NotFound
from langchain_core.messages import SystemMessage

from app import charts
from app.charts import (
    _CHART_SPEC_INSTRUCTIONS,
    _D3_NUMBER_FORMAT,
    MAX_CHART_SPEC_ATTEMPTS,
    VEGA_LITE_SCHEMA,
    ChartHandleNotFound,
    ChartResultTooLarge,
    ChartSpec,
    ChartSpecInvalid,
    ChartSpecTimeout,
    _chart_spec_user_prompt,
    _collect,
    _collect_derived_fields,
    _collect_invisible_white_marks,
    _collect_number_formats,
    _collect_scheme_names,
    _collect_split_lines,
    _collect_temporal_fields,
    _collect_unlabeled_layer_series,
    _collect_unsupported_encoding_properties,
    _describe_result,
    _fetch_rows,
    _geo_stub_node,
    _geo_url_node,
    _integer_year_columns,
    _is_valid_number_format,
    _is_whiteish,
    _normalize_chart_spec,
    _resolve_named_data,
    _sanitize_chart_spec,
    _spread_sample,
    _summarize_column,
    _validate_chart_spec,
    fetch_chart_data,
    generate_chart_spec,
    inject_chart_data,
    unbind_chart_spec,
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


class TestSanitizeDataGenerators:
    @pytest.mark.parametrize(
        "generator",
        [
            {"sequence": {"start": 0, "stop": 10, "step": 1, "as": "x"}},
            {"graticule": {"step": [15, 15]}},
            {"sphere": True},
        ],
    )
    def test_generators_are_kept(self, generator):
        assert _sanitize_chart_spec({"data": generator}) == {"data": generator}

    def test_a_generator_name_is_dropped(self):
        """A named generator could otherwise shadow the `query_result` dataset."""
        spec = {"data": {"sequence": {"start": 0, "stop": 3}, "name": "query_result"}}

        assert _sanitize_chart_spec(spec) == {
            "data": {"sequence": {"start": 0, "stop": 3}}
        }

    def test_a_generator_mixed_with_inline_data_is_dropped(self):
        spec = {"data": {"sphere": True, "values": [{"a": 1}]}}

        assert _sanitize_chart_spec(spec) == {}

    def test_a_sequence_chart_validates(self):
        spec = {
            "data": {"sequence": {"start": 0, "stop": 5}},
            "mark": "line",
            "encoding": {"x": {"field": "data", "type": "quantitative"}},
        }

        assert _validate_chart_spec(_sanitize_chart_spec(spec), []) == []


class TestFetchRows:
    def _client(self, rows, columns=("col1",)):
        client = MagicMock()
        client.get_table.return_value = SimpleNamespace(
            schema=[
                SimpleNamespace(name=name, field_type="STRING") for name in columns
            ],
        )
        client.list_rows.return_value = iter(rows)
        return client

    def test_reads_columns_and_rows(self, mocker):
        client = self._client([{"col1": "a"}, {"col1": "b"}])
        mocker.patch("app.charts._bq_client", return_value=client)

        columns, rows = _fetch_rows(DESTINATION)

        assert columns == {"col1": "STRING"}
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

    def test_whole_number_decimal_becomes_int(self, mocker):
        """A whole-number Decimal becomes an int, not a float.

        A NUMERIC year (`Decimal("2019")`) must arrive as 2019, not 2019.0,
        or `_integer_year_columns` would not recognize the column as years.
        """
        rows = [{"ano": Decimal("2019"), "total": Decimal("10.00")}]
        client = self._client(rows, columns=("ano", "total"))
        mocker.patch("app.charts._bq_client", return_value=client)

        columns, got = _fetch_rows(DESTINATION)

        assert got == [{"ano": 2019, "total": 10}]
        assert all(isinstance(value, int) for value in got[0].values())
        assert _integer_year_columns(list(columns), got) == {"ano"}

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
            "url": f"{settings.GEO_ASSET_URL_BASE}/brasil_estados.topojson",
            "format": {"type": "topojson", "feature": "uf"},
        }

    def test_carries_no_inline_geometry(self):
        # The whole point: geometry travels by URL, never inline in the spec.
        node = _geo_url_node("brazil_municipalities")

        assert "values" not in node
        assert node["url"].endswith("/brasil_municipios.topojson")


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
    def test_resolves_a_geo_name_via_the_given_resolver(self):
        node = _resolve_named_data({"name": "brazil_states"}, _geo_url_node)

        assert node == _geo_url_node("brazil_states")

    def test_resolves_geo_references_nested_anywhere(self):
        spec = {"layer": [{"data": {"name": "brazil_municipalities"}}]}

        resolved = _resolve_named_data(spec, _geo_stub_node)

        assert resolved["layer"][0]["data"] == _geo_stub_node("brazil_municipalities")

    def test_keeps_query_result_references(self):
        """They read the rows bound once under top-level `datasets`."""
        spec = {"transform": [{"from": {"data": {"name": "query_result"}}}]}

        assert _resolve_named_data(spec, _geo_url_node) == spec

    def test_leaves_unknown_names_and_plain_nodes_untouched(self):
        # An unknown name is not resolved here (sanitize drops it earlier).
        assert _resolve_named_data({"name": "secret"}, _geo_url_node) == {
            "name": "secret"
        }
        # A node with no named source passes through unchanged.
        assert _resolve_named_data({"mark": "bar"}, _geo_url_node) == {"mark": "bar"}


class TestInjectChartData:
    def test_binds_rows_for_a_plain_chart(self):
        # inject_chart_data trusts an already-sanitized spec (see generate_chart_spec);
        # it does not strip — it binds the rows a plain chart declared no data for.
        spec = {"mark": "bar", "encoding": {"x": {"field": "ano"}}}
        rows = [{"ano": 2025, "total": 10}]

        chart = inject_chart_data(spec, rows)

        assert chart["$schema"] == VEGA_LITE_SCHEMA
        assert chart["datasets"] == {"query_result": rows}
        assert chart["data"] == {"name": "query_result"}
        assert chart["mark"] == "bar"

    def test_resolves_named_sources_in_a_choropleth(self):
        rows = [{"sigla_uf": "SP", "valor": 1}]

        chart = inject_chart_data(_choropleth_spec(), rows)

        # Top-level geometry becomes a URL node (fetched client-side); the lookup reads
        # the rows from the bound dataset.
        assert chart["data"] == _geo_url_node("brazil_states")
        assert chart["transform"][0]["from"]["data"] == {"name": "query_result"}
        assert chart["datasets"] == {"query_result": rows}

    def test_rows_appear_once_however_many_views_use_them(self):
        rows = [{"ano": 2025, "total": 10}]
        spec = {
            "layer": [
                {"data": {"name": "query_result"}, "mark": "bar"},
                {"data": {"name": "query_result"}, "mark": "line"},
            ]
        }

        chart = inject_chart_data(spec, rows)

        assert json.dumps(chart).count('"total": 10') == 1

    def test_bound_chart_renders(self):
        rows = [{"ano": 2024, "total": 10}, {"ano": 2025, "total": 12}]
        spec = {
            "layer": [
                {"data": {"name": "query_result"}, "mark": "bar"},
                {"mark": "line"},
            ],
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "total", "type": "quantitative"},
            },
        }

        svg = vlc.vegalite_to_svg(json.dumps(inject_chart_data(spec, rows)))

        assert "<path" in svg


class TestUnbindChartSpec:
    def test_round_trips_a_plain_chart(self):
        spec = {"mark": "bar", "encoding": {"x": {"field": "ano"}}}

        unbound = unbind_chart_spec(inject_chart_data(spec, [{"ano": 2025}]))

        assert unbound == {**spec, "data": {"name": "query_result"}}

    def test_round_trips_a_choropleth(self):
        unbound = unbind_chart_spec(
            inject_chart_data(_choropleth_spec(), [{"sigla_uf": "SP", "valor": 1}])
        )

        assert unbound["data"] == {"name": "brazil_states"}
        assert unbound["transform"][0]["from"]["data"] == {"name": "query_result"}
        assert "datasets" not in unbound and "$schema" not in unbound


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
            charts, "_fetch_rows", lambda dest: ({"ano": "INTEGER"}, [{"ano": 2025}])
        )

        got_handle, columns, rows = await fetch_chart_data("qr_1", "test-thread")

        assert got_handle is handle
        assert columns == {"ano": "INTEGER"}
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


class TestCollectDerivedFields:
    def test_collects_string_as(self):
        spec = {"transform": [{"calculate": "datum.total * 2", "as": "dobro"}]}

        assert _collect_derived_fields(spec) == {"dobro"}

    def test_collects_list_as(self):
        """`fold`, `stack`, `regression`, … name several outputs in a list."""
        spec = {
            "transform": [{"fold": ["homens", "mulheres"], "as": ["sexo", "valor"]}]
        }

        assert _collect_derived_fields(spec) == {"sexo", "valor"}

    def test_collects_as_inside_aggregate_ops(self):
        spec = {
            "transform": [
                {
                    "aggregate": [{"op": "sum", "field": "v", "as": "total"}],
                    "groupby": ["uf"],
                }
            ]
        }

        assert _collect_derived_fields(spec) == {"total"}

    def test_bin_transform_also_creates_end_field(self):
        spec = {"transform": [{"bin": True, "field": "renda", "as": "faixa"}]}

        assert _collect_derived_fields(spec) == {"faixa", "faixa_end"}

    @pytest.mark.parametrize(
        "transform, expected",
        [
            ({"fold": ["a", "b"]}, {"key", "value"}),
            ({"density": "renda"}, {"value", "density"}),
            ({"quantile": "renda"}, {"prob", "value"}),
        ],
    )
    def test_default_outputs_when_as_is_omitted(self, transform, expected):
        assert _collect_derived_fields({"transform": [transform]}) == expected

    def test_collects_across_layers(self):
        spec = {
            "layer": [
                {"transform": [{"calculate": "1", "as": "a"}]},
                {"transform": [{"regression": "y", "on": "x", "as": ["rx", "ry"]}]},
            ]
        }

        assert _collect_derived_fields(spec) == {"a", "rx", "ry"}


class TestCollectUnsupportedEncodingProperties:
    def test_valid_channels_have_no_problems(self):
        spec = {
            "mark": "bar",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal", "axis": {"labelAngle": 0}},
                "y": {
                    "field": "total",
                    "type": "quantitative",
                    "axis": {"format": "~s"},
                },
                "color": {"datum": "Homens", "type": "nominal"},
                "opacity": {"value": 0.5},
                "tooltip": [
                    {"field": "total", "type": "quantitative", "format": ",.0f"}
                ],
                "text": {
                    "condition": {
                        "test": "datum.ano === 2020",
                        "value": {"expr": "'x'"},
                    },
                    "value": "",
                },
            },
        }

        assert _collect_unsupported_encoding_properties(spec) == []

    def test_channels_nested_inside_another_channel_are_reported(self):
        """The production case: `color`/`tooltip` ended up inside `y`, so a series lost its color."""
        spec = {
            "layer": [
                {"mark": "bar", "encoding": {"x": {"field": "ano", "type": "ordinal"}}},
                {
                    "mark": "bar",
                    "encoding": {
                        "y": {
                            "field": "mulheres",
                            "type": "quantitative",
                            "color": {"datum": "Mulheres", "type": "nominal"},
                            "tooltip": [{"field": "mulheres"}],
                        }
                    },
                },
            ]
        }

        assert _collect_unsupported_encoding_properties(spec) == [
            "layer[1].encoding.y has unsupported ['color', 'tooltip']"
        ]

    def test_bare_expr_on_text_is_reported(self):
        """`text: {"expr": ...}` is ignored; a computed text goes in `value: {"expr": ...}`."""
        spec = {"mark": "text", "encoding": {"text": {"expr": "'R$ ' + datum.v"}}}

        assert _collect_unsupported_encoding_properties(spec) == [
            "encoding.text has unsupported ['expr']"
        ]

    def test_bare_expr_in_condition_is_reported_without_blaming_value(self):
        """`value` beside a `condition` is valid; the misplaced `expr` is the problem."""
        spec = {
            "mark": "text",
            "encoding": {
                "text": {
                    "condition": {
                        "test": "datum.ano === 2020",
                        "expr": "'R$ ' + datum.v",
                    },
                    "value": "",
                }
            },
        }

        problems = _collect_unsupported_encoding_properties(spec)

        assert any(
            "encoding.text.condition has unsupported ['expr']" in p for p in problems
        )
        assert not any("['value']" in p for p in problems)

    def test_nested_unsupported_property_is_located(self):
        spec = {
            "mark": "bar",
            "encoding": {
                "y": {"field": "v", "type": "quantitative", "axis": {"tooltip": True}}
            },
        }

        assert _collect_unsupported_encoding_properties(spec) == [
            "encoding.y.axis has unsupported ['tooltip']"
        ]

    def test_unknown_channel_is_reported(self):
        spec = {
            "mark": "bar",
            "encoding": {"config": {"axis": {"labelColor": "white"}}},
        }

        assert _collect_unsupported_encoding_properties(spec) == [
            "encoding.config is not a Vega-Lite encoding channel"
        ]

    def test_facet_channels_are_allowed(self):
        spec = {"mark": "bar", "encoding": {"row": {"field": "uf", "type": "nominal"}}}

        assert _collect_unsupported_encoding_properties(spec) == []


class TestNormalizeChartSpec:
    def test_drops_the_point_overlay_of_a_dashed_line(self):
        """Its legend would draw point symbols, which cannot show the dash."""
        spec = {
            "mark": {"type": "line", "point": True},
            "encoding": {"strokeDash": {"field": "indicador", "type": "nominal"}},
        }

        assert _normalize_chart_spec(spec)["mark"] == {"type": "line"}

    def test_keeps_the_point_overlay_without_a_dash(self):
        spec = {
            "mark": {"type": "line", "point": True},
            "encoding": {"color": {"field": "governo", "type": "nominal"}},
        }

        assert _normalize_chart_spec(spec)["mark"] == {"type": "line", "point": True}

    def test_a_dash_inherited_from_the_layer_counts(self):
        spec = {
            "encoding": {"strokeDash": {"field": "grupo", "type": "nominal"}},
            "layer": [{"mark": {"type": "line", "point": True}}],
        }

        assert _normalize_chart_spec(spec)["layer"][0]["mark"] == {"type": "line"}


def _line_by(field: str) -> dict:
    return {
        "mark": "line",
        "encoding": {
            "x": {"field": "ano", "type": "ordinal"},
            "y": {"field": "valor", "type": "quantitative"},
            "color": {"field": field, "type": "nominal"},
        },
    }


class TestCollectSplitLines:
    def test_a_highlight_category_inside_the_span_splits_the_line(self):
        """The production case: the one surplus year dropped out of the deficit line."""
        rows = [
            {"ano": ano, "valor": -1, "tipo": "Déficit" if ano != 2022 else "Superávit"}
            for ano in range(2019, 2026)
        ]

        assert _collect_split_lines(_line_by("tipo"), rows) == ["tipo"]

    def test_groups_side_by_side_are_not_split(self):
        """Two governments, one after the other, are two intentional lines."""
        rows = [
            {"ano": ano, "valor": 1, "governo": "A" if ano < 2023 else "B"}
            for ano in range(2019, 2027)
        ]

        assert _collect_split_lines(_line_by("governo"), rows) == []

    def test_parallel_series_over_the_same_x_are_not_split(self):
        rows = [
            {"ano": ano, "valor": 1, "uf": uf}
            for ano in range(2019, 2026)
            for uf in "AB"
        ]

        assert _collect_split_lines(_line_by("uf"), rows) == []

    def test_derived_grouping_fields_are_not_checked(self):
        rows = [{"ano": 2020, "valor": 1}]

        assert _collect_split_lines(_line_by("calculado"), rows) == []


class TestCollectUnlabeledLayerSeries:
    def test_layers_with_fixed_colors_and_no_color_encoding_are_reported(self):
        spec = {
            "encoding": {"x": {"field": "mes", "type": "ordinal"}},
            "layer": [
                {
                    "mark": {"type": "line", "color": "#1f77b4"},
                    "encoding": {"y": {"field": "abertas"}},
                },
                {
                    "mark": {"type": "line", "color": "#d62728"},
                    "encoding": {"y": {"field": "fechadas"}},
                },
            ],
        }

        assert _collect_unlabeled_layer_series(spec) == [["abertas", "fechadas"]]

    def test_layers_labeled_with_a_color_datum_are_fine(self):
        spec = {
            "layer": [
                {
                    "mark": "line",
                    "encoding": {
                        "y": {"field": "abertas"},
                        "color": {"datum": "Abertas"},
                    },
                },
                {
                    "mark": "line",
                    "encoding": {
                        "y": {"field": "fechadas"},
                        "color": {"datum": "Fechadas"},
                    },
                },
            ]
        }

        assert _collect_unlabeled_layer_series(spec) == []

    def test_a_series_with_a_reference_rule_is_fine(self):
        spec = {
            "layer": [
                {"mark": "bar", "encoding": {"y": {"field": "valor"}}},
                {"mark": "rule", "encoding": {"y": {"datum": 0}}},
                {
                    "mark": "text",
                    "encoding": {"y": {"field": "valor"}, "text": {"field": "valor"}},
                },
            ]
        }

        assert _collect_unlabeled_layer_series(spec) == []


class TestIsWhiteish:
    @pytest.mark.parametrize(
        "color", ["white", "#fff", "#FFFFFF", "#f5f5f5", "rgb(255, 255, 255)"]
    )
    def test_white_like_colors(self, color):
        assert _is_whiteish(color)

    @pytest.mark.parametrize(
        "color", ["#2B8C4D", "black", "#1f2937", "red", None, {"expr": "x"}]
    )
    def test_visible_colors(self, color):
        assert not _is_whiteish(color)


class TestCollectInvisibleWhiteMarks:
    def test_a_white_line_on_the_default_background_is_reported(self):
        spec = {"mark": {"type": "line", "stroke": "white"}}

        assert _collect_invisible_white_marks(spec) == ["line"]

    def test_a_white_color_value_is_reported(self):
        spec = {"mark": "bar", "encoding": {"color": {"value": "#ffffff"}}}

        assert _collect_invisible_white_marks(spec) == ["bar"]

    def test_a_dark_background_makes_white_visible(self):
        spec = {"background": "#1f2937", "mark": {"type": "line", "color": "white"}}

        assert _collect_invisible_white_marks(spec) == []

    def test_white_text_over_bars_is_fine(self):
        """Labels inside bars are commonly white."""
        spec = {
            "layer": [
                {"mark": "bar", "encoding": {"y": {"field": "v"}}},
                {
                    "mark": {"type": "text", "color": "white"},
                    "encoding": {"text": {"field": "v"}},
                },
            ]
        }

        assert _collect_invisible_white_marks(spec) == []

    def test_white_text_over_a_line_is_reported(self):
        spec = {
            "layer": [
                {"mark": "line"},
                {
                    "mark": {"type": "text", "color": "white"},
                    "encoding": {"text": {"field": "v"}},
                },
            ]
        }

        assert _collect_invisible_white_marks(spec) == ["text"]


class TestCollectSchemeNames:
    def test_collects_a_string_scheme(self):
        spec = {"encoding": {"color": {"field": "v", "scale": {"scheme": "reds"}}}}

        assert _collect_scheme_names(spec) == {"reds"}

    def test_collects_a_scheme_params_object_by_name(self):
        # `scale.scheme` may be a SchemeParams object; the name lives under `name`.
        spec = {
            "encoding": {
                "color": {
                    "field": "v",
                    "scale": {"scheme": {"name": "viridis", "extent": [0.2, 1]}},
                }
            }
        }

        assert _collect_scheme_names(spec) == {"viridis"}

    def test_collects_every_scheme_across_layers(self):
        spec = {
            "layer": [
                {"encoding": {"color": {"scale": {"scheme": "reds"}}}},
                {"encoding": {"color": {"scale": {"scheme": {"name": "blues"}}}}},
            ]
        }

        assert _collect_scheme_names(spec) == {"reds", "blues"}

    def test_returns_empty_when_no_scheme(self):
        spec = {
            "mark": "bar",
            "encoding": {"color": {"scale": {"range": ["red", "blue"]}}},
        }

        assert _collect_scheme_names(spec) == set()

    def test_ignores_a_scheme_object_without_a_string_name(self):
        # A malformed scheme object (no usable name) yields nothing, not a crash.
        spec = {"encoding": {"color": {"scale": {"scheme": {"extent": [0, 1]}}}}}

        assert _collect_scheme_names(spec) == set()


class TestCollectNumberFormats:
    def test_collects_formats_on_a_quantitative_field_its_axis_and_legend(self):
        spec = {
            "encoding": {
                "y": {"field": "v", "type": "quantitative", "axis": {"format": "~s"}},
                "color": {
                    "field": "v",
                    "type": "quantitative",
                    "legend": {"format": ".1%"},
                },
                "tooltip": [{"field": "v", "type": "quantitative", "format": ",.0f"}],
            }
        }

        assert _collect_number_formats(spec) == {"~s", ".1%", ",.0f"}

    def test_ignores_formats_on_temporal_fields(self):
        # `%b %Y` is a d3-time-format string, not a number format.
        spec = {
            "encoding": {
                "x": {"field": "data", "type": "temporal", "axis": {"format": "%b %Y"}},
                "tooltip": [
                    {"field": "data", "type": "temporal", "format": "%B de %Y"}
                ],
            }
        }

        assert _collect_number_formats(spec) == set()

    def test_ignores_a_quantitative_field_with_a_time_format_type(self):
        spec = {
            "field": "ts",
            "type": "quantitative",
            "formatType": "time",
            "format": "%Y",
        }

        assert _collect_number_formats(spec) == set()

    def test_collects_an_explicit_number_format_type(self):
        spec = {
            "field": "v",
            "type": "ordinal",
            "formatType": "number",
            "format": ".2f",
        }

        assert _collect_number_formats(spec) == {".2f"}

    def test_collects_across_layers(self):
        spec = {
            "layer": [
                {"encoding": {"y": {"type": "quantitative", "format": ".1f"}}},
                {"encoding": {"text": {"type": "quantitative", "format": "d"}}},
            ]
        }

        assert _collect_number_formats(spec) == {".1f", "d"}


class TestCollectTemporalFields:
    def test_collects_temporal_and_time_unit_fields_across_views(self):
        spec = {
            "transform": [{"timeUnit": "year", "field": "data_tu", "as": "a"}],
            "layer": [
                {"encoding": {"x": {"field": "ano", "type": "temporal"}}},
                {"encoding": {"x": {"field": "mes", "timeUnit": "month"}}},
            ],
            "encoding": {"y": {"field": "total", "type": "quantitative"}},
        }

        assert _collect_temporal_fields(spec) == {"ano", "mes", "data_tu"}


class TestIntegerYearColumns:
    def test_finds_only_whole_number_year_columns(self):
        rows = [
            {
                "ano": 2019,
                "total": 5,
                "data": "2019-01-01",
                "flag": True,
                "taxa": 2019.5,
            },
            {
                "ano": None,
                "total": 2020,
                "data": "2020-01-01",
                "flag": True,
                "taxa": 1.0,
            },
        ]

        assert _integer_year_columns(list(rows[0]), rows) == {"ano"}

    def test_an_all_null_column_is_not_a_year(self):
        assert _integer_year_columns(["ano"], [{"ano": None}]) == set()


class TestIsValidNumberFormat:
    @pytest.mark.parametrize(
        "specifier", ["", ".1f", ",.0f", "~s", ".1%", "$,.2f", "+.1f", "08.2f", ",.2~f"]
    )
    def test_accepts_d3_specifiers(self, specifier):
        assert _is_valid_number_format(specifier)

    @pytest.mark.parametrize("specifier", [".2f%", ".1F", "%Y", ".2f mil", "abc"])
    def test_rejects_invalid_specifiers(self, specifier):
        assert not _is_valid_number_format(specifier)

    @pytest.mark.parametrize(
        "specifier",
        [
            *("", ".1f", ",.0f", "~s", ".1%", "$,.2f", "+.1f", "08.2f", ",.2~f"),
            *("*^10", "(.2f", "#x", ".1F", ".1z"),
            *(".2f%", "%Y", ".2f mil", "abc", ".f", "1.2.3f", ",,d"),
        ],
    )
    def test_grammar_matches_the_bundled_d3_format(self, specifier):
        """The grammar must reject exactly what vl-convert's bundled d3-format throws on.

        Guards against a vl-convert upgrade shipping a d3-format with a different grammar.
        """
        spec = {
            "$schema": VEGA_LITE_SCHEMA,
            "data": {"values": [{"ano": 2019, "total": 1234.5}]},
            "mark": "bar",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {
                    "field": "total",
                    "type": "quantitative",
                    "axis": {"format": specifier},
                },
            },
        }
        try:
            vlc.vegalite_to_svg(json.dumps(spec))
            d3_throws = False
        except ValueError:
            d3_throws = True

        assert d3_throws == (_D3_NUMBER_FORMAT.match(specifier) is None)


class TestValidateChartSpec:
    """Uses the real vl-convert compiler (in-process)."""

    def test_invalid_legend_number_format_is_rejected(self):
        """The `.2f%` legend format that shipped on a choropleth.

        d3 throws on it only when a scale with data formats a label, which the empty-data
        render never does — so the format check, not the compile step, must catch it.
        """
        spec = _choropleth_spec()
        spec["encoding"]["color"]["legend"] = {"format": ".2f%"}

        errors = _validate_chart_spec(spec, ["sigla_uf", "valor"])

        assert any(".2f%" in error and "format" in error for error in errors)
        assert not any("compile" in error for error in errors)

    def test_valid_number_formats_pass(self):
        spec = {
            "mark": "line",
            "encoding": {
                "x": {"field": "data", "type": "temporal", "axis": {"format": "%b %Y"}},
                "y": {
                    "field": "total",
                    "type": "quantitative",
                    "axis": {"format": "~s"},
                },
            },
        }

        assert _validate_chart_spec(spec, ["data", "total"]) == []

    def test_temporal_integer_year_is_rejected(self):
        """An integer year as `temporal` renders every year as 1970, with no render error."""
        spec = {
            "mark": "line",
            "encoding": {
                "x": {"field": "ano", "type": "temporal"},
                "y": {"field": "total", "type": "quantitative"},
            },
        }

        errors = _validate_chart_spec(
            spec, ["ano", "total"], year_columns=frozenset({"ano"})
        )

        assert any("ano" in error and "temporal" in error for error in errors)
        assert not any("compile" in error for error in errors)

    def test_quantitative_integer_year_passes(self):
        spec = {
            "mark": "line",
            "encoding": {
                "x": {"field": "ano", "type": "quantitative", "axis": {"format": "d"}},
                "y": {"field": "total", "type": "quantitative"},
            },
        }

        assert (
            _validate_chart_spec(
                spec, ["ano", "total"], year_columns=frozenset({"ano"})
            )
            == []
        )

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

    def test_fold_into_grouped_bars_is_valid(self):
        """A wide result folded into long form — the production spec this used to reject."""
        spec = {
            "transform": [
                {
                    "fold": ["remuneracao_media_homens", "remuneracao_media_mulheres"],
                    "as": ["sexo", "remuneracao_media"],
                },
                {
                    "calculate": "datum.sexo === 'remuneracao_media_homens' ? 'Homens' : 'Mulheres'",
                    "as": "sexo_label",
                },
            ],
            "mark": "bar",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "xOffset": {"field": "sexo_label", "type": "nominal"},
                "y": {"field": "remuneracao_media", "type": "quantitative"},
                "color": {"field": "sexo_label", "type": "nominal"},
            },
        }
        columns = ["ano", "remuneracao_media_homens", "remuneracao_media_mulheres"]

        assert _validate_chart_spec(spec, columns) == []

    def test_fold_default_output_names_are_valid(self):
        spec = {
            "transform": [{"fold": ["homens", "mulheres"]}],
            "mark": "line",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "value", "type": "quantitative"},
                "color": {"field": "key", "type": "nominal"},
            },
        }

        assert _validate_chart_spec(spec, ["ano", "homens", "mulheres"]) == []

    def test_pivot_output_fields_are_not_flagged(self):
        """A pivot's new columns are named after data values, unknown from the spec."""
        spec = {
            "transform": [{"pivot": "sexo", "value": "total", "groupby": ["ano"]}],
            "mark": "bar",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "Homens", "type": "quantitative"},
            },
        }

        assert _validate_chart_spec(spec, ["ano", "sexo", "total"]) == []

    def test_missing_column_still_flagged_alongside_a_fold(self):
        spec = {
            "transform": [{"fold": ["homens", "mulheres"], "as": ["sexo", "valor"]}],
            "mark": "bar",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "remuneracao", "type": "quantitative"},
            },
        }

        errors = _validate_chart_spec(spec, ["ano", "homens", "mulheres"])

        assert any("remuneracao" in error for error in errors)

    def test_misplaced_channel_property_is_rejected(self):
        """It compiles and renders, so only the schema check catches it."""
        spec = {
            "mark": "bar",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {
                    "field": "total",
                    "type": "quantitative",
                    "color": {"datum": "Total", "type": "nominal"},
                },
            },
        }

        errors = _validate_chart_spec(spec, ["ano", "total"])

        assert any("encoding.y has unsupported ['color']" in error for error in errors)

    def test_split_line_is_rejected_with_the_rows(self):
        rows = [
            {"ano": ano, "valor": 1, "tipo": "A" if ano != 2021 else "B"}
            for ano in range(2019, 2024)
        ]

        errors = _validate_chart_spec(
            _line_by("tipo"), ["ano", "valor", "tipo"], rows=rows
        )

        assert any("splits it into pieces" in error for error in errors)

    def test_geometry_fields_are_allowed_in_a_map(self):
        spec = {
            **_choropleth_spec(),
            "encoding": {
                "color": {"field": "valor", "type": "quantitative"},
                "tooltip": [{"field": "properties.name"}, {"field": "id"}],
            },
        }

        missing = [
            e
            for e in _validate_chart_spec(spec, ["sigla_uf", "valor"])
            if "not in the result" in e
        ]

        assert missing == []

    def test_id_is_flagged_outside_a_map(self):
        """Only a map's geometry has an `id`; elsewhere it is a missing column."""
        spec = {"mark": "bar", "encoding": {"x": {"field": "id"}}}

        errors = _validate_chart_spec(spec, ["ano", "total"])

        assert any("['id']" in error for error in errors)

    def test_invalid_mark_fails_to_compile(self):
        """A structurally invalid spec is caught by the vl-convert compile step."""
        spec = {"mark": "notamark", "encoding": {"x": {"field": "ano"}}}

        errors = _validate_chart_spec(spec, ["ano"])

        assert any("compile" in error for error in errors)

    def test_unknown_color_scheme_is_rejected(self):
        """A scheme name outside Vega's built-in set (here d3's `RdBu`) is flagged."""
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

        assert any("RdBu" in error and "scheme" in error for error in errors)

    def test_unknown_scheme_in_choropleth_is_caught_by_the_allowlist(self):
        """The regression that shipped a blank chart.

        A geoshape's bad scheme does not raise in vl-convert (it logs to stderr but still
        returns an SVG), so the compile step alone never catches it — the allowlist does.
        """
        spec = _choropleth_spec()
        spec["encoding"]["color"]["scale"] = {"scheme": "lightpinkdark"}

        errors = _validate_chart_spec(spec, ["sigla_uf", "valor"])

        assert any("lightpinkdark" in error for error in errors)
        # The compile step swallows this one, so it is not what caught it.
        assert not any("compile" in error for error in errors)

    def test_unknown_scheme_params_object_is_rejected(self):
        """A bad scheme given as a SchemeParams object is caught too."""
        spec = {
            "mark": "bar",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "total", "type": "quantitative"},
                "color": {
                    "field": "total",
                    "type": "quantitative",
                    "scale": {"scheme": {"name": "notarealscheme"}},
                },
            },
        }

        errors = _validate_chart_spec(spec, ["ano", "total"])

        assert any("notarealscheme" in error for error in errors)

    def test_valid_scheme_passes(self):
        """A real Vega scheme (`reds`) is accepted."""
        spec = {
            "mark": "rect",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "mes", "type": "ordinal"},
                "color": {
                    "field": "temp",
                    "type": "quantitative",
                    "scale": {"scheme": "reds"},
                },
            },
        }

        assert _validate_chart_spec(spec, ["ano", "mes", "temp"]) == []

    def test_valid_scheme_params_object_passes(self):
        """A real scheme given as a SchemeParams object is accepted."""
        spec = {
            "mark": "bar",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "total", "type": "quantitative"},
                "color": {
                    "field": "total",
                    "type": "quantitative",
                    "scale": {"scheme": {"name": "viridis"}},
                },
            },
        }

        assert _validate_chart_spec(spec, ["ano", "total"]) == []

    def test_explicit_color_range_needs_no_scheme(self):
        """Explicit `range` colors (no scheme) pass the scheme check."""
        spec = {
            "mark": "bar",
            "encoding": {
                "x": {"field": "ano", "type": "nominal"},
                "y": {"field": "total", "type": "quantitative"},
                "color": {
                    "field": "ano",
                    "type": "nominal",
                    "scale": {"domain": ["2023", "2024"], "range": ["red", "blue"]},
                },
            },
        }

        assert _validate_chart_spec(spec, ["ano", "total"]) == []

    def test_choropleth_spec_is_valid(self):
        """A choropleth compiles (real geometry) and its geo properties are allowed."""
        assert _validate_chart_spec(_choropleth_spec(), ["sigla_uf", "valor"]) == []


class TestChartSpecUserPrompt:
    def test_carries_task_data_on_the_first_attempt(self):
        prompt = _chart_spec_user_prompt(
            "a bar chart", "pt", "Rows: 1 ... ano ... total", None, None, []
        )

        assert "a bar chart" in prompt
        assert "Portuguese" in prompt
        assert "ano" in prompt and "total" in prompt
        assert "rejected" not in prompt  # no retry feedback on the first attempt

    def test_echoes_the_rejected_spec_and_errors_on_retry(self):
        prompt = _chart_spec_user_prompt(
            "a bar chart", "en", "Rows: 1", None, {"mark": "nope"}, ["bad column"]
        )

        assert '"nope"' in prompt  # the model's own rejected spec, echoed back
        assert "bad column" in prompt  # the validator's reason
        assert "rejected" in prompt

    def test_carries_the_chart_to_change(self):
        prompt = _chart_spec_user_prompt(
            "make the line blue", "pt", "Rows: 1", {"mark": "line"}, None, []
        )

        assert "Chart to change" in prompt
        assert '"mark": "line"' in prompt


class TestSpreadSample:
    def test_returns_every_row_when_few(self):
        rows = [{"i": i} for i in range(3)]

        assert _spread_sample(rows, 10) == rows

    def test_spreads_across_the_whole_result(self):
        """A result sorted by group shows every group, not just the first one."""
        rows = [{"grupo": "pobres", "i": i} for i in range(50)] + [
            {"grupo": "ricos", "i": i} for i in range(50)
        ]

        sample = _spread_sample(rows, 10)

        assert len(sample) == 10
        assert sample[0] == rows[0] and sample[-1] == rows[-1]
        assert {row["grupo"] for row in sample} == {"pobres", "ricos"}


class TestSummarizeColumn:
    def test_numbers_get_a_range(self):
        summary = _summarize_column("FLOAT", [96.25, None, 1731.84, 715.52])

        assert summary == {
            "type": "FLOAT",
            "distinct": 3,
            "nulls": 1,
            "min": 96.25,
            "max": 1731.84,
        }

    def test_dates_get_a_range(self):
        summary = _summarize_column("DATE", ["2025-03-01", "2024-01-01"])

        assert summary["min"] == "2024-01-01" and summary["max"] == "2025-03-01"

    def test_low_cardinality_text_lists_every_value(self):
        summary = _summarize_column("STRING", ["SP", "MG", "SP"])

        assert summary["distinct"] == 2
        assert summary["values"] == ["MG", "SP"]

    def test_high_cardinality_text_shows_a_few_examples(self):
        summary = _summarize_column("STRING", [f"m{i:04d}" for i in range(5570)])

        assert summary["distinct"] == 5570
        assert summary["examples"] == ["m0000", "m0001", "m0002"]
        assert "values" not in summary

    def test_booleans_are_listed_not_ranged(self):
        summary = _summarize_column("BOOLEAN", [True, False, True])

        assert summary["values"] == [False, True]

    def test_all_null_column(self):
        assert _summarize_column("STRING", [None, None]) == {
            "type": "STRING",
            "distinct": 0,
            "nulls": 2,
        }


class TestDescribeResult:
    def test_carries_row_count_types_summary_and_sample(self):
        rows = [{"uf": "SP", "valor": 1}, {"uf": "MG", "valor": 3}]

        description = _describe_result({"uf": "STRING", "valor": "INTEGER"}, rows)

        assert "Rows: 2" in description
        assert '"type": "STRING"' in description and '"type": "INTEGER"' in description
        assert '"min": 1, "max": 3' in description
        assert '"MG"' in description


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
            lambda spec, columns, year_columns, rows: []
            if spec["encoding"] == good["encoding"]
            else ["bad column"],
        )

        result = await generate_chart_spec(
            {"ano": "INTEGER"}, [{"ano": 2025}], "a bar chart", "pt"
        )

        assert result["encoding"] == good["encoding"]
        assert model.ainvoke.await_count == 2

    async def test_year_columns_are_detected_from_all_rows(self, monkeypatch):
        """The year check sees every row, not just the sample shown to the model."""
        spec = {"mark": "bar", "encoding": {"x": {"field": "ano"}}}
        self._model(monkeypatch, _structured_reply(spec))
        seen = []
        monkeypatch.setattr(
            charts,
            "_validate_chart_spec",
            lambda spec, columns, year_columns, rows: seen.append(year_columns) or [],
        )
        rows = [{"ano": 2000 + i, "n": 1} for i in range(10)] + [{"ano": 1, "n": 1}]

        await generate_chart_spec(
            {"ano": "INTEGER", "n": "INTEGER"}, rows, "a bar chart", "pt"
        )

        assert seen == [frozenset()]  # the 11th row is not a year, so `ano` is not one

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
            lambda spec, columns, year_columns, rows: []
            if spec["encoding"] == good["encoding"]
            else ["column 'nope' is not in the result"],
        )

        await generate_chart_spec(
            {"ano": "INTEGER"}, [{"ano": 2025}], "a bar chart", "pt"
        )

        system_message, user_message = model.ainvoke.await_args_list[1].args[0]
        # The durable how-to is a system message; the retry data rides the user message.
        assert isinstance(system_message, SystemMessage)
        assert system_message.content == _CHART_SPEC_INSTRUCTIONS
        assert '"nope"' in user_message.content  # its own rejected spec, echoed back
        assert "not in the result" in user_message.content  # the validator's reason

    async def test_prompt_carries_the_chart_to_change(self, monkeypatch):
        spec = {"mark": "line", "encoding": {"x": {"field": "ano"}}}
        model = self._model(monkeypatch, _structured_reply(spec))
        monkeypatch.setattr(
            charts, "_validate_chart_spec", lambda spec, columns, year_columns, rows: []
        )

        await generate_chart_spec(
            {"ano": "INTEGER"},
            [{"ano": 2025}],
            "make the line blue",
            "pt",
            base_spec={"mark": "line", "title": "Antes"},
        )

        _, user_message = model.ainvoke.await_args.args[0]
        assert "Chart to change" in user_message.content
        assert '"title": "Antes"' in user_message.content

    async def test_prompt_carries_language_and_data_summary(self, monkeypatch):
        spec = {"mark": "bar", "encoding": {"x": {"field": "ano"}}}
        model = self._model(monkeypatch, _structured_reply(spec))
        monkeypatch.setattr(
            charts, "_validate_chart_spec", lambda spec, columns, year_columns, rows: []
        )

        await generate_chart_spec(
            {"ano": "INTEGER"}, [{"ano": 2024}, {"ano": 2025}], "a bar chart", "es"
        )

        _, user_message = model.ainvoke.await_args.args[0]
        assert "Spanish" in user_message.content
        assert "Rows: 2" in user_message.content
        assert '"min": 2024, "max": 2025' in user_message.content

    async def test_model_supplied_data_is_stripped(self, monkeypatch):
        """A model spec carrying data/url is sanitized (not rejected) before returning."""
        spec = {
            "mark": "bar",
            "encoding": {"x": {"field": "ano"}},
            "data": {"url": "https://evil.example/x.json"},
        }
        model = self._model(monkeypatch, _structured_reply(spec))
        monkeypatch.setattr(
            charts, "_validate_chart_spec", lambda spec, columns, year_columns, rows: []
        )

        result = await generate_chart_spec(
            {"ano": "INTEGER"}, [{"ano": 2025}], "a bar chart", "pt"
        )

        assert "data" not in result
        assert "url" not in json.dumps(result)
        assert model.ainvoke.await_count == 1

    async def test_missing_spec_is_retried(self, monkeypatch):
        """A reply the parser could not turn into a spec is treated as a failure and retried."""
        empty = _structured_reply(None)
        good = _structured_reply({"mark": "bar", "encoding": {"x": {"field": "ano"}}})
        model = self._model(monkeypatch, empty, good)
        monkeypatch.setattr(
            charts, "_validate_chart_spec", lambda spec, columns, year_columns, rows: []
        )

        result = await generate_chart_spec(
            {"ano": "INTEGER"}, [{"ano": 2025}], "a bar chart", "pt"
        )

        assert result["mark"] == "bar"
        assert model.ainvoke.await_count == 2

    async def test_raises_when_generation_exceeds_the_deadline(self, monkeypatch):
        """A hung model call is cut off instead of blocking the turn indefinitely."""
        monkeypatch.setattr(
            charts,
            "settings",
            charts.settings.model_copy(update={"CHART_SPEC_TIMEOUT_SECONDS": 0.05}),
        )

        async def hang(messages):
            await asyncio.sleep(10)

        model = MagicMock()
        model.ainvoke = hang
        monkeypatch.setattr(charts, "_chart_spec_model", lambda: model)

        with pytest.raises(ChartSpecTimeout):
            await generate_chart_spec(
                {"ano": "INTEGER"}, [{"ano": 2025}], "a bar chart", "pt"
            )

    async def test_raises_after_max_attempts(self, monkeypatch):
        spec = {"mark": "bar", "encoding": {}}
        messages = [_structured_reply(spec) for _ in range(MAX_CHART_SPEC_ATTEMPTS)]
        model = self._model(monkeypatch, *messages)
        monkeypatch.setattr(
            charts,
            "_validate_chart_spec",
            lambda spec, columns, year_columns, rows: ["always bad"],
        )

        with pytest.raises(ChartSpecInvalid):
            await generate_chart_spec(
                {"ano": "INTEGER"}, [{"ano": 2025}], "a bar chart", "pt"
            )

        assert model.ainvoke.await_count == MAX_CHART_SPEC_ATTEMPTS
