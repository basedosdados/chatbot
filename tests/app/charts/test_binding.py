import json

from app.charts.binding import (
    VEGA_LITE_SCHEMA,
    _geo_url_node,
    inject_chart_data,
    resolve_named_data,
    sanitize_chart_spec,
)
from app.settings import settings
from tests.app.charts.specs import choropleth_spec


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

        clean = sanitize_chart_spec(raw)

        assert "data" not in clean
        assert "datasets" not in clean
        assert "data" not in clean["layer"][0]
        # No URL survives anywhere (external-fetch / SSRF vector).
        assert "url" not in json.dumps(clean)
        # The presentational parts are untouched.
        assert clean["layer"][0]["encoding"] == {"x": {"field": "ano"}}

    def test_keeps_allowlisted_named_sources(self):
        clean = sanitize_chart_spec(choropleth_spec())

        assert clean["data"] == {"name": "brazil_states"}
        assert clean["transform"][0]["from"]["data"] == {"name": "query_result"}

    def test_drops_unknown_named_source_inline_values_and_url(self):
        assert "data" not in sanitize_chart_spec({"data": {"name": "secret"}})
        assert "data" not in sanitize_chart_spec({"data": {"values": [{"x": 1}]}})
        assert "data" not in sanitize_chart_spec({"data": {"url": "http://x"}})

    def test_strips_href_click_link_but_keeps_a_field_named_href(self):
        raw = {
            "mark": {"type": "point", "href": "javascript:alert(1)"},
            "encoding": {
                "href": {"field": "link"},  # the href channel is dropped
                "x": {"field": "href"},  # a column named "href" is a value, not a key
            },
        }

        clean = sanitize_chart_spec(raw)

        assert "href" not in clean["mark"]
        assert "href" not in clean["encoding"]
        assert clean["encoding"]["x"] == {"field": "href"}
        # No clickable-link href survives anywhere (click-XSS vector).
        assert "javascript:" not in json.dumps(clean)


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


class TestResolveNamedData:
    def test_resolves_query_result_to_rows(self):
        rows = [{"sigla_uf": "SP", "valor": 1}]

        assert resolve_named_data({"name": "query_result"}, rows, _geo_url_node) == {
            "values": rows
        }

    def test_resolves_a_geo_name_via_the_given_resolver(self):
        node = resolve_named_data({"name": "brazil_states"}, [], _geo_url_node)

        assert node == _geo_url_node("brazil_states")

    def test_resolves_references_nested_anywhere(self):
        rows = [{"sigla_uf": "SP", "valor": 1}]
        spec = {"transform": [{"from": {"data": {"name": "query_result"}}}]}

        resolved = resolve_named_data(spec, rows, _geo_url_node)

        assert resolved["transform"][0]["from"]["data"] == {"values": rows}

    def test_leaves_unknown_names_and_plain_nodes_untouched(self):
        # An unknown name is not resolved here (sanitize drops it earlier).
        assert resolve_named_data({"name": "secret"}, [], _geo_url_node) == {
            "name": "secret"
        }
        # A node with no named source passes through unchanged.
        assert resolve_named_data({"mark": "bar"}, [], _geo_url_node) == {"mark": "bar"}


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

        chart = inject_chart_data(choropleth_spec(), rows)

        # Top-level geometry becomes a URL node (fetched client-side); the rows fill the
        # lookup source inline.
        assert chart["data"] == _geo_url_node("brazil_states")
        assert chart["transform"][0]["from"]["data"] == {"values": rows}
