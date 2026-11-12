import json

import vl_convert as vlc

from app.charts.rendering import (
    _BRAZIL_STATES,
    NO_MARKS_PROBLEM,
    _collect_marks,
    _geo_stub_ids,
    _geo_stub_node,
    _rendered_text,
    render_problems,
)
from tests.app.charts.specs import ROWS, choropleth_spec, line_spec


class TestGeoStubNode:
    def test_has_one_feature_per_id(self):
        node = _geo_stub_node("brazil_states", ["SP", "RJ", "MG"])
        geometries = node["values"]["objects"]["uf"]["geometries"]

        assert node["format"] == {"type": "topojson", "feature": "uf"}
        assert node["values"]["type"] == "Topology"
        assert [g["id"] for g in geometries] == ["SP", "RJ", "MG"]
        assert [g["arcs"] for g in geometries] == [[[0]], [[1]], [[2]]]
        assert len(node["values"]["arcs"]) == 3

    def test_features_sit_inside_brazil(self):
        """Each square winds clockwise, so its centroid is its own middle, not the
        far side of the globe's."""
        spec = {
            "data": _geo_stub_node("brazil_states", ["SP", "RJ"]),
            "transform": [{"calculate": "geoCentroid(null, datum)[0]", "as": "lon"}],
            "mark": "text",
            "encoding": {"text": {"field": "lon", "type": "quantitative"}},
        }

        scenegraph = vlc.vegalite_to_scenegraph(json.dumps(spec))["scenegraph"]
        longitudes = [
            float(text.replace("−", "-"))
            for mark in _collect_marks(scenegraph)
            if mark.get("role") == "mark"
            for text in _rendered_text(mark)
        ]

        assert len(longitudes) == 2 and all(-75 < lon < -30 for lon in longitudes)

    def test_keys_the_stub_by_the_assets_feature(self):
        node = _geo_stub_node("brazil_municipalities", ["3550308"])

        assert node["format"]["feature"] == "Munic"
        assert set(node["values"]["objects"]) == {"Munic"}


class TestGeoStubIds:
    def test_states_are_the_real_27(self):
        ids = _geo_stub_ids("brazil_states", choropleth_spec(), [])

        assert len(ids) == 27 and "SP" in ids

    def test_municipalities_are_the_joined_codes_plus_one_without_data(self):
        spec = {
            "transform": [
                {
                    "lookup": "id",
                    "from": {
                        "data": {"name": "query_result"},
                        "key": "id_municipio",
                        "fields": ["valor"],
                    },
                }
            ]
        }
        rows = [
            {"id_municipio": 3550308, "valor": 1, "outro": "x"},
            {"id_municipio": 3304557, "valor": 2, "outro": "y"},
            {"id_municipio": None, "valor": 3, "outro": "z"},
        ]

        ids = _geo_stub_ids("brazil_municipalities", spec, rows)

        assert ids == ["3304557", "3550308", "0"]


class TestRenderProblems:
    """What only shows once the spec is drawn with the result's rows."""

    def test_valid_spec_has_no_problems(self):
        assert render_problems(line_spec(), ROWS) == []

    def test_valid_spec_without_rows_has_no_problems(self):
        assert render_problems(line_spec(), []) == []

    def test_unknown_color_scheme_draws_nothing(self):
        """A scheme outside Vega's built-in set (here d3's `RdBu`) fails on real rows."""
        spec = line_spec(
            color={"field": "sigla_uf", "type": "nominal", "scale": {"scheme": "RdBu"}}
        )

        problems = render_problems(spec, ROWS)

        assert len(problems) == 1
        assert problems == [NO_MARKS_PROBLEM] and "scheme" in problems[0]

    def test_invalid_axis_number_format_draws_nothing(self):
        spec = line_spec(
            y={"field": "total", "type": "quantitative", "axis": {"format": ".2f%"}}
        )

        problems = render_problems(spec, ROWS)

        assert NO_MARKS_PROBLEM in problems

    def test_invalid_tooltip_format_leaves_marks_undrawn(self):
        """The `R$,.2f` tooltip format that shipped a blank map: the items exist but
        the failed encoding never placed them."""
        spec = {
            "mark": "bar",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "total", "type": "quantitative"},
                "tooltip": [
                    {"field": "total", "type": "quantitative", "format": "R$,.2f"}
                ],
            },
        }

        problems = render_problems(spec, ROWS)

        assert any("none of the chart's marks are drawn" in p for p in problems)
        assert NO_MARKS_PROBLEM not in problems

    def test_one_failed_layer_is_reported_as_partial(self):
        spec = {
            "layer": [
                {"mark": "bar"},
                {
                    "mark": "text",
                    "encoding": {
                        "text": {
                            "field": "total",
                            "type": "quantitative",
                            "format": ".2f%",
                        }
                    },
                },
            ],
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "total", "type": "quantitative"},
            },
        }

        problems = render_problems(spec, ROWS)

        assert any("only part of the chart's marks" in p for p in problems)

    def test_layer_with_no_rows_beside_a_drawn_one_passes(self):
        """A layer whose filter keeps no row is not a failure while others draw."""
        spec = {
            "layer": [
                {"mark": "bar"},
                {"mark": "point", "transform": [{"filter": "datum.total < 0"}]},
            ],
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "total", "type": "quantitative"},
            },
        }

        assert render_problems(spec, ROWS) == []

    def test_filter_removing_every_row_draws_nothing(self):
        spec = {**line_spec(), "transform": [{"filter": "datum.sigla_uf == 'XX'"}]}

        problems = render_problems(spec, ROWS)

        assert problems == [NO_MARKS_PROBLEM] and "filter" in problems[0]

    def test_failed_dataflow_is_an_empty_frame(self):
        spec = {
            **line_spec(),
            "transform": [{"calculate": "datum.nope.x * 2", "as": "z"}],
        }

        problems = render_problems(spec, ROWS)

        assert len(problems) == 1 and "fails when it runs" in problems[0]

    def test_invalid_mark_fails_to_compile(self):
        spec = {"mark": "notamark", "encoding": {"x": {"field": "ano"}}}

        problems = render_problems(spec, ROWS)

        assert len(problems) == 1 and "compile" in problems[0]

    def test_compile_error_leaves_out_the_spec_and_the_stack_trace(self):
        """An invalid specification's error quotes the whole spec, rows included."""
        rows = [{"ano": i, "total": i} for i in range(400)]

        problems = render_problems({"encoding": {"x": {"field": "ano"}}}, rows)

        assert len(problems) == 1 and "Invalid specification" in problems[0]
        assert "at least one of the following properties" in problems[0]
        assert len(problems[0]) < 800 and "\n    at " not in problems[0]

    def test_undefined_text_is_rejected(self):
        """A misspelled field in a text expression shows "undefined" on the chart."""
        spec = {
            "transform": [{"calculate": "'Ano ' + datum.anno", "as": "rotulo"}],
            "mark": "text",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "total", "type": "quantitative"},
                "text": {"field": "rotulo"},
            },
        }

        problems = render_problems(spec, ROWS)

        assert any("undefined" in p for p in problems)

    def test_transparent_text_is_not_read(self):
        spec = {
            "transform": [{"calculate": "'Ano ' + datum.anno", "as": "rotulo"}],
            "mark": {"type": "text", "opacity": 0},
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "total", "type": "quantitative"},
                "text": {"field": "rotulo"},
            },
        }

        assert render_problems(spec, ROWS) == []

    def test_axis_with_one_label_on_every_tick_is_rejected(self):
        """A `labelExpr` that discards the value labels every tick alike."""
        spec = line_spec(
            y={
                "field": "total",
                "type": "quantitative",
                "axis": {"labelExpr": "format(datum.valu | 0, 'd')"},
            }
        )

        problems = render_problems(spec, ROWS)

        assert any("same label '0'" in p for p in problems)

    def test_log_scale_over_zero_has_no_tick_labels(self):
        rows = [*ROWS, {"ano": 2014, "sigla_uf": "SP", "total": 0}]
        spec = line_spec(
            y={"field": "total", "type": "quantitative", "scale": {"type": "log"}}
        )

        problems = render_problems(spec, rows)

        assert any("no tick labels" in p for p in problems)

    def test_axis_without_labels_by_choice_passes(self):
        spec = line_spec(
            x={"field": "ano", "type": "quantitative", "axis": {"labels": False}}
        )

        assert render_problems(spec, ROWS) == []

    def test_fixed_domain_far_from_the_values_has_no_tick_labels(self):
        """Values in the thousands on a domain fixed at [0, 8.8] leave the axis bare."""
        spec = line_spec(
            y={"field": "total", "type": "quantitative", "scale": {"domain": [0, 8.8]}}
        )

        problems = render_problems(spec, ROWS)

        assert any("no tick labels" in p and "domain" in p for p in problems)

    def test_chart_far_too_large_is_rejected(self):
        problems = render_problems({**line_spec(), "height": 20000}, ROWS)

        assert any("far too large" in p for p in problems)

    def test_chart_without_size_is_rejected(self):
        """A facet beside a top-level layer is drawn 0 pixels wide."""
        bars = {
            "mark": "bar",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "total", "type": "quantitative"},
            },
        }
        spec = {
            "facet": {"row": {"field": "sigla_uf", "type": "nominal"}},
            "spec": {"width": "container", **bars},
            "layer": [{"mark": "bar"}],
        }

        problems = render_problems(spec, ROWS)

        assert any("no size" in p for p in problems)

    def test_bar_without_an_x_encoding_passes(self):
        """A position the spec does not encode is absent from every item."""
        spec = {
            "mark": "bar",
            "encoding": {"y": {"field": "total", "type": "quantitative"}},
        }

        assert render_problems(spec, ROWS[:1]) == []

    def test_text_layer_that_writes_nothing_is_rejected(self):
        """A `text` channel given as `{"expr": ...}` is ignored: the labels the request
        asked for are silently missing."""
        spec = {
            "layer": [
                {"mark": "bar"},
                {
                    "mark": "text",
                    "encoding": {"text": {"expr": "format(datum.total, '.1f')"}},
                },
            ],
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "total", "type": "quantitative"},
            },
        }

        problems = render_problems(spec, ROWS)

        assert any("only part of the chart's marks" in p for p in problems)

    def test_choropleth_joined_to_its_rows_is_valid(self):
        rows = [{"sigla_uf": "SP", "valor": 10.5}, {"sigla_uf": "RJ", "valor": 7.25}]

        assert render_problems(choropleth_spec(), rows) == []

    def test_choropleth_with_labels_at_each_states_centroid_is_valid(self):
        """The stub geometry has a real shape per state, so centroids and labels work."""
        rows = [{"sigla_uf": uf, "valor": 1.5} for uf in _BRAZIL_STATES]
        choropleth = choropleth_spec()
        spec = {
            "data": choropleth["data"],
            "transform": [
                *choropleth["transform"],
                {"calculate": "geoCentroid(null, datum)[0]", "as": "lon"},
                {"calculate": "geoCentroid(null, datum)[1]", "as": "lat"},
                {"calculate": "format(datum.valor, '.1f') + '%'", "as": "rotulo"},
            ],
            "projection": choropleth["projection"],
            "layer": [
                {"mark": "geoshape", "encoding": choropleth["encoding"]},
                {
                    "mark": "text",
                    "encoding": {
                        "longitude": {"field": "lon", "type": "quantitative"},
                        "latitude": {"field": "lat", "type": "quantitative"},
                        "text": {"field": "rotulo"},
                    },
                },
            ],
        }

        assert render_problems(spec, rows) == []

    def test_choropleth_label_from_a_misspelled_field_is_rejected(self):
        rows = [{"sigla_uf": uf, "valor": 1.5} for uf in _BRAZIL_STATES]
        choropleth = choropleth_spec()
        spec = {
            **choropleth,
            "transform": [
                *choropleth["transform"],
                {"calculate": "format(datum.valr, '.1f')", "as": "rotulo"},
            ],
        }
        spec["encoding"] = {
            **choropleth["encoding"],
            "tooltip": [{"field": "rotulo", "type": "nominal"}],
        }
        labels = {
            "data": choropleth["data"],
            "transform": spec["transform"],
            "mark": "text",
            "encoding": {"text": {"field": "rotulo"}},
        }

        problems = render_problems({"vconcat": [choropleth, labels]}, rows)

        assert any("NaN" in p for p in problems)

    def test_municipality_choropleth_joins_integer_codes(self):
        """The real geometry's ids are text; an integer code column still joins."""
        spec = json.loads(
            json.dumps(choropleth_spec())
            .replace("brazil_states", "brazil_municipalities")
            .replace("sigla_uf", "id_municipio")
        )
        rows = [
            {"id_municipio": 3550308, "valor": 10.5},
            {"id_municipio": 3304557, "valor": 7.25},
        ]

        assert render_problems(spec, rows) == []
