from app.charts import validation
from app.charts.data import spread_sample
from app.charts.rendering import NO_MARKS_PROBLEM
from app.charts.validation import (
    _VALIDATION_MAX_ROWS,
    _integer_year_columns,
    validate_chart_spec,
)
from tests.app.charts.specs import ROW_COLUMNS, ROWS, choropleth_spec, line_spec


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


# Rows whose `ano` holds whole-number years.
YEAR_ROWS = [{"ano": 2020, "total": 1.5}, {"ano": 2021, "total": 2.5}]


class TestValidateChartSpec:
    """Uses the real vl-convert compiler (in-process)."""

    def test_temporal_integer_year_is_rejected(self):
        """An integer year as `temporal` collapses every year onto one 1969/1970 tick."""
        spec = {
            "mark": "line",
            "encoding": {
                "x": {"field": "ano", "type": "temporal"},
                "y": {"field": "total", "type": "quantitative"},
            },
        }

        errors = validate_chart_spec(spec, ["ano", "total"], rows=YEAR_ROWS)

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

        assert validate_chart_spec(spec, ["ano", "total"], rows=YEAR_ROWS) == []

    def test_year_converted_to_a_date_in_place_passes(self):
        """A `calculate` that rewrites the year as a date under its own name is valid."""
        spec = {
            "mark": "line",
            "transform": [{"calculate": "datetime(datum.ano, 0, 1)", "as": "ano"}],
            "encoding": {
                "x": {"field": "ano", "type": "temporal"},
                "y": {"field": "total", "type": "quantitative"},
            },
        }

        assert validate_chart_spec(spec, ["ano", "total"], rows=YEAR_ROWS) == []

    def test_year_time_unit_transform_in_place_is_rejected(self):
        """A `timeUnit` transform still reads the integer year as epoch milliseconds."""
        spec = {
            "mark": "line",
            "transform": [{"timeUnit": "year", "field": "ano", "as": "ano"}],
            "encoding": {
                "x": {"field": "ano", "type": "temporal"},
                "y": {"field": "total", "type": "quantitative"},
            },
        }

        errors = validate_chart_spec(spec, ["ano", "total"], rows=YEAR_ROWS)

        assert any("ano" in error and "temporal" in error for error in errors)

    def test_valid_spec_has_no_errors(self):
        spec = {
            "mark": "bar",
            "encoding": {"x": {"field": "ano"}, "y": {"field": "total"}},
        }
        assert validate_chart_spec(spec, ["ano", "total"]) == []

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
        assert validate_chart_spec(spec, ["ano", "media", "var"]) == []

    def test_missing_column_flagged_across_layers(self):
        """A field not in the result, even nested in a layer — else it renders empty."""
        spec = {"layer": [{"mark": "bar", "encoding": {"x": {"field": "vendas"}}}]}

        errors = validate_chart_spec(spec, ["ano", "total"])

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
        assert validate_chart_spec(spec, ["ano", "total"]) == []

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

        assert validate_chart_spec(spec, columns) == []

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

        assert validate_chart_spec(spec, ["ano", "homens", "mulheres"]) == []

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

        assert validate_chart_spec(spec, ["ano", "sexo", "total"]) == []

    def test_missing_column_still_flagged_alongside_a_fold(self):
        spec = {
            "transform": [{"fold": ["homens", "mulheres"], "as": ["sexo", "valor"]}],
            "mark": "bar",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "remuneracao", "type": "quantitative"},
            },
        }

        errors = validate_chart_spec(spec, ["ano", "homens", "mulheres"])

        assert any("remuneracao" in error for error in errors)

    def test_invalid_mark_fails_to_compile(self):
        """A structurally invalid spec is caught by the vl-convert compile step."""
        spec = {"mark": "notamark", "encoding": {"x": {"field": "ano"}}}

        errors = validate_chart_spec(spec, ["ano"])

        assert any("compile" in error for error in errors)

    def test_explicit_color_range_passes(self):
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

        assert validate_chart_spec(spec, ["ano", "total"]) == []

    def test_rows_are_rendered_and_their_problems_returned(self):
        spec = line_spec(
            color={"field": "sigla_uf", "type": "nominal", "scale": {"scheme": "RdBu"}}
        )

        assert validate_chart_spec(spec, ROW_COLUMNS) == []  # builds without rows
        errors = validate_chart_spec(spec, ROW_COLUMNS, rows=ROWS)

        assert errors == [NO_MARKS_PROBLEM]

    def test_large_result_is_rendered_with_a_sample(self, monkeypatch):
        rows = [{"ano": i, "total": i} for i in range(_VALIDATION_MAX_ROWS * 3)]
        rendered = []
        monkeypatch.setattr(
            validation,
            "render_problems",
            lambda spec, rows: rendered.append(rows) or [],
        )

        assert validate_chart_spec({"mark": "bar"}, ["ano", "total"], rows=rows) == []
        assert [len(r) for r in rendered] == [_VALIDATION_MAX_ROWS]
        assert rendered[0][0] == rows[0] and rendered[0][-1] == rows[-1]

    def test_integer_years_are_looked_for_in_every_row(self):
        """A column is a year column only if every row holds a year, not just the
        rows the chart is rendered with."""
        rows = [{"ano": 2020, "total": i} for i in range(_VALIDATION_MAX_ROWS * 3)]
        rows[7] = {"ano": 1, "total": 7}  # not a year, and not in the sample
        spec = {
            "mark": "point",
            "encoding": {
                "x": {"field": "ano", "type": "temporal"},
                "y": {"field": "total", "type": "quantitative"},
            },
        }

        assert spread_sample(rows, _VALIDATION_MAX_ROWS).count(rows[7]) == 0
        assert not any(
            "integer years" in e
            for e in validate_chart_spec(spec, ["ano", "total"], rows)
        )

    def test_filter_keeping_rows_the_sample_misses_passes(self):
        """A sample with none of the rows a filter keeps draws no marks; every row
        decides then."""
        rows = [{"ano": i, "total": i} for i in range(_VALIDATION_MAX_ROWS * 3)]
        spec = {
            "transform": [{"filter": "datum.ano == 7"}],
            "mark": "bar",
            "encoding": {
                "x": {"field": "ano", "type": "ordinal"},
                "y": {"field": "total", "type": "quantitative"},
            },
        }

        assert spread_sample(rows, _VALIDATION_MAX_ROWS).count(rows[7]) == 0
        assert validate_chart_spec(spec, ["ano", "total"], rows=rows) == []

    def test_only_a_chart_without_marks_is_rendered_again_with_every_row(
        self, monkeypatch
    ):
        """Any other problem does not come from sampling, and every row could take
        many seconds to draw."""
        rows = [{"ano": i, "total": i} for i in range(_VALIDATION_MAX_ROWS * 3)]
        rendered = []

        def render_problems(spec, rows):
            rendered.append(len(rows))
            return ["shows 'undefined' as text"]

        monkeypatch.setattr(validation, "render_problems", render_problems)

        errors = validate_chart_spec({"mark": "bar"}, ["ano", "total"], rows=rows)

        assert errors == ["shows 'undefined' as text"]
        assert rendered == [_VALIDATION_MAX_ROWS]

    def test_choropleth_spec_is_valid(self):
        """A choropleth compiles (real geometry) and its geo properties are allowed."""
        assert validate_chart_spec(choropleth_spec(), ["sigla_uf", "valor"]) == []
