import pytest

from app.charts.spec import (
    collect,
    collect_date_calculated_fields,
    collect_derived_fields,
    collect_temporal_fields,
)


class TestCollect:
    def test_collects_string_values_under_key_recursively(self):
        node = {
            "encoding": {"x": {"field": "ano"}, "y": {"field": "total"}},
            "layer": [{"encoding": {"color": {"field": "uf"}}}],
        }

        assert collect(node, "field") == {"ano", "total", "uf"}

    def test_ignores_non_string_values_under_the_key(self):
        assert collect({"field": 5, "x": {"field": "ano"}}, "field") == {"ano"}


class TestCollectDerivedFields:
    def test_collects_string_as(self):
        spec = {"transform": [{"calculate": "datum.total * 2", "as": "dobro"}]}

        assert collect_derived_fields(spec) == {"dobro"}

    def test_collects_list_as(self):
        """`fold`, `stack`, `regression`, … name several outputs in a list."""
        spec = {
            "transform": [{"fold": ["homens", "mulheres"], "as": ["sexo", "valor"]}]
        }

        assert collect_derived_fields(spec) == {"sexo", "valor"}

    def test_collects_as_inside_aggregate_ops(self):
        spec = {
            "transform": [
                {
                    "aggregate": [{"op": "sum", "field": "v", "as": "total"}],
                    "groupby": ["uf"],
                }
            ]
        }

        assert collect_derived_fields(spec) == {"total"}

    def test_bin_transform_also_creates_end_field(self):
        spec = {"transform": [{"bin": True, "field": "renda", "as": "faixa"}]}

        assert collect_derived_fields(spec) == {"faixa", "faixa_end"}

    @pytest.mark.parametrize(
        "transform, expected",
        [
            ({"fold": ["a", "b"]}, {"key", "value"}),
            ({"density": "renda"}, {"value", "density"}),
            ({"quantile": "renda"}, {"prob", "value"}),
        ],
    )
    def test_default_outputs_when_as_is_omitted(self, transform, expected):
        assert collect_derived_fields({"transform": [transform]}) == expected

    def test_collects_across_layers(self):
        spec = {
            "layer": [
                {"transform": [{"calculate": "1", "as": "a"}]},
                {"transform": [{"regression": "y", "on": "x", "as": ["rx", "ry"]}]},
            ]
        }

        assert collect_derived_fields(spec) == {"a", "rx", "ry"}


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

        assert collect_temporal_fields(spec) == {"ano", "mes", "data_tu"}


class TestCollectDateCalculatedFields:
    # Each expression was rendered with integer years in `ano` (vl-convert): these drew
    # the years on a time axis, the others drew them as epoch milliseconds or not at all.
    @pytest.mark.parametrize(
        ("expression", "writes_a_date"),
        [
            ("datetime(datum.ano, 0, 1)", True),
            ("utc(datum.ano, 0, 1)", True),
            ("time(datetime(datum.ano, 0, 1))", True),
            ("timeParse(toString(datum.ano), '%Y')", True),
            ("utcParse(toString(datum.ano), '%Y')", True),
            ("toDate(toString(datum.ano))", True),
            ("toDate(datum.ano + '-01-01')", True),
            ("toDate(datum.ano)", False),
            ("toDate(datum['ano'])", False),
            ("toString(datum.ano)", False),
            ("datum.ano + '-01-01'", False),
            ("datum.ano + 0", False),
            ("utcyear(datum.ano)", False),
        ],
    )
    def test_counts_only_expressions_that_write_a_date(self, expression, writes_a_date):
        spec = {"transform": [{"calculate": expression, "as": "ano"}]}

        assert collect_date_calculated_fields(spec) == (
            {"ano"} if writes_a_date else set()
        )

    def test_collects_across_layers_and_ignores_other_transforms(self):
        spec = {
            "transform": [{"timeUnit": "year", "field": "data", "as": "a"}],
            "layer": [
                {"transform": [{"calculate": "datetime(datum.ano, 0, 1)", "as": "b"}]},
                {"transform": [{"calculate": "datum.total * 2", "as": "c"}]},
            ],
        }

        assert collect_date_calculated_fields(spec) == {"b"}
