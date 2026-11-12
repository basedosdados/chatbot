import pytest

from app.charts.spec import collect, collect_derived_fields, collect_temporal_fields


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
