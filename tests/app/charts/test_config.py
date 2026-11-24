import json

import vl_convert as vlc

from app.charts.config import _NUMBER_LOCALES, apply_server_config
from app.charts.rendering import (
    _collect_marks,
    _rendered_text,
)
from tests.app.charts.specs import ROWS, line_spec


class TestServerConfig:
    def test_sets_locale_and_tooltips(self):
        spec = apply_server_config({"mark": "bar"}, "pt")

        assert spec["config"]["locale"]["number"]["decimal"] == ","
        assert spec["config"]["locale"]["time"]["shortMonths"][1] == "Fev"
        assert spec["config"]["bar"] == {"tooltip": {"content": "encoding"}}
        assert "line" not in spec["config"]

    def test_english_keeps_default_dates(self):
        spec = apply_server_config({"mark": "bar"}, "en")

        assert spec["config"]["locale"] == {"number": _NUMBER_LOCALES["en"]}

    def test_keeps_the_models_other_config_and_overrides_its_locale(self):
        spec = {
            "mark": "bar",
            "config": {
                "background": "#111111",
                "bar": {"cornerRadius": 2},
                "locale": {"number": {"decimal": "."}},
            },
        }

        configured = apply_server_config(spec, "pt")

        assert configured["config"]["background"] == "#111111"
        assert configured["config"]["bar"] == {
            "cornerRadius": 2,
            "tooltip": {"content": "encoding"},
        }
        assert configured["config"]["locale"]["number"]["decimal"] == ","
        assert spec["config"]["locale"] == {"number": {"decimal": "."}}  # not mutated

    def test_numbers_are_drawn_in_the_languages_format(self):
        spec = apply_server_config(
            line_spec(
                y={"field": "total", "type": "quantitative", "axis": {"format": ",.1f"}}
            ),
            "pt",
        )
        spec["data"] = {"values": ROWS}

        scenegraph = vlc.vegalite_to_scenegraph(json.dumps(spec))["scenegraph"]
        labels = [
            text
            for mark in _collect_marks(scenegraph)
            if mark.get("role") == "axis-label"
            for text in _rendered_text(mark)
        ]

        assert "10.000,0" in labels
