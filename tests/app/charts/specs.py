"""Chart specs and rows shared across the chart tests."""


def choropleth_spec() -> dict:
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


# Rows for the render checks: one value per year and state.
ROWS = [
    {"ano": ano, "sigla_uf": uf, "total": (ano - 2000) * (index + 1) * 1000.5}
    for ano in range(2015, 2024)
    for index, uf in enumerate(["SP", "RJ", "MG"])
]

ROW_COLUMNS = ["ano", "sigla_uf", "total"]


def line_spec(**encoding) -> dict:
    return {
        "mark": "line",
        "encoding": {
            "x": {"field": "ano", "type": "quantitative"},
            "y": {"field": "total", "type": "quantitative"},
            "color": {"field": "sigla_uf", "type": "nominal"},
            **encoding,
        },
    }
