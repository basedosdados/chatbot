"""The data a spec may reference: sanitizing it, and binding the real data in."""

from collections.abc import Callable
from typing import Any

from pydantic import JsonValue

from app.settings import settings

VEGA_LITE_SCHEMA = "https://vega.github.io/schema/vega-lite/v6.json"

# Keys always stripped anywhere in a model-generated spec for security: `url`/`datasets`
# are external/inline data (SSRF, provenance); `href` makes a mark a clickable link,
# whose `javascript:` URI would be a click-XSS in the viewer's browser.
_UNTRUSTED_KEYS = frozenset({"datasets", "url", "href"})

# Geographic assets a spec may inject by name for choropleth maps. The TopoJSON
# files are served as static assets by the website (see settings.GEO_ASSET_URL_BASE).
GEO_ASSETS: dict[str, dict[str, str]] = {
    "brazil_states": {
        "file": "brasil_estados.topojson",
        "feature": "uf",
    },
    "brazil_municipalities": {
        "file": "brasil_municipios.topojson",
        "feature": "Munic",
    },
}

# The result rows, referenced from a spec as {"data": {"name": "query_result"}}.
_RESULT_SOURCE = "query_result"

# Every data-source name a spec may reference; anything else in a `data` node is dropped.
_ALLOWED_SOURCES = frozenset({_RESULT_SOURCE, *GEO_ASSETS})


def sanitize_chart_spec(node: JsonValue) -> JsonValue:
    """Sanitize a model spec: drop untrusted keys, keep only allowlisted named sources.

    Every `_UNTRUSTED_KEYS` key (`datasets`, `url`, `href`) is removed outright, and a `data`
    node survives only when it is exactly `{"name": <allowlisted source>}`, so the model can
    never supply its own data or a clickable link.

    Args:
        node (JsonValue): A spec, or any node within it, to walk.

    Returns:
        JsonValue: The same node with untrusted data removed.
    """
    if isinstance(node, dict):
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _UNTRUSTED_KEYS:
                continue
            elif key == "data":
                if (
                    isinstance(value, dict)
                    and set(value) == {"name"}
                    and value["name"] in _ALLOWED_SOURCES
                ):
                    out[key] = {"name": value["name"]}
                # any other data node (inline values, url, unknown name) is dropped
                continue
            out[key] = sanitize_chart_spec(value)
        return out
    if isinstance(node, list):
        return [sanitize_chart_spec(item) for item in node]
    return node


def _geo_url_node(name: str) -> dict[str, Any]:
    """Build a Vega-Lite TopoJSON data node that points at the geometry by URL.

    The URL is the website's static path for the file; the browser fetches (and caches) the
    ~megabytes of geometry directly, so it never inflates the spec, the SSE payload, or the
    persisted `events`.

    Args:
        name (str): A geo source name from `GEO_ASSETS` (e.g. "brazil_states").

    Returns:
        dict[str, Any]: The `data` node — a topojson `url` plus its topojson `format`.
    """
    asset = GEO_ASSETS[name]
    return {
        "url": f"{settings.GEO_ASSET_URL_BASE.rstrip('/')}/{asset['file']}",
        "format": {"type": "topojson", "feature": asset["feature"]},
    }


def resolve_named_data(
    node: JsonValue,
    rows: list[dict[str, Any]],
    geo_node: Callable[[str], dict[str, Any]],
) -> JsonValue:
    """Replace every `{"name": <allowlisted>}` data reference with real data.

    `query_result` resolves to the rows; a geo name resolves via a `geo_node` — a URL
    node for a render-ready spec, a tiny inline stub for offline validation. Any other
    `{"name": ...}` was already dropped by `sanitize_chart_spec`.

    Args:
        node (JsonValue): A spec, or any node within it, to walk.
        rows (list[dict[str, Any]]): The exact result rows to bind for `query_result`.
        geo_node (Callable[[str], dict[str, Any]]): Resolver for a geo source name.

    Returns:
        JsonValue: The same node with its named data references resolved.
    """
    if isinstance(node, dict):
        if set(node) == {"name"} and node["name"] in _ALLOWED_SOURCES:
            if node["name"] == _RESULT_SOURCE:
                return {"values": rows}
            return geo_node(node["name"])
        return {
            key: resolve_named_data(value, rows, geo_node)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [resolve_named_data(item, rows, geo_node) for item in node]
    return node


def inject_chart_data(
    spec: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Inject the real geometry and result rows into an already-sanitized spec.

    Takes an already sanitized spec from `generate_chart_spec` and makes it render-ready:
    resolves its named sources (a geo name to a TopoJSON URL, `query_result` to the rows)
    and binds the rows as the default data for a plain chart that declared none.

    Args:
        spec (dict[str, Any]): A sanitized, data-less spec from `generate_chart_spec`.
        rows (list[dict[str, Any]]): The exact result rows to render.

    Returns:
        dict[str, Any]: A render-ready Vega-Lite spec with the real data bound in.
    """
    chart = resolve_named_data(spec, rows, _geo_url_node)
    chart["$schema"] = VEGA_LITE_SCHEMA
    chart.setdefault("data", {"values": rows})
    return chart
