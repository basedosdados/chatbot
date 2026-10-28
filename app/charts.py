import asyncio
import json
import re
from collections.abc import Callable, Iterator
from decimal import Decimal
from functools import cache
from pathlib import Path
from typing import Any

import vl_convert as vlc
from google.api_core.exceptions import NotFound
from google.cloud import bigquery as bq
from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, JsonValue

from app.db.database import AsyncDatabase, sessionmaker
from app.db.models import QueryHandle
from app.exports import ResultTableExpired, is_result_expired
from app.i18n import LanguageCode, language_name
from app.settings import settings

VEGA_LITE_SCHEMA = "https://vega.github.io/schema/vega-lite/v6.json"

# The JSON schema of the Vega-Lite version vl-convert bundles (6.4.1 in vl-convert 1.9.0);
# keep it in sync when upgrading vl-convert.
_VEGA_LITE_SCHEMA_FILE = (
    Path(__file__).parent / "assets" / "vega-lite-v6.4.1.schema.json"
)

# Rows shown to the spec generator, evenly spaced across the result. It also gets a
# per-column summary over every row; the full result is bound afterwards by
# inject_chart_data.
CHART_SPEC_SAMPLE_ROWS = 10

# A text column's distinct values are listed in full up to this many; beyond it, only a
# few examples are shown.
_SUMMARY_MAX_LISTED_VALUES = 12
_SUMMARY_EXAMPLES = 3

# Temporal BigQuery types arrive as ISO strings, which order like the dates they encode.
_TEMPORAL_TYPES = frozenset({"DATE", "DATETIME", "TIMESTAMP", "TIME"})

# How many times the generator may retry after a spec fails validation before giving up.
MAX_CHART_SPEC_ATTEMPTS = 3

# Keys always stripped anywhere in a model-generated spec for security: `url`/`datasets`
# are external/inline data (SSRF, provenance); `href` makes a mark a clickable link,
# whose `javascript:` URI would be a click-XSS in the viewer's browser.
_UNTRUSTED_KEYS = frozenset({"datasets", "url", "href"})

# Geographic assets a spec may inject by name for choropleth maps. The TopoJSON
# files are served as static assets by the website (see settings.GEO_ASSET_URL_BASE).
_GEO_ASSETS: dict[str, dict[str, str]] = {
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
_ALLOWED_SOURCES = frozenset({_RESULT_SOURCE, *_GEO_ASSETS})

# Vega-Lite data generators: they synthesize values (a number sequence, a map graticule,
# the globe's outline) without fetching anything, so a spec may keep them.
_DATA_GENERATORS = frozenset({"sequence", "graticule", "sphere"})

# Vega's built-in color scheme names (https://vega.github.io/vega/docs/schemes/). A scheme
# name outside this set is rejected. Includes the dark-/light-background ramp variants for
# completeness, though the prompt steers away from them. Keep in sync with the scheme list
# in _CHART_SPEC_INSTRUCTIONS.
_VEGA_COLOR_SCHEMES = frozenset(
    {
        # Categorical
        "accent",
        "category10",
        "category20",
        "category20b",
        "category20c",
        "observable10",
        "dark2",
        "paired",
        "pastel1",
        "pastel2",
        "set1",
        "set2",
        "set3",
        "tableau10",
        "tableau20",
        # Sequential single-hue
        "blues",
        "tealblues",
        "teals",
        "greens",
        "browns",
        "oranges",
        "reds",
        "purples",
        "warmgreys",
        "greys",
        # Sequential multi-hue
        "viridis",
        "magma",
        "inferno",
        "plasma",
        "cividis",
        "turbo",
        "bluegreen",
        "bluepurple",
        "goldgreen",
        "goldorange",
        "goldred",
        "greenblue",
        "orangered",
        "purplebluegreen",
        "purpleblue",
        "purplered",
        "redpurple",
        "yellowgreenblue",
        "yellowgreen",
        "yelloworangebrown",
        "yelloworangered",
        # Sequential multi-hue for dark / light backgrounds
        "darkblue",
        "darkgold",
        "darkgreen",
        "darkmulti",
        "darkred",
        "lightgreyred",
        "lightgreyteal",
        "lightmulti",
        "lightorange",
        "lighttealblue",
        # Diverging
        "blueorange",
        "brownbluegreen",
        "purplegreen",
        "pinkyellowgreen",
        "purpleorange",
        "redblue",
        "redgrey",
        "redyellowblue",
        "redyellowgreen",
        "spectral",
        # Cyclical
        "rainbow",
        "sinebow",
    }
)

# d3-format's specifier grammar ([[fill]align][sign][symbol][0][width][,][.precision][~][type]),
# copied from d3-format 3.1.0 as bundled by vl-convert; a string outside it makes d3 throw
# "invalid format" at render. d3 does not throw on an unknown type letter (e.g. ".1F") but
# silently falls back to ".12~g" (labels like "2e+2"), so its real types are checked too.
_D3_NUMBER_FORMAT = re.compile(
    pattern=r"^(?:(.)?([<>=^]))?([+\-( ])?([$#])?(0)?(\d+)?(,)?(\.\d+)?(~)?([a-z%])?$",
    flags=re.IGNORECASE,
)
_D3_NUMBER_FORMAT_TYPES = frozenset("%bcdefgnoprsXx")

# Whole-number values in this range are read as calendar years (e.g. `ano` = 2019).
_YEAR_RANGE = range(1000, 3001)

# Marks one series is commonly drawn with, one layer per series.
_SERIES_MARKS = frozenset(
    {"line", "area", "bar", "point", "circle", "square", "trail", "tick"}
)

# Channels that tell series apart, and so give a chart its legend.
_SERIES_CHANNELS = ("color", "fill", "stroke", "strokeDash", "shape")

# Marks with an area other marks (e.g. white labels) can sit on.
_FILLED_MARKS = frozenset({"bar", "area", "rect", "arc"})

# The properties that carry a mark's visible color, by mark type (default: `color`).
_MARK_COLOR_KEYS: dict[str, tuple[str, ...]] = {
    "line": ("color", "stroke"),
    "rule": ("color", "stroke"),
    "trail": ("color", "stroke"),
    "bar": ("color", "fill"),
    "area": ("color", "fill"),
    "rect": ("color", "fill"),
    "arc": ("color", "fill"),
    "circle": ("color", "fill"),
    "square": ("color", "fill"),
    "tick": ("color", "fill"),
    "text": ("color", "fill"),
}

# Colors that vanish on the chat panel's white background: white-like names, and any color
# whose darkest RGB channel is at least this light.
_WHITE_COLOR_NAMES = frozenset({"white", "snow", "ghostwhite", "whitesmoke", "ivory"})
_MIN_INVISIBLE_CHANNEL = 235

# Fields a transform creates when its `as` is omitted (Vega-Lite's defaults).
_DEFAULT_TRANSFORM_OUTPUTS = {
    "sequence": ("data",),
    "fold": ("key", "value"),
    "density": ("value", "density"),
    "quantile": ("prob", "value"),
}

# System prompt for the chart spec generation.
_CHART_SPEC_INSTRUCTIONS = """\
You are a data visualization specialist. Given an already-aggregated query result — its row count, each column's BigQuery type with a summary over every row, and an evenly spaced sample of rows — and a description of the chart to build, return one complete Vega-Lite v6 spec. Reference the result's columns by their exact names, and never include a data source, dataset, or URL — the exact rows are injected separately. You may use composite views and transforms when they make the chart clearer.

# Guidelines

- Follow the description when it is specific. When it leaves the chart type open, pick the clearest form for the data's shape.
- When a chart to change is given, start from it: apply only the requested changes and keep its other choices (chart type, encodings, colors, labels, sizes).
- Add a short title, label both axes with human-readable text rather than the raw column names.
- Add a legend whenever a view shows two or more series.
- Write every title, axis label and legend in the language the request names.
- Base scale, size and layout choices on the column summaries (ranges, distinct counts), not just the sample rows. For example, panels whose value ranges barely overlap should not share an axis.

## Color

By default, set no color-related property — no `scale`, `scheme`, `sort`, `range`, or explicit color value — on any encoding, mark, or config, whether the field is categorical, quantitative, or there's only one series. Just map the field to the color channel and leave everything else about its color to the frontend's brand palette.

The one exception is when the description explicitly asks for a specific color. Then honor exactly what was asked, choosing the encoding that fits the chart and the number of series being colored, and set only that — leave every other color to the frontend. Do not invent a color, scheme, or palette the description did not ask for.

Give the color as explicit CSS colors (a hex string or a standard color keyword) whenever the request names a color rather than a palette: a single fixed color as a constant color `value`, and specific per-category colors as a `range` on the color encoding's `scale`. For a gradient over a quantitative or ordinal field, give a `range` of two or more explicit colors, e.g. light-to-dark for low-to-high.

Use a named `scheme` only when the request names or clearly maps to one of the built-in Vega schemes below, and then use its name exactly. Never guess or invent a scheme name. If no built-in scheme matches the requested color, use an explicit `range` instead.

- Categorical: `accent`, `category10`, `category20`, `category20b`, `category20c`, `observable10`, `dark2`, `paired`, `pastel1`, `pastel2`, `set1`, `set2`, `set3`, `tableau10`, `tableau20`.
- Sequential single-hue: `blues`, `tealblues`, `teals`, `greens`, `browns`, `oranges`, `reds`, `purples`, `warmgreys`, `greys`.
- Sequential multi-hue: `viridis`, `magma`, `inferno`, `plasma`, `cividis`, `turbo`, `bluegreen`, `bluepurple`, `goldgreen`, `goldorange`, `goldred`, `greenblue`, `orangered`, `purplebluegreen`, `purpleblue`, `purplered`, `redpurple`, `yellowgreenblue`, `yellowgreen`, `yelloworangebrown`, `yelloworangered`.
- Diverging: `blueorange`, `brownbluegreen`, `purplegreen`, `pinkyellowgreen`, `purpleorange`, `redblue`, `redgrey`, `redyellowblue`, `redyellowgreen`, `spectral`.
- Cyclical: `rainbow`, `sinebow`.

## Number formatting

Abbreviate any large-magnitude axis or label with SI notation (the `~s` / `s` format), then rewrite its prefix letters (k, M, G, T) to follow the naming convention the measured quantity uses in the requested language, via an axis `labelExpr` that reformats `datum.value`. Do this on every axis whose numbers reach the thousands or beyond.

## Years

A year stored as a plain integer (e.g. `2019`) is not a date: Vega-Lite reads a number in a `temporal` field as milliseconds since 1970, so every year collapses onto one "1969"/"1970" tick. Never encode an integer year column as `temporal` or give it a `timeUnit`; encode it as `quantitative` (or `ordinal` for bars) with the integer label format on its axis, `"axis": {"format": "d"}` — a `format` on the field itself does not reach axis labels — and never abbreviate it with SI notation. Use `temporal` only for real date or datetime columns.

## Log scales

A log scale is undefined at zero and below, so a single non-positive value in the encoded field collapses the whole axis — no ticks, every point crushed into a corner. Real-world columns routinely carry zeros, nulls, and occasional negatives. So whenever you set `"scale": {"type": "log"}`, first drop the non-positive rows with a `filter` transform over every log-scaled field. If a field genuinely spans zero and cannot be filtered, use a linear scale instead.

## Scale domains

Never write `null` inside a literal `scale.domain` array (e.g. `{"domain": [0, null]}`) — Vega-Lite treats a literal domain as authoritative and does not fill a `null` entry in from the data, so the scale silently collapses to a zero-width range and the chart renders empty (no bars, no marks) with no error. To pin one end while leaving the other data-driven, use `domainMin`/`domainMax` instead (e.g. `{"domainMin": 0}`). Quantitative scales already default to including zero, so most of the time no domain override is needed at all.

## Layered charts

In a `layer`, layers that share an axis are merged into one axis, and a property hidden on any layer is hidden on the merged axis. Never set `"labels": false`, `"ticks": false`, `"title": null` or `"axis": null` on a secondary layer to avoid duplicates — that removes the labels from the whole chart. Define each axis once, on the first layer or in a shared top-level `encoding`, and leave the other layers' axes unset.

## Value labels

To show values on bars, lines or points, add a separate layer with a `text` mark; a `text` channel on any other mark is ignored, so the labels silently disappear.

## Size

The chart is rendered at the container's width, so never set a width. For any chart with many categories along one axis (see that column's distinct count), set an explicit height so the cells or bands are not overly tall — a height noticeably smaller than the chart's width reads best. Otherwise leave the height to default.

## Long labels

The chart renders inside a chat panel a few hundred pixels wide, not a full-page dashboard. Never set `labelLimit` on an axis or legend — leave it to Vega-Lite's default (180px), which already ellipsis-truncates an overly long label and keeps the full text on hover. An axis's label column shares the container's width with the plot itself, so a large override (e.g. several hundred pixels) steals space directly from the marks, leaving them cramped or invisible.

## Choropleth maps

To shade a Brazilian map by a value, use a `geoshape` mark whose data is the map geometry, injected by name, joined to your result with a `lookup` transform. Never use a URL. Available geometry (each feature's `id` is the join key, its `properties.name` the label):

- `brazil_states`: the 27 states; `id` is the UF code as text (e.g. "SP"). Join it to your result's UF column (usually `sigla_uf`).
- `brazil_municipalities`: the municipalities; `id` is the 7-digit IBGE code as text (e.g. "3550308"). Join it to your result's municipality id column (usually `id_municipio`).

Set the geoshape's data to `{"name": "brazil_states"}` (or `"brazil_municipalities"`) and reference your result as `{"name": "query_result"}`; the exact geometry and rows are injected separately. Join with a `lookup` on the geometry's `id`, matching your result's location column as the `key`, and pull the value column via `fields`. Color by that value. Use a `mercator` projection, add a `properties.name` + value tooltip, and set an explicit `height` (Brazil is roughly square, so a height close to the width works well). Examples:

{
  "data": {"name": "brazil_states"},
  "transform": [{"lookup": "id", "from": {"data": {"name": "query_result"}, "key": "sigla_uf", "fields": ["valor"]}}],
  "projection": {"type": "mercator"},
  "mark": "geoshape",
  "encoding": {
    "color": {"field": "valor", "type": "quantitative", "title": "…"},
    "tooltip": [{"field": "properties.name", "type": "nominal", "title": "Estado"}, {"field": "valor", "type": "quantitative"}]
  },
}

{
  "data": {"name": "brazil_municipalities"},
  "transform": [{"lookup": "id", "from": {"data": {"name": "query_result"}, "key": "id_municipio", "fields": ["valor"]}}],
  "projection": {"type": "mercator"},
  "mark": "geoshape",
  "encoding": {
    "color": {"field": "valor", "type": "quantitative", "title": "…"},
    "tooltip": [{"field": "properties.name", "type": "nominal", "title": "Município"}, {"field": "valor", "type": "quantitative"}]
  },
}
"""


class ChartHandleNotFound(Exception):
    """No query result exists for the given query_ref in this thread."""


class ChartResultTooLarge(Exception):
    """The result has more rows than a chart should bind."""


class ChartSpecInvalid(Exception):
    """The generator could not produce a spec that passes validation."""


class ChartSpecTimeout(Exception):
    """The generator did not produce a valid spec within the deadline."""


class ChartSpec(BaseModel):
    """The tool the charting model calls: a free-form Vega-Lite spec."""

    # Description kept to a concise label per OpenAI's gpt-5.x guidance ("state each
    # instruction once"); the how-to lives once in _CHART_SPEC_INSTRUCTIONS (system prompt).
    spec: dict[str, Any] = Field(description="The complete Vega-Lite v6 specification.")


# ===================================================================
# CHART DATA FETCHING
# ===================================================================
@cache
def _bq_client() -> bq.Client:  # pragma: no cover
    return bq.Client(
        project=settings.GOOGLE_BILLING_PROJECT,
        credentials=settings.GOOGLE_CREDENTIALS,
    )


def _json_default(value: Any) -> Any:
    """Coerce a BigQuery value that JSON can't render natively into a JSON-native one.

    `Decimal` (NUMERIC/BIGNUMERIC) becomes a number so quantitative encodings and color
    scales see a number, not a string. Everything else (date/datetime/…) falls back to
    its string form, which Vega-Lite parses for temporal encodings.

    Args:
        value (Any): The value json.dumps could not serialize.

    Returns:
        Any: A JSON-native replacement.
    """
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    return str(value)


def _fetch_rows(
    destination_table: dict[str, Any],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Fetch a result table's rows for charting, capped at settings.CHART_MAX_BYTES.

    Fetches row by row and stops as soon as the JSON that would be bound into the spec
    exceeds the budget, so a huge result is rejected without materializing all of it.

    Args:
        destination_table (dict[str, Any]): `TableReference.to_api_repr()` of the result table.

    Returns:
        tuple[dict[str, str], list[dict[str, Any]]]: Column names mapped to their BigQuery
            types, in schema order, and row dicts.

    Raises:
        ChartResultTooLarge: The bound data would exceed settings.CHART_MAX_BYTES.
        ResultTableExpired: The result table no longer exists (~24h TTL).
    """
    table_ref = bq.TableReference.from_api_repr(destination_table)

    try:
        table = _bq_client().get_table(table_ref)

        columns = {field.name: field.field_type for field in table.schema}

        rows = []
        size = 0
        for row in _bq_client().list_rows(table_ref):
            # Serialize once to both measure the row and coerce BigQuery types
            # (date/datetime/Decimal/…) to JSON-native values. Raw objects are
            # not valid JSON and would fail serialization when the spec is bound.
            serialized = json.dumps(
                dict(row), ensure_ascii=False, default=_json_default
            )
            size += len(serialized.encode())
            if size > settings.CHART_MAX_BYTES:
                raise ChartResultTooLarge(
                    f"The result is too large to chart (over "
                    f"{settings.CHART_MAX_BYTES // (1024 * 1024)} MB of data). Aggregate "
                    "or summarize it in SQL first, then chart the smaller result."
                )
            rows.append(json.loads(serialized))
    except NotFound as e:
        raise ResultTableExpired(str(e)) from e

    return columns, rows


async def fetch_chart_data(
    query_ref: str, thread_id: str
) -> tuple[QueryHandle, dict[str, str], list[dict[str, Any]]]:
    """Resolve a chartable result: authorize the handle, then read its capped rows.

    Args:
        query_ref (str): The handle of the result to chart.
        thread_id (str): The thread the handle must belong to (authorization).

    Returns:
        tuple[QueryHandle, dict[str, str], list[dict[str, Any]]]: The handle, its columns
            (mapped to their BigQuery types), and rows.

    Raises:
        ChartHandleNotFound: No such result in this thread.
        ResultTableExpired: The result expired (by age or missing table).
    """
    async with sessionmaker() as session:
        handle = await AsyncDatabase(session).get_query_handle_from_thread(
            query_ref, thread_id
        )

    if handle is None:
        raise ChartHandleNotFound(
            f"No query result found for '{query_ref}'. "
            "Call list_query_results to see the available results."
        )

    if is_result_expired(handle.created_at):
        raise ResultTableExpired(
            f"The result for '{query_ref}' has expired (results are kept ~24h). "
            "Re-run the query, then chart the new result."
        )

    columns, rows = await asyncio.to_thread(_fetch_rows, handle.destination_table)

    return handle, columns, rows


# ===================================================================
# CHART DATA INJECTION
# ===================================================================
def _geo_url_node(name: str) -> dict[str, Any]:
    """Build a Vega-Lite TopoJSON data node that points at the geometry by URL.

    The URL is the website's static path for the file; the browser fetches (and caches) the
    ~megabytes of geometry directly, so it never inflates the spec, the SSE payload, or the
    persisted `events`.

    Args:
        name (str): A geo source name from `_GEO_ASSETS` (e.g. "brazil_states").

    Returns:
        dict[str, Any]: The `data` node — a topojson `url` plus its topojson `format`.
    """
    asset = _GEO_ASSETS[name]
    return {
        "url": f"{settings.GEO_ASSET_URL_BASE.rstrip('/')}/{asset['file']}",
        "format": {"type": "topojson", "feature": asset["feature"]},
    }


def _geo_stub_node(name: str) -> dict[str, Any]:
    """Build a minimal single-feature TopoJSON node for offline spec validation.

    Args:
        name (str): A geo source name from `_GEO_ASSETS` (e.g. "brazil_states").

    Returns:
        dict[str, Any]: A `data` node with a tiny inline TopoJSON and its topojson `format`.
    """
    feature = _GEO_ASSETS[name]["feature"]
    stub = {
        "type": "Topology",
        "objects": {
            feature: {
                "type": "GeometryCollection",
                "geometries": [
                    {"type": "Polygon", "id": "0", "properties": {}, "arcs": [[0]]}
                ],
            }
        },
        "arcs": [[[0, 0], [1, 0], [0, 1], [0, 0]]],
    }
    return {"values": stub, "format": {"type": "topojson", "feature": feature}}


def _resolve_named_data(
    node: JsonValue,
    geo_node: Callable[[str], dict[str, Any]],
) -> JsonValue:
    """Replace every `{"name": <geo source>}` data reference with real geometry.

    A geo name resolves via a `geo_node` — a URL node for a render-ready spec, a tiny
    inline stub for offline validation. `{"name": "query_result"}` references are kept:
    they resolve to the rows bound once under top-level `datasets` (see `_bind_rows`).
    Any other `{"name": ...}` was already dropped by `_sanitize_chart_spec`.

    Args:
        node (JsonValue): A spec, or any node within it, to walk.
        geo_node (Callable[[str], dict[str, Any]]): Resolver for a geo source name.

    Returns:
        JsonValue: The same node with its geo data references resolved.
    """
    if isinstance(node, dict):
        if set(node) == {"name"} and node["name"] in _GEO_ASSETS:
            return geo_node(node["name"])
        return {
            key: _resolve_named_data(value, geo_node) for key, value in node.items()
        }
    if isinstance(node, list):
        return [_resolve_named_data(item, geo_node) for item in node]
    return node


def _bind_rows(spec: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Bind the result rows once, as the `query_result` dataset, and set the schema.

    Every `{"name": "query_result"}` reference — in any layer, facet or lookup — reads the
    same top-level dataset, so the rows appear once in the payload however many views use
    them. A plain chart that declared no data reads them by default.

    Args:
        spec (dict[str, Any]): A sanitized spec with its geo sources resolved.
        rows (list[dict[str, Any]]): The rows to bind.

    Returns:
        dict[str, Any]: The spec with the rows bound.
    """
    chart = {
        **spec,
        "$schema": VEGA_LITE_SCHEMA,
        "datasets": {_RESULT_SOURCE: rows},
    }
    chart.setdefault("data", {"name": _RESULT_SOURCE})
    return chart


def inject_chart_data(
    spec: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Inject the real geometry and result rows into an already-sanitized spec.

    Takes an already sanitized spec from `generate_chart_spec` and makes it render-ready:
    resolves its geo sources to TopoJSON URLs and binds the rows once as the
    `query_result` dataset (see `_bind_rows`).

    Args:
        spec (dict[str, Any]): A sanitized, data-less spec from `generate_chart_spec`.
        rows (list[dict[str, Any]]): The exact result rows to render.

    Returns:
        dict[str, Any]: A render-ready Vega-Lite spec with the real data bound in.
    """
    return _bind_rows(_resolve_named_data(spec, _geo_url_node), rows)


def describe_chart(spec: dict[str, Any]) -> dict[str, Any]:
    """Summarize what a chart draws, so the agent can describe it as it is.

    Args:
        spec (dict[str, Any]): A sanitized spec.

    Returns:
        dict[str, Any]: The title, any facet fields, and each view's mark with the field (or
            constant datum) on each channel; tooltips are left out.
    """
    title = spec.get("title")
    if isinstance(title, dict):
        title = title.get("text")

    views = []
    for unit, encoding in _iter_units(spec):
        channels = {}
        for channel, definition in encoding.items():
            if channel == "tooltip" or not isinstance(definition, dict):
                continue
            if isinstance(definition.get("field"), str):
                channels[channel] = definition["field"]
            elif "datum" in definition:
                channels[channel] = {"datum": definition["datum"]}
        views.append({"mark": _mark_type(unit["mark"]), "encoding": channels})

    summary: dict[str, Any] = {"title": title, "views": views}
    facet = spec.get("facet")
    if isinstance(facet, dict):
        if isinstance(facet.get("field"), str):
            summary["facet"] = facet["field"]
        else:
            summary["facet"] = {
                key: value["field"]
                for key, value in facet.items()
                if isinstance(value, dict) and isinstance(value.get("field"), str)
            }
    return summary


def _unresolve_geo_urls(node: JsonValue) -> JsonValue:
    """Turn every geometry URL node (see `_geo_url_node`) back into its `{"name": ...}`."""
    if isinstance(node, dict):
        url = node.get("url")
        if isinstance(url, str):
            filename = url.rsplit("/", 1)[-1]
            for name, asset in _GEO_ASSETS.items():
                if asset["file"] == filename:
                    return {"name": name}
        return {key: _unresolve_geo_urls(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_unresolve_geo_urls(item) for item in node]
    return node


def unbind_chart_spec(chart: dict[str, Any]) -> dict[str, Any]:
    """Turn a render-ready chart back into the data-less spec it was made from.

    Undoes `inject_chart_data` — drops the bound rows and turns geometry URLs back into
    their names — so an earlier chart can be handed to the generator to change.

    Args:
        chart (dict[str, Any]): A chart from `inject_chart_data`.

    Returns:
        dict[str, Any]: The sanitized, data-less spec.
    """
    spec = {
        key: value for key, value in chart.items() if key not in ("$schema", "datasets")
    }
    unbound = _sanitize_chart_spec(_unresolve_geo_urls(spec))
    assert isinstance(unbound, dict)  # an object spec sanitizes to an object
    return unbound


# ===================================================================
# CHART SPEC GENERATION
# ===================================================================
@cache
def _chart_spec_model():  # pragma: no cover
    """A model that returns a parsed `ChartSpec` via forced function-calling.

    `method="function_calling"` (not strict json_schema, which 400s on the open spec) and
    `include_raw=True` (a bad call surfaces as `parsed=None`, not a raise) are required.
    A request that hangs is cut off by the timeout and retried once.
    """
    return ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.MODEL_URI,
        reasoning={"effort": "medium"},
        timeout=settings.CHART_SPEC_REQUEST_TIMEOUT_SECONDS,
        max_retries=1,
    ).with_structured_output(
        ChartSpec,
        method="function_calling",
        include_raw=True,
    )


def _spread_sample(rows: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Pick `size` rows evenly spaced across the result, first and last included.

    A result is usually sorted by its grouping columns, so the first rows often all share
    one group; an even spread shows every part of it.

    Args:
        rows (list[dict[str, Any]]): The result rows.
        size (int): How many rows to pick.

    Returns:
        list[dict[str, Any]]: The sample, in result order.
    """
    if len(rows) <= size:
        return rows
    step = (len(rows) - 1) / (size - 1)
    return [rows[round(i * step)] for i in range(size)]


def _summarize_column(bq_type: str, values: list[Any]) -> dict[str, Any]:
    """Summarize one column over every row, for the spec generator.

    Args:
        bq_type (str): The column's BigQuery type.
        values (list[Any]): The column's value in every row.

    Returns:
        dict[str, Any]: The type, distinct and null counts, plus min/max for numbers and
            dates, or the distinct values (all of them, or a few examples) for other columns.
    """
    present = [value for value in values if value is not None]
    distinct = sorted({json.dumps(value, sort_keys=True) for value in present})
    summary: dict[str, Any] = {
        "type": bq_type,
        "distinct": len(distinct),
        "nulls": len(values) - len(present),
    }

    if not present:
        return summary

    numeric = all(
        isinstance(value, int | float) and not isinstance(value, bool)
        for value in present
    )
    if numeric or bq_type in _TEMPORAL_TYPES:
        summary["min"], summary["max"] = min(present), max(present)
    elif len(distinct) <= _SUMMARY_MAX_LISTED_VALUES:
        summary["values"] = [json.loads(value) for value in distinct]
    else:
        summary["examples"] = [
            json.loads(value) for value in distinct[:_SUMMARY_EXAMPLES]
        ]

    return summary


def _describe_result(columns: dict[str, str], rows: list[dict[str, Any]]) -> str:
    """Describe a result for the spec generator: size, column summaries, and a sample.

    Args:
        columns (dict[str, str]): The result's columns mapped to their BigQuery types.
        rows (list[dict[str, Any]]): The result rows.

    Returns:
        str: The data section of the spec generator's user message.
    """
    summary = {
        column: _summarize_column(bq_type, [row[column] for row in rows])
        for column, bq_type in columns.items()
    }
    sample = _spread_sample(rows, CHART_SPEC_SAMPLE_ROWS)
    return (
        f"Rows: {len(rows)}\n\n"
        "Columns (BigQuery type, then a summary over every row): "
        f"{json.dumps(summary, ensure_ascii=False, default=str)}\n\n"
        "Sample rows (evenly spaced across the result): "
        f"{json.dumps(sample, ensure_ascii=False, default=str)}"
    )


def _chart_spec_user_prompt(
    instructions: str,
    language: LanguageCode,
    data_description: str,
    base_spec: dict[str, Any] | None,
    previous_spec: dict[str, Any] | None,
    errors: list[str],
) -> str:
    """Build the task-specific user message.

    On a retry, appends the model's own rejected spec and the validator's problems so it
    can fix that spec rather than regenerate from scratch.

    Args:
        instructions (str): The natural-language description of the chart to produce.
        language (LanguageCode): The language of the chart's text (the user's language).
        data_description (str): The result's size, column summaries and sample rows.
        base_spec (dict[str, Any] | None): An earlier chart to change, or None for a new one.
        previous_spec (dict[str, Any] | None): The last rejected spec, echoed back on a retry.
        errors (list[str]): The reasons the previous spec was rejected; empty on the first try.

    Returns:
        str: The user message for the chart-spec model.
    """
    prompt = (
        f"What to chart: {instructions}\n\n"
        f"Language of the chart's text: {language_name(language)}\n\n"
        f"{data_description}"
    )

    if base_spec is not None:
        prompt += (
            "\n\nChart to change (keep everything the request does not ask to change):\n"
            f"{json.dumps(base_spec, ensure_ascii=False, default=str)}"
        )

    if errors:
        # Show the model its own rejected spec so it can edit that spec to fix the
        # problems, rather than regenerating from scratch on each retry.
        if previous_spec is not None:
            prompt += (
                "\n\nYour previous spec:\n"
                f"{json.dumps(previous_spec, ensure_ascii=False, default=str)}"
            )
        feedback = "- " + "\n- ".join(errors)
        prompt += f"\n\nIt was rejected — fix these problems:\n{feedback}"

    return prompt


def _sanitize_chart_spec(node: JsonValue) -> JsonValue:
    """Sanitize a model spec: drop untrusted keys, keep only allowlisted named sources.

    Every `_UNTRUSTED_KEYS` key (`datasets`, `url`, `href`) is removed outright, and a `data`
    node survives only when it is exactly `{"name": <allowlisted source>}` or a data
    generator (whose own `name` is dropped, so it cannot shadow a source), so the model can
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
                elif (
                    isinstance(value, dict)
                    and len(set(value) & _DATA_GENERATORS) == 1
                    and set(value) - {"name"} <= _DATA_GENERATORS
                ):
                    out[key] = {
                        generator: _sanitize_chart_spec(value[generator])
                        for generator in set(value) & _DATA_GENERATORS
                    }
                # any other data node (inline values, url, unknown name) is dropped
                continue
            out[key] = _sanitize_chart_spec(value)
        return out
    if isinstance(node, list):
        return [_sanitize_chart_spec(item) for item in node]
    return node


def _collect(node: JsonValue, key: str) -> set[str]:
    """Collect every string value stored under `key`, anywhere in the spec.

    Args:
        node (JsonValue): A spec, or any node within it, to walk.
        key (str): The key whose string values to collect (e.g. "field").

    Returns:
        set[str]: Every string found under `key`.
    """
    found: set[str] = set()
    if isinstance(node, dict):
        for k, value in node.items():
            if k == key and isinstance(value, str):
                found.add(value)
            else:
                found |= _collect(value, key)
    elif isinstance(node, list):
        for item in node:
            found |= _collect(item, key)
    return found


def _collect_derived_fields(node: JsonValue) -> set[str]:
    """Collect every field a spec's transforms create, anywhere within it.

    A transform names its outputs in `as`: a string (`calculate`, `aggregate`, `window`, …)
    or a list (`fold`, `stack`, `regression`, …). A bin transform also creates `<as>_end`,
    and `fold`, `density`, `quantile` and a `sequence` generator fall back to default names
    when `as` is omitted.

    Args:
        node (JsonValue): A spec, or any node within it, to walk.

    Returns:
        set[str]: Every field name a transform creates.
    """
    found: set[str] = set()
    if isinstance(node, dict):
        output = node.get("as")
        if isinstance(output, str):
            found.add(output)
            if "bin" in node:
                found.add(f"{output}_end")
        elif isinstance(output, list):
            found |= {name for name in output if isinstance(name, str)}
        else:
            for transform, defaults in _DEFAULT_TRANSFORM_OUTPUTS.items():
                if transform in node:
                    found.update(defaults)
        for value in node.values():
            found |= _collect_derived_fields(value)
    elif isinstance(node, list):
        for item in node:
            found |= _collect_derived_fields(item)
    return found


def _collect_scheme_names(node: JsonValue) -> set[str]:
    """Collect every color-scheme name a spec references, anywhere within it.

    A `scheme` is either a string (`"scheme": "reds"`) or a `SchemeParams` object whose
    `name` holds the scheme (`"scheme": {"name": "reds", "extent": [...]}`); this collects
    the name from both forms so `_validate_chart_spec` can reject an unknown one.

    Args:
        node (JsonValue): A spec, or any node within it, to walk.

    Returns:
        set[str]: Every scheme name found under a `scheme` key.
    """
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "scheme":
                if isinstance(value, str):
                    found.add(value)
                elif isinstance(value, dict) and isinstance(value.get("name"), str):
                    found.add(value["name"])
            found |= _collect_scheme_names(value)
    elif isinstance(node, list):
        for item in node:
            found |= _collect_scheme_names(item)
    return found


def _collect_number_formats(node: JsonValue) -> set[str]:
    """Collect every number-format string a spec applies, anywhere within it.

    A format is a number format when it sits on a quantitative field definition (or one
    with `formatType: "number"`), either directly or under that definition's `axis` or
    `legend`. Formats on temporal fields, or with a time `formatType`, are date formats
    and are not collected.

    Args:
        node (JsonValue): A spec, or any node within it, to walk.

    Returns:
        set[str]: Every number-format string found.
    """
    found: set[str] = set()
    if isinstance(node, dict):
        format_type = node.get("formatType")
        if format_type == "number" or (
            node.get("type") == "quantitative" and format_type not in ("time", "utc")
        ):
            for holder in (node, node.get("axis"), node.get("legend")):
                if isinstance(holder, dict) and isinstance(holder.get("format"), str):
                    found.add(holder["format"])
        for value in node.values():
            found |= _collect_number_formats(value)
    elif isinstance(node, list):
        for item in node:
            found |= _collect_number_formats(item)
    return found


def _collect_temporal_fields(node: JsonValue) -> set[str]:
    """Collect every field a spec treats as a date, anywhere within it.

    A field is treated as a date when its definition is `temporal` or carries a
    `timeUnit` (which also covers a `timeUnit` transform).

    Args:
        node (JsonValue): A spec, or any node within it, to walk.

    Returns:
        set[str]: Every field name treated as a date.
    """
    found: set[str] = set()
    if isinstance(node, dict):
        if isinstance(node.get("field"), str) and (
            node.get("type") == "temporal" or "timeUnit" in node
        ):
            found.add(node["field"])
        for value in node.values():
            found |= _collect_temporal_fields(value)
    elif isinstance(node, list):
        for item in node:
            found |= _collect_temporal_fields(item)
    return found


@cache
def _encoding_channel_validators() -> dict[str, Draft7Validator]:
    """One JSON-schema validator per Vega-Lite encoding channel, keyed by channel name."""
    schema = json.loads(_VEGA_LITE_SCHEMA_FILE.read_text())
    definitions = schema["definitions"]
    return {
        channel: Draft7Validator({**channel_schema, "definitions": definitions})
        for channel, channel_schema in definitions["FacetedEncoding"][
            "properties"
        ].items()
    }


def _error_leaves(error: ValidationError) -> Iterator[ValidationError]:
    """The innermost errors under an `anyOf`/`oneOf` error, or the error itself."""
    if not error.context:
        yield error
    for sub_error in error.context:
        yield from _error_leaves(sub_error)


def _unsupported_properties(validator: Draft7Validator, value: JsonValue) -> list[str]:
    """Describe the properties a channel definition carries that its schema does not allow.

    A channel accepts several shapes (field, datum, value, …), so the schema tries each in
    an `anyOf`. The shape the model meant is the one whose own top-level keys fit best
    (then the one with the fewest errors), and its unexpected properties are reported.
    Only unexpected properties count: Vega-Lite silently ignores them, while other schema
    violations either fail the compile step or are tolerated by Vega-Lite.

    Args:
        validator (Draft7Validator): The channel's validator.
        value (JsonValue): The channel definition.

    Returns:
        list[str]: "<location> has unsupported <keys>" messages; empty if none.
    """
    branches: dict[tuple, list[ValidationError]] = {}
    for error in validator.iter_errors(value):
        for leaf in _error_leaves(error):
            schema_path = list(leaf.absolute_schema_path)
            branch = (
                tuple(schema_path[: schema_path.index("anyOf") + 2])
                if "anyOf" in schema_path
                else ()
            )
            branches.setdefault(branch, []).append(leaf)

    if not branches:
        return []

    def fit(leaves: list[ValidationError]) -> tuple[int, int]:
        top_level_unexpected = sum(
            len(set(leaf.instance) - set(leaf.schema.get("properties", {})))
            for leaf in leaves
            if leaf.validator == "additionalProperties" and not leaf.absolute_path
        )
        return top_level_unexpected, len(leaves)

    unsupported: dict[str, list[str]] = {}
    for leaf in min(branches.values(), key=fit):
        if leaf.validator != "additionalProperties":
            continue
        location = "".join(f".{part}" for part in leaf.absolute_path)
        keys = sorted(set(leaf.instance) - set(leaf.schema.get("properties", {})))
        # The same location can fail in several nested shapes; keep the closest one.
        if location not in unsupported or len(keys) < len(unsupported[location]):
            unsupported[location] = keys

    return [
        f"{location} has unsupported {keys}" for location, keys in unsupported.items()
    ]


def _collect_unsupported_encoding_properties(
    node: JsonValue, path: str = ""
) -> list[str]:
    """Find encoding channels, anywhere in a spec, with properties Vega-Lite ignores.

    Vega-Lite drops an unknown property without an error, so a misplaced one (e.g. `color`
    nested inside `y`) compiles, renders, and silently loses what it was meant to do.

    Args:
        node (JsonValue): A spec, or any node within it, to walk.
        path (str): The location of `node` within the spec, for the messages.

    Returns:
        list[str]: One message per offending channel location.
    """
    found: list[str] = []
    if isinstance(node, dict):
        encoding = node.get("encoding")
        if isinstance(encoding, dict):
            validators = _encoding_channel_validators()
            for channel, definition in encoding.items():
                location = f"{path}.encoding.{channel}".lstrip(".")
                if channel not in validators:
                    found.append(f"{location} is not a Vega-Lite encoding channel")
                    continue
                found += [
                    f"{location}{problem}"
                    for problem in _unsupported_properties(
                        validators[channel], definition
                    )
                ]
        for key, value in node.items():
            found += _collect_unsupported_encoding_properties(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found += _collect_unsupported_encoding_properties(item, f"{path}[{index}]")
    return found


def _integer_year_columns(columns: list[str], rows: list[dict[str, Any]]) -> set[str]:
    """Find the columns whose every non-null value is a whole-number calendar year.

    Args:
        columns (list[str]): The result's column names.
        rows (list[dict[str, Any]]): The result rows.

    Returns:
        set[str]: Columns holding only integers within `_YEAR_RANGE` (and at least one).
    """
    year_columns: set[str] = set()
    for column in columns:
        values = [row[column] for row in rows if row[column] is not None]
        if values and all(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value in _YEAR_RANGE
            for value in values
        ):
            year_columns.add(column)
    return year_columns


def _is_valid_number_format(specifier: str) -> bool:
    """Return whether d3-format accepts a number-format specifier.

    Args:
        specifier (str): The format string (e.g. ".1f", ",.0f", "~s").

    Returns:
        bool: True if d3-format can parse it.
    """
    match = _D3_NUMBER_FORMAT.match(specifier)
    if match is None:
        return False
    format_type = match.group(10)
    return format_type is None or format_type in _D3_NUMBER_FORMAT_TYPES


def _mark_type(mark: JsonValue) -> str | None:
    """The type of a `mark`, written either as a string or as a mark definition."""
    if isinstance(mark, str):
        return mark
    if isinstance(mark, dict) and isinstance(mark.get("type"), str):
        return mark["type"]
    return None


def _iter_units(
    node: JsonValue, inherited: dict[str, Any] | None = None
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Yield every unit view (a node with a `mark`) with its effective encoding.

    A layer's views inherit the layer's encoding; facets, repeats and concatenations
    do not share theirs with the views inside.

    Args:
        node (JsonValue): A spec, or any node within it, to walk.
        inherited (dict[str, Any] | None): The encoding inherited from enclosing layers.

    Yields:
        tuple[dict[str, Any], dict[str, Any]]: A unit view and its effective encoding.
    """
    if isinstance(node, list):
        for item in node:
            yield from _iter_units(item, inherited)
        return
    if not isinstance(node, dict):
        return

    own = node.get("encoding")
    encoding = {**(inherited or {}), **(own if isinstance(own, dict) else {})}
    if "mark" in node:
        yield node, encoding
    if "layer" in node:
        yield from _iter_units(node["layer"], encoding)
    for key in ("spec", "concat", "hconcat", "vconcat"):
        if key in node:
            yield from _iter_units(node[key])


def _normalize_chart_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Fix, instead of rejecting, mistakes that have one obvious correction.

    - A line with a point overlay (`"point": true`) and a `strokeDash` encoding: the
      legend draws the overlay's point symbols, which cannot show a dash, so the dashed
      series look identical in it. The overlay is dropped so the legend draws lines.

    Args:
        spec (dict[str, Any]): A sanitized spec; it is changed in place.

    Returns:
        dict[str, Any]: The same spec, normalized.
    """
    for unit, encoding in _iter_units(spec):
        mark = unit["mark"]
        dash = encoding.get("strokeDash")
        if (
            _mark_type(mark) == "line"
            and isinstance(mark, dict)
            and mark.get("point")
            and isinstance(dash, dict)
            and dash.get("field")
        ):
            del mark["point"]
    return spec


def _groups_interleave(groups: list[set[Any]]) -> bool:
    """Whether disjoint groups of x values take turns along the axis.

    Args:
        groups (list[set[Any]]): Each group's x values.

    Returns:
        bool: True when no x value is shared and some group has another group's value
            strictly inside its range (e.g. a highlighted year inside the others' span).
    """
    if len(groups) < 2:
        return False
    seen: set[Any] = set()
    for group in groups:
        if seen & group:
            return False  # parallel series over shared x values, not one split line
        seen |= group
    try:
        return any(
            min(group) < x < max(group)
            for group in groups
            for other in groups
            if other is not group
            for x in other
        )
    except TypeError:  # x values that cannot be ordered
        return False


def _collect_split_lines(spec: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    """Find lines that a grouping channel breaks into pieces taking turns along x.

    Coloring a line by a per-point category (e.g. "maior crescimento" vs "demais anos")
    makes each category its own series, so the line skips every point of the other
    categories and the highlighted points float unconnected. Only grouping and x fields
    that are result columns are checked, against the bound rows.

    Args:
        spec (dict[str, Any]): The candidate spec.
        rows (list[dict[str, Any]]): The result rows.

    Returns:
        list[str]: The grouping fields that split a line.
    """
    if not rows:
        return []
    found: list[str] = []
    for unit, encoding in _iter_units(spec):
        if _mark_type(unit["mark"]) not in ("line", "trail", "area"):
            continue
        x = encoding.get("x")
        x_field = x.get("field") if isinstance(x, dict) else None
        if not isinstance(x_field, str) or x_field not in rows[0]:
            continue
        for channel in ("color", "strokeDash", "detail"):
            definition = encoding.get(channel)
            field = definition.get("field") if isinstance(definition, dict) else None
            if not isinstance(field, str) or field not in rows[0] or field == x_field:
                continue
            groups: dict[str, set[Any]] = {}
            try:
                for row in rows:
                    if row[x_field] is not None:
                        key = json.dumps(row[field], sort_keys=True)
                        groups.setdefault(key, set()).add(row[x_field])
            except TypeError:  # unhashable x values
                continue
            if _groups_interleave(list(groups.values())) and field not in found:
                found.append(field)
    return found


def _collect_unlabeled_layer_series(spec: JsonValue) -> list[list[str]]:
    """Find layers that draw several series with no channel telling them apart.

    One layer per column (e.g. one line for `abertas`, one for `fechadas`), each with a
    fixed color and no color encoding, gives the chart no legend, so readers cannot tell
    the series apart. Series drawn with different marks (a bar and a line) or on
    independent y axes (each with its own title) can be told apart, so they are not
    reported.

    Args:
        spec (JsonValue): A spec, or any node within it, to walk.

    Returns:
        list[list[str]]: For each such layer, the y fields of its unlabeled series.
    """
    found: list[list[str]] = []
    if isinstance(spec, dict):
        if isinstance(spec.get("layer"), list):
            own = spec.get("encoding")
            units = list(
                _iter_units(spec["layer"], own if isinstance(own, dict) else {})
            )
            labeled = any(
                isinstance(encoding.get(channel), dict)
                and ("field" in encoding[channel] or "datum" in encoding[channel])
                for _, encoding in units
                for channel in _SERIES_CHANNELS
            )
            resolve = spec.get("resolve")
            independent_y = (
                isinstance(resolve, dict)
                and isinstance(resolve.get("scale"), dict)
                and resolve["scale"].get("y") == "independent"
            )
            if not labeled and not independent_y:
                y_fields_by_mark: dict[str, set[str]] = {}
                for unit, encoding in units:
                    mark_type = _mark_type(unit["mark"])
                    y = encoding.get("y")
                    if (
                        mark_type in _SERIES_MARKS
                        and isinstance(y, dict)
                        and isinstance(y.get("field"), str)
                    ):
                        y_fields_by_mark.setdefault(mark_type, set()).add(y["field"])
                found += [
                    sorted(fields)
                    for fields in y_fields_by_mark.values()
                    if len(fields) >= 2
                ]
        for value in spec.values():
            found += _collect_unlabeled_layer_series(value)
    elif isinstance(spec, list):
        for item in spec:
            found += _collect_unlabeled_layer_series(item)
    return found


def _is_whiteish(color: JsonValue) -> bool:
    """Whether a CSS color is white or close enough to vanish on a white background."""
    if not isinstance(color, str):
        return False
    value = color.strip().lower()
    if value in _WHITE_COLOR_NAMES:
        return True
    match = re.fullmatch(r"#([0-9a-f]{3}|[0-9a-f]{6})", value) or re.fullmatch(
        r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+).*\)", value
    )
    if match is None:
        return False
    if value.startswith("#"):
        digits = match.group(1)
        if len(digits) == 3:
            digits = "".join(digit * 2 for digit in digits)
        channels = [int(digits[i : i + 2], 16) for i in (0, 2, 4)]
    else:
        channels = [int(group) for group in match.groups()]
    return min(channels) >= _MIN_INVISIBLE_CHANNEL


def _collect_invisible_white_marks(spec: dict[str, Any]) -> list[str]:
    """Find marks drawn in white on the chat's default white background.

    White text is allowed when the chart has a filled mark (bars, areas, …) it can sit on,
    as labels inside bars commonly are.

    Args:
        spec (dict[str, Any]): The candidate spec.

    Returns:
        list[str]: The types of the white marks; empty when a dark background is set.
    """
    config = spec.get("config")
    background = spec.get("background") or (
        config.get("background") if isinstance(config, dict) else None
    )
    if (
        isinstance(background, str)
        and background.strip().lower() != "transparent"
        and not _is_whiteish(background)
    ):
        return []

    units = list(_iter_units(spec))
    has_filled_mark = any(
        _mark_type(unit["mark"]) in _FILLED_MARKS for unit, _ in units
    )

    found: list[str] = []
    for unit, encoding in units:
        mark = unit["mark"]
        mark_type = _mark_type(mark)
        if mark_type is None or (mark_type == "text" and has_filled_mark):
            continue
        keys = _MARK_COLOR_KEYS.get(mark_type, ("color",))
        colors = [mark.get(key) for key in keys] if isinstance(mark, dict) else []
        colors += [
            encoding[key].get("value")
            for key in keys
            if isinstance(encoding.get(key), dict)
        ]
        if any(_is_whiteish(color) for color in colors) and mark_type not in found:
            found.append(mark_type)
    return found


def _validate_chart_spec(
    spec: dict[str, Any],
    columns: list[str],
    year_columns: frozenset[str] = frozenset(),
    rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Return the reasons a spec would not render a correct chart (empty if valid).

    Args:
        spec (dict[str, Any]): The candidate spec (already sanitized).
        columns (list[str]): The result's real column names.
        year_columns (frozenset[str]): Columns holding whole-number years (see `_integer_year_columns`).
        rows (list[dict[str, Any]] | None): The result rows, for the checks that need data.

    Returns:
        list[str]: Human-readable problems to feed back to the generator; empty if valid.
    """
    errors: list[str] = []

    referenced = _collect(spec, "field")
    derived = _collect_derived_fields(spec)
    unknown = referenced - set(columns) - derived - {"*"}
    # A map references its geometry's own fields — the top-level `id` and its labels under
    # `properties.*` — which are not result columns; those are allowed only in a map.
    if _collect(spec, "name") & set(_GEO_ASSETS):
        unknown = {
            field
            for field in unknown
            if field != "id" and not field.startswith("properties.")
        }
    missing = sorted(unknown)

    # A pivot names its new columns after the pivoted field's values, which the spec alone
    # does not reveal, so the check is skipped rather than reject a valid spec.
    if missing and not _collect(spec, "pivot"):
        errors.append(
            f"Encoding references column(s) not in the result: {missing}. "
            f"Available columns: {sorted(columns)}."
        )

    # A `scheme` name outside Vega's built-in set errors only once real data instantiates the
    # scale — which vl_convert swallows below (it logs to stderr but still returns an SVG), so
    # the render check never catches it and the chart ships blank. Reject unknown names here.
    unknown_schemes = sorted(_collect_scheme_names(spec) - _VEGA_COLOR_SCHEMES)
    if unknown_schemes:
        errors.append(
            f"Unknown color scheme(s): {unknown_schemes}. Use an exact Vega scheme name, "
            "or set explicit colors via a `range` instead."
        )

    # An invalid number format (e.g. ".2f%") also throws only once a scale has data — the
    # empty-data render below never formats a tick or legend label — so check the grammar here.
    invalid_formats = sorted(
        f for f in _collect_number_formats(spec) if not _is_valid_number_format(f)
    )
    if invalid_formats:
        errors.append(
            f"Invalid number format(s): {invalid_formats}. Use a d3-format specifier such "
            "as '.1f', ',.0f', '~s' or '.1%' (the '%' type multiplies by 100, so use it "
            "only on fractions). To append a unit or a '%' sign to values that are already "
            "percentages, use a `labelExpr` instead."
        )

    # An integer year in a temporal field is read as epoch milliseconds, so every year
    # lands on 1970 — the chart renders without any error, so no render check catches it.
    temporal_years = sorted(_collect_temporal_fields(spec) & year_columns)
    if temporal_years:
        errors.append(
            f"Column(s) {temporal_years} hold integer years, not dates, but are encoded "
            "as `temporal` or with a `timeUnit`, so every year renders as 1970. Encode "
            'them as `quantitative` (or `ordinal` for bars) with `"axis": {"format": "d"}` '
            "instead."
        )

    # A misplaced property (e.g. `color` inside `y`) is silently ignored by Vega-Lite, so
    # the chart compiles and renders without it; only the schema reveals it.
    unsupported = _collect_unsupported_encoding_properties(spec)
    if unsupported:
        errors.append(
            f"Properties Vega-Lite does not allow here, which it would silently ignore: "
            f"{unsupported}. Move each to where Vega-Lite expects it (e.g. a channel such as "
            "`color` or `tooltip` belongs directly under `encoding`, not inside another "
            'channel; a computed constant goes in `value: {"expr": ...}`).'
        )

    # The checks below catch charts that compile and render without error but read wrong.
    for field in _collect_split_lines(spec, rows or []):
        errors.append(
            f"Grouping the line by {field!r} splits it into pieces that take turns along "
            "the x axis, so each piece skips the other groups' points (a highlighted point "
            "drops out of the line). Draw one unsplit line, and mark the highlighted points "
            f"with a separate point layer colored by {field!r}."
        )

    for y_fields in _collect_unlabeled_layer_series(spec):
        errors.append(
            f"The series {y_fields} are drawn as separate layers with no color encoding, "
            "so the chart has no legend and readers cannot tell them apart. Fold those "
            "columns into one series field and encode color by it, or give each layer "
            '`"color": {"datum": "<series name>"}`.'
        )

    white_marks = _collect_invisible_white_marks(spec)
    if white_marks:
        errors.append(
            f"White {white_marks} mark(s) are invisible on the chat's white background. "
            "Use a visible color, or, if the user asked for a dark theme, also set a dark "
            "top-level `background`."
        )

    # Resolve named sources (stub geometry, empty rows) so a geoshape/lookup spec renders
    # its geometry; a normal chart just gets empty data. Rendering (not just VL→Vega compiling)
    # is the only way to validate expression strings in the spec.
    compile_spec = _bind_rows(_resolve_named_data(spec, _geo_stub_node), [])

    try:
        vlc.vegalite_to_svg(json.dumps(compile_spec))
    except Exception as e:
        # vl-convert raises on any spec the renderer rejects at build time
        errors.append(f"The chart spec does not compile: {e}")

    return errors


async def generate_chart_spec(
    columns: dict[str, str],
    rows: list[dict[str, Any]],
    instructions: str,
    language: LanguageCode,
    base_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a sanitized Vega-Lite spec for a result, within a deadline.

    Bounds `_generate_chart_spec` by settings.CHART_SPEC_TIMEOUT_SECONDS.

    Args:
        columns (dict[str, str]): The result's columns mapped to their BigQuery types.
        rows (list[dict[str, Any]]): The result rows; the model sees a summary and a sample.
        instructions (str): A natural-language description of the chart to produce.
        language (LanguageCode): The language of the chart's text (the user's language).
        base_spec (dict[str, Any] | None): An earlier chart to change; None for a new chart.

    Returns:
        dict[str, Any]: A validated, sanitized spec (no data bound yet).

    Raises:
        ChartSpecInvalid: No attempt produced a spec that validates.
        ChartSpecTimeout: No valid spec within the deadline.
    """
    try:
        async with asyncio.timeout(settings.CHART_SPEC_TIMEOUT_SECONDS):
            return await _generate_chart_spec(
                columns, rows, instructions, language, base_spec
            )
    except TimeoutError as e:
        raise ChartSpecTimeout(
            f"Generating the chart took over {settings.CHART_SPEC_TIMEOUT_SECONDS:.0f} "
            "seconds and was stopped. Try again, or describe a simpler chart."
        ) from e


async def _generate_chart_spec(
    columns: dict[str, str],
    rows: list[dict[str, Any]],
    instructions: str,
    language: LanguageCode,
    base_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a sanitized Vega-Lite spec for a result, retrying until it validates.

    Args:
        columns (dict[str, str]): The result's columns mapped to their BigQuery types.
        rows (list[dict[str, Any]]): The result rows; the model sees a summary and a sample.
        instructions (str): A natural-language description of the chart to produce.
        language (LanguageCode): The language of the chart's text (the user's language).
        base_spec (dict[str, Any] | None): An earlier chart to change (see
            `unbind_chart_spec`); None for a new chart.

    Returns:
        dict[str, Any]: A validated, sanitized spec (no data bound yet).

    Raises:
        ChartSpecInvalid: No attempt produced a spec that validates.
    """
    data_description = _describe_result(columns, rows)
    year_columns = frozenset(_integer_year_columns(list(columns), rows))
    previous_spec: dict[str, Any] | None = None
    errors: list[str] = []

    for _attempt in range(MAX_CHART_SPEC_ATTEMPTS):
        response = await _chart_spec_model().ainvoke(
            [
                SystemMessage(_CHART_SPEC_INSTRUCTIONS),
                HumanMessage(
                    _chart_spec_user_prompt(
                        instructions,
                        language,
                        data_description,
                        base_spec,
                        previous_spec,
                        errors,
                    )
                ),
            ]
        )

        parsed: ChartSpec | None = response["parsed"]

        if parsed is None:
            errors = ["No chart was produced — return a complete Vega-Lite v6 spec."]
        else:
            spec = _sanitize_chart_spec(parsed.spec)
            assert isinstance(spec, dict)  # an object spec sanitizes to an object
            spec = _normalize_chart_spec(spec)
            previous_spec = spec  # feed the spec back if it fails to validate
            errors = await asyncio.to_thread(
                _validate_chart_spec, spec, list(columns), year_columns, rows
            )
            if not errors:
                return spec

    raise ChartSpecInvalid(
        f"Could not produce a valid chart after {MAX_CHART_SPEC_ATTEMPTS} attempts: "
        + "; ".join(errors)
    )
