import asyncio
import json
import math
import re
from collections.abc import Callable
from decimal import Decimal
from functools import cache
from typing import Any

import vl_convert as vlc
from google.api_core.exceptions import NotFound
from google.cloud import bigquery as bq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, JsonValue

from app.db.database import AsyncDatabase, sessionmaker
from app.db.models import QueryHandle
from app.exports import ResultTableExpired, is_result_expired
from app.i18n import LanguageCode, language_name
from app.settings import settings

VEGA_LITE_SCHEMA = "https://vega.github.io/schema/vega-lite/v6.json"

# Rows shown to the spec generator, evenly spaced across the result. It also gets
# a per-column summary over every row; the full result is bound afterwards by
# inject_chart_data.
CHART_SPEC_SAMPLE_ROWS = 10

# A text column's distinct values are listed in full up to this many.
# Beyond it, only a few examples are shown.
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

# Fields a transform creates when its `as` is omitted (Vega-Lite's defaults).
_DEFAULT_TRANSFORM_OUTPUTS = {
    "fold": ("key", "value"),
    "density": ("value", "density"),
    "quantile": ("prob", "value"),
}

# Whole-number values in this range are read as calendar years.
_YEAR_RANGE = range(1000, 3001)

# Most rows a spec is validated with; a larger result is validated with an even spread
# of this many. Render time grows with the marks drawn, and renders in a process run one
# at a time: a bar and a text label per row took 1.2 s at 500 rows, 11 s at 5,000 and
# 44 s at 20,000. Most charts draw far fewer marks and render in about 40 ms.
_VALIDATION_MAX_ROWS = 500

# What to check when an encoding fails on the rows, for the spec generator.
_ENCODING_HINT = (
    "Check that every color `scheme` is an exact Vega scheme name, that every number "
    "`format` is a d3-format specifier such as '.1f', ',.0f', '~s' or '.1%' (to add a "
    "unit or a '%' sign use a `labelExpr`, not the format), and that dates are parsed "
    "from the values the column really holds."
)

# The one render problem a sample of the rows can cause by itself: it may hold none of
# the rows a filter keeps.
_NO_MARKS_PROBLEM = (
    "With the result's rows, the chart draws no marks at all: every row is filtered "
    f"out, or an encoding fails. Check that no `filter` removes every row. {_ENCODING_HINT}"
)

# The width a spec is validated at: about the chat panel's, where the website draws
# every chart at its container's width.
_VALIDATION_WIDTH = 560

# A chart larger than this, in pixels on either side, is rejected: no screen shows it
# and a browser canvas cannot hold it (e.g. a fixed scale domain far from the data).
_MAX_CHART_PIXELS = 10_000

# Text a chart shows where an expression or field produced no value (`format(undefined)`
# renders "NaN", a missing field "undefined"). `null` is left out: a NULL category in the
# data legitimately shows as "null", and no spec change can fix it.
_MISSING_VALUE_TEXT = re.compile(r"\b(?:undefined|NaN|Infinity)\b")

# What a drawn item of each mark type carries in the scenegraph: at least one of these.
# An encoding that fails on the real rows (an invalid format, an unknown scheme, a broken
# expression) stops before the item gets any, so it is drawn nowhere. A position the
# spec does not encode (e.g. a bar with only `y`) is legitimately absent.
_DRAWN_ITEM_KEYS = {"shape": {"shape"}, "text": {"text"}}
_DRAWN_ITEM_DEFAULT_KEYS = {"x", "y"}

# The 27 federative units' codes: the feature ids of the `brazil_states` geometry.
_BRAZIL_STATES = (
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA",
    "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
)  # fmt: skip

# Number and date formats per language, set on every chart (see `_server_config`).
_NUMBER_LOCALES: dict[LanguageCode, dict[str, Any]] = {
    "pt": {"decimal": ",", "thousands": ".", "grouping": [3], "currency": ["R$ ", ""]},
    "es": {"decimal": ",", "thousands": ".", "grouping": [3], "currency": ["$", ""]},
    "en": {"decimal": ".", "thousands": ",", "grouping": [3], "currency": ["$", ""]},
}

_TIME_LOCALES: dict[LanguageCode, dict[str, Any]] = {
    "pt": {
        "dateTime": "%A, %e de %B de %Y. %X",
        "date": "%d/%m/%Y",
        "time": "%H:%M:%S",
        "periods": ["AM", "PM"],
        "days": ["Domingo", "Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado"],
        "shortDays": ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sáb"],
        "months": [
            "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
            "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro",
        ],
        "shortMonths": [
            "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
            "Jul", "Ago", "Set", "Out", "Nov", "Dez",
        ],
    },
    "es": {
        "dateTime": "%A, %e de %B de %Y, %X",
        "date": "%d/%m/%Y",
        "time": "%H:%M:%S",
        "periods": ["AM", "PM"],
        "days": ["domingo", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado"],
        "shortDays": ["dom", "lun", "mar", "mié", "jue", "vie", "sáb"],
        "months": [
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
        ],
        "shortMonths": [
            "ene", "feb", "mar", "abr", "may", "jun",
            "jul", "ago", "sep", "oct", "nov", "dic",
        ],
    },
}  # fmt: skip

# Marks that show a tooltip of their encoded fields unless the spec sets its own. Lines
# and areas are left out: one mark spans many rows, so its tooltip would show only one.
_TOOLTIP_MARKS = ("bar", "point", "circle", "square", "rect", "arc", "geoshape")

# System prompt for the chart spec generation.
_CHART_SPEC_INSTRUCTIONS = """\
You are a data visualization specialist. You receive instructions on what to chart and how, and a profile of an already-aggregated BigQuery query result: its row count, each column's BigQuery type and a summary computed over every row, and sample rows spread evenly across the result. Return one complete Vega-Lite v6 spec that charts the result as instructed. Reference the result's columns by their exact names, and never include a data source, dataset, or URL — the exact rows are injected separately. You may use composite views and transforms when they make the chart clearer.

# Guidelines

- Follow the instructions where they are specific. When they leave the chart type open, pick the clearest form for the data's shape.
- Add a short title, label both axes with human-readable text rather than the raw column names.
- A `fold` turns column names into values, which then appear as they are in the legend and labels. Map each to a human-readable label with a `calculate` after the `fold`, and encode that field instead.
- Add a legend whenever a view shows two or more series.
- Write every title, axis label and legend in the language the request names.
- Base scale, size and layout choices on the column summaries (ranges, distinct counts), not just the sample rows. For example, panels whose value ranges barely overlap should not share an axis.

## Color

By default, set no color-related property — no `scale`, `scheme`, `sort`, `range`, or explicit color value — on any encoding, mark, or config, whether the field is categorical, quantitative, or there's only one series. Just map the field to the color channel and leave everything else about its color to the frontend's brand palette.

The one exception is when the instructions explicitly ask for a specific color. Then honor exactly what was asked, choosing the encoding that fits the chart and the number of series being colored, and set only that — leave every other color to the frontend. Do not invent a color, scheme, or palette the instructions did not ask for.

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
        tuple[QueryHandle, dict[str, str], list[dict[str, Any]]]: The handle,
            its columns (mapped to their BigQuery types), and rows.

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


def _geo_stub_ids(
    name: str, spec: dict[str, Any], rows: list[dict[str, Any]]
) -> list[str]:
    """Pick the feature ids a stub geometry needs, so a map's joins behave as in the browser.

    The states' ids are the real ones, all 27. The municipalities are too many to list,
    so they are the ones the rows join on, plus one no row matches, as the real geometry
    almost always has places without data.

    Args:
        name (str): A geo source name from `_GEO_ASSETS` (e.g. "brazil_states").
        spec (dict[str, Any]): The spec, whose `lookup` transforms name the join columns.
        rows (list[dict[str, Any]]): The rows the spec is rendered with.

    Returns:
        list[str]: The feature ids, as text like the real geometry's.
    """
    if name == "brazil_states":
        return list(_BRAZIL_STATES)
    join_columns = _collect(spec, "lookup") | _collect(spec, "key")
    ids = {
        str(row[column])
        for row in rows
        for column in join_columns & row.keys()
        if row[column] is not None
    }
    return [*sorted(ids), "0"]


def _geo_stub_node(name: str, ids: list[str]) -> dict[str, Any]:
    """Build a small inline TopoJSON node for offline spec validation.

    Each id gets its own square, laid out on a grid inside Brazil's bounding box, so
    the stub projects and draws like the real geometry at a fraction of its size.

    Args:
        name (str): A geo source name from `_GEO_ASSETS` (e.g. "brazil_states").
        ids (list[str]): The feature ids (see `_geo_stub_ids`).

    Returns:
        dict[str, Any]: A `data` node with the inline TopoJSON and its topojson `format`.
    """
    feature = _GEO_ASSETS[name]["feature"]
    per_side = math.isqrt(max(len(ids) - 1, 0)) + 1
    cell, size = 30 / per_side, 20 / per_side
    arcs, geometries = [], []
    for index, feature_id in enumerate(ids):
        x = -70 + (index % per_side) * cell
        y = -30 + (index // per_side) * cell
        # Clockwise, as spherical geometry winds a polygon's outer ring; the other way
        # round it would cover the rest of the globe.
        arcs.append(
            [[x, y], [x, y + size], [x + size, y + size], [x + size, y], [x, y]]
        )
        geometries.append(
            {
                "type": "Polygon",
                "id": feature_id,
                "properties": {"name": feature_id},
                "arcs": [[index]],
            }
        )
    stub = {
        "type": "Topology",
        "objects": {feature: {"type": "GeometryCollection", "geometries": geometries}},
        "arcs": arcs,
    }
    return {"values": stub, "format": {"type": "topojson", "feature": feature}}


def _resolve_named_data(
    node: JsonValue,
    rows: list[dict[str, Any]],
    geo_node: Callable[[str], dict[str, Any]],
) -> JsonValue:
    """Replace every `{"name": <allowlisted>}` data reference with real data.

    `query_result` resolves to the rows; a geo name resolves via a `geo_node` — a URL
    node for a render-ready spec, a tiny inline stub for offline validation. Any other
    `{"name": ...}` was already dropped by `_sanitize_chart_spec`.

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
            key: _resolve_named_data(value, rows, geo_node)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_resolve_named_data(item, rows, geo_node) for item in node]
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
    chart = _resolve_named_data(spec, rows, _geo_url_node)
    chart["$schema"] = VEGA_LITE_SCHEMA
    chart.setdefault("data", {"values": rows})
    return chart


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

    A result is usually sorted by its grouping columns, so the first rows often
    all share one group; an even spread shows every part of it.

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
        "null_pct": round(100 * (len(values) - len(present)) / len(values), 2),
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


def _profile_result(columns: dict[str, str], rows: list[dict[str, Any]]) -> str:
    """Profile a result for the spec generator: size, column summaries, and a sample.

    Args:
        columns (dict[str, str]): The result's columns mapped to their BigQuery types.
        rows (list[dict[str, Any]]): The result rows.

    Returns:
        str: The data profile section of the spec generator's user message.
    """
    summary = {
        column: _summarize_column(bq_type, [row[column] for row in rows])
        for column, bq_type in columns.items()
    }
    sample = _spread_sample(rows, CHART_SPEC_SAMPLE_ROWS)
    return (
        f"Rows: {len(rows)}\n\n"
        "Columns (BigQuery type and a summary computed over every row): "
        f"{json.dumps(summary, ensure_ascii=False, default=str)}\n\n"
        "Sample rows (spread evenly across the result): "
        f"{json.dumps(sample, ensure_ascii=False, default=str)}"
    )


def _chart_spec_user_prompt(
    instructions: str,
    language: LanguageCode,
    data_profile: str,
    previous_spec: dict[str, Any] | None,
    errors: list[str],
) -> str:
    """Build the task-specific user message.

    On a retry, appends the model's own rejected spec and the validator's problems so it
    can fix that spec rather than regenerate from scratch.

    Args:
        instructions (str): What to chart and how, in natural language.
        language (LanguageCode): The language of the chart's text (the user's language).
        data_profile (str): The result's size, column summaries and sample rows.
        previous_spec (dict[str, Any] | None): The last rejected spec, echoed back on a retry.
        errors (list[str]): The reasons the previous spec was rejected; empty on the first try.

    Returns:
        str: The user message for the chart-spec model.
    """
    prompt = (
        f"# What to chart and how\n\n{instructions}\n\n"
        f"Language of the chart's text: {language_name(language)}\n\n"
        f"# Data profile\n\n{data_profile}"
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
    and `fold`, `density` and `quantile` fall back to default names when `as` is omitted.

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


def _collect_temporal_fields(node: JsonValue) -> set[str]:
    """Collect every field a spec treats as a date, anywhere within it.

    A field is treated as a date when its definition is `temporal` or
    carries a `timeUnit` (which also covers a `timeUnit` transform).

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


def _collect_calculated_fields(node: JsonValue) -> set[str]:
    """Collect every field a `calculate` transform writes, anywhere within a spec.

    Args:
        node (JsonValue): A spec, or any node within it, to walk.

    Returns:
        set[str]: Every `as` name of a `calculate` transform.
    """
    found: set[str] = set()
    if isinstance(node, dict):
        if "calculate" in node and isinstance(node.get("as"), str):
            found.add(node["as"])
        for value in node.values():
            found |= _collect_calculated_fields(value)
    elif isinstance(node, list):
        for item in node:
            found |= _collect_calculated_fields(item)
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


def _server_config(language: LanguageCode) -> dict[str, Any]:
    """Build the Vega-Lite `config` the server sets on every chart.

    It holds what does not depend on the chart and the model should not have to get
    right: number and date formats in the user's language, and a tooltip on marks that
    stand for one row. Colors, fonts and sizes stay with the website, which merges its
    own config under the spec's.

    Args:
        language (LanguageCode): The language of the chart's text.

    Returns:
        dict[str, Any]: The config.
    """
    locale: dict[str, Any] = {"number": _NUMBER_LOCALES[language]}
    if language in _TIME_LOCALES:
        locale["time"] = _TIME_LOCALES[language]
    tooltips = {mark: {"tooltip": {"content": "encoding"}} for mark in _TOOLTIP_MARKS}
    return {"locale": locale, **tooltips}


def _merge_config(base: Any, override: Any) -> Any:
    """Merge two config objects key by key; `override` wins where both set a value.

    Args:
        base (Any): The config to start from.
        override (Any): The config whose values take precedence.

    Returns:
        Any: The merged config.
    """
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _merge_config(base[key], value) if key in base else value
    return merged


def _apply_server_config(
    spec: dict[str, Any], language: LanguageCode
) -> dict[str, Any]:
    """Set the server's config on a spec, over any value the model gave the same keys.

    Args:
        spec (dict[str, Any]): A sanitized spec.
        language (LanguageCode): The language of the chart's text.

    Returns:
        dict[str, Any]: The spec with the server's config merged into its own.
    """
    config = _merge_config(spec.get("config") or {}, _server_config(language))
    return {**spec, "config": config}


def _collect_marks(node: JsonValue) -> list[dict[str, Any]]:
    """Collect every mark a rendered chart draws, from its Vega scenegraph.

    Args:
        node (JsonValue): A scenegraph (from `vlc.vegalite_to_scenegraph`), or any node
            within it.

    Returns:
        list[dict[str, Any]]: Each mark that is not a group, in drawing order: its
            `marktype`, its `role` (e.g. "mark", "axis-label") and its `items`.
    """
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("marktype") not in (None, "group"):
            found.append(node)
        items = node.get("items")
        if isinstance(items, list):
            for item in items:
                found += _collect_marks(item)
    return found


def _is_drawn(mark: dict[str, Any]) -> bool:
    """Return whether a mark has at least one item with a property that draws it."""
    needed = _DRAWN_ITEM_KEYS.get(mark["marktype"], _DRAWN_ITEM_DEFAULT_KEYS)
    return any(not needed.isdisjoint(item) for item in mark.get("items") or [])


def _rendered_text(mark: dict[str, Any]) -> list[str]:
    """Return the visible text of a text mark's items, leaving out transparent ones."""
    texts = []
    for item in mark.get("items") or []:
        if "text" not in item:
            continue
        if item.get("opacity", 1) == 0 or item.get("fillOpacity", 1) == 0:
            continue
        text = item["text"]
        texts.append(" ".join(map(str, text)) if isinstance(text, list) else str(text))
    return texts


def _render_problems(spec: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
    """Render a spec with rows and return what is wrong with the chart it draws.

    Rendering, not just compiling Vega-Lite to Vega, is the only way to run the spec's
    expressions, and the rows show what the user would see. A dataflow that fails does
    not raise: Vega logs the error and draws what it had, so the checks read the
    scenegraph. They look at the output only, whatever in the spec caused it.

    Args:
        spec (dict[str, Any]): The candidate spec (already sanitized).
        rows (list[dict[str, Any]]): The rows to render it with. Without any, only the
            checks that hold for an empty chart run.

    Returns:
        list[str]: Human-readable problems to feed back to the generator; empty if none.
    """
    render_spec = _resolve_named_data(
        spec,
        rows,
        lambda name: _geo_stub_node(name, _geo_stub_ids(name, spec, rows)),
    )
    assert isinstance(render_spec, dict)  # an object spec resolves to an object
    # Drawn as the website draws it: at the chat panel's width, fitted to it.
    render_spec = {
        **render_spec,
        "$schema": VEGA_LITE_SCHEMA,
        "width": _VALIDATION_WIDTH,
        "autosize": {
            "type": "fit",
            "contains": "padding",
            **(spec.get("autosize") if isinstance(spec.get("autosize"), dict) else {}),
        },
    }
    render_spec.setdefault("data", {"values": rows})

    try:
        rendered = vlc.vegalite_to_scenegraph(json.dumps(render_spec))
    except Exception as e:
        # vl-convert raises on any spec the renderer rejects at build time
        return [f"The chart spec does not compile: {e}"]

    root = rendered["scenegraph"]
    if not any(group.get("items") for group in root.get("items") or []):
        return [
            "The chart compiles but fails when it runs, so it would show nothing at "
            "all: one of its transforms, lookups or expressions errors. Check each "
            "against the columns and their types, use only operations Vega-Lite "
            "supports in that place, and simplify where you can."
        ]

    problems: list[str] = []

    width, height = rendered.get("width") or 0, rendered.get("height") or 0
    if width <= 0 or height <= 0:
        problems.append(
            "The chart has no size when drawn at its container's width, so nothing "
            "would show. Do not mix `facet` (or a `row`/`column` encoding) with a "
            "top-level `mark` or `layer`: put the marks inside the facet's `spec`."
        )
    elif max(width, height) > _MAX_CHART_PIXELS:
        problems.append(
            f"The chart would be {width:.0f}x{height:.0f} pixels, far too large to "
            "show. Use a smaller `height` or `step`, or fewer facet panels."
        )

    if not rows:
        return problems

    marks = _collect_marks(root)
    data_marks = [mark for mark in marks if mark.get("role") == "mark"]
    drawn = [mark for mark in data_marks if _is_drawn(mark)]
    # Items an encoding failed on exist, but carry nothing that draws them.
    failed = [mark for mark in data_marks if mark.get("items") and not _is_drawn(mark)]
    if failed:
        problems.append(
            "With the result's rows, "
            + ("none" if not drawn else "only part")
            + " of the chart's marks are drawn: an encoding fails on the others, or "
            f"a text mark has no text to write. {_ENCODING_HINT}"
        )
    elif not drawn:
        problems.append(_NO_MARKS_PROBLEM)

    text_marks = [
        (mark.get("role"), _rendered_text(mark))
        for mark in marks
        if mark["marktype"] == "text"
    ]

    missing = sorted(
        {
            text
            for _, texts in text_marks
            for text in texts
            if _MISSING_VALUE_TEXT.search(text)
        }
    )
    if missing:
        problems.append(
            f"The rendered chart shows {missing} as text (in an axis, legend or label): "
            "a field or expression yields no value for the real rows. Check every field "
            "name and every `calculate`, `labelExpr` and text expression against the "
            "columns and their values."
        )

    for role, labels in text_marks:
        if role != "axis-label":
            continue
        if len(labels) >= 3 and len(set(labels)) == 1:
            problems.append(
                f"An axis shows the same label {labels[0]!r} on all of its {len(labels)} "
                "ticks, so its `labelExpr` or `format` discards the value. Fix that axis."
            )
        elif not labels and drawn:
            problems.append(
                "An axis has no tick labels with the result's rows: its scale has no valid "
                "domain, or the values fall far outside it (e.g. a log scale over zero "
                "or negative values, a fixed `domain` that does not fit the values, "
                "series in different units on one scale, or values that are not valid "
                "dates or numbers). Fix that axis's scale."
            )

    return problems


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
        rows (list[dict[str, Any]] | None): The result rows the chart is rendered with
            (an even spread of `_VALIDATION_MAX_ROWS` of them, for a larger result);
            None renders it with no data, which checks only that it builds.

    Returns:
        list[str]: Human-readable problems to feed back to the generator; empty if valid.
    """
    errors: list[str] = []

    referenced = _collect(spec, "field")
    derived = _collect_derived_fields(spec)
    # A geoshape references its geometry's own fields — the top-level `id` and its labels
    # under `properties.*` — which are not result columns; we allow those and flag the rest.
    unknown = referenced - set(columns) - derived - {"*", "id"}
    missing = sorted(field for field in unknown if not field.startswith("properties."))

    # A pivot names its new columns after the pivoted field's values, which the spec alone
    # does not reveal, so the check is skipped rather than reject a valid spec.
    if missing and not _collect(spec, "pivot"):
        errors.append(
            f"Encoding references column(s) not in the result: {missing}. "
            f"Available columns: {sorted(columns)}."
        )

    # An integer year in a temporal field is read as epoch milliseconds, so every year
    # lands in the first seconds of 1970 UTC, one tick labeled with the viewer's local
    # year: 1969 west of UTC, as in Brazil. The chart renders without any error, and its
    # output looks like any chart with a single date, so the render check cannot tell.
    # A year a `calculate` rewrites under its own name (e.g. `datetime(datum.ano, 0, 1)`
    # as `ano`) is a real date by then, so it is not flagged.
    temporal_years = sorted(
        (_collect_temporal_fields(spec) & year_columns)
        - _collect_calculated_fields(spec)
    )
    if temporal_years:
        errors.append(
            f"Column(s) {temporal_years} hold integer years, not dates, but are encoded "
            "as `temporal` or with a `timeUnit`, so every year collapses onto a single "
            '"1969"/"1970" tick. Encode them as `quantitative` (or `ordinal` for bars) '
            'with `"axis": {"format": "d"}` instead.'
        )

    rows = rows or []
    sample = _spread_sample(rows, _VALIDATION_MAX_ROWS)
    problems = _render_problems(spec, sample)
    if _NO_MARKS_PROBLEM in problems and len(sample) < len(rows):
        # The sample may hold none of the rows a filter keeps, so every row decides.
        # That render is cheap: either few rows pass the filter, or the encoding fails
        # before anything is drawn. No other problem comes from sampling.
        problems = _render_problems(spec, rows)

    return errors + problems


async def generate_chart_spec(
    columns: dict[str, str],
    rows: list[dict[str, Any]],
    instructions: str,
    language: LanguageCode,
) -> dict[str, Any]:
    """Generate a sanitized Vega-Lite spec for a result, within a deadline.

    Bounds `_generate_chart_spec` by settings.CHART_SPEC_TIMEOUT_SECONDS.

    Args:
        columns (dict[str, str]): The result's columns mapped to their BigQuery types.
        rows (list[dict[str, Any]]): The result rows; the model sees a summary and a sample.
        instructions (str): What to chart and how, in natural language.
        language (LanguageCode): The language of the chart's text (the user's language).

    Returns:
        dict[str, Any]: A validated, sanitized spec carrying the server's config (no
            data bound yet).

    Raises:
        ChartSpecInvalid: No attempt produced a spec that validates.
        ChartSpecTimeout: No valid spec within the deadline.
    """
    try:
        async with asyncio.timeout(settings.CHART_SPEC_TIMEOUT_SECONDS):
            return await _generate_chart_spec(columns, rows, instructions, language)
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
) -> dict[str, Any]:
    """Generate a sanitized Vega-Lite spec for a result, retrying until it validates.

    Args:
        columns (dict[str, str]): The result's columns mapped to their BigQuery types.
        rows (list[dict[str, Any]]): The result rows; the model sees a summary and a sample.
        instructions (str): What to chart and how, in natural language.
        language (LanguageCode): The language of the chart's text (the user's language).

    Returns:
        dict[str, Any]: A validated, sanitized spec carrying the server's config (no
            data bound yet).

    Raises:
        ChartSpecInvalid: No attempt produced a spec that validates.
    """
    data_profile = _profile_result(columns, rows)
    year_columns = frozenset(_integer_year_columns(list(columns), rows))
    previous_spec: dict[str, Any] | None = None
    errors: list[str] = []

    for _attempt in range(MAX_CHART_SPEC_ATTEMPTS):
        response = await _chart_spec_model().ainvoke(
            [
                SystemMessage(_CHART_SPEC_INSTRUCTIONS),
                HumanMessage(
                    _chart_spec_user_prompt(
                        instructions, language, data_profile, previous_spec, errors
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
            previous_spec = spec  # feed the spec back if it fails to validate
            # Validated as it ships, with the server's config; the model is shown only
            # what it wrote.
            chart = _apply_server_config(spec, language)
            errors = await asyncio.to_thread(
                _validate_chart_spec, chart, list(columns), year_columns, rows
            )
            if not errors:
                return chart

    raise ChartSpecInvalid(
        f"Could not produce a valid chart after {MAX_CHART_SPEC_ATTEMPTS} attempts: "
        + "; ".join(errors)
    )
