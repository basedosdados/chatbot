import asyncio
import json
from functools import cache
from pathlib import Path
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
from app.settings import settings

VEGA_LITE_SCHEMA = "https://vega.github.io/schema/vega-lite/v6.json"

# Rows shown to the spec generator. It only needs the shape (columns + a few
# example values); the full result is bound afterwards by inject_chart_data.
CHART_SPEC_SAMPLE_ROWS = 10

# How many times the generator may retry after a spec fails validation before giving up.
MAX_CHART_SPEC_ATTEMPTS = 3

# Keys that are always stripped anywhere in a model-generated spec for security.
_UNTRUSTED_KEYS = frozenset({"datasets", "url"})

# Static geographic assets a spec may inject by name for choropleth maps.
_GEO_DIR = Path(__file__).parent / "assets"
_GEO_ASSETS: dict[str, dict[str, str]] = {
    "brazil_states": {
        "file": "brazil_states.topojson",
        "feature": "uf",
    },
    "brazil_municipalities": {
        "file": "brazil_municipalities.topojson",
        "feature": "Munic",
    },
}

# The result rows, referenced from a spec as {"data": {"name": "query_result"}}.
_RESULT_SOURCE = "query_result"

# Every data-source name a spec may reference; anything else in a `data` node is dropped.
_ALLOWED_SOURCES = frozenset({_RESULT_SOURCE, *_GEO_ASSETS})

# System prompt for the chart spec generation.
_CHART_SPEC_INSTRUCTIONS = """\
You are a data visualization specialist. Given a small, already-aggregated query result and a description of the chart to build, return one complete Vega-Lite v6 spec. Reference the result's columns by their exact names, and never include a data source, dataset, or URL — the exact rows are injected separately. You may use composite views and transforms when they make the chart clearer.

# Guidelines

- Follow the description when it is specific. When it leaves the chart type open, pick the clearest form for the data's shape.
- Add a short title, label both axes with human-readable text rather than the raw column names.
- Add a legend whenever a view shows two or more series.

## Color

Do not set colors, scales, or ranges for a categorical or single-series encoding — just map a field to the color channel to distinguish series and leave the palette to the defaults.

For a numeric value mapped to color (any mark colored by a number), set the color scale's `scheme` only when the value's semantic meaning has a color convention worth matching; otherwise leave it to the default. For example:

- Temperature → blue for cold, red for hot: use "redblue" or "redyellowblue" (add `"reverse": true` if low and high land on the wrong ends).
- A quantity that diverges around a meaningful midpoint → a diverging scheme such as "redblue" or "blueorange" with the scale's `domainMid` set to that midpoint.

When you do name a scheme, use a real Vega scheme name — never one from another library. Prefer colorblind-safe schemes.

## Size

The chart is rendered at the container's width, so never set a width. For any chart with many categories along one axis, set an explicit height so the cells or bands are not overly tall — a height noticeably smaller than the chart's width reads best. Otherwise leave the height to default.

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
    "color": {"field": "valor", "type": "quantitative", "title": "…", "scale": {"scheme": "yellowgreenblue"}},
    "tooltip": [{"field": "properties.name", "type": "nominal", "title": "Estado"}, {"field": "valor", "type": "quantitative"}]
  },
}

{
  "data": {"name": "brazil_municipalities"},
  "transform": [{"lookup": "id", "from": {"data": {"name": "query_result"}, "key": "id_municipio", "fields": ["valor"]}}],
  "projection": {"type": "mercator"},
  "mark": "geoshape",
  "encoding": {
    "color": {"field": "valor", "type": "quantitative", "title": "…", "scale": {"scheme": "yellowgreenblue"}},
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


def _fetch_rows(
    destination_table: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Fetch a result table's rows for charting, capped at settings.CHART_MAX_BYTES.

    Fetches row by row and stops as soon as the JSON that would be bound into the spec
    exceeds the budget, so a huge result is rejected without materializing all of it.

    Args:
        destination_table (dict[str, Any]): `TableReference.to_api_repr()` of the result table.

    Returns:
        tuple[list[str], list[dict[str, Any]]]: Column names and row dicts.

    Raises:
        ChartResultTooLarge: The bound data would exceed settings.CHART_MAX_BYTES.
        ResultTableExpired: The result table no longer exists (~24h TTL).
    """
    table_ref = bq.TableReference.from_api_repr(destination_table)

    try:
        table = _bq_client().get_table(table_ref)

        columns = [field.name for field in table.schema]

        rows = []
        size = 0
        for row in _bq_client().list_rows(table_ref):
            # Serialize once to both measure the row and coerce BigQuery types
            # (date/datetime/Decimal/…) to JSON-native values. Raw objects are
            # not valid JSON and would fail serialization when the spec is bound.
            serialized = json.dumps(dict(row), ensure_ascii=False, default=str)
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
) -> tuple[QueryHandle, list[str], list[dict[str, Any]]]:
    """Resolve a chartable result: authorize the handle, then read its capped rows.

    Args:
        query_ref (str): The handle of the result to chart.
        thread_id (str): The thread the handle must belong to (authorization).

    Returns:
        tuple[QueryHandle, list[str], list[dict[str, Any]]]: The handle, its columns, and rows.

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
@cache
def _geo_data_node(name: str) -> dict[str, Any]:
    """Build a Vega-Lite inline-TopoJSON data node for an allowlisted geo source.

    Args:
        name (str): A geo source name from `_GEO_ASSETS` (e.g. "brazil_states").

    Returns:
        dict[str, Any]: The `data` node — the TopoJSON `values` plus its topojson `format`.
    """
    asset = _GEO_ASSETS[name]
    topojson = json.loads((_GEO_DIR / asset["file"]).read_text(encoding="utf-8"))
    return {
        "values": topojson,
        "format": {"type": "topojson", "feature": asset["feature"]},
    }


def _resolve_named_data(node: JsonValue, rows: list[dict[str, Any]]) -> JsonValue:
    """Replace every `{"name": <allowlisted>}` data reference with the real data.

    `query_result` resolves to the rows, a geo name to its inline TopoJSON; any other
    `{"name": ...}` was already dropped by `_sanitize_chart_spec`.

    Args:
        node (JsonValue): A spec, or any node within it, to walk.
        rows (list[dict[str, Any]]): The exact result rows to bind for `query_result`.

    Returns:
        JsonValue: The same node with its named data references resolved.
    """
    if isinstance(node, dict):
        if set(node) == {"name"} and node["name"] in _ALLOWED_SOURCES:
            if node["name"] == _RESULT_SOURCE:
                return {"values": rows}
            return _geo_data_node(node["name"])
        return {key: _resolve_named_data(value, rows) for key, value in node.items()}
    if isinstance(node, list):
        return [_resolve_named_data(item, rows) for item in node]
    return node


def inject_chart_data(
    spec: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Inject the real geometry and result rows into an already-sanitized spec.

    Takes an already sanitized spec from `generate_chart_spec` and makes it render-ready:
    resolves its named sources (a geo name to inline TopoJSON, `query_result` to the rows)
    and binds the rows as the default data for a plain chart that declared none.

    Args:
        spec (dict[str, Any]): A sanitized, data-less spec from `generate_chart_spec`.
        rows (list[dict[str, Any]]): The exact result rows to render.

    Returns:
        dict[str, Any]: A render-ready Vega-Lite spec with the real data bound in.
    """
    chart = _resolve_named_data(spec, rows)
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
    """
    return ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.MODEL_URI,
        reasoning={
            "effort": "medium",
            "summary": "auto",
        },
    ).with_structured_output(
        ChartSpec,
        method="function_calling",
        include_raw=True,
    )


def _chart_spec_user_prompt(
    columns: list[str],
    sample: list[dict[str, Any]],
    instructions: str,
    previous_spec: dict[str, Any] | None,
    errors: list[str],
) -> str:
    """Build the task-specific user message.

    On a retry, appends the model's own rejected spec and the validator's problems so it
    can fix that spec rather than regenerate from scratch.

    Args:
        columns (list[str]): The result's column names.
        sample (list[dict[str, Any]]): A few example rows, for shape only.
        instructions (str): The natural-language description of the chart to produce.
        previous_spec (dict[str, Any] | None): The last rejected spec, echoed back on a retry.
        errors (list[str]): The reasons the previous spec was rejected; empty on the first try.

    Returns:
        str: The user message for the chart-spec model.
    """
    prompt = (
        f"What to chart: {instructions}\n\n"
        f"Columns: {json.dumps(columns, ensure_ascii=False)}\n\n"
        f"Sample rows: {json.dumps(sample, ensure_ascii=False, default=str)}"
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
    """Sanitize a model spec: drop external/inline data, keep only allowlisted named sources.

    `datasets` and any `url` are removed outright; a `data` node survives only when it is
    exactly `{"name": <allowlisted source>}`, so the model can never supply its own data.

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


def _validate_chart_spec(spec: dict[str, Any], columns: list[str]) -> list[str]:
    """Return the reasons a spec would not render a correct chart (empty if valid).

    Args:
        spec (dict[str, Any]): The candidate spec (already sanitized).
        columns (list[str]): The result's real column names.

    Returns:
        list[str]: Human-readable problems to feed back to the generator; empty if valid.
    """
    errors: list[str] = []

    referenced = _collect(spec, "field")
    derived = _collect(spec, "as")  # fields produced by transforms
    # A geoshape references its geometry's own fields — the top-level `id` and its labels
    # under `properties.*` — which are not result columns; we allow those and flag the rest.
    unknown = referenced - set(columns) - derived - {"*", "id"}
    missing = sorted(field for field in unknown if not field.startswith("properties."))

    if missing:
        errors.append(
            f"Encoding references column(s) not in the result: {missing}. "
            f"Available columns: {sorted(columns)}."
        )

    # Resolve named sources (real geometry, empty rows) so a geoshape/lookup spec renders
    # its geometry; a normal chart just gets empty data. Rendering (not just VL→Vega compiling)
    # is the only way to validate expression strings in the spec.
    compile_spec = _resolve_named_data(spec, [])
    compile_spec = {**compile_spec, "$schema": VEGA_LITE_SCHEMA}
    compile_spec.setdefault("data", {"values": []})

    try:
        vlc.vegalite_to_svg(json.dumps(compile_spec))
    except Exception as e:
        # vl-convert raises on any spec the renderer rejects at build time
        errors.append(f"The chart spec does not compile: {e}")

    return errors


async def generate_chart_spec(
    columns: list[str], rows: list[dict[str, Any]], instructions: str
) -> dict[str, Any]:
    """Generate a sanitized Vega-Lite spec for a result, retrying until it validates.

    Args:
        columns (list[str]): The result's column names.
        rows (list[dict[str, Any]]): The result rows; only a sample is shown to the model.
        instructions (str): A natural-language description of the chart to produce.

    Returns:
        dict[str, Any]: A validated, sanitized spec (no data bound yet).

    Raises:
        ChartSpecInvalid: No attempt produced a spec that validates.
    """
    sample = rows[:CHART_SPEC_SAMPLE_ROWS]
    previous_spec: dict[str, Any] | None = None
    errors: list[str] = []

    for _attempt in range(MAX_CHART_SPEC_ATTEMPTS):
        response = await _chart_spec_model().ainvoke(
            [
                SystemMessage(_CHART_SPEC_INSTRUCTIONS),
                HumanMessage(
                    _chart_spec_user_prompt(
                        columns, sample, instructions, previous_spec, errors
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
            errors = await asyncio.to_thread(_validate_chart_spec, spec, columns)
            if not errors:
                return spec

    raise ChartSpecInvalid(
        f"Could not produce a valid chart after {MAX_CHART_SPEC_ATTEMPTS} attempts: "
        + "; ".join(errors)
    )
