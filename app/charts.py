import asyncio
import json
from functools import cache
from typing import Any

import vl_convert as vlc
from google.api_core.exceptions import NotFound
from google.cloud import bigquery as bq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.db.database import AsyncDatabase, sessionmaker
from app.db.models import QueryHandle
from app.exports import ResultTableExpired, is_result_expired
from app.settings import settings

VEGA_LITE_SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"

# Rows shown to the spec generator. It only needs the shape (columns + a few
# example values); the full result is bound afterwards by build_chart_spec.
CHART_SPEC_SAMPLE_ROWS = 10

# How many times the generator may retry after a spec fails validation before giving up.
MAX_CHART_SPEC_ATTEMPTS = 3

# The chart description used by the chart button, where the user gives no instruction.
DEFAULT_CHART_INSTRUCTIONS = "Choose the clearest chart for this result."

# Keys stripped anywhere in a model spec: `data`/`datasets` (the server binds the
# real rows) and `url` (any external fetch — data.url, lookup sources, image marks).
_UNTRUSTED_KEYS = frozenset({"data", "datasets", "url"})

# System prompt for the chart spec generation. Detailed on purpose: the chart button
# sends only DEFAULT_CHART_INSTRUCTIONS, so the model must choose a sensible chart itself.
_CHART_SPEC_INSTRUCTIONS = """\
You are a data visualization specialist. Given a small, already-aggregated query result and a description of the chart to build, return one complete Vega-Lite v5 spec. Reference the result's columns by their exact names, and never include a data source, dataset, or URL — the exact rows are bound separately. You may use composite views and transforms when they make the chart clearer.

Follow the description when it is specific. When it leaves the chart type open, pick the clearest form for the data's shape:
- A quantity across categories → a bar chart, sorted by value unless the categories have an inherent order.
- A value over time → a line chart.
- One quantity against another → a scatter plot.
- The distribution of a single quantity → a binned histogram.

Make it readable: add a short title, label both axes with human-readable text rather than the raw column names, and begin a bar chart's quantitative axis at zero.

Color: use the Base dos Dados green #2B8C4D for a single series; to distinguish two series, set the color scale's range to ["#2B8C4D", "#0068C5"]; for three or more series, use a clear, colorblind-friendly scheme instead of repeating those two. Tie each color to its category rather than its rank, and add a legend whenever a view shows two or more series.
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
    spec: dict[str, Any] = Field(description="The complete Vega-Lite v5 specification.")


@cache
def _bq_client() -> bq.Client:  # pragma: no cover
    return bq.Client(
        project=settings.GOOGLE_BILLING_PROJECT,
        credentials=settings.GOOGLE_CREDENTIALS,
    )


def read_chart_data(
    destination_table: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Read a result table's rows for charting, capped at settings.CHART_MAX_BYTES.

    Reads row by row and stops as soon as the JSON that would be bound into the spec
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
            row = dict(row)
            size += len(json.dumps(row, ensure_ascii=False, default=str).encode())
            if size > settings.CHART_MAX_BYTES:
                raise ChartResultTooLarge(
                    f"The result is too large to chart (over "
                    f"{settings.CHART_MAX_BYTES // (1024 * 1024)} MB of data). Aggregate "
                    "or summarize it in SQL first, then chart the smaller result."
                )
            rows.append(row)
    except NotFound as e:
        raise ResultTableExpired(str(e)) from e

    return columns, rows


async def load_chart_source(
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

    columns, rows = await asyncio.to_thread(read_chart_data, handle.destination_table)

    return handle, columns, rows


def _strip_untrusted(node: Any) -> Any:
    """Recursively drop data sources and any URL from a spec."""
    if isinstance(node, dict):
        return {
            key: _strip_untrusted(value)
            for key, value in node.items()
            if key not in _UNTRUSTED_KEYS
        }
    if isinstance(node, list):
        return [_strip_untrusted(item) for item in node]
    return node


def build_chart_spec(
    spec: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Sanitize the model's spec and bind the exact result rows.

    Args:
        spec (dict[str, Any]): The model's Vega-Lite spec.
        rows (list[dict[str, Any]]): The exact result rows to render.

    Returns:
        dict[str, Any]: A sanitized Vega-Lite spec with the data bound in.
    """
    chart = _strip_untrusted(spec)
    chart["$schema"] = VEGA_LITE_SCHEMA
    chart["data"] = {"values": rows}
    return chart


def _collect(node: Any, key: str) -> set[str]:
    """Collect every string value stored under `key`, anywhere in the spec."""
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
            "effort": "none",
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
    """The task-specific user message; the durable how-to is the system prompt."""
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
    missing = sorted(referenced - set(columns) - derived - {"*"})

    if missing:
        errors.append(
            f"Encoding references column(s) not in the result: {missing}. "
            f"Available columns: {sorted(columns)}."
        )

    # Render the structure with empty data. Rendering (not just VL→Vega compiling)
    # is the only way to validate expression strings in the spec.
    compile_spec = {**spec, "$schema": VEGA_LITE_SCHEMA, "data": {"values": []}}
    try:
        vlc.vegalite_to_svg(json.dumps(compile_spec))
    except (
        Exception
    ) as e:  # vl-convert raises on any spec the renderer rejects at build time
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
            errors = ["No chart was produced — return a complete Vega-Lite spec."]
        else:
            spec = _strip_untrusted(parsed.spec)
            previous_spec = spec  # feed the spec back if it fails to validate
            errors = await asyncio.to_thread(_validate_chart_spec, spec, columns)
            if not errors:
                return spec

    raise ChartSpecInvalid(
        f"Could not produce a valid chart after {MAX_CHART_SPEC_ATTEMPTS} attempts: "
        + "; ".join(errors)
    )
