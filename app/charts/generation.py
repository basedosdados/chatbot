import asyncio
from functools import cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.charts.binding import sanitize_chart_spec
from app.charts.config import apply_server_config
from app.charts.prompts import CHART_SPEC_INSTRUCTIONS, chart_spec_user_prompt
from app.charts.validation import validate_chart_spec
from app.i18n import LanguageCode
from app.settings import settings

# How many times the generator may retry after a spec fails validation before giving up.
MAX_CHART_SPEC_ATTEMPTS = 3


class ChartSpecInvalid(Exception):
    """The generator could not produce a spec that passes validation."""


class ChartSpecTimeout(Exception):
    """The generator did not produce a valid spec within the deadline."""


class ChartSpec(BaseModel):
    """The tool the charting model calls: a free-form Vega-Lite spec."""

    # Description kept to a concise label per OpenAI's gpt-5.x guidance ("state each
    # instruction once"); the how-to lives once in CHART_SPEC_INSTRUCTIONS (system prompt).
    spec: dict[str, Any] = Field(description="The complete Vega-Lite v6 specification.")


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
    previous_spec: dict[str, Any] | None = None
    errors: list[str] = []

    for _attempt in range(MAX_CHART_SPEC_ATTEMPTS):
        response = await _chart_spec_model().ainvoke(
            [
                SystemMessage(CHART_SPEC_INSTRUCTIONS),
                HumanMessage(
                    chart_spec_user_prompt(
                        instructions, language, columns, rows, previous_spec, errors
                    )
                ),
            ]
        )

        parsed: ChartSpec | None = response["parsed"]

        if parsed is None:
            errors = ["No chart was produced — return a complete Vega-Lite v6 spec."]
        else:
            spec = sanitize_chart_spec(parsed.spec)
            assert isinstance(spec, dict)  # an object spec sanitizes to an object
            previous_spec = spec  # feed the spec back if it fails to validate
            # Validated as it ships, with the server's config; the model is shown only
            # what it wrote.
            chart = apply_server_config(spec, language)
            errors = await asyncio.to_thread(
                validate_chart_spec, chart, list(columns), rows
            )
            if not errors:
                return chart

    raise ChartSpecInvalid(
        f"Could not produce a valid chart after {MAX_CHART_SPEC_ATTEMPTS} attempts: "
        + "; ".join(errors)
    )


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
        dict[str, Any]: A validated, sanitized spec carrying the server's config
            (no data bound yet).

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
