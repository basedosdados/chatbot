"""Charting a query result as a Vega-Lite spec.

The pipeline, as the `chart_query_result` tool runs it:

1. `fetch_chart_data` (data.py): authorize the result and read its rows, capped in size.
2. `generate_chart_spec` (generation.py): prompt the model (prompts.py), then sanitize
   its spec (binding.py), set the server's config on it (config.py) and validate it
   (validation.py), retrying with the problems fed back. Validation checks what the
   spec references (spec.py) and what it draws with the result's rows (rendering.py).
3. `inject_chart_data` (binding.py): bind the real rows and geometry into the spec.
"""

from app.charts.binding import inject_chart_data
from app.charts.data import ChartHandleNotFound, ChartResultTooLarge, fetch_chart_data
from app.charts.generation import (
    ChartSpecInvalid,
    ChartSpecTimeout,
    generate_chart_spec,
)

__all__ = [
    "ChartHandleNotFound",
    "ChartResultTooLarge",
    "ChartSpecInvalid",
    "ChartSpecTimeout",
    "fetch_chart_data",
    "generate_chart_spec",
    "inject_chart_data",
]
