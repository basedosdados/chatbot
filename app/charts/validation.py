from typing import Any

from app.charts.binding import GEO_ASSETS
from app.charts.data import spread_sample
from app.charts.rendering import NO_MARKS_PROBLEM, render_problems
from app.charts.spec import (
    collect,
    collect_date_calculated_fields,
    collect_derived_fields,
    collect_temporal_fields,
)

# Whole-number values in this range are read as calendar years.
_YEAR_RANGE = range(1000, 3001)

# Most rows a spec is validated with; a larger result is validated with an even spread
# of this many. Render time grows with the marks drawn, and renders in a process run one
# at a time: a bar and a text label per row took 1.2 s at 500 rows, 11 s at 5,000 and
# 44 s at 20,000. Most charts draw far fewer marks and render in about 40 ms.
_VALIDATION_MAX_ROWS = 500


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


def validate_chart_spec(
    spec: dict[str, Any],
    columns: list[str],
    rows: list[dict[str, Any]],
) -> list[str]:
    """Return the reasons a spec would not render a correct chart (empty if valid).

    Args:
        spec (dict[str, Any]): The candidate spec (already sanitized).
        columns (list[str]): The result's real column names.
        rows (list[dict[str, Any]]): The result rows. Integer years are looked for in
            all of them, and the chart is rendered with them (an even spread of
            `_VALIDATION_MAX_ROWS`, for a larger result). With no rows, only that the
            chart builds is checked.

    Returns:
        list[str]: Human-readable problems to feed back to the generator; empty if valid.
    """
    errors: list[str] = []

    referenced = collect(spec, "field")
    derived = collect_derived_fields(spec)
    # `*` stands for "every row" (e.g. in a `count`), never a column.
    unknown = referenced - set(columns) - derived - {"*"}
    # A map's data is its geometry, whose features carry fields of their own: the top-level
    # `id` and the labels under `properties.*`. They are not result columns, so they are
    # allowed, but only in a spec that reads a geometry (a map need not draw a geoshape:
    # it may draw only points or labels at the features' centroids).
    if collect(spec, "name") & GEO_ASSETS.keys():
        unknown = {
            field
            for field in unknown
            if field != "id" and not field.startswith("properties.")
        }
    missing = sorted(unknown)

    # A pivot names its new columns after the pivoted field's values, which the spec alone
    # does not reveal, so the check is skipped rather than reject a valid spec.
    if missing and not collect(spec, "pivot"):
        errors.append(
            f"Encoding references column(s) not in the result: {missing}. "
            f"Available columns: {sorted(columns)}."
        )

    # An integer year in a temporal field is read as epoch milliseconds, so every year
    # lands in the first seconds of 1970 UTC, one tick labeled with the viewer's local
    # year. The chart renders without any error, and its output looks like any chart
    # with a single date, so the render check cannot tell.
    # A year a `calculate` rewrites as a date under its own name
    # (e.g. `datetime(datum.ano, 0, 1)` as `ano`) is a real date by then, so it is not flagged;
    # one it rewrites as anything else (e.g. `datum.ano + 0`) still is.
    temporal_years = sorted(
        (collect_temporal_fields(spec) & _integer_year_columns(columns, rows))
        - collect_date_calculated_fields(spec)
    )
    if temporal_years:
        errors.append(
            f"Column(s) {temporal_years} hold integer years, not dates, but are encoded "
            "as `temporal` or with a `timeUnit`, so every year collapses onto a single "
            '"1969"/"1970" tick. Encode them as `quantitative` (or `ordinal` for bars) '
            'with `"axis": {"format": "d"}` instead.'
        )

    sample = spread_sample(rows, _VALIDATION_MAX_ROWS)
    problems = render_problems(spec, sample)
    if NO_MARKS_PROBLEM in problems and len(sample) < len(rows):
        # The sample may hold none of the rows a filter keeps, so every row decides.
        # That render is cheap: either few rows pass the filter, or the encoding
        # fails before anything is drawn. No other problem comes from sampling.
        problems = render_problems(spec, rows)

    return errors + problems
