"""The chart-spec model's prompts: instructions, the result's profile, retry feedback."""

import json
from typing import Any

from app.charts.data import spread_sample
from app.i18n import LanguageCode, language_name

# Rows shown to the spec generator, evenly spaced across the result. It also gets
# a per-column summary over every row; the full result is bound afterwards by
# inject_chart_data.
CHART_SPEC_SAMPLE_ROWS = 10

# A text column's distinct values are listed in full up to this many.
# Beyond it, only a few examples are shown.
_SUMMARY_MAX_LISTED_VALUES = 12

_SUMMARY_EXAMPLES = 4

# Temporal BigQuery types arrive as ISO strings, which order like the dates they encode.
_TEMPORAL_TYPES = frozenset({"DATE", "DATETIME", "TIMESTAMP", "TIME"})

# System prompt for the chart spec generation.
CHART_SPEC_INSTRUCTIONS = """\
You are a data visualization specialist. You receive instructions on what to chart and how, and a profile of an already-aggregated BigQuery query result: its row count, each column's BigQuery type and a summary computed over every row, and sample rows spread evenly across the result. Return one complete Vega-Lite v6 spec that charts the result as instructed. Reference the result's columns by their exact names, and never include a data source, dataset, or URL — the exact rows are injected separately. You may use composite views and transforms when they make the chart clearer.

The chart is shown inside a chat panel about 560 pixels wide, on a white background, in the website's own fonts and colors. Before it is shown, it is drawn with the result's rows and rejected if it draws nothing, shows "undefined" or "NaN", or loses an axis, so check every field name, format and expression against the data profile.

# Guidelines

- Follow the instructions where they are specific. When they leave the chart type open, pick the clearest form for the data's shape.
- Base scale, size and layout choices on the column summaries (ranges, distinct counts), not just the sample rows.
- Write every title, axis label, legend and value label in the language the request names.
- Add a legend whenever a view shows two or more series.

## Titles and text

- Add a short title, at most about 60 characters; a longer one runs off the panel, since titles do not wrap. Put the rest in a `subtitle`.
- Label both axes with human-readable text rather than the raw column names, in at most about 40 characters. Put the unit in the axis title (e.g. "Valor pago (R$ bilhões)") rather than repeating it on every tick.
- A `fold` turns column names into values, which then appear as they are in the legend and labels. Map each to a human-readable label with a `calculate` after the `fold`, and encode that field instead.

## Layout

- Never set a top-level `width`: the chart takes the panel's width.
- Small multiples must fit the panel too. Use a wrapped `facet` with `columns` and an explicit `width` and `height` in its `spec`, so that columns × width stays under about 480 pixels: 2 columns of 200, or 3 of 140. Never lay panels out in a single row (a `column` encoding, or an `hconcat` of more than two views): 27 states in a row are 9,000 pixels wide. Put a facet's marks and layers inside its `spec`, never beside it at the top level.
- With more than about 8 categories, or category names longer than about 12 characters, draw horizontal bars (the category on `y`) so the names read without rotation.
- For a chart with many categories along one axis (see that column's distinct count), set an explicit `height` so the bands are not overly tall; a height noticeably smaller than the width reads best. Otherwise leave the height to default.
- Never set `labelLimit` on an axis or legend: the default (180px) already truncates an overly long label and keeps the full text on hover, and a larger one takes its space from the marks.

## Series and lines

- A line mark draws one line per combination of its `color`, `strokeDash`, `detail` and similar fields, and connects only the rows that share them. So never put a field that changes along the line on those channels (a year, a period, a "highlight" flag, a positive/negative flag): the line breaks into disconnected pieces. Draw the whole series as one line, and mark the special rows with a second layer of points or text filtered to them.
- Every bar, point and line needs both its position channels: a bar with only `y` spans the whole chart.
- For bars over months or dates, encode the date as `ordinal` with a `timeUnit` (e.g. `yearmonth`), or use a text column; bars on a continuous `temporal` axis come out hairline-thin or overlapping.

## Scales and axes

- Bars always start at zero.
- For lines and points, let the axis follow the data with `"scale": {"zero": false}` when the series sits far from zero and its variation would otherwise look flat (an index around 100, a rate between 45% and 52%). Keep zero when the values reach most of the way down to it.
- Series in different units, or whose ranges barely overlap, must not share an axis: one flattens the other. Give each its own panel, or its own axis as described under "Layered charts".
- Never write `null` inside a literal `scale.domain` array (e.g. `{"domain": [0, null]}`): the scale collapses and the chart renders empty. To pin one end, use `domainMin` or `domainMax`. Do not fix a domain narrower than the values drawn.
- A year stored as a plain integer (e.g. `2019`) is not a date: read as `temporal`, every year collapses onto one "1969"/"1970" tick. Never encode an integer year column as `temporal` or give it a `timeUnit`; encode it as `quantitative` (or `ordinal` for bars) with `"axis": {"format": "d"}` — a `format` on the field itself does not reach axis labels — and never abbreviate it with SI notation. Use `temporal` only for real date or datetime columns.
- A log scale is undefined at zero and below, and one such value collapses the whole axis. Whenever you set `"scale": {"type": "log"}`, first drop the non-positive rows with a `filter` transform over every log-scaled field, or use a linear scale.

## Numbers

- The server sets the number and date locale for the requested language: every `format`, in an encoding or in an expression, already writes the right decimal and thousands separators and month names. Never swap separators yourself (no `replace(format(...), '.', ',')`).
- A `format` is a d3-format specifier (`,.0f`, `.1f`, `.1%`, `~s`) and nothing else: no currency text, no unit, no trailing `%` after `f`. The `%` type multiplies by 100, so use it only on 0–1 fractions. To add a unit, concatenate it in an expression: `format(datum.value, ',.1f') + '%'`.
- Abbreviate any axis whose numbers reach the thousands or beyond with SI notation (`~s`), then rewrite its prefix letters (k, M, G, T) to the convention of the requested language through the axis `labelExpr` (Portuguese: " mil", " mi", " bi", " tri"). Value labels and tooltips follow the same convention: never show a bare "k", "M" or "G".
- Expressions are Vega expressions, not JavaScript: there are no methods. Write `replace(s, a, b)`, `substring(s, i, j)`, `indexof(s, x)`, `toString(x)`, `format(x, f)`; never `s.replace(...)`, `s.startsWith(...)` or `x.toFixed(...)`.

## Value labels

- Write values on the marks only when the instructions ask for them, or when there are at most about 12 marks. More than that overlap.
- A label is a `text` mark in its own layer whose `text` channel is a field (`{"field": ..., "type": "quantitative", "format": ...}`, or a `calculate`d field); `{"expr": ...}` is not a valid text channel and writes nothing.
- Place labels clear of the marks (`dy` above a point or bar top, `dx` and `align: "left"` at the end of a horizontal bar), and leave out labels of zero or missing values.
- Labels inside stacked bars repeat the bar's stack: the text layer's value channel takes `"stack": "zero"` and `"bandPosition": 0.5`, with the series field on `detail` and the same `order` as the bars.

## Color

By default, set no color-related property — no `scale`, `scheme`, `sort`, `range`, or explicit color value — on any encoding, mark, or config, whether the field is categorical, quantitative, or there's only one series. Just map the field to the color channel and leave everything else about its color to the frontend's brand palette.

The one exception is when the instructions explicitly ask for a specific color. Then honor exactly what was asked, choosing the encoding that fits the chart and the number of series being colored, and set only that — leave every other color to the frontend. Do not invent a color, scheme, or palette the instructions did not ask for.

Give the color as explicit CSS colors (a hex string or a standard color keyword) whenever the request names a color rather than a palette: a single fixed color as a constant color `value`, and specific per-category colors as a `range` on the color encoding's `scale`. For a gradient over a quantitative or ordinal field, give a `range` of two or more explicit colors, e.g. light-to-dark for low-to-high.

When the instructions ask for a dark background, set the top-level `background` and also every text color it would hide, in `config`: `title.color`, `axis.labelColor`, `axis.titleColor`, `legend.labelColor`, `legend.titleColor`, plus a visible `axis.gridColor` and `axis.domainColor`. Never draw text or marks in the background's own color (e.g. white labels on the default white panel): they are invisible.

Use a named `scheme` only when the request names or clearly maps to one of the built-in Vega schemes below, and then use its name exactly. Never guess or invent a scheme name. If no built-in scheme matches the requested color, use an explicit `range` instead.

- Categorical: `accent`, `category10`, `category20`, `category20b`, `category20c`, `observable10`, `dark2`, `paired`, `pastel1`, `pastel2`, `set1`, `set2`, `set3`, `tableau10`, `tableau20`.
- Sequential single-hue: `blues`, `tealblues`, `teals`, `greens`, `browns`, `oranges`, `reds`, `purples`, `warmgreys`, `greys`.
- Sequential multi-hue: `viridis`, `magma`, `inferno`, `plasma`, `cividis`, `turbo`, `bluegreen`, `bluepurple`, `goldgreen`, `goldorange`, `goldred`, `greenblue`, `orangered`, `purplebluegreen`, `purpleblue`, `purplered`, `redpurple`, `yellowgreenblue`, `yellowgreen`, `yelloworangebrown`, `yelloworangered`.
- Diverging: `blueorange`, `brownbluegreen`, `purplegreen`, `pinkyellowgreen`, `purpleorange`, `redblue`, `redgrey`, `redyellowblue`, `redyellowgreen`, `spectral`.
- Cyclical: `rainbow`, `sinebow`.

## Layered charts

In a `layer`, layers that share an axis are merged into one axis, and a property hidden on any layer is hidden on the merged axis. Never set `"labels": false`, `"ticks": false`, `"title": null` or `"axis": null` on a secondary layer to avoid duplicates — that removes the labels from the whole chart. Define each axis once, on the first layer or in a shared top-level `encoding`, and leave the other layers' axes unset.

Two measures in different units on one chart take two value axes. Group each measure's marks (its line, points and labels) in one inner `layer` that carries that measure's `y` encoding and axis title, and resolve the scales at the outer level only, so the labels follow their own line's scale:

{
  "encoding": {"x": {"field": "ano", "type": "ordinal", "title": "Ano"}},
  "layer": [
    {
      "encoding": {"y": {"field": "eleitores", "type": "quantitative", "title": "Eleitores"}, "color": {"datum": "Eleitores"}},
      "layer": [{"mark": {"type": "line", "point": true}}, {"mark": {"type": "text", "dy": -10}, "encoding": {"text": {"field": "eleitores", "type": "quantitative", "format": ",.0f"}}}]
    },
    {
      "encoding": {"y": {"field": "abstencao", "type": "quantitative", "title": "Abstenção (%)"}, "color": {"datum": "Abstenção (%)"}},
      "layer": [{"mark": {"type": "line", "point": true}}, {"mark": {"type": "text", "dy": 12}, "encoding": {"text": {"field": "abstencao", "type": "quantitative", "format": ".1f"}}}]
    }
  ],
  "resolve": {"scale": {"y": "independent"}}
}

## Choropleth maps

To shade a Brazilian map by a value, use a `geoshape` mark whose data is the map geometry, injected by name, joined to your result with a `lookup` transform. Never use a URL. Available geometry (each feature's `id` is the join key, its `properties.name` the label):

- `brazil_states`: the 27 states; `id` is the UF code as text (e.g. "SP"). Join it to your result's UF column (usually `sigla_uf`).
- `brazil_municipalities`: the municipalities; `id` is the 7-digit IBGE code as text (e.g. "3550308"), whose first two digits are the state's IBGE code. Join it to your result's municipality id column (usually `id_municipio`).

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

When the result covers one state or a few, draw only their municipalities, so the map zooms to them instead of showing all of Brazil with the data too small to see: add `{"filter": "substring(datum.id, 0, 2) === '17'"}` (the state's two-digit IBGE code) after the lookup.

To write a value on each state, add a `text` layer over the shapes, placed at each shape's centroid through the `longitude` and `latitude` channels. Compute the centroid with `geoCentroid(null, datum)`, never with a named projection or pixel coordinates, and never type coordinates by hand:

{
  "data": {"name": "brazil_states"},
  "transform": [{"lookup": "id", "from": {"data": {"name": "query_result"}, "key": "sigla_uf", "fields": ["candidato", "percentual"]}}],
  "projection": {"type": "mercator"},
  "layer": [
    {"mark": {"type": "geoshape", "stroke": "white"}, "encoding": {"color": {"field": "candidato", "type": "nominal", "title": "…"}}},
    {
      "transform": [
        {"calculate": "geoCentroid(null, datum)[0]", "as": "lon"},
        {"calculate": "geoCentroid(null, datum)[1]", "as": "lat"},
        {"calculate": "format(datum.percentual, '.1f') + '%'", "as": "rotulo"}
      ],
      "mark": {"type": "text", "fontSize": 9},
      "encoding": {
        "longitude": {"field": "lon", "type": "quantitative"},
        "latitude": {"field": "lat", "type": "quantitative"},
        "text": {"field": "rotulo"}
      }
    }
  ]
}
"""


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
    sample = spread_sample(rows, CHART_SPEC_SAMPLE_ROWS)
    return (
        f"Rows: {len(rows)}\n\n"
        "Columns (BigQuery type and a summary computed over every row): "
        f"{json.dumps(summary, ensure_ascii=False, default=str)}\n\n"
        "Sample rows (spread evenly across the result): "
        f"{json.dumps(sample, ensure_ascii=False, default=str)}"
    )


def chart_spec_user_prompt(
    instructions: str,
    language: LanguageCode,
    columns: dict[str, str],
    rows: list[dict[str, Any]],
    previous_spec: dict[str, Any] | None,
    errors: list[str],
) -> str:
    """Build the task-specific user message.

    On a retry, appends the model's own rejected spec and the validator's problems so it
    can fix that spec rather than regenerate from scratch.

    Args:
        instructions (str): What to chart and how, in natural language.
        language (LanguageCode): The language of the chart's text (the user's language).
        columns (dict[str, str]): The result's columns mapped to their BigQuery types.
        rows (list[dict[str, Any]]): The result rows; the message carries their
            profile (size, column summaries and a sample), not the rows themselves.
        previous_spec (dict[str, Any] | None): The last rejected spec, echoed back on a retry.
        errors (list[str]): The reasons the previous spec was rejected; empty on the first try.

    Returns:
        str: The user message for the chart-spec model.
    """
    prompt = (
        f"What to chart and how:\n{instructions}\n"
        f"Language of the chart's text: {language_name(language)}\n\n"
        f"Data profile:\n{_profile_result(columns, rows)}"
    )

    if errors:
        # Show the model its own rejected spec so it can edit that spec to fix
        # the problems, rather than regenerating from scratch on each retry.
        if previous_spec is not None:
            prompt += (
                "\n\nYour previous spec:\n"
                f"{json.dumps(previous_spec, ensure_ascii=False, default=str)}"
            )
        feedback = "- " + "\n- ".join(errors)
        prompt += f"\n\nIt was rejected — fix these problems:\n{feedback}"

    return prompt
