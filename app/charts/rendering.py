import json
import math
import re
from typing import Any

import vl_convert as vlc
from pydantic import JsonValue

from app.charts.binding import GEO_ASSETS, VEGA_LITE_SCHEMA, resolve_named_data
from app.charts.spec import collect

# Most characters of a compile error fed back to the spec generator.
_COMPILE_ERROR_MAX_CHARS = 600

# What to check when an encoding fails on the rows, for the spec generator.
_ENCODING_HINT = (
    "Check that every color `scheme` is an exact Vega scheme name, that every number "
    "`format` is a d3-format specifier such as '.1f', ',.0f', '~s' or '.1%' (to add a "
    "unit or a '%' sign use a `labelExpr`, not the format), and that dates are parsed "
    "from the values the column really holds."
)

# The one render problem a sample of the rows can cause by itself: it may hold none of
# the rows a filter keeps.
NO_MARKS_PROBLEM = (
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
# expression) stops before the item gets any, so it is drawn nowhere. A position the spec
# does not encode (e.g. a bar with only `y`) is legitimately absent.
_DRAWN_ITEM_KEYS = {"shape": {"shape"}, "text": {"text"}}

_DRAWN_ITEM_DEFAULT_KEYS = {"x", "y"}

# The 27 federative units' codes: the feature ids of the `brazil_states` geometry.
_BRAZIL_STATES = (
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT", "PA",
    "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
)  # fmt: skip


def _geo_stub_ids(
    name: str, spec: dict[str, Any], rows: list[dict[str, Any]]
) -> list[str]:
    """Pick the feature ids a stub geometry needs, so a map's joins behave as in the browser.

    The states' ids are the real ones, all 27. The municipalities are too many to list,
    so they are the ones the rows join on, plus one no row matches, as the real geometry
    almost always has places without data.

    Args:
        name (str): A geo source name from `GEO_ASSETS` (e.g. "brazil_states").
        spec (dict[str, Any]): The spec, whose `lookup` transforms name the join columns.
        rows (list[dict[str, Any]]): The rows the spec is rendered with.

    Returns:
        list[str]: The feature ids, as text like the real geometry's.
    """
    if name == "brazil_states":
        return list(_BRAZIL_STATES)
    join_columns = collect(spec, "lookup") | collect(spec, "key")
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
        name (str): A geo source name from `GEO_ASSETS` (e.g. "brazil_states").
        ids (list[str]): The feature ids (see `_geo_stub_ids`).

    Returns:
        dict[str, Any]: A `data` node with the inline TopoJSON and its topojson `format`.
    """
    feature = GEO_ASSETS[name]["feature"]
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


def _compile_error(error: Exception) -> str:
    """Return a compile error's message without what the generator cannot use.

    vl-convert's message carries a JavaScript stack trace and, for an invalid
    specification, the whole spec with its rows: tens of thousands of characters
    that would go back to the model on the retry.

    Args:
        error (Exception): The error vl-convert raised.

    Returns:
        str: The message without the stack trace, its middle cut if still long.
    """
    lines = [
        line for line in str(error).splitlines() if not line.lstrip().startswith("at ")
    ]
    message = "\n".join(lines).strip()
    if len(message) > _COMPILE_ERROR_MAX_CHARS:
        half = _COMPILE_ERROR_MAX_CHARS // 2
        message = f"{message[:half]} … {message[-half:]}"
    return message


def render_problems(spec: dict[str, Any], rows: list[dict[str, Any]]) -> list[str]:
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
    render_spec = resolve_named_data(
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
        return [f"The chart spec does not compile: {_compile_error(e)}"]

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
        problems.append(NO_MARKS_PROBLEM)

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
                "domain, or the values fall far outside it (e.g. a log scale over zero or "
                "negative values, a fixed `domain` that does not fit the values, series "
                "in different units on one scale, or values that are not valid dates "
                "or numbers). Fix that axis's scale."
            )

    return problems
