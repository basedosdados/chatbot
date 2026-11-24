import json
import math
import re
from typing import Any

import vl_convert as vlc
from pydantic import JsonValue

from app.charts.binding import GEO_ASSETS, VEGA_LITE_SCHEMA, resolve_named_data
from app.charts.config import merge_config
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

# The width a spec is drawn at to be checked. The website lays the chart out for the
# reader's screen itself (its views' sizes, the columns of small multiples, the legends),
# so nothing checked here depends on it: it only gives a single view a size to fit.
_RENDER_WIDTH = 694

# The colors the website gives a chart's text, under the spec's own `config`. They
# decide what text can be read on a background the spec sets. The subtitle is judged
# in Vega's default black, so a spec on a dark background must give it a color itself.
_SITE_TEXT_COLORS: dict[str, Any] = {
    "axis": {"labelColor": "#464A51", "titleColor": "#252A32"},
    "legend": {"labelColor": "#464A51", "titleColor": "#252A32"},
    "title": {"color": "#252A32"},
}

# What is behind a chart that sets no background: the website's white chart card.
_SITE_BACKGROUND = "white"

# A chart larger than this, in pixels on either side, is rejected: no screen shows it
# and a browser canvas cannot hold it (e.g. a fixed scale domain far from the data).
_MAX_CHART_PIXELS = 10_000

# Text whose contrast ratio (WCAG) against what is behind it is below this cannot be
# read: 1 is the same color, white on black is 21. A muted gray label on white ("#999",
# 2.8) stays above it; the website's text colors on a dark background do not.
_MIN_TEXT_CONTRAST = 2

# The color names a spec most often gives text or a background. Any other name is not
# judged, like a gradient.
_NAMED_COLORS = {
    "white": (255, 255, 255),
    "whitesmoke": (245, 245, 245),
    "lightgray": (211, 211, 211),
    "lightgrey": (211, 211, 211),
    "silver": (192, 192, 192),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "dimgray": (105, 105, 105),
    "dimgrey": (105, 105, 105),
    "black": (0, 0, 0),
}

_RGB_COLOR = re.compile(
    r"rgba?\(\s*(\d+)[\s,]+(\d+)[\s,]+(\d+)(?:\s*[,/]\s*(\d*\.?\d+)(%?))?"
)

# What each guide's text is called in a problem fed back to the generator.
_GUIDE_TEXT = {
    "title-text": "The title",
    "title-subtitle": "The subtitle",
    "axis-label": "An axis's labels",
    "axis-title": "An axis's title",
    "legend-label": "A legend's labels",
    "legend-title": "A legend's title",
}

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


def _parse_color(value: Any) -> tuple[int, int, int] | None:
    """Parse a CSS color into its red, green and blue, or None if it is not a plain one.

    A translucent color shows what is behind it, which cannot be told from the color
    alone, so it is not judged.

    Args:
        value (Any): A fill as the scenegraph holds it: a hex string, an `rgb()` or
            `rgba()` call, a color name, or something else (a gradient, `transparent`,
            nothing).

    Returns:
        tuple[int, int, int] | None: The color, or None for one that cannot be judged.
    """
    if not isinstance(value, str):
        return None
    color = value.strip().lower()
    if color in _NAMED_COLORS:
        return _NAMED_COLORS[color]
    if match := _RGB_COLOR.match(color):
        alpha, percent = match.group(4, 5)
        if alpha is not None and float(alpha) < (100 if percent else 1):
            return None
        red, green, blue = (min(int(part), 255) for part in match.group(1, 2, 3))
        return red, green, blue
    if re.fullmatch(r"#[0-9a-f]{3}(?:[0-9a-f]{3})?", color):
        digits = color[1:]
        if len(digits) == 3:
            digits = "".join(digit * 2 for digit in digits)
        return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)
    return None


def _contrast(color: tuple[int, int, int], other: tuple[int, int, int]) -> float:
    """Return the WCAG contrast ratio between two colors: 1 (the same) to 21."""

    def luminance(rgb: tuple[int, int, int]) -> float:
        linear = [
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            for c in (channel / 255 for channel in rgb)
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    lighter, darker = sorted((luminance(color), luminance(other)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def _unreadable(text_mark: dict[str, Any], backdrop: tuple[int, int, int]) -> bool:
    """Return whether a guide's text mark writes text and none of it stands out.

    Args:
        text_mark (dict[str, Any]): A text mark of the scenegraph (a title, an axis's or
            a legend's labels or title).
        backdrop (tuple[int, int, int]): The color behind the group the mark is in.

    Returns:
        bool: True if the mark writes visible text in colors all lost in the backdrop.
    """
    lost = False
    for item in text_mark.get("items") or []:
        if not str(item.get("text") or "").strip():
            continue
        if item.get("opacity", 1) == 0 or item.get("fillOpacity", 1) == 0:
            continue
        color = _parse_color(item.get("fill"))
        if color is None or _contrast(color, backdrop) >= _MIN_TEXT_CONTRAST:
            return False
        lost = True
    return lost


def _find_unreadable_text(
    node: JsonValue,
    backdrop: tuple[int, int, int],
) -> set[str]:
    """Find the guides of a rendered chart whose text is lost in what is behind it.

    Only titles, axes and legends are judged. A value label may sit on a bar, a point or
    a map's shape, whose color the scenegraph does not reliably give, so it is not.

    Args:
        node (JsonValue): A group mark of the scenegraph, starting from its root.
        backdrop (tuple[int, int, int]): The color behind the group: the chart's
            background, or the fill of a group around it.

    Returns:
        set[str]: The roles of the guides whose text cannot be read (e.g. "title-subtitle").
    """
    unreadable: set[str] = set()
    if not isinstance(node, dict) or node.get("marktype") != "group":
        return unreadable

    for group in node.get("items") or []:
        if not isinstance(group, dict):
            continue
        behind = _parse_color(group.get("fill")) or backdrop
        for child in group.get("items") or []:
            if not isinstance(child, dict):
                continue
            if child.get("marktype") == "group":
                unreadable |= _find_unreadable_text(child, behind)
            elif (
                child.get("marktype") == "text"
                and child.get("role") in _GUIDE_TEXT
                and _unreadable(child, behind)
            ):
                unreadable.add(child["role"])

    return unreadable


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

    How the chart fits the screen is not checked: the website sizes its views and lays
    out its panels for the reader's screen, and shows a chart too large for it as a
    preview that opens in full. Only a chart with no size, or one too large for any
    screen, is rejected.

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
    # Drawn with the website's text colors under the spec's own.
    render_spec = {
        **render_spec,
        "$schema": VEGA_LITE_SCHEMA,
        "width": _RENDER_WIDTH,
        "config": merge_config(_SITE_TEXT_COLORS, spec.get("config") or {}),
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

    config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
    backdrop = spec.get("background") or config.get("background")
    if not backdrop or str(backdrop).strip().lower() == "transparent":
        # The website draws a chart with no background of its own on its white card.
        backdrop = _SITE_BACKGROUND
    background = _parse_color(backdrop)
    # A background that cannot be judged (a gradient, a translucent or unusual color)
    # is left alone.
    unreadable = _find_unreadable_text(root, background) if background else set()

    for role in sorted(unreadable):
        problems.append(
            f"{_GUIDE_TEXT[role]} cannot be read: the text is drawn in the color of "
            "what is behind it (e.g. white text on the white panel, or the default dark "
            "text on a dark `background`). Give that text a color that contrasts with "
            "the background (e.g. `subtitleColor` in `title`, or `labelColor` and "
            "`titleColor` in `config.axis` and `config.legend`)."
        )

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

    # Collected as a set: a faceted chart has one axis per panel, all alike.
    axis_problems: set[str] = set()
    for role, labels in text_marks:
        if role != "axis-label":
            continue
        # Labels blanked on purpose (`labelExpr: "''"`, e.g. a sparkline) are not broken.
        if len(labels) >= 3 and len(set(labels)) == 1 and labels[0].strip():
            axis_problems.add(
                f"An axis shows the same label {labels[0]!r} on all of its {len(labels)} "
                "ticks, so its `labelExpr` or `format` discards the value. Fix that axis."
            )
        elif not labels and drawn:
            axis_problems.add(
                "An axis has no tick labels with the result's rows: its scale has no valid "
                "domain, or the values fall far outside it (e.g. a log scale over zero or "
                "negative values, a fixed `domain` that does not fit the values, series "
                "in different units on one scale, or values that are not valid dates "
                "or numbers). Fix that axis's scale."
            )
    problems.extend(sorted(axis_problems))

    return problems
