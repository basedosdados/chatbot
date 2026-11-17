import re

from pydantic import JsonValue

# Fields a transform creates when its `as` is omitted (Vega-Lite's defaults).
_DEFAULT_TRANSFORM_OUTPUTS = {
    "fold": ("key", "value"),
    "density": ("value", "density"),
    "quantile": ("prob", "value"),
}

# A call in a Vega expression that yields a real date from an integer year: one that
# builds or parses a date, or a `toDate` of anything but a bare field.
_DATE_EXPRESSION = re.compile(
    r"\b(?:datetime|utc|timeParse|utcParse)\s*\("
    r"|\btoDate\s*\((?!\s*datum(?:\.\w+|\[[^\]]*\])\s*\))"
)


def place_facet_columns(node: JsonValue) -> JsonValue:
    """Move a `columns` written inside a `facet` operator to where Vega-Lite reads it.

    A wrapped facet takes its `columns` beside `facet`, not inside it (only the `facet`
    encoding channel takes it inside). Vega-Lite ignores a misplaced one and lays every
    panel out in a single row, far wider than the panel, though what was meant is clear.

    Args:
        node (JsonValue): A spec, or any node within it, to walk.

    Returns:
        JsonValue: The same node with every misplaced `columns` moved beside its `facet`.
    """
    if isinstance(node, list):
        return [place_facet_columns(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {key: place_facet_columns(value) for key, value in node.items()}
    facet = out.get("facet")
    # A facet operator sits beside the `spec` it repeats; a facet channel does not.
    if isinstance(facet, dict) and "spec" in out and "columns" in facet:
        columns = facet["columns"]
        out["facet"] = {key: value for key, value in facet.items() if key != "columns"}
        out.setdefault("columns", columns)
    return out


def collect(node: JsonValue, key: str) -> set[str]:
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
                found |= collect(value, key)
    elif isinstance(node, list):
        for item in node:
            found |= collect(item, key)
    return found


def collect_derived_fields(node: JsonValue) -> set[str]:
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
            found |= collect_derived_fields(value)
    elif isinstance(node, list):
        for item in node:
            found |= collect_derived_fields(item)
    return found


def collect_temporal_fields(node: JsonValue) -> set[str]:
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
            found |= collect_temporal_fields(value)
    elif isinstance(node, list):
        for item in node:
            found |= collect_temporal_fields(item)
    return found


def collect_date_calculated_fields(node: JsonValue) -> set[str]:
    """Collect every field a `calculate` transform writes as a date, anywhere within a spec.

    An expression writes a date when it builds one (`datetime`, `utc`), parses one from
    a string (`timeParse`, `utcParse`) or converts anything but a bare field with
    `toDate`: `toDate(datum.ano)` reads an integer year as epoch milliseconds.

    Args:
        node (JsonValue): A spec, or any node within it, to walk.

    Returns:
        set[str]: Every `as` name of a `calculate` transform whose expression writes a date.
    """
    found: set[str] = set()
    if isinstance(node, dict):
        expression = node.get("calculate")
        output = node.get("as")
        if (
            isinstance(expression, str)
            and isinstance(output, str)
            and _DATE_EXPRESSION.search(expression)
        ):
            found.add(output)
        for value in node.values():
            found |= collect_date_calculated_fields(value)
    elif isinstance(node, list):
        for item in node:
            found |= collect_date_calculated_fields(item)
    return found
