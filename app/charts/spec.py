"""Reading a spec: the fields it references, creates and treats as dates."""

from pydantic import JsonValue

# Fields a transform creates when its `as` is omitted (Vega-Lite's defaults).
_DEFAULT_TRANSFORM_OUTPUTS = {
    "fold": ("key", "value"),
    "density": ("value", "density"),
    "quantile": ("prob", "value"),
}


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


def collect_calculated_fields(node: JsonValue) -> set[str]:
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
            found |= collect_calculated_fields(value)
    elif isinstance(node, list):
        for item in node:
            found |= collect_calculated_fields(item)
    return found
