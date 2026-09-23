"""Parsing of Base dos Dados coverage-period values.

A table's `period_start` / `period_end` (from `get_table_details`) arrive in one of
three formats, and the format itself carries the granularity::

    "2026"       -> year
    "2026-05"    -> month
    "2026-05-01" -> day

These helpers recover that granularity and the leading year, so a scorer can compare a
period against a table's own coverage at the right grain without hardcoding any date.
"""

import re
from typing import Literal, NamedTuple

Granularity = Literal["year", "month", "day"]

# Rank (coarse -> fine) and the ISO-prefix length of each granularity, for truncation.
_GRANULARITY_RANK: dict[str, int] = {"year": 0, "month": 1, "day": 2}
_GRANULARITY_PREFIX: dict[str, int] = {"year": 4, "month": 7, "day": 10}


class ParsedPeriod(NamedTuple):
    """A coverage-period value split into its granularity and its original text."""

    granularity: Granularity
    value: str


# Ordered most-specific first: "2026-01-01" must match `day` before `year` matches its prefix.
_PERIOD_FORMATS: tuple[tuple[Granularity, str], ...] = (
    ("day", r"\d{4}-\d{2}-\d{2}"),
    ("month", r"\d{4}-\d{2}"),
    ("year", r"\d{4}"),
)

_LEADING_YEAR_RE = re.compile(r"\s*(\d{4})")


def parse_period(value: object) -> ParsedPeriod | None:
    """Split a coverage-period value into (granularity, text).

    Strict: a value that is not exactly one of the three formats returns `None` rather
    than guessing. Surrounding quotes and whitespace are tolerated ("'2026-01-01'").

    Args:
        value: The raw period value (str or None).

    Returns:
        The parsed period, or `None` when the value is missing or malformed.
    """
    text = str(value or "").strip().strip("'").strip('"')
    for granularity, pattern in _PERIOD_FORMATS:
        if re.fullmatch(pattern, text):
            return ParsedPeriod(granularity, text)
    return None


def leading_year(value: object) -> int | None:
    """The leading four-digit year of a period value ("2026", "2026-01-01", "2026-01").

    Looser than :func:`parse_period`: it only needs a value to *start* with four digits,
    so it recovers a year from an otherwise irregular value.

    Args:
        value: The raw period value (str or None).

    Returns:
        The leading year, or `None``when the value has no leading four digits.
    """
    match = _LEADING_YEAR_RE.match(str(value or ""))
    return int(match.group(1)) if match else None


def is_coarser_or_equal(granularity: Granularity, than: Granularity) -> bool:
    """Whether `granularity` is no finer than `than` (year ≤ month ≤ day)."""
    return _GRANULARITY_RANK[granularity] <= _GRANULARITY_RANK[than]


def coarser(a: Granularity, b: Granularity) -> Granularity:
    """The coarser of two granularities (year < month < day)."""
    return a if _GRANULARITY_RANK[a] <= _GRANULARITY_RANK[b] else b


def truncate(value: object, granularity: Granularity) -> str | None:
    """Reduce a period value to a coarser (or equal) granularity's ISO prefix.

    `"2026-05-01"` → `"2026-05"` (month) / `"2026"` (year). Returns ``None`` when the value
    can't reach that granularity — either it's malformed, or it's *coarser* than requested
    (a year can't be truncated to a month), which is how a caller detects "the query didn't
    pin the period this finely" and abstains rather than guessing.

    Args:
        value: The period value (str or None).
        granularity: The target granularity to truncate to.

    Returns:
        The truncated ISO-prefix string, or ``None`` if the value can't reach it.
    """
    parsed = parse_period(value)
    if parsed is None or not is_coarser_or_equal(granularity, parsed.granularity):
        return None
    return parsed.value[: _GRANULARITY_PREFIX[granularity]]


def complete_year(value: object) -> int | None:
    """The most recent fully-covered calendar year implied by a coverage `period_end`.

    The prompt's period rule anchors a **flow** on the most recent *complete* year, which is
    not always `period_end`'s year: an annual `period_end` completes its year, but a
    sub-annual one only does so if it reaches December — otherwise the current year is partial
    and the last complete year is the one before. So `"2025"` → 2025, `"2026-12"` → 2026, but
    `"2026-06"` → 2025.

    Args:
        value: The coverage `period_end` value (str or None).

    Returns:
        The most recent complete year, or ``None`` when the value has no parseable year.
    """
    parsed = parse_period(value)
    if parsed is None:
        return None
    year = int(parsed.value[:4])
    if parsed.granularity == "year":
        return year
    if parsed.granularity == "month":
        return year if parsed.value[5:7] == "12" else year - 1
    return year if parsed.value[5:10] == "12-31" else year - 1
