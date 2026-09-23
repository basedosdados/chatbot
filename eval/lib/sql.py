"""Reading the SQL the agent executed.

Two kinds of readers live here:

* **Structural** readers parse the query with sqlglot's BigQuery dialect and walk the AST
  (tables hit, statement shape, columns used in each clause). Parse once with :func:`parse`
  and pass the resulting expression to these — they assume a valid tree, so a caller
  abstains on a query that :func:`parse` couldn't handle (`None`) rather than guessing.
  On the real transcripts every executed query parsed, so parse failure is an edge case,
  not the norm.
* **Text** readers scan raw strings and result-row values: year detection (robust for
  `BETWEEN` and date literals) and a `project.dataset.table` pattern match used to spot
  a table id leaked into prose — a job for a regex, since prose is not parseable SQL.

The column-usage readers are named for the prompt's own vocabulary — a coded column may not
be *filtered*, *grouped*, or *displayed* untranslated — so a check reads the way the rule
reads. Groups are covered via :func:`projected_columns` (a grouped column is virtually
always also projected); a column grouped only by ordinal and never projected is the one
gap, and it errs toward not flagging.
"""

import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

# ---- text-scan patterns ------------------------------------------------------
# A BigQuery id segment is an identifier (letter/underscore-led), never all-digit — so a
# `project.dataset.table` ref matches but a PT-formatted number like 213.421.037 does not.
_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_-]*"
_GCP_REF = rf"{_IDENTIFIER}\.{_IDENTIFIER}\.{_IDENTIFIER}"
_GCP_REF_FULL_RE = re.compile(rf"^{_GCP_REF}$")
_GCP_REF_WORD_RE = re.compile(rf"\b{_GCP_REF}\b")
_LINE_COMMENT_RE = re.compile(r"--[^\n]*")

# A year appears in SQL as a bare 19xx/20xx literal. In this domain that is unambiguous:
# municipio codes are 7-digit, NCM 8-digit, and LIMIT/GROUP values are small.
_YEAR_LITERAL_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# Result-row year detection: a value is a year when it sits in a year/date-named column and
# looks like one, or is an unambiguous date string in any column.
_YEAR_COLUMN_RE = re.compile(r"ano|year|exercicio|periodo|competencia", re.I)
_DATE_COLUMN_RE = re.compile(r"data|date", re.I)
_YEAR_VALUE_RE = re.compile(r"(?:19|20)\d{2}")
_DATE_VALUE_RE = re.compile(r"(\d{4})-\d{2}(?:-\d{2})?")

# A quoted date/month literal in SQL, e.g. WHERE data = '2026-06-01' or '2026-06'.
_DATE_LITERAL_RE = re.compile(r"['\"](\d{4}-\d{2}(?:-\d{2})?)['\"]")
_YEAR_ONLY_RE = re.compile(r"(?:19|20)\d{2}")

# Statements that only read data; anything else (INSERT/UPDATE/DELETE/MERGE/CREATE/…) does not.
_READ_ONLY_STATEMENTS = (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)


# =============================================================================
# Parsing
# =============================================================================
def parse(sql: str | None) -> exp.Expression | None:
    """Parse a query with the BigQuery dialect, returning `None` if it can't be parsed.

    Parse once and pass the result to the structural readers below, so a whole query's
    checks share one parse and a parse failure is handled in exactly one place.

    Args:
        sql: The executed SQL text.

    Returns:
        The parsed expression, or `None` when the text is empty or not parseable.
    """
    if not sql or not sql.strip():
        return None
    try:
        return sqlglot.parse_one(sql, dialect="bigquery")
    except SqlglotError:
        return None


# =============================================================================
# Structural readers (operate on a parsed expression)
# =============================================================================
def cte_names(tree: exp.Expression) -> set[str]:
    """The names introduced by `WITH name AS (...)` common table expressions."""
    return {cte.alias for cte in tree.find_all(exp.CTE)}


def _base_table_refs(tree: exp.Expression) -> list[str]:
    """Every base-table reference, rendered `[catalog.]db.name`, excluding CTE references.

    sqlglot represents a CTE reference as a one-part table, so a one-part ref whose name is
    a CTE is dropped; a genuine base table keeps whatever qualification it was written with
    (so an under-qualified real table is still reported, for the shape check).
    """
    ctes = cte_names(tree)
    refs = []
    for table in tree.find_all(exp.Table):
        parts = [part for part in (table.catalog, table.db, table.name) if part]
        is_cte_reference = table.name in ctes and len(parts) == 1
        if not is_cte_reference:
            refs.append(".".join(parts))
    return refs


def referenced_tables(tree: exp.Expression) -> set[str]:
    """The base tables a query reads, each as its `project.dataset.table` ref.

    CTE references are excluded; a self-join that names a table twice yields it once.
    """
    return set(_base_table_refs(tree))


def unqualified_table_refs(tree: exp.Expression) -> set[str]:
    """Base tables written with fewer than three parts (not a full `project.dataset.table`).

    The prompt requires every table be referenced by its full `gcp_id`; a non-empty result
    means at least one table was under-qualified.
    """
    return {ref for ref in _base_table_refs(tree) if ref.count(".") < 2}


def is_read_only(tree: exp.Expression) -> bool:
    """Whether the statement only reads data (`SELECT` / `WITH` / set operation).

    Any write or DDL statement (`INSERT`/`UPDATE`/`DELETE`/`MERGE`/`CREATE`/…)
    returns `False`.
    """
    return isinstance(tree, _READ_ONLY_STATEMENTS)


def has_select_star(tree: exp.Expression) -> bool:
    """Whether any SELECT projects `*` or a qualified `alias.*`.

    `COUNT(*)` and other stars nested inside a function do not count — only a star that is
    itself a projected item.
    """
    for select in tree.find_all(exp.Select):
        for projection in select.expressions:
            projects_star = isinstance(projection, exp.Star) or (
                isinstance(projection, exp.Column)
                and isinstance(projection.this, exp.Star)
            )
            if projects_star:
                return True
    return False


# =========================================================================================
# Column-usage readers — named for the prompt's "filters / groups / displays" vocabulary
# =========================================================================================
def where_columns(tree: exp.Expression) -> set[str]:
    """Column names that appear in a `WHERE` clause (i.e. the columns the query filters on)."""
    columns: set[str] = set()
    for where in tree.find_all(exp.Where):
        columns |= {column.name for column in where.find_all(exp.Column)}
    return columns


def projected_columns(tree: exp.Expression) -> set[str]:
    """Column names that appear in any SELECT list (the columns the query displays/derives).

    Includes columns inside a projected expression (`SUM(valor)` contributes `valor`),
    which is the conservative reading for "is this coded column used untranslated".
    """
    columns: set[str] = set()
    for select in tree.find_all(exp.Select):
        for projection in select.expressions:
            columns |= {column.name for column in projection.find_all(exp.Column)}
    return columns


def min_max_columns(tree: exp.Expression) -> set[str]:
    """Column names used as an argument to `MIN` or `MAX`.

    Paired with :func:`distinct_columns`, this locates the period-probing the prompt forbids
    (`MIN`/`MAX`/`DISTINCT` over a coverage column) when a check applies it to a
    table's temporal column.
    """
    columns: set[str] = set()
    for aggregate in tree.find_all((exp.Min, exp.Max)):
        columns |= {column.name for column in aggregate.find_all(exp.Column)}
    return columns


def distinct_columns(tree: exp.Expression) -> set[str]:
    """Column names read under a `DISTINCT`.

    Covers both forms: `SELECT DISTINCT a, b` (the distinct marker carries no columns, so
    the SELECT's own projections are taken) and `COUNT(DISTINCT a)` (the `Distinct` node
    carries the column).
    """
    columns: set[str] = set()
    for select in tree.find_all(exp.Select):
        if select.args.get("distinct"):
            for projection in select.expressions:
                columns |= {column.name for column in projection.find_all(exp.Column)}
    for distinct in tree.find_all(exp.Distinct):
        columns |= {column.name for column in distinct.find_all(exp.Column)}
    return columns


# =============================================================================
# Text readers — raw SQL strings and result-row values (no parse needed)
# =============================================================================
def strip_comments(sql: str | None) -> str:
    """Remove `-- ...` line comments from a SQL string (`None` -> "")."""
    return _LINE_COMMENT_RE.sub("", sql or "")


def is_gcp_ref(ref: object) -> bool:
    """Whether a string is exactly a resolved `project.dataset.table` reference.

    An unresolved UUID or a bare number returns `False`.
    """
    return bool(_GCP_REF_FULL_RE.fullmatch(str(ref)))


def gcp_refs_in_text(text: str | None) -> set[str]:
    """The `project.dataset.table` refs mentioned in arbitrary text (e.g. a prose answer).

    A text scan, not a parse: used to detect a table id leaked into prose, which is not
    parseable SQL. Column refs (`t1.ano`) and PT-formatted numbers do not match.
    """
    return set(_GCP_REF_WORD_RE.findall((text or "").replace("`", "")))


def year_literals(sql: str | None) -> set[int]:
    """The year literals (19xx/20xx) written in a query's SQL, column-agnostic.

    `ano = 2020`, `ano_campeonato BETWEEN 2012 AND 2024` and a `'2020-01-01'` date
    literal all yield their year(s); comments are stripped first.
    """
    return {int(year) for year in _YEAR_LITERAL_RE.findall(strip_comments(sql))}


def years_in_rows(rows: list[dict] | None) -> set[int]:
    """Years evidenced by a query's result rows.

    For `evolução` queries that select every year (no year in the `WHERE`), the period
    lives in the rows, not the SQL. Reads year values from year/date-named columns, plus any
    unambiguous `YYYY-MM(-DD)` string value in any column.

    Args:
        rows: The result rows (each a column-name -> value mapping).

    Returns:
        The distinct years evidenced by the rows.
    """
    years: set[int] = set()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for column, value in row.items():
            if value is None:
                continue
            text = str(value)
            if date_match := _DATE_VALUE_RE.fullmatch(text):  # date string, any column
                years.add(int(date_match.group(1)))
            elif _YEAR_COLUMN_RE.search(column) and _YEAR_VALUE_RE.fullmatch(text):
                years.add(int(text))
            elif _DATE_COLUMN_RE.search(column) and (
                year_match := _YEAR_VALUE_RE.match(text)
            ):
                years.add(int(year_match.group(0)))
    return years


def years_evidenced(
    sql: str | None, rows: list[dict] | None = None, row_count: int | None = None
) -> set[int]:
    """Years one executed query evidences: SQL literals plus values from its result rows.

    Truncated results are skipped — their min/max would understate the true span, so only
    the SQL literals count when the rows were capped.

    Args:
        sql: The executed SQL text.
        rows: The result rows, if persisted.
        row_count: The true row count; when it exceeds `len(rows)` the rows are truncated.

    Returns:
        The distinct years the query evidences.
    """
    years = year_literals(sql)
    is_untruncated = row_count is None or (rows is not None and row_count <= len(rows))
    if rows and is_untruncated:
        years |= years_in_rows(rows)
    return years


def date_literals(sql: str | None) -> set[str]:
    """The quoted date/month literals in a query's SQL, e.g. `'2026-06'` / `'2026-06-01'`.

    These carry the month/day granularity that bare `ano`/`mes` filters don't; comments are
    stripped first.
    """
    return set(_DATE_LITERAL_RE.findall(strip_comments(sql)))


def period_values_in_rows(rows: list[dict] | None) -> set[str]:
    """Period value strings evidenced by result rows, at their native granularity.

    A `YYYY-MM(-DD)` value in any column (unambiguous), or a bare `YYYY` in a year/date-named
    column. Unlike :func:`years_in_rows` this keeps the full string (`"2026-05"`), so a scorer
    can compare at month/day granularity when the query actually surfaced it.

    Args:
        rows: The result rows (each a column-name -> value mapping).

    Returns:
        The distinct period value strings the rows evidence.
    """
    values: set[str] = set()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for column, value in row.items():
            if value is None:
                continue
            text = str(value)
            if _DATE_VALUE_RE.fullmatch(text):  # YYYY-MM or YYYY-MM-DD, any column
                values.add(text)
            elif (
                _YEAR_COLUMN_RE.search(column) or _DATE_COLUMN_RE.search(column)
            ) and _YEAR_ONLY_RE.fullmatch(text):
                values.add(text)  # a bare year in a temporal-named column
    return values


def periods_evidenced(
    sql: str | None, rows: list[dict] | None = None, row_count: int | None = None
) -> set[str]:
    """Period value strings one query evidences, at native granularity (year/month/day).

    Combines year literals (`"2025"`) and quoted date/month literals from the SQL with the
    period values in its (untruncated) result rows. This is the granularity-aware companion
    to :func:`years_evidenced`: a scorer truncates these to a target granularity and compares.

    Args:
        sql: The executed SQL text.
        rows: The result rows, if persisted.
        row_count: The true row count; when it exceeds `len(rows)` the rows are truncated.

    Returns:
        The distinct period value strings the query evidences.
    """
    periods = {str(year) for year in year_literals(sql)}
    periods |= date_literals(sql)
    is_untruncated = row_count is None or (rows is not None and row_count <= len(rows))
    if rows and is_untruncated:
        periods |= period_values_in_rows(rows)
    return periods
