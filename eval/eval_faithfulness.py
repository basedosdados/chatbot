"""Faithfulness eval: does the structured output honestly reflect the agent's run?

Gold-free and judge-free. Reads a transcript produced by eval_output.py and, for each
answered turn, checks that the STRUCTURED fields are self-consistent and match what the
agent actually did (the executed SQL persisted in each turn's `queries`). It answers a
different question than the correctness evals: not "did the agent do the right thing?"
but "do data_sources / temporal_coverage truthfully describe THIS run?"

Contract assumed (see app/agent/schemas.py): a no-query turn (clarification / platform
explanation) has no temporal_coverage, but MAY list the tables it explored in
data_sources; a query turn fills both from what it actually queried.

Checks (each -> a pass-rate over the turns it applies to; `·`/None = not applicable):

  cross-field / contract
    clarify_clean            no query this turn -> temporal_coverage absent (data_sources
                             may still list the tables the agent explored)
    query_has_sources        queried -> data_sources is non-empty

  data_sources faithfulness
    sources_resolve          every reported table_id resolved to a gcp_id, not invented
                             (query OR clarify turns)
    sources_match_sql        reported tables are a subset of the tables the SQL actually hit

  temporal_coverage validity / faithfulness
    temporal_gran_valid      granularity matches the period_start/end format
    temporal_order_valid     period_start <= period_end
    temporal_matches_sql     reported year range == years evidenced by the run: SQL literals
                             + result-row values (covers evolução queries that filter no year;
                             column-agnostic; abstains + reports when neither evidences a year)

  prose
    response_nonempty        the prose answer is non-empty
    prose_no_leak            the prose doesn't embed SQL or a raw table id (each has its field)

  informational (reported, not counted as faithfulness bugs)
    sources_exact_match      reported tables == tables the SQL hit (flags omitted joins)
    followups_3              follow_up_questions is 3 non-empty items

SQL-derived checks use the thread's executed queries: this turn's own when it ran any,
else earlier turns' (a follow-up may answer from data fetched on a previous turn without
re-querying). No agent, no BQ, no LLM: pure analysis of the transcript. Runs over ALL repeats.

Pipeline: run AFTER eval_output.py — it scores the transcript eval_output.py produced.
Needs no gold, LLM or BQ, so it's the cheapest scorer. Sibling scorers over the same
transcript, independent of this one and of each other:
  eval_quality.py   answer quality vs the gold — LLM judge + live reference_sql
  eval_queries.py   source/period from the executed SQL vs the gold — reuses this
                    module's SQL/period helpers (_leading_year, _tables_and_years, ...)

    uv run eval/eval_faithfulness.py --in eval/<transcript>.json
    uv run eval/eval_faithfulness.py --in eval/<transcript>.json --show-failures
"""

import argparse
import json
import re
from collections import defaultdict, namedtuple
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent

# Checks in display order; `info` ones are reported but excluded from the headline
# (they flag stylistic/definitional choices, not faithfulness bugs).
CORE_CHECKS = (
    "clarify_clean",
    "query_has_sources",
    "sources_resolve",
    "sources_match_sql",
    "temporal_gran_valid",
    "temporal_order_valid",
    "temporal_matches_sql",
    "response_nonempty",
    "prose_no_leak",
)
INFO_CHECKS = ("sources_exact_match", "followups_3")
ALL_CHECKS = CORE_CHECKS + INFO_CHECKS

# A period value split into its granularity and its original text (e.g. "2026-05" -> month).
Period = namedtuple("Period", ["granularity", "value"])


# =============================================================================
# SQL / period parsing (best-effort, self-contained — no agent deps)
# =============================================================================
def _strip_comments(sql: str) -> str:
    """Remove `-- ...` line comments from a SQL string.

    Args:
        sql (str): The SQL text (may be None).

    Returns:
        str: The SQL with every `--` comment stripped to end-of-line.
    """
    return re.sub(r"--[^\n]*", "", sql or "")


# A BigQuery id segment is an identifier (letter/underscore-led), never all-digit — so a
# `project.dataset.table` ref matches but a PT-formatted number like 213.421.037 does not.
_IDENT = r"[A-Za-z_][A-Za-z0-9_-]*"
_GCP_PAT = rf"{_IDENT}\.{_IDENT}\.{_IDENT}"


def _is_gcp_ref(ref: str) -> bool:
    """Whether a string is a resolved BigQuery table ref `project.dataset.table`.

    Args:
        ref (str): The candidate string (an unresolved UUID or a number returns False).

    Returns:
        bool: True if `ref` is exactly a 3-part dotted identifier.
    """
    return bool(re.fullmatch(_GCP_PAT, str(ref)))


def _sql_tables(sql: str) -> set[str]:
    """The `project.dataset.table` refs a query touches.

    Matches 3-part dotted identifiers only, so column refs (t1.ano), CTE aliases and
    PT-formatted numbers are excluded; comments are stripped first.

    Args:
        sql (str): The SQL text.

    Returns:
        set[str]: The distinct fully-qualified table references found.
    """
    without_comments = _strip_comments(sql).replace("`", "")
    return set(re.findall(rf"\b{_GCP_PAT}\b", without_comments))


# A year shows up in a query's SQL as a literal, or in its RESULT rows as a value in a
# year/date-named column (or an unambiguous date string). Column matching is liberal — a
# false hit still has to BE a 19xx/20xx value to count.
_YEAR_COLUMN_RE = re.compile(r"ano|year|exercicio|periodo|competencia", re.I)
_DATE_COLUMN_RE = re.compile(r"data|date", re.I)
_YEAR_VALUE_RE = re.compile(r"(?:19|20)\d{2}")
_DATE_VALUE_RE = re.compile(r"(\d{4})-\d{2}(?:-\d{2})?")


def _years_in_sql(sql: str) -> set[int]:
    """Year literals in a query's SQL, read column-agnostically.

    `ano`, `ano_campeonato`, `mes` + year, or a `'2020-01-01'` date literal all yield their
    year. In this domain a standalone 19xx/20xx is a year (municipio codes are 7-digit, NCM
    8-digit, LIMIT/GROUP small).

    Args:
        sql (str): The SQL text.

    Returns:
        set[int]: The distinct years mentioned as literals.
    """
    return {
        int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", _strip_comments(sql))
    }


def _years_in_rows(rows: list[dict]) -> set[int]:
    """Years evidenced by a query's RESULT rows.

    For evolução queries that SELECT all years (no year in the WHERE), the period lives
    here, not in the SQL. Reads year values from year/date-named columns, plus any
    unambiguous 'YYYY-MM(-DD)' string value in any column.

    Args:
        rows (list[dict]): The query's result rows (column name -> value).

    Returns:
        set[int]: The distinct years evidenced by the rows.
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


def _years_evidenced_by_query(query: dict) -> set[int]:
    """Years one executed query evidences: SQL literals + values from its result rows.

    Truncated results are skipped — their min/max would understate the true span.

    Args:
        query (dict): An executed query record with `sql`, `rows` and `row_count`.

    Returns:
        set[int]: The distinct years evidenced by the query's SQL and (untruncated) rows.
    """
    years = _years_in_sql(query.get("sql") or "")
    rows, row_count = query.get("rows"), query.get("row_count")
    is_untruncated = row_count is None or row_count <= len(rows)
    if rows and is_untruncated:
        years |= _years_in_rows(rows)
    return years


def _leading_year(value) -> int | None:
    """The leading 4-digit year of a period value ('2015', '2015-03-01', '2026-05').

    Args:
        value: The period value (str or None).

    Returns:
        int | None: The leading year, or None when the value has no leading 4 digits.
    """
    match = re.match(r"\s*(\d{4})", str(value or ""))
    return int(match.group(1)) if match else None


def _parse_period(period) -> Period | None:
    """Split a period value into its granularity and text, in its own format:

    'YYYY' -> year
    'YYYY-MM' -> month
    'YYYY-MM-DD' -> day

    Strict: a malformed value returns None.

    Args:
        period: The period value (str or None).

    Returns:
        Period | None: (granularity, value), or None when the value is malformed.
    """
    text = str(period or "").strip().strip("'").strip('"')
    for granularity, pattern in (
        ("day", r"\d{4}-\d{2}-\d{2}"),
        ("month", r"\d{4}-\d{2}"),
        ("year", r"\d{4}"),
    ):
        if re.fullmatch(pattern, text):
            return Period(granularity, text)
    return None


# =============================================================================
# Per-turn checks
# =============================================================================
def turn_success_queries(turn: dict) -> list[dict]:
    """This turn's successfully-executed queries.

    Args:
        turn (dict): A per-turn record from the transcript.

    Returns:
        list[dict]: The query records (each with `sql` + result `rows`) whose status is
            "success".
    """
    return [
        query
        for query in (turn.get("queries") or [])
        if query.get("status") == "success" and query.get("sql")
    ]


def _tables_and_years(queries: list[dict]) -> tuple[set[str], tuple[int, int] | None]:
    """Aggregate the tables hit and the overall year range across a set of executed queries.

    Args:
        queries (list[dict]): Executed query records (each with `sql` + result `rows`).

    Returns:
        tuple[set[str], tuple[int, int] | None]: (tables referenced, (min_year, max_year)),
            where the year range is None when no query evidenced any year.
    """
    tables: set[str] = set()
    years: set[int] = set()
    for query in queries:
        tables |= _sql_tables(query.get("sql") or "")
        years |= _years_evidenced_by_query(query)
    year_range = (min(years), max(years)) if years else None
    return tables, year_range


def check_turn(turn: dict, executed_queries: list[dict]) -> dict:
    """Run every faithfulness check for one `ok` turn.

    Each check is True (passed), False (failed), or None (not applicable to this turn).

    Args:
        turn (dict): The per-turn record (its `structured` fields, `is_query`, and the
            `tables` resolved from data_sources).
        executed_queries (list[dict]): The executed queries (sql + rows) to check
            faithfulness against — this turn's own when it ran any, else the thread's earlier
            ones (a follow-up may answer from data fetched on a previous turn without
            re-querying).

    Returns:
        dict: Maps each check name in ALL_CHECKS to True / False / None.
    """
    structured = turn.get("structured") or {}
    is_query = bool(turn.get("is_query"))
    data_sources = structured.get("data_sources") or []
    temporal_coverage = structured.get("temporal_coverage")
    follow_ups = structured.get("follow_up_questions")
    response = structured.get("response") or ""
    reported_tables = list(
        turn.get("tables") or []
    )  # data_sources resolved to gcp/uuid
    queried_tables, queried_year_range = _tables_and_years(executed_queries)

    results: dict = {check: None for check in ALL_CHECKS}

    # cross-field / contract
    if not is_query:
        # a no-query turn may surface explored tables in data_sources, but with no SQL there
        # is no filtered interval, so temporal_coverage must stay empty.
        results["clarify_clean"] = temporal_coverage is None
    else:
        results["query_has_sources"] = bool(data_sources)
    results["followups_3"] = (
        isinstance(follow_ups, list)
        and len(follow_ups) == 3
        and all((question or "").strip() for question in follow_ups)
    )

    # data_sources faithfulness
    if data_sources:
        # every reported table must have been fetched this run (resolves to a gcp ref) —
        # applies to clarify turns too, which may list the tables they explored.
        results["sources_resolve"] = all(
            _is_gcp_ref(table) for table in reported_tables
        )
        # matching against the executed SQL only makes sense on a query turn.
        if is_query and queried_tables:
            reported_gcp = {table for table in reported_tables if _is_gcp_ref(table)}
            results["sources_match_sql"] = reported_gcp <= queried_tables
            results["sources_exact_match"] = reported_gcp == queried_tables

    # temporal_coverage validity / faithfulness
    if temporal_coverage:
        start = _parse_period(temporal_coverage.get("period_start"))
        end = _parse_period(temporal_coverage.get("period_end"))
        granularity = temporal_coverage.get("granularity")
        results["temporal_gran_valid"] = (
            start is not None
            and end is not None
            and start.granularity == granularity
            and end.granularity == granularity
        )
        if (
            start is not None
            and end is not None
            and start.granularity == end.granularity
        ):
            results["temporal_order_valid"] = start.value <= end.value
        else:
            start_year = _leading_year(temporal_coverage.get("period_start"))
            end_year = _leading_year(temporal_coverage.get("period_end"))
            results["temporal_order_valid"] = (
                start_year is not None
                and end_year is not None
                and start_year <= end_year
            )
        if is_query and queried_year_range is not None:
            start_year = _leading_year(temporal_coverage.get("period_start"))
            end_year = _leading_year(temporal_coverage.get("period_end"))
            results["temporal_matches_sql"] = (
                start_year,
                end_year,
            ) == queried_year_range

    # prose
    results["response_nonempty"] = bool(response.strip())
    results["prose_no_leak"] = not (
        "```sql" in response.lower()
        or any(_is_gcp_ref(table) for table in _sql_tables(response))
    )
    return results


# =============================================================================
# Aggregation & reporting
# =============================================================================
def aggregate(turn_results: list[dict]) -> dict:
    """Compute the pass-rate per check across all scored turns.

    A None value (check not applicable to that turn) is skipped, so each check's `n` is the
    number of turns it actually applied to.

    Args:
        turn_results (list[dict]): One check_turn() result dict per scored turn.

    Returns:
        dict: Maps each check name to {"rate": float, "n": int}.
    """
    tally = defaultdict(lambda: {"passed": 0, "applicable": 0})
    for result in turn_results:
        for check, outcome in result.items():
            if outcome is None:
                continue
            tally[check]["passed"] += int(bool(outcome))
            tally[check]["applicable"] += 1
    return {
        check: {
            "rate": round(counts["passed"] / counts["applicable"], 3),
            "n": counts["applicable"],
        }
        for check, counts in tally.items()
    }


def print_scorecard(
    summary: dict, processed: int, skipped: int, notes: dict | None = None
) -> None:
    """Print the faithfulness scorecard, core checks then informational ones.

    Args:
        summary (dict): The per-check {"rate", "n"} mapping from aggregate().
        processed (int): Number of `ok` turns scored.
        skipped (int): Number of non-`ok` turns skipped.
        notes (dict | None): Optional per-check suffix strings (e.g. an abstention note).

    Returns:
        None
    """
    notes = notes or {}
    print(
        f"\n=== Faithfulness scorecard ({processed} turns; {skipped} non-ok skipped) ==="
    )
    print("  -- faithfulness / contract --")
    for check in CORE_CHECKS:
        if check in summary:
            print(
                f"    {check:<22} {summary[check]['rate']:.0%}  (n={summary[check]['n']}){notes.get(check, '')}"
            )
    print("  -- informational (not faithfulness bugs) --")
    for check in INFO_CHECKS:
        if check in summary:
            print(
                f"    {check:<22} {summary[check]['rate']:.0%}  (n={summary[check]['n']}){notes.get(check, '')}"
            )


def main() -> None:
    """Score a transcript's structured-output faithfulness and write a JSON report."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--in", dest="transcript", required=True, help="thread_eval_*.json"
    )
    parser.add_argument(
        "--thread", action="append", help="Only this thread id (repeatable)"
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="List each failing (thread, repeat, turn) and which checks failed",
    )
    args = parser.parse_args()

    transcript = json.load(open(args.transcript))
    units = transcript["units"]
    if args.thread:
        units = [unit for unit in units if unit["thread"] in args.thread]

    turn_results, failures, skipped, temporal_abstained = [], [], 0, 0
    for unit in units:
        prior_queries: list[
            dict
        ] = []  # successful queries from earlier turns of THIS thread
        for turn in unit["turns"]:
            if turn.get("status") != "ok":
                skipped += 1
                continue
            this_queries = turn_success_queries(turn)
            result = check_turn(turn, this_queries or prior_queries)
            turn_results.append(result)
            # a query turn that reported a period but whose run evidenced no year (no literal
            # in SQL, none in the rows): unverifiable, NOT a pass — surface it so the rate
            # isn't misread.
            structured = turn.get("structured") or {}
            if (
                turn.get("is_query")
                and structured.get("temporal_coverage")
                and result["temporal_matches_sql"] is None
            ):
                temporal_abstained += 1
            prior_queries += this_queries
            failed_checks = [
                check
                for check, outcome in result.items()
                if outcome is False and check in CORE_CHECKS
            ]
            if failed_checks:
                failures.append(
                    (unit["thread"], unit["repeat"], turn["turn_index"], failed_checks)
                )

    summary = aggregate(turn_results)
    notes = {}
    if temporal_abstained:
        notes["temporal_matches_sql"] = (
            f"  [+{temporal_abstained} unverifiable: no year literal in SQL]"
        )
    print(
        f"transcript={args.transcript!r}  branch={transcript.get('branch')!r}  model={transcript.get('model')!r}"
    )
    print_scorecard(summary, len(turn_results), skipped, notes)

    if args.show_failures and failures:
        print("\n=== Failing turns (core checks) ===")
        for thread, repeat, turn_index, failed_checks in failures:
            print(f"  {thread:<20} #{repeat} t{turn_index}: {', '.join(failed_checks)}")

    out_path = args.out or str(
        EVAL_DIR / f"{Path(args.transcript).stem}_faithfulness.json"
    )
    with open(out_path, "w") as file:
        json.dump(
            {
                "transcript": args.transcript,
                "summary": summary,
                "turn_results": turn_results,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
