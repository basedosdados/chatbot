"""Deterministic scorer: grade a transcript against the gold, no LLM and no BigQuery.

One pass over a runner transcript, folding what used to be three separate scripts
(structured-field scoring, executed-SQL source/period, and structured-output faithfulness)
into a single set of per-turn checks. Every check is derived from the run's OWN trace and
its OWN retrieved metadata (`get_table_details` outputs recorded per unit), so assertions
track each run rather than hardcoded constants.

Each check is True (pass), False (fail), a float in [0, 1] (partial, for `source`), or None
(not applicable to this turn). A check's headline rate is its pass-fraction over the turns
it applied to. Groups:

  run/routing   completed, action
  sources       source_reported (from data_sources), source_executed (from the SQL)
  period        period (at the target's granularity, from the SQL/rows vs the gold rule)
  sql rules     select_shape_ok, partition_filtered, no_period_probing, coded_cols_translated
  faithfulness  query_has_sources, sources_resolve, sources_match_sql, response_nonempty,
                prose_no_leak, followups_3
  informational sources_exact_match (reported == queried; flags omitted joins, not a bug)

`action` is deterministic only as query-vs-not: `query` means "ran execute_bigquery_sql",
`ask` means "ran none" — whether the agent explored the catalog first, asked directly,
described available data, or reported it can't help is response-intent the judge scores, not
this pass. `period` is checked at the target's granularity (year for `last_years`, the
table's own `period_end` granularity for `latest`, the value's granularity for a concrete
point/range); a month/day target the query pinned only to the year abstains (None) and
defers to the judge.

    uv run eval/score_deterministic.py --in eval/transcript_*.json
    uv run eval/score_deterministic.py --in eval/transcript_*.json --show-failures
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.lib import gold, metadata, period, sql  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent

# A coverage/period column, matched by name. Restricted at the call site to a table's
# PARTITION columns (its coverage dimension), so an ordinary date column like
# `data_nascimento` is not mistaken for the period.
TEMPORAL_COLUMN_RE = re.compile(
    r"\b(ano|mes|dia|data|trimestre|semestre|exercicio|periodo|competencia|year)", re.I
)

# Checks grouped for the scorecard; every key here is scored per turn.
CHECK_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("run / routing", ("completed", "action")),
    ("sources", ("source_reported", "source_executed")),
    ("period", ("period",)),
    (
        "sql rules",
        (
            "select_shape_ok",
            "partition_filtered",
            "no_period_probing",
            "coded_cols_translated",
        ),
    ),
    (
        "faithfulness",
        (
            "query_has_sources",
            "sources_resolve",
            "sources_match_sql",
            "response_nonempty",
            "prose_no_leak",
            "followups_3",
        ),
    ),
)
CORE_CHECKS = tuple(check for _, checks in CHECK_GROUPS for check in checks)
INFO_CHECKS = ("sources_exact_match",)
ALL_CHECKS = CORE_CHECKS + INFO_CHECKS


# =============================================================================
# Per-turn SQL analysis (aggregated across a turn's effective queries)
# =============================================================================
@dataclass(frozen=True)
class QueryAnalysis:
    """What a turn's executed queries evidence, aggregated across all of them."""

    tables: set[str]  # base tables referenced (gcp refs)
    periods: set[
        str
    ]  # period values evidenced, native granularity ("2025", "2026-05", ...)
    shape_ok: bool | None  # every query read-only, no SELECT *, fully qualified
    where_columns: set[str]  # columns filtered (WHERE)
    projected_columns: set[str]  # columns displayed/derived (SELECT lists)
    period_probes: set[str]  # columns under MIN/MAX/DISTINCT


def analyze_queries(queries: list[dict]) -> QueryAnalysis:
    """Aggregate the structural facts of a turn's successful queries.

    Args:
        queries: The turn's effective successful queries (each ``{sql, rows, row_count}``).

    Returns:
        The aggregated :class:`QueryAnalysis`. Period detection uses the text/rows readers
        (so it survives an unparseable query); the structural facts come from the parsed AST
        and simply omit any query that failed to parse.
    """
    tables: set[str] = set()
    periods: set[str] = set()
    where: set[str] = set()
    projected: set[str] = set()
    probes: set[str] = set()
    shape_flags: list[bool] = []
    for query in queries:
        periods |= sql.periods_evidenced(
            query["sql"], query["rows"], query["row_count"]
        )
        tree = sql.parse(query["sql"])
        if tree is None:
            continue
        tables |= sql.referenced_tables(tree)
        where |= sql.where_columns(tree)
        projected |= sql.projected_columns(tree)
        probes |= sql.min_max_columns(tree) | sql.distinct_columns(tree)
        shape_flags.append(
            sql.is_read_only(tree)
            and not sql.has_select_star(tree)
            and not sql.unqualified_table_refs(tree)
        )
    return QueryAnalysis(
        tables=tables,
        periods=periods,
        shape_ok=all(shape_flags) if shape_flags else None,
        where_columns=where,
        projected_columns=projected,
        period_probes=probes,
    )


# =============================================================================
# Individual checks
# =============================================================================
def score_action(gold_turn: dict, is_query: bool) -> bool:
    """Whether the turn ran a query exactly when the gold expected one.

    Query-vs-not is the only routing signal the trace proves: `query` must call
    execute_bigquery_sql, `ask` must not. Whether an `ask` turn explored the catalog first,
    asked directly, described available data, or reported it can't help is the judge's.
    """
    return is_query == (gold_turn["action"] == gold.ACTION_QUERY)


def _source_overlap(observed: set[str], expected: set[str]) -> float:
    """1.0 exact / 0.5 partial / 0.0 none, comparing observed tables to the gold's."""
    if not observed:
        return 0.0
    return 1.0 if observed == expected else (0.5 if observed & expected else 0.0)


def score_source(observed_tables: set[str], gold_turn: dict) -> float | None:
    """Score tables (reported or executed) against the gold `sources`; None off query turns."""
    if gold_turn["action"] != gold.ACTION_QUERY or not gold_turn.get("sources"):
        return None
    return _source_overlap(observed_tables, set(gold_turn["sources"]))


def _matches(periods: set[str], start: str, end: str, granularity: str) -> float | None:
    """Compare the evidenced periods to a [start, end] target at a given granularity.

    0.0 if the query evidenced no period at all; None (abstain) if it evidenced one but not
    at this granularity (e.g. a month target when only the year was pinned — deferred to the
    judge); else 1.0/0.0 on whether the evidenced span equals the target span.
    """
    if not periods:
        return 0.0  # a scored period turn that pinned no period at all
    at_grain = sorted(
        truncated
        for value in periods
        if (truncated := period.truncate(value, granularity)) is not None
    )
    if not at_grain:
        return None  # evidenced a period, but not finely enough to check this target
    return 1.0 if (at_grain[0], at_grain[-1]) == (start, end) else 0.0


def _target_span(rule: object) -> tuple[str, str, str] | None:
    """Read a concrete `period` value into (start, end, granularity), or None if it isn't one.

    A point (`2025` / `"2026-05"` / `"2026-05-01"`) -> (v, v, grain); a `{start, end}` range
    -> both truncated to the coarser of the two granularities. Named rules (latest /
    latest / last_years) are handled by the caller, not here.
    """
    if isinstance(rule, dict) and {"start", "end"} <= rule.keys():
        start, end = (
            period.parse_period(rule["start"]),
            period.parse_period(rule["end"]),
        )
        if start is None or end is None:
            return None
        grain = period.coarser(start.granularity, end.granularity)
        return (
            period.truncate(start.value, grain),
            period.truncate(end.value, grain),
            grain,
        )
    if isinstance(rule, (int, str)):
        point = period.parse_period(rule)
        return (point.value, point.value, point.granularity) if point else None
    return None


def _queried_period_bounds(
    table_meta: dict[str, metadata.TableMetadata], queried_tables: set[str], attr: str
) -> list[str]:
    """The `period_start`/`period_end` values of the queried tables (all tables as fallback)."""
    scoped = [
        value
        for gcp in queried_tables
        if (meta := table_meta.get(gcp)) and (value := getattr(meta, attr))
    ]
    return scoped or [
        value for meta in table_meta.values() if (value := getattr(meta, attr))
    ]


def score_period(
    gold_turn: dict,
    periods: set[str],
    table_meta: dict[str, metadata.TableMetadata],
    queried_tables: set[str],
) -> float | None:
    """Score the period the SQL evidenced against the gold `period` rule.

    Compares at the target's granularity (year for `last_years`, the `period_end`'s own
    granularity for `latest`, the value's granularity for a concrete point/range). None when
    no rule applies or the trace can't be checked at that grain.
    """
    if not gold.scores_period(gold_turn):
        return None
    rule = gold_turn["period"]

    if rule == gold.PERIOD_LATEST:
        # Run-relative: the queried period must equal the queried tables' latest period_end,
        # at that period_end's OWN granularity (read from the run's metadata, never hardcoded).
        ends = _queried_period_bounds(table_meta, queried_tables, "period_end")
        latest = max(
            (
                period.parse_period(value)
                for value in ends
                if period.parse_period(value)
            ),
            key=lambda parsed: parsed.value,
            default=None,
        )
        if latest is None:
            return None  # can't determine the tables' latest period from the trace
        return _matches(periods, latest.value, latest.value, latest.granularity)

    if isinstance(rule, dict) and "last_years" in rule:
        window = _last_years_window(rule["last_years"], table_meta, queried_tables)
        if window is None:
            return None
        start_year, end_year = window
        return _matches(periods, str(start_year), str(end_year), "year")

    span = _target_span(rule)
    if span is not None:
        start, end, granularity = span
        return _matches(periods, start, end, granularity)
    return None  # unrecognized period value


def _last_years_window(
    n: int, table_meta: dict[str, metadata.TableMetadata], queried_tables: set[str]
) -> tuple[int, int] | None:
    """The run-relative N most-recent-COMPLETE-year window, clamped to coverage start.

    This is the prompt's **flow** anchor: it ends at the most recent *complete* year (not
    `period_end` itself — a sub-annual `period_end` like `2026-06` completes only 2025), so
    `{last_years: 1}` means "the latest complete year". The start is floored at the earliest
    `period_start` year so a short series isn't asked for years it can't have.
    """
    end_years = [
        year
        for value in _queried_period_bounds(table_meta, queried_tables, "period_end")
        if (year := period.complete_year(value)) is not None
    ]
    if not end_years:
        return None
    end = max(end_years)
    start_years = [
        year
        for value in _queried_period_bounds(table_meta, queried_tables, "period_start")
        if (year := period.leading_year(value)) is not None
    ]
    start = end - (n - 1)
    if start_years:
        start = max(start, min(start_years))
    return start, end


def score_select_shape(analysis: QueryAnalysis) -> bool | None:
    """Every executed query is read-only, names no ``SELECT *``, and fully qualifies its tables."""
    return analysis.shape_ok


def score_partition_filtered(
    analysis: QueryAnalysis, table_meta: dict[str, metadata.TableMetadata]
) -> bool | None:
    """Every partitioned table the query hit is filtered on one of its partition columns.

    None when no queried table is partitioned (nothing to require). Column matching is
    name-based, so two partitioned tables sharing a partition name can't be told apart — a
    documented limitation that errs toward crediting a filter.
    """
    saw_partitioned = False
    for gcp in analysis.tables:
        meta = table_meta.get(gcp)
        if meta is None or not meta.partitioned_by:
            continue
        saw_partitioned = True
        if not (set(meta.partitioned_by) & analysis.where_columns):
            return False
    return True if saw_partitioned else None


def score_no_period_probing(
    analysis: QueryAnalysis, table_meta: dict[str, metadata.TableMetadata]
) -> bool | None:
    """No ``MIN``/``MAX``/``DISTINCT`` over a queried table's temporal partition column.

    None when no queried table has a temporal partition column (nothing to probe).
    """
    period_columns: set[str] = set()
    for gcp in analysis.tables:
        meta = table_meta.get(gcp)
        if meta is None:
            continue
        period_columns |= {
            column
            for column in meta.partitioned_by
            if TEMPORAL_COLUMN_RE.search(column)
        }
    if not period_columns:
        return None
    return period_columns.isdisjoint(analysis.period_probes)


def _joins_a_directory(tables: set[str]) -> bool:
    """Whether any referenced table is a Base dos Dados directory (a `br_bd_diretorios_*` dataset)."""
    return any(".br_bd_diretorios" in gcp for gcp in tables)


def score_coded_cols_translated(
    analysis: QueryAnalysis,
    table_meta: dict[str, metadata.TableMetadata],
    tools_used: list[str],
) -> bool | None:
    """A coded column used raw (filtered or displayed) has some translation in the run.

    Translation evidence is turn-level: a `decode_table_values` call, or a directory JOIN.
    None when the query references no coded column. This is the fuzziest check — it reads the
    prompt strictly (translate via the dictionary or a JOIN), so an inline ``CASE`` mapping a
    code counts as *untranslated*.
    """
    used_columns = analysis.where_columns | analysis.projected_columns
    references_coded = any(
        column.name in used_columns
        for gcp in analysis.tables
        if (meta := table_meta.get(gcp)) is not None
        for column in meta.coded_columns
    )
    if not references_coded:
        return None
    translated = "decode_table_values" in tools_used or _joins_a_directory(
        analysis.tables
    )
    return translated


# =============================================================================
# Per-turn scoring
# =============================================================================
def score_turn(
    turn: dict,
    gold_turn: dict,
    analysis: QueryAnalysis,
    table_meta: dict[str, metadata.TableMetadata],
) -> dict:
    """All deterministic checks for one turn. Values are True/False/float/None (n/a)."""
    checks: dict = {check: None for check in ALL_CHECKS}
    status = turn["status"]
    if status == "skipped":
        return checks  # never ran

    checks["completed"] = status == "ok"
    if status == "error":
        return checks  # crashed; nothing else to read

    # ok or no_structured: the trace exists, so trace-based checks apply.
    is_query = turn["is_query"]
    tools_used = turn["tools_used"]
    checks["action"] = score_action(gold_turn, is_query)
    checks["source_executed"] = score_source(analysis.tables, gold_turn)
    checks["period"] = score_period(
        gold_turn, analysis.periods, table_meta, analysis.tables
    )
    if is_query:
        checks["select_shape_ok"] = score_select_shape(analysis)
        checks["partition_filtered"] = score_partition_filtered(analysis, table_meta)
        checks["no_period_probing"] = score_no_period_probing(analysis, table_meta)
        checks["coded_cols_translated"] = score_coded_cols_translated(
            analysis, table_meta, tools_used
        )
        checks["query_has_sources"] = (
            bool(turn["structured"]["data_sources"]) if turn["structured"] else False
        )

    # Structured-field checks only when the agent produced a structured response.
    structured = turn["structured"]
    if structured is not None:
        reported = set(turn["tables"])
        checks["source_reported"] = score_source(reported, gold_turn)
        if reported:
            checks["sources_resolve"] = all(sql.is_gcp_ref(table) for table in reported)
            reported_gcp = {table for table in reported if sql.is_gcp_ref(table)}
            if is_query and analysis.tables:
                checks["sources_match_sql"] = reported_gcp <= analysis.tables
                checks["sources_exact_match"] = reported_gcp == analysis.tables
        response = structured["response"] or ""
        checks["response_nonempty"] = bool(response.strip())
        checks["prose_no_leak"] = not (
            "```sql" in response.lower() or bool(sql.gcp_refs_in_text(response))
        )
        follow_ups = structured["follow_up_prompts"]
        checks["followups_3"] = (
            isinstance(follow_ups, list)
            and len(follow_ups) == 3
            and all((prompt or "").strip() for prompt in follow_ups)
        )
    return checks


def _successful_queries(turn: dict) -> list[dict]:
    """The turn's successfully-executed queries (status success, SQL present)."""
    return [
        query
        for query in turn["queries"]
        if query["status"] == "success" and query["sql"]
    ]


def score_unit(unit: dict, gold_index: dict) -> list[dict]:
    """Score every gold-matched turn of one replay unit, threading query reuse.

    A turn with no query of its own reuses the thread's earlier queries (a follow-up may
    answer from data already fetched), matching how the agent is allowed to work.
    """
    table_meta = metadata.index_by_gcp_id(
        metadata.parse_table_details(payload) for payload in unit["tables"].values()
    )
    rows: list[dict] = []
    prior_queries: list[dict] = []

    for turn in unit["turns"]:
        gold_turn = gold_index.get((unit["thread"], turn["turn_index"]))
        if gold_turn is None:
            continue
        this_queries = (
            _successful_queries(turn)
            if turn["status"] != "skipped" and "queries" in turn
            else []
        )
        analysis = analyze_queries(this_queries or prior_queries)
        checks = score_turn(turn, gold_turn, analysis, table_meta)
        rows.append(
            {
                "thread": unit["thread"],
                "repeat": unit["repeat"],
                "turn_index": turn["turn_index"],
                "gold_action": gold_turn["action"],
                **checks,
            }
        )
        prior_queries = prior_queries + this_queries
    return rows


# =============================================================================
# Aggregation & reporting
# =============================================================================
def aggregate(rows: list[dict]) -> dict:
    """Pass-rate and applicable-count per check over all scored turns (None values skipped)."""
    tally: dict = defaultdict(lambda: {"hit": 0.0, "n": 0})
    for row in rows:
        for check in ALL_CHECKS:
            value = row[check]
            if value is None:
                continue
            tally[check]["hit"] += float(value)
            tally[check]["n"] += 1
    return {
        check: {"rate": round(counts["hit"] / counts["n"], 3), "n": counts["n"]}
        for check, counts in tally.items()
        if counts["n"]
    }


def print_scorecard(summary: dict, transcript: dict, rows: list[dict]) -> None:
    """Print the grouped scorecard plus run status counts."""
    statuses = defaultdict(int)
    for row in rows:
        # `completed is False` means error/no_structured; None means skipped.
        statuses["scored_turns"] += 1
    print(
        f"\n=== Deterministic scorecard "
        f"(branch={transcript['branch']!r} effort={transcript['effort']!r} "
        f"repeats={transcript['repeats']}) ==="
    )
    for group_name, checks in CHECK_GROUPS:
        shown = [check for check in checks if check in summary]
        if not shown:
            continue
        print(f"  -- {group_name} --")
        for check in shown:
            print(
                f"    {check:<22} {summary[check]['rate']:.0%}  (n={summary[check]['n']})"
            )
    info_shown = [check for check in INFO_CHECKS if check in summary]
    if info_shown:
        print("  -- informational (not counted as failures) --")
        for check in info_shown:
            print(
                f"    {check:<22} {summary[check]['rate']:.0%}  (n={summary[check]['n']})"
            )


def print_failures(rows: list[dict]) -> None:
    """List each turn with a failing (False or <1.0) core check, and which checks failed."""
    print("\n=== Turns with a failing core check ===")
    any_failure = False
    for row in rows:
        failed = [
            check
            for check in CORE_CHECKS
            if row[check] is not None and row[check] is not True and row[check] != 1.0
        ]
        if failed:
            any_failure = True
            marks = ", ".join(f"{check}={row[check]}" for check in failed)
            print(
                f"  {row['thread']:<24} #{row['repeat']} t{row['turn_index']} "
                f"({row['gold_action']}): {marks}"
            )
    if not any_failure:
        print("  (none)")


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """Score a transcript against the gold, print the scorecard, and write the scores JSON."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--in", dest="transcript", required=True, help="runner transcript JSON"
    )
    parser.add_argument(
        "--gold",
        default=None,
        help="gold YAML (default: the transcript's own gold_path)",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="List every turn with a failing check",
    )
    args = parser.parse_args()

    with open(args.transcript) as file:
        transcript = json.load(file)
    gold_index = gold.load_indexed(args.gold or transcript["gold_path"])

    rows = [row for unit in transcript["units"] for row in score_unit(unit, gold_index)]
    summary = aggregate(rows)

    print_scorecard(summary, transcript, rows)
    if args.show_failures:
        print_failures(rows)

    out_path = args.out or str(EVAL_DIR / f"{Path(args.transcript).stem}_scores.json")
    with open(out_path, "w") as file:
        json.dump(
            {
                "transcript": args.transcript,
                "branch": transcript["branch"],
                "effort": transcript["effort"],
                "summary": summary,
                "rows": rows,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
