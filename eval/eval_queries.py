"""Trace-based source/period eval: did the agent query the right tables and period?

Unlike eval_output (which scores source/period from the agent's STRUCTURED fields), this
reads them from the agent's EXECUTED SQL — the `queries` persisted in the transcript — so
it works for BOTH the structured branch and the free-text baseline (--no-structured /
main). Run it on both transcripts to compare "did the agent query the right inputs" on
equal footing, independent of whether the agent emits structured output.

It answers a different question than eval_output's source/period: those check "did the
agent correctly REPORT its sources/period"; this checks "did the agent actually QUERY the
right ones". Because it reads the executed SQL, trace `source` counts directory JOINs the
agent may omit from data_sources, so it tends to score higher than the reported version.

Per gold QUERY turn:
  source: the tables the executed SQL referenced (FROM/JOIN) vs gold `tables`
          (1.0 exact / 0.5 partial / 0.0 none)
  period: the year range the executed SQL/rows evidence vs the gold `temporal` rule
          (range / exact / match_previous / latest — all at YEAR granularity)

`latest` is a year-level check here (the trace gives years, not the table's exact
period_end granularity): the single queried year must equal the queried tables' latest
period_end year. Reuse turns (no SQL this turn) fall back to the thread's earlier queries;
period abstains (None) when the run evidences no year.

No agent, no BQ, no LLM: pure analysis of the transcript + gold. Runs over ALL repeats.

Pipeline: run AFTER eval_output.py — it scores the transcript eval_output.py produced.
Needs no agent, LLM or BQ. Sibling scorers over the same transcript, independent of this
one and of each other:
  eval_quality.py       answer quality vs the gold — LLM judge + live reference_sql
  eval_faithfulness.py  structured output's self-consistency — gold-free; this module
                        reuses its SQL/period helpers

    uv run eval/eval_queries.py --in eval/<transcript>.json
    uv run eval/eval_queries.py --in eval/<transcript>.json --show-failures
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

# Make the repo root importable so `eval.eval_faithfulness` resolves whether run as a
# module (python -m eval.eval_queries) or directly (python eval/eval_queries.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_faithfulness import (  # noqa: E402
    _leading_year,
    _tables_and_years,
    turn_success_queries,
)

EVAL_DIR = Path(__file__).resolve().parent
DIMENSIONS = ("source", "period")


# =============================================================================
# Gold
# =============================================================================
def load_gold(path: str) -> dict[tuple[str, int], dict]:
    """(thread_id, turn_index) -> gold turn dict (carries action / tables / temporal).

    Args:
        path (str): Path to eval_gold.yaml.

    Returns:
        dict[tuple[str, int], dict]: Gold turn keyed by (thread id, turn index).
    """
    gold = yaml.safe_load(open(path))
    return {
        (thread["id"], i): turn
        for thread in gold
        for i, turn in enumerate(thread["turns"])
    }


# =============================================================================
# Per-turn scoring (source / period, from the agent's executed-SQL trace)
# =============================================================================
def _period_rule_applies(gold: dict) -> bool:
    """Whether the gold expects a scored period this turn (a real rule on a query turn).

    Args:
        gold (dict): The gold turn (its `action` and `temporal` rule).

    Returns:
        bool: True when the turn is a query turn whose `temporal` rule should be scored.
    """
    rule = gold.get("temporal")
    return gold["action"] == "query" and rule not in (None, "none", "any", "", {})


def score_source(trace_tables: set[str], gold: dict) -> float | None:
    """Score the tables the SQL actually hit against the gold `tables`.

    Args:
        trace_tables (set[str]): gcp_ids referenced by the turn's executed SQL.
        gold (dict): The gold turn (action / tables).

    Returns:
        float | None: 1.0 exact, 0.5 partial, 0.0 none; None when not a scored query turn.
    """
    if gold["action"] != "query" or not gold.get("tables"):
        return None
    expected, observed = set(gold["tables"]), set(trace_tables)
    if not observed:
        return 0.0  # gold expects a query but the trace hit no tables
    return 1.0 if observed == expected else (0.5 if observed & expected else 0.0)


def score_period(
    trace_range: tuple[int, int] | None,
    gold: dict,
    period_ends: dict[str, str],
    queried_tables: set[str],
    prev_range: tuple[int, int] | None,
) -> float | None:
    """Score the year range the SQL evidenced against the gold `temporal` rule (year level).

    Args:
        trace_range (tuple[int, int] | None): (min_year, max_year) evidenced by the run.
        gold (dict): The gold turn (action / temporal).
        period_ends (dict[str, str]): gcp_id -> table period_end seen so far this thread.
        queried_tables (set[str]): gcp_ids the turn's SQL referenced (for the `latest` target).
        prev_range (tuple[int, int] | None): the previous query turn's year range.

    Returns:
        float | None: 1.0 / 0.0, or None when the rule doesn't apply or can't be checked.
    """
    rule = gold.get("temporal")
    if gold["action"] != "query" or rule in (None, "none", "any", "", {}):
        return None
    if trace_range is None:
        return 0.0  # queried but the run evidenced no year
    start, end = trace_range
    if rule == "match_previous":
        if prev_range is None:
            return None
        prev_start, prev_end = prev_range
        return 1.0 if prev_start <= start and end <= prev_end else 0.0
    if rule == "range":
        return 1.0 if end > start else 0.0
    if rule == "latest":
        # Year-level: the single queried year must equal the queried tables' latest
        # period_end year (fall back to all seen period_ends if none match by gcp_id).
        candidates = [
            period_ends[t] for t in queried_tables if t in period_ends
        ] or list(period_ends.values())
        targets = [y for value in candidates if (y := _leading_year(value)) is not None]
        if not targets:
            return None  # can't determine the tables' latest year from the trace
        target = max(targets)
        return 1.0 if start == target == end else 0.0
    span = rule.get("exact", rule) if isinstance(rule, dict) else None
    if isinstance(span, dict) and "start" in span and "end" in span:
        return 1.0 if start == int(span["start"]) and end == int(span["end"]) else 0.0
    return None


# =============================================================================
# Transcript scoring & aggregation
# =============================================================================
def score_transcript(units: list[dict], gold: dict) -> list[dict]:
    """Score every `ok` turn's source/period from its executed SQL.

    SQL-derived checks use the thread's executed queries: this turn's own when it ran any,
    else earlier turns' (a follow-up may answer from data fetched on a previous turn).

    Args:
        units (list[dict]): The transcript's units (thread replays).
        gold (dict): (thread, turn) -> gold turn from load_gold().

    Returns:
        list[dict]: One {thread, repeat, turn_index, source, period} per scored turn.
    """
    rows = []
    for unit in units:
        prior_queries: list[dict] = []
        period_ends: dict[str, str] = {}
        prev_range: tuple[int, int] | None = None
        for turn in unit["turns"]:
            if turn.get("status") != "ok":
                continue
            gold_turn = gold.get((unit["thread"], turn["turn_index"]))
            if gold_turn is None:
                continue
            period_ends.update(turn.get("table_period_ends") or {})
            this_queries = turn_success_queries(turn)
            trace_tables, trace_range = _tables_and_years(this_queries or prior_queries)
            period = score_period(
                trace_range, gold_turn, period_ends, trace_tables, prev_range
            )
            rows.append(
                {
                    "thread": unit["thread"],
                    "repeat": unit["repeat"],
                    "turn_index": turn["turn_index"],
                    "source": score_source(trace_tables, gold_turn),
                    "period": period,
                    # a real period rule that couldn't be scored (no year / no period_end in
                    # the trace) is unverifiable, NOT a pass — surfaced so a 100% can't hide it.
                    "period_unverifiable": _period_rule_applies(gold_turn)
                    and period is None,
                }
            )
            # the period a later `match_previous` turn must stay within
            if gold_turn["action"] == "query" and trace_range is not None:
                prev_range = trace_range
            prior_queries += this_queries
    return rows


def aggregate(rows: list[dict]) -> dict:
    """Pass-rate per dimension over the turns it applied to (None = not applicable).

    Args:
        rows (list[dict]): Per-turn score records from score_transcript().

    Returns:
        dict: {dimension: {"rate": float, "n": int}}.
    """
    tally = defaultdict(lambda: {"hit": 0.0, "n": 0})
    for row in rows:
        for dim in DIMENSIONS:
            value = row[dim]
            if value is None:
                continue
            tally[dim]["hit"] += float(value)
            tally[dim]["n"] += 1
    return {
        dim: {"rate": round(counts["hit"] / counts["n"], 3), "n": counts["n"]}
        for dim, counts in tally.items()
        if counts["n"]
    }


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """Score a transcript's trace-based source/period against the gold and write a report."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--in", dest="transcript", required=True, help="thread_eval_*.json"
    )
    parser.add_argument("--gold", default=str(EVAL_DIR / "eval_gold.yaml"))
    parser.add_argument(
        "--thread", action="append", help="Only this thread id (repeatable)"
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--show-failures",
        action="store_true",
        help="List each turn where source or period is below 1.0",
    )
    args = parser.parse_args()

    transcript = json.load(open(args.transcript))
    units = transcript["units"]
    if args.thread:
        units = [unit for unit in units if unit["thread"] in args.thread]
    gold = load_gold(args.gold)

    rows = score_transcript(units, gold)
    summary = aggregate(rows)

    print(
        f"transcript={args.transcript!r}  branch={transcript.get('branch')!r}  "
        f"structured_output={transcript.get('structured_output')}"
    )
    period_unverifiable = sum(1 for row in rows if row["period_unverifiable"])
    notes = {}
    if period_unverifiable:
        notes["period"] = (
            f"  [+{period_unverifiable} unverifiable: no period_end/year in the trace]"
        )

    print(
        "\n=== Trace-based inputs scorecard (source/period from executed SQL vs gold) ==="
    )
    for dim in DIMENSIONS:
        if dim in summary:
            print(
                f"  {dim:<8} {summary[dim]['rate']:.0%}  (n={summary[dim]['n']}){notes.get(dim, '')}"
            )

    if args.show_failures:
        flagged = [
            row
            for row in rows
            if (row["source"] is not None and row["source"] < 1.0)
            or (row["period"] is not None and row["period"] < 1.0)
            or row["period_unverifiable"]
        ]
        if flagged:
            print("\n=== Turns below 1.0 or unverifiable ===")
            for row in flagged:
                tag = "  (period unverifiable)" if row["period_unverifiable"] else ""
                print(
                    f"  {row['thread']:<20} #{row['repeat']} t{row['turn_index']}: "
                    f"source={row['source']}  period={row['period']}{tag}"
                )

    out_path = args.out or str(EVAL_DIR / f"{Path(args.transcript).stem}_inputs.json")
    with open(out_path, "w") as file:
        json.dump(
            {
                "transcript": args.transcript,
                "summary": summary,
                "period_unverifiable": period_unverifiable,
                "rows": rows,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
