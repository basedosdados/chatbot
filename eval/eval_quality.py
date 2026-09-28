"""LLM judge over the transcripts that eval_output.py saved.

One judge call per answered turn (query OR ask), emitting SEPARATE scores
(not a blended verdict):

  correct           : the answer's figures/entities/ranking are CONSISTENT with the
                      reference_sql result (the gold, re-run live) — a different but
                      valid slice/period/grouping isn't penalized, only contradictions.
                      null when the turn carries no reference_sql (correctness unverifiable)
  grounded          : every claim traces to data the AGENT ACTUALLY RETRIEVED, this turn
                      OR an earlier turn (its query results across the thread, persisted in
                      the transcript) — an extra metric absent from the reference but
                      present in the agent's own results is grounded; a fabricated value is not
  answers_question  : the prose addresses what the user actually asked
  stated_assumption : on an ambiguous turn, the interpretation is made explicit
                      (null when the turn wasn't ambiguous)

The two data axes are DECOUPLED: `grounded` is anchored on the agent's own retrieved
data, `correct` on the gold reference (and only scored when a reference_sql is present);
the rest are rubric. Prose format (no Markdown headers) is checked deterministically in
eval_output, not here. It reads a transcript JSON (--in) + the gold (for reference_sql),
executes each distinct reference_sql once (cached) via the project's BQ client, then judges.

Repeats with IDENTICAL agent output are judged once and weighted by their count,
so a 20-repeat temp-0 transcript costs ~1 judge call per turn, not 20.

Pipeline: run AFTER eval_output.py — it scores the transcript eval_output.py produced.
This is the only post-hoc scorer that needs an LLM (the judge) and BQ (to run
reference_sql). Sibling scorers over the same transcript, independent of this one and of
each other:
  eval_faithfulness.py  structured output's self-consistency — gold-free, no LLM/BQ
  eval_queries.py       source/period from the executed SQL vs the gold — no LLM/BQ

    uv run python -m eval.eval_quality --in eval/<transcript>.json --dry-run
    uv run python -m eval.eval_quality --in eval/<transcript>.json --judge-model google_genai:gemini-3.1-pro-preview

NOTE: the judge should ideally be a stronger / different-family model than the
agent (less self-preference). Default --judge-model is google_genai:gemini-3.1-pro-preview.
"""

import argparse
import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

import yaml

from app.settings import settings
from eval.lib.judge import (
    DEFAULT_JUDGE_MODEL,
    JUDGE_SYSTEM,
    SCORE_FIELDS,
    ajudge,
    build_judge,
    build_tasks,
    dedup_tasks,
    render_turn,
    run_reference_sql,
)

# This script's folder — gold input and result files default here, so the eval
# works regardless of the current working directory.
EVAL_DIR = Path(__file__).resolve().parent


# Dry-run cost estimate only (a real run reports exact usage). Gemini measures
# ~3.5 chars/token on this mixed PT-prose/JSON/SQL content; a verdict is small.
DRY_CHARS_PER_TOKEN = 3.5
DRY_VERDICT_TOKENS = 100


# =============================================================================
# Gold
# =============================================================================
def load_gold_refs(path: str) -> dict[tuple[str, int], dict]:
    """Load the gold spec, keyed by (thread id, turn index).

    Args:
        path (str): Path to the gold YAML file.

    Returns:
        dict[tuple[str, int], dict]: Maps (thread_id, turn_index) to that turn's gold dict
            (which carries reference_sql and notes).
    """
    gold = yaml.safe_load(open(path))
    return {
        (thread["id"], turn_index): turn
        for thread in gold
        for turn_index, turn in enumerate(thread["turns"])
    }


# =============================================================================
# Judging
# =============================================================================
async def judge_turn(judge, semaphore: asyncio.Semaphore, task: dict) -> dict:
    """Judge one task, returning its verdict record (and printing a one-line mark).

    Args:
        judge: The structured-output judge model from build_judge().
        semaphore (asyncio.Semaphore): Concurrency limiter.
        task (dict): A deduped task (carries weight/repeats and reference_rows).

    Returns:
        dict: The task's identity fields plus the parsed verdict (or None + an error) and
            token usage.
    """
    base_record = {
        key: task[key] for key in ("thread", "turn_index", "weight", "repeats")
    }
    async with semaphore:
        verdict_record = {**base_record, **await ajudge(judge, task)}
    mark = (
        "ERR"
        if verdict_record["verdict"] is None
        else " ".join(
            f"{field[:4]}{'·' if verdict_record['verdict'][field] is None else ('✓' if verdict_record['verdict'][field] else '✗')}"
            for field in SCORE_FIELDS
        )
    )
    print(f"  [{task['thread']:<12} t{task['turn_index']} ×{task['weight']:<2}] {mark}")
    return verdict_record


# =============================================================================
# Aggregation & reporting
# =============================================================================
def aggregate(verdicts: list[dict]) -> dict:
    """Compute the pass-rate per score field, weighting each verdict by its repeat count.

    Args:
        verdicts (list[dict]): Verdict records from judge_turn().

    Returns:
        dict: {field: {"rate": float, "n": int}} plus "_errors" (weighted count of verdicts
            that failed to parse).
    """
    totals = defaultdict(lambda: {"hit": 0, "n": 0})
    errors = 0
    for verdict_record in verdicts:
        weight = verdict_record.get("weight", 1)
        if verdict_record["verdict"] is None:
            errors += weight
            continue
        for field in SCORE_FIELDS:
            value = verdict_record["verdict"][field]
            if value is None:
                continue
            totals[field]["hit"] += weight * int(bool(value))
            totals[field]["n"] += weight
    summary = {
        field: {"rate": round(counts["hit"] / counts["n"], 3), "n": counts["n"]}
        for field, counts in totals.items()
    }
    summary["_errors"] = errors
    return summary


def configure_tracing(args: argparse.Namespace) -> tuple[str, bool]:
    """Forward LangSmith config from `settings` into os.environ before the judge runs.

    LangChain's tracer reads the environment, but pydantic-settings only populates
    `settings`. Judge runs get their own project by default so their traces don't mix with
    the agent-eval traces.

    Args:
        args (argparse.Namespace): Parsed CLI args (langsmith_project, no_trace).

    Returns:
        tuple[str, bool]: (project name, tracing enabled).
    """
    ls_project = args.langsmith_project or f"{settings.LANGSMITH_PROJECT}-eval-judge"
    tracing = settings.LANGSMITH_TRACING and not args.no_trace
    os.environ["LANGSMITH_TRACING"] = "true" if tracing else "false"
    if tracing:
        os.environ["LANGSMITH_PROJECT"] = ls_project
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    return ls_project, tracing


# =============================================================================
# CLI
# =============================================================================
async def main() -> None:
    """Parse args, judge the transcript (or dry-run), and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in", dest="transcript", required=True, help="thread_eval_*.json"
    )
    parser.add_argument("--gold", default=str(EVAL_DIR / "eval_gold.yaml"))
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--max-repeats",
        type=int,
        default=None,
        help="cap repeats considered before dedup (rarely needed; dedup already collapses identical outputs)",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--thread", action="append", help="Only this thread id (repeatable)"
    )
    parser.add_argument(
        "--price-in",
        type=float,
        default=2.0,
        help="USD per 1M input tokens (default: gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--price-out",
        type=float,
        default=12.0,
        help="USD per 1M output tokens (default: gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--langsmith-project",
        default=None,
        help="LangSmith project for judge traces (default: <settings project>-eval-judge)",
    )
    parser.add_argument(
        "--no-trace", action="store_true", help="Disable LangSmith tracing"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print how many judge calls and reference SQLs would run, then exit without calling the judge or BQ",
    )
    args = parser.parse_args()

    transcript = json.load(open(args.transcript))
    if args.thread:
        transcript["units"] = [
            unit for unit in transcript["units"] if unit["thread"] in args.thread
        ]
    gold = load_gold_refs(args.gold)
    tasks = build_tasks(transcript, gold, args.max_repeats)
    deduped_tasks = dedup_tasks(tasks)

    print(f"judge_model={args.judge_model!r}  (agent was {transcript.get('model')!r})")
    if args.judge_model == settings.MODEL_URI:
        print(
            "  WARNING: judging with the same model as the agent — prefer a stronger/different one via --judge-model"
        )
    print(
        f"turn-instances={len(tasks)}  ->  distinct outputs to judge={len(deduped_tasks)}\n"
    )

    suffix = f"_{'-'.join(args.thread)}" if args.thread else ""
    out_path = args.out or str(
        EVAL_DIR / f"{Path(args.transcript).stem}{suffix}_judged.json"
    )
    if args.dry_run:
        reference_sql_count = len(
            {task["reference_sql"] for task in deduped_tasks if task["reference_sql"]}
        )
        estimated_input_tokens = (
            sum(
                len(JUDGE_SYSTEM) + len(render_turn(dict(task, reference_rows=None)))
                for task in deduped_tasks
            )
            / DRY_CHARS_PER_TOKEN
        )
        estimated_output_tokens = DRY_VERDICT_TOKENS * len(deduped_tasks)
        estimated_cost = (
            estimated_input_tokens / 1e6 * args.price_in
            + estimated_output_tokens / 1e6 * args.price_out
        )
        print("DRY RUN — no reference SQLs executed and no judge calls made.")
        print(f"  judge calls that would run: {len(deduped_tasks)}")
        print(
            f"  distinct reference SQLs that would execute (BQ): {reference_sql_count}"
        )
        print(
            f"  est. tokens: ~{estimated_input_tokens:,.0f} in (char-based, excl. reference rows) "
            f"+ ~{estimated_output_tokens:,.0f} out"
        )
        print(
            f"  est. cost @ ${args.price_in}/M in, ${args.price_out}/M out: "
            f"~${estimated_cost:.3f}  (real run reports exact)"
        )
        print(f"  output would be: {out_path}")
        return

    ls_project, tracing = configure_tracing(args)
    print("langsmith: " + (f"project={ls_project!r}" if tracing else "disabled"))

    print("executing reference SQLs ...")
    reference_cache: dict[str, list] = {}
    for task in deduped_tasks:
        task["reference_rows"] = (
            run_reference_sql(task["reference_sql"], reference_cache)
            if task["reference_sql"]
            else None
        )
    print(f"  {len(reference_cache)} distinct reference queries run\n")

    judge = build_judge(args.judge_model)
    semaphore = asyncio.Semaphore(args.concurrency)
    verdicts = await asyncio.gather(
        *(judge_turn(judge, semaphore, task) for task in deduped_tasks)
    )
    summary = aggregate(verdicts)

    print("\n=== Judge scorecard (weighted over all repeats) ===")
    for field in SCORE_FIELDS:
        if field in summary:
            print(
                f"  {field:<18} {summary[field]['rate']:.0%}  (n={summary[field]['n']})"
            )
    print(f"  judge_errors={summary['_errors']}")

    total_input_tokens = sum(v.get("input_tokens") or 0 for v in verdicts)
    total_output_tokens = sum(v.get("output_tokens") or 0 for v in verdicts)
    cost = (
        total_input_tokens / 1e6 * args.price_in
        + total_output_tokens / 1e6 * args.price_out
    )
    print("\n=== Cost ===")
    print(f"  judge calls:   {len(verdicts)}")
    print(
        f"  input tokens:  {total_input_tokens:,}    output tokens: {total_output_tokens:,}"
    )
    print(f"  @ ${args.price_in}/M in, ${args.price_out}/M out  ->  ${cost:.4f}")
    if total_input_tokens == 0 and any(v["verdict"] for v in verdicts):
        print("  (note: judge model did not report token usage — cost unavailable)")

    report = {
        "transcript": args.transcript,
        "judge_model": args.judge_model,
        "agent_model": transcript.get("model"),
        "langsmith_project": ls_project if tracing else None,
        "cost": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "price_in_per_m": args.price_in,
            "price_out_per_m": args.price_out,
            "usd": round(cost, 4),
        },
        "summary": summary,
        "verdicts": verdicts,
    }
    with open(out_path, "w") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
