"""LLM judge over the transcripts that eval_output.py saved.

One judge call per answered turn (query OR clarify), emitting SEPARATE scores
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

    uv run eval/eval_quality.py --in eval/<transcript>.json --dry-run
    uv run eval/eval_quality.py --in eval/<transcript>.json --judge-model google_genai:gemini-3.1-pro-preview

NOTE: the judge should ideally be a stronger / different-family model than the
agent (less self-preference). Default --judge-model is google_genai:gemini-3.1-pro-preview.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import yaml
from google.cloud import bigquery as bq
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

# Make the repo root importable so `app` resolves whether this file is run as a
# module (python -m eval.eval_quality) or directly (python eval/eval_quality.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.tools.bigquery import MAX_BYTES_BILLED, _bq_client  # noqa: E402
from app.settings import settings  # noqa: E402

# This script's folder — gold input and result files default here, so the eval
# works regardless of the current working directory.
EVAL_DIR = Path(__file__).resolve().parent

SCORE_FIELDS = (
    "correct",
    "grounded",
    "answers_question",
    "stated_assumption",
)
MAX_REFERENCE_ROWS = 50  # cap rows shown to the judge to bound tokens

# Dry-run cost estimate only (a real run reports exact usage). Gemini measures
# ~3.5 chars/token on this mixed PT-prose/JSON/SQL content; a verdict is small.
DRY_CHARS_PER_TOKEN = 3.5
DRY_VERDICT_TOKENS = 100


# =============================================================================
# Judge schema & prompt
# =============================================================================
class JudgeVerdict(BaseModel):
    correct: bool | None = Field(
        default=None,
        description="(query turns WITH a reference result) Are the main figures/entities/ranking CONSISTENT with the REFERENCE RESULT? A different-but-valid slice/grouping/period the user didn't specify, extra metrics, rounding, or language differences are OK — fail ONLY on a direct contradiction (a different value for the same quantity, wrong entity, wrong ordering). null on clarify turns AND when no reference result is provided (correctness can't be verified).",
    )
    grounded: bool = Field(
        description="Does every quantitative/factual claim trace to data the assistant ACTUALLY RETRIEVED (this turn OR an earlier turn of the conversation)? Check the prose against the ASSISTANT'S RETRIEVED DATA section (its own query results), NOT the reference — a value present there is grounded even if absent from the reference. Inventing numbers/trends/datasets, or asserting a value with no supporting query, fails. (clarify: backed by real exploration, no assumed value)."
    )
    answers_question: bool = Field(
        description="Does the answer address what the user needed this turn? (query: answers the question; clarify: does the right clarification/guidance per the note)."
    )
    stated_assumption: bool | None = Field(
        default=None,
        description="If the request was ambiguous (e.g., an unspecified breakdown/interpretation), did the answer make its chosen interpretation explicit? null if there was NO ambiguity.",
    )
    rationale: str = Field(
        description="1-2 sentence justification; name the main discrepancy, if any."
    )


JUDGE_SYSTEM = """\
You are a strict evaluator of a Brazilian open-data assistant's answer, for ONE turn of a conversation. The assistant answer and the data are in Portuguese.

You receive: the turn's expected type, the conversation so far, an internal note (what the turn tests), the tools the assistant used, the ASSISTANT'S RETRIEVED DATA (the results of its OWN SQL, from this turn AND earlier turns of the conversation — a follow-up may rely on data queried in a previous turn), its final answer (prose + the structured fields it reported), and — WHEN AVAILABLE — a REFERENCE RESULT from a trusted SQL query (the gold). Some data turns have no reference; the deterministic eval covers their inputs (right tables/period) separately.

You have TWO independent data anchors — do NOT conflate them:
- The ASSISTANT'S RETRIEVED DATA anchors `grounded`: did every claim come from data the assistant actually queried (at any point in the conversation)?
- The REFERENCE RESULT anchors `correct`: is the answer consistent with the gold? (only when a reference is present)

Score each criterion as a boolean (or null when it does not apply).

If the type is `query` (it should answer with data):
- correct: the main figures/entities/ordering are CONSISTENT with the reference. A different-but-valid decomposition, extra metrics, or a different period the user did NOT specify are NOT failures — fail only on a direct contradiction (a different value for the SAME quantity, a wrong entity, wrong ordering). A value the assistant computed that the reference simply doesn't contain is NOT a correctness failure — judge it under `grounded` instead. If the assistant's scope/period differs so much that its figures aren't comparable to the reference, don't invent a contradiction: lean on whether the trend/entities are consistent. If NO REFERENCE RESULT is provided for this turn, set correct=null (you cannot verify correctness) and judge only `grounded` and `answers_question`.
- grounded: every quantitative/factual claim traces to the ASSISTANT'S RETRIEVED DATA shown to you (from this turn OR an earlier turn — a follow-up legitimately reuses data queried before). A number that appears in the assistant's own query results IS grounded, even if it's absent from the reference. Fabricating figures/trends/comparisons that are in NEITHER the assistant's results nor a legitimate calculation over them fails.
- answers_question: the prose addresses what the user asked in this turn.
- stated_assumption: if the request was ambiguous, it made its chosen interpretation explicit; null if there was no ambiguity.

If the type is `clarify` (it should NOT query data; per the note, it should ask for the missing detail OR explore the catalog and guide):
- correct: null (there is no data answer to verify).
- grounded: the answer does NOT invent datasets/tables/values. If it describes available data, that must be backed by real exploration — check the tools used (search_datasets/get_dataset_details/get_table_details). If it assumes a value the user did not provide (e.g., a specific município), grounded=false.
- answers_question: it does the right clarification/guidance per the note (e.g., asks which município; or describes what exists and suggests specific refinements), without having queried data.
- stated_assumption: null.

Be strict but fair: wording/rounding differences, extra valid metrics, and a differently-but-validly-scoped query are acceptable; a wrong value for the same quantity, wrong entities, claims absent from the assistant's retrieved data (fabrications), unrequested assumptions, or not doing what the turn required are failures. Give a one- to two-sentence rationale."""


# =============================================================================
# Gold & reference SQL
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


def run_reference_sql(sql: str, cache: dict[str, list]) -> list[dict]:
    """Execute a reference SQL against BigQuery, caching the result by SQL text.

    Args:
        sql (str): The reference SQL to run.
        cache (dict[str, list]): SQL-text -> rows cache, mutated in place so each distinct
            reference SQL runs at most once.

    Returns:
        list[dict]: The query result rows (each row as a dict).
    """
    if sql not in cache:
        job = _bq_client().query(
            sql, job_config=bq.QueryJobConfig(maximum_bytes_billed=MAX_BYTES_BILLED)
        )
        cache[sql] = [dict(row) for row in job.result()]
    return cache[sql]


# =============================================================================
# Task building & rendering
# =============================================================================
def _render_agent_queries(queries: list[dict]) -> str:
    """Render the assistant's own executed queries + result rows for the judge prompt.

    Args:
        queries (list[dict]): Executed query records (sql / status / rows / row_count /
            message).

    Returns:
        str: A human-readable block (the anchor for `grounded`), or an N/A notice when the
            assistant executed no SQL this turn or in any earlier turn.
    """
    if not queries:
        return "N/A — the assistant executed no SQL this turn or in any earlier turn."
    blocks = []
    for index, query in enumerate(queries, 1):
        sql = (query.get("sql") or "").strip()
        if query.get("rows") is not None:
            rows, total = query["rows"], query.get("row_count")
            omitted = (
                f"\n(... showing {len(rows)} of {total} rows)"
                if total and total > len(rows)
                else ""
            )
            body = json.dumps(rows, ensure_ascii=False, default=str, indent=2) + omitted
        else:
            body = query.get("message") or "(no result)"
        blocks.append(
            f"## Assistant query {index} (status: {query.get('status')})\n{sql}\nResult:\n{body}"
        )
    return "\n\n".join(blocks)


def render_turn(task: dict) -> str:
    """Build the judge's human-message prompt for one task (turn).

    Args:
        task (dict): A task from build_tasks(), augmented with `reference_rows`.

    Returns:
        str: The rendered turn — expected type, conversation, note, tools, the assistant's
            answer + structured fields, its retrieved data, and the reference result.
    """
    reference_rows = task["reference_rows"]
    if reference_rows is None and task["turn_type"] == "query":
        ref_block = (
            "N/A — no reference query was provided for this turn, so correctness cannot "
            "be verified against a gold result. Set `correct = null` and judge only "
            "`grounded` (against the assistant's retrieved data) and `answers_question`."
        )
    elif reference_rows is None:
        ref_block = "N/A — clarification turn; there is no data answer to verify."
    else:
        shown = reference_rows[:MAX_REFERENCE_ROWS]
        omitted = (
            f"\n(... {len(reference_rows) - len(shown)} rows omitted; {len(reference_rows)} total)"
            if len(reference_rows) > len(shown)
            else ""
        )
        ref_block = (
            json.dumps(shown, ensure_ascii=False, default=str, indent=2) + omitted
        )
    agent_data_block = _render_agent_queries(task.get("agent_queries") or [])
    structured = task["agent_structured"]
    data_source_names = [d.get("name") for d in (structured.get("data_sources") or [])]
    conversation = "\n".join(
        f"  {turn_number + 1}. {user_message}"
        for turn_number, user_message in enumerate(task["user_turns"])
    )
    return f"""# Expected turn type: {task["turn_type"]}

# Conversation (user turns so far)
{conversation}

# This turn's question
{task["user_turns"][-1]}

# What this turn tests (internal note)
{task["note"] or "—"}

# Tools the assistant used this turn
{task["tools_used"]}

# Assistant's answer (prose)
{task["agent_response"]}

# Structured fields the assistant reported
- data_sources: {data_source_names}
- temporal_coverage: {structured.get("temporal_coverage")}
- follow_up_questions: {structured.get("follow_up_questions")}

# ASSISTANT'S RETRIEVED DATA (its OWN queries + results, this turn AND earlier turns of the conversation — anchor for `grounded`)
{agent_data_block}

# REFERENCE RESULT (gold, from the trusted query — anchor for `correct`)
{ref_block}"""


def build_tasks(transcript: dict, gold: dict, max_repeats: int | None) -> list[dict]:
    """Build one judge task per answered (`ok`) turn that has a gold expectation.

    Args:
        transcript (dict): The transcript produced by eval_output.
        gold (dict): (thread_id, turn_index) -> gold turn, from load_gold_refs().
        max_repeats (int | None): If set, skip repeats whose index is >= this value.

    Returns:
        list[dict]: One task per judged turn, carrying the conversation so far, the agent's
            answer + structured fields, the thread-cumulative executed queries, and the
            reference_sql.
    """
    tasks = []
    for unit in transcript["units"]:
        if max_repeats is not None and unit["repeat"] >= max_repeats:
            continue
        user_messages = []
        thread_queries: list[
            dict
        ] = []  # executed queries accumulated across the thread
        for turn in unit["turns"]:
            user_messages.append(turn["user"])
            thread_queries += turn.get("queries") or []
            gold_turn = gold.get((unit["thread"], turn["turn_index"]))
            # Judge any turn the agent actually answered (query or clarify) for which
            # we have a gold expectation. Query turns carry a reference_sql to anchor
            # correctness; clarify turns are judged on the rubric only.
            if gold_turn is None or turn["status"] != "ok":
                continue
            tasks.append(
                {
                    "thread": unit["thread"],
                    "repeat": unit["repeat"],
                    "turn_index": turn["turn_index"],
                    "turn_type": gold_turn["action"],
                    "user_turns": list(user_messages),
                    "note": gold_turn.get("notes"),
                    "agent_response": (turn.get("structured") or {}).get("response")
                    or turn.get("response_text"),
                    "agent_structured": turn.get("structured") or {},
                    "agent_queries": list(
                        thread_queries
                    ),  # this turn + all earlier turns
                    "tools_used": turn.get("tools_used", []),
                    "reference_sql": gold_turn.get("reference_sql"),
                }
            )
    return tasks


# =============================================================================
# Dedup & judging
# =============================================================================
def _output_signature(task: dict) -> str:
    """Hash the parts of a task that vary across repeats and that the judge reads.

    Covers the agent's structured fields, prose response, tools used, and retrieved query
    results. `response` is included explicitly so free-text answers (no structured fields,
    e.g. the main branch) still dedup by their prose rather than collapsing to one.

    Args:
        task (dict): A task from build_tasks().

    Returns:
        str: A SHA-1 hex digest identifying this agent output.
    """
    payload = json.dumps(
        {
            "structured": task["agent_structured"],
            "response": task["agent_response"],
            "tools_used": task["tools_used"],
            "queries": task["agent_queries"],
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()


def dedup_tasks(tasks: list[dict]) -> list[dict]:
    """Collapse repeats with identical agent output (per thread+turn) into one task each.

    Judging each distinct output once, weighted by how many repeats produced it, reflects
    the full distribution without an N-x judge bill (a big win at temperature 0).

    Args:
        tasks (list[dict]): Tasks from build_tasks().

    Returns:
        list[dict]: One task per distinct output, with added `weight` (repeat count) and
            `repeats` (sorted repeat indices).
    """
    groups: dict[tuple, dict] = {}
    for task in tasks:
        key = (task["thread"], task["turn_index"], _output_signature(task))
        groups.setdefault(key, {"task": task, "repeats": []})["repeats"].append(
            task["repeat"]
        )
    return [
        dict(
            group["task"],
            weight=len(group["repeats"]),
            repeats=sorted(group["repeats"]),
        )
        for group in groups.values()
    ]


def build_judge(model_uri: str):
    """Build the structured-output judge model.

    Args:
        model_uri (str): The judge model URI (Google models get the service-account creds).

    Returns:
        A LangChain model configured to return a JudgeVerdict (with the raw response
        included).
    """
    kwargs = {"temperature": 0}
    if model_uri.startswith("google"):
        kwargs["credentials"] = settings.GOOGLE_CREDENTIALS
    return init_chat_model(model_uri, **kwargs).with_structured_output(
        JudgeVerdict, include_raw=True
    )


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
        try:
            judge_output = await judge.ainvoke(
                [SystemMessage(JUDGE_SYSTEM), HumanMessage(render_turn(task))]
            )
            verdict: JudgeVerdict | None = judge_output["parsed"]
            usage = getattr(judge_output["raw"], "usage_metadata", None) or {}
            verdict_record = {
                **base_record,
                "verdict": verdict.model_dump() if verdict is not None else None,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            }
            if verdict is None:
                verdict_record["error"] = f"parse: {judge_output.get('parsing_error')}"
        except Exception as exc:
            verdict_record = {
                **base_record,
                "verdict": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
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
    parser.add_argument("--judge-model", default="google_genai:gemini-3.1-pro-preview")
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
