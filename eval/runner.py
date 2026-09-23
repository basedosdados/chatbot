"""Drive the agent over the gold threads and write a transcript the scorers read.

This is the only eval component that runs the live agent (real BigQuery, real LLM), so it
does one job and does it faithfully: build the agent exactly as production does, replay
each gold thread, and record everything a scorer or the judge could need — then stop. It
does **no** scoring; the deterministic scorer and the judge are separate passes over the
transcript it writes.

What it mirrors from `app/main.py`: a `ChatOpenAI` model with reasoning effort, the
system-prompt / summarization / call-limit middleware, `response_format=StructuredResponse`
and `context_schema=AgentContext`. What it changes for eval: an in-memory checkpointer,
and the reasoning **effort** is chosen per run (production pins one; the eval runs a single
effort per invocation), because a GPT-5.6 reasoning model has no temperature to sweep.

Each gold thread is replayed on a shared `thread_id` (so follow-ups see prior context),
`--repeats` times, at one `--effort` (default medium). One replay is a *unit*; every unit
records its turns plus the table metadata the run retrieved (`get_table_details` outputs),
so run-relative checks read each run's own `partitioned_by` / coded-column flags rather
than constants. To compare efforts, run the script once per effort — each writes its own
transcript — then diff the scored results.

    uv run eval/runner.py --dry-run
    uv run eval/runner.py --repeats 5 --effort high
    uv run eval/runner.py --thread comex-stat --repeats 1

Every turn is a live multi-step agent run — cost is (threads x turns x repeats).
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
)
from langchain.messages import AnyMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

# Make the repo root importable so `app` and `eval.lib` resolve whether this file
# is run as a module (python -m eval.runner) or directly (python eval/runner.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.context import AgentContext  # noqa: E402
from app.agent.middleware import system_prompt_middleware  # noqa: E402
from app.agent.prompts import SYSTEM_PROMPT  # noqa: E402
from app.agent.schemas import StructuredResponse  # noqa: E402
from app.agent.tools import BDToolkit  # noqa: E402
from app.i18n import DEFAULT_LANGUAGE  # noqa: E402
from app.settings import settings  # noqa: E402
from eval.lib import gold  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent

# Middleware tunables, kept in step with app/main.py.
# If production changes these, change them here too.
SUMMARIZE_TRIGGER_TOKENS = 500_000
SUMMARIZE_KEEP_TOKENS = 100_000
MODEL_CALL_RUN_LIMIT = 20

# The reasoning effort to run at by default (see module docstring for why effort, not temp).
DEFAULT_EFFORT = "medium"

# Rows persisted per executed query. The true row count is stored alongside, so a scorer can
# still tell the result was truncated (the agent's own tool already caps context at 1000).
QUERY_ROWS_CAP = 50

# A synthetic user id for the run context's BigQuery job labels (no real user in eval).
EVAL_USER_ID = "eval-harness"


# =============================================================================
# Agent construction (mirrors app/main.py, parameterized by reasoning effort)
# =============================================================================
def current_branch() -> str:
    """The current git branch name, or "unknown" if git can't be queried."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def build_agent(effort: str) -> CompiledStateGraph:
    """Build the agent under test at a given reasoning effort, mirroring production.

    Args:
        effort: The reasoning effort to run at ("medium", "high", ...).

    Returns:
        The compiled agent graph, with a fresh in-memory checkpointer.
    """
    model = ChatOpenAI(
        api_key=settings.OPENAI_API_KEY,
        model=settings.MODEL_URI,
        reasoning={"effort": effort, "summary": "auto"},
    )

    middleware = [
        system_prompt_middleware,
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", SUMMARIZE_TRIGGER_TOKENS),
            keep=("tokens", SUMMARIZE_KEEP_TOKENS),
            trim_tokens_to_summarize=None,
        ),
        ModelCallLimitMiddleware(run_limit=MODEL_CALL_RUN_LIMIT, exit_behavior="end"),
    ]

    return create_agent(
        model=model,
        tools=BDToolkit.get_tools(),
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        response_format=StructuredResponse,
        context_schema=AgentContext,
        checkpointer=InMemorySaver(),
    )


# =============================================================================
# Trace extraction from the agent's final state
# =============================================================================
def _messages_since_last_human(messages: list[AnyMessage]) -> list[AnyMessage]:
    """The messages produced in the latest turn (everything after the last human message).

    The checkpointer returns the whole thread, so a per-turn view slices off the tail.
    """
    last_human = max(
        (i for i, message in enumerate(messages) if message.type == "human"), default=-1
    )
    return messages[last_human + 1 :]


def _tools_used(turn_messages: list[AnyMessage]) -> list[str]:
    """The distinct tool names the agent called this turn, sorted.

    Lets a scorer tell `explore` (metadata tools, no query) from `clarify` (no tools).
    """
    names = {
        tool_call["name"]
        for message in turn_messages
        for tool_call in getattr(message, "tool_calls", None) or []
    }
    return sorted(names)


def _last_ai_text(messages: list[AnyMessage]) -> str | None:
    """The text of the last AI message — a debugging aid when no structured response comes back."""
    for message in reversed(messages):
        if message.type == "ai":
            return message.text
    return None


def _parse_query_result(content: str) -> tuple[int | None, list | None, str | None]:
    """Parse an `execute_bigquery_sql` result body into (row_count, capped_rows, error).

    Success is `{"row_count", "rows", ...}` -> (count, first `QUERY_ROWS_CAP` rows, None).
    A failed query is a serialized `ToolError` (`{"status": "error", "message": ...}`) ->
    (None, None, message). Anything unexpected -> (None, None, raw text), never a crash.

    Args:
        content: The raw tool-message content.

    Returns:
        (row_count, capped_rows, error_message).
    """
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None, None, content
    if not isinstance(payload, dict):
        return None, None, content
    # A success payload carries `rows`/`row_count`; a failed query is a ToolError with
    # `status`/`message` and no `rows`. `status` is absent on success (it's the discriminator,
    # not a guaranteed key), so it's read with .get; the keys guaranteed *within* each branch
    # are read strictly.
    if "rows" in payload:
        return payload["row_count"], payload["rows"][:QUERY_ROWS_CAP], None
    if payload.get("status") == "error":
        return None, None, payload["message"]
    return None, None, content


def _executed_queries(turn_messages: list[AnyMessage]) -> list[dict]:
    """The `execute_bigquery_sql` calls made this turn, each paired with its result.

    This is the agent's OWN retrieved data — what the judge scores grounding against and the
    deterministic checks read the SQL from.

    Returns:
        One `{sql, status, row_count, rows, error}` per query call this turn.
    """
    sql_by_call_id = {
        tool_call["id"]: tool_call["args"].get("sql_query")
        for message in turn_messages
        for tool_call in getattr(message, "tool_calls", None) or []
        if tool_call["name"] == "execute_bigquery_sql"
    }

    queries = []

    for message in turn_messages:
        if message.type == "tool" and message.name == "execute_bigquery_sql":
            row_count, rows, error = _parse_query_result(message.content)
            queries.append(
                {
                    "sql": sql_by_call_id.get(message.tool_call_id),
                    "status": "error" if error is not None else "success",
                    "row_count": row_count,
                    "rows": rows,
                    "error": error,
                }
            )

    return queries


def _collect_table_metadata(
    messages: list[AnyMessage],
) -> tuple[dict[str, str], dict[str, dict]]:
    """Gather table identity + full metadata from the whole thread's exploration calls.

    Returns two maps, both accumulated across every turn seen so far:
      * `uuid -> gcp_id` (from `get_table_details` and `get_dataset_details`), to
        resolve the table UUIDs the agent reports in `data_sources` to their gcp ids.
      * `gcp_id -> raw get_table_details payload`, the run's own metadata (partitioned_by,
        period_start/end, per-column coded flags) the scorer parses via `lib.metadata`.

    A failed exploration call serializes to a `ToolError` (`status: error`) and is skipped.
    """
    uuid_to_gcp: dict[str, str] = {}
    table_details: dict[str, dict] = {}

    for message in messages:
        if message.type != "tool" or message.name not in (
            "get_table_details",
            "get_dataset_details",
        ):
            continue
        try:
            payload = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict) or payload.get("status") == "error":
            continue
        # Past the error guard, a success payload is a full Table/Dataset serialization, so
        # its keys are guaranteed; `gcp_id` may still be None (an unmaterialized table).
        if message.name == "get_table_details":
            if payload["gcp_id"]:
                uuid_to_gcp[payload["id"]] = payload["gcp_id"]
                table_details[payload["gcp_id"]] = payload
        else:  # get_dataset_details: table overviews (no columns/partition metadata)
            for table in payload["tables"]:
                if table["gcp_id"]:
                    uuid_to_gcp[table["id"]] = table["gcp_id"]

    return uuid_to_gcp, table_details


def extract_turn(result: dict, uuid_to_gcp: dict[str, str]) -> dict:
    """Build the per-turn record from the agent's final state.

    Args:
        result: The agent's `ainvoke` result (`messages` and `structured_response`).
        uuid_to_gcp: The thread's accumulated uuid -> gcp_id map, to resolve data_sources.

    Returns:
        The per-turn record. Status "ok", or "no_structured" when the agent produced no
        structured response (a contract failure the scorer counts, not a crash).
    """
    messages = result["messages"]
    turn_messages = _messages_since_last_human(messages)
    queries = _executed_queries(turn_messages)
    base = {
        "status": "ok",
        "is_query": bool(queries),
        "tools_used": _tools_used(turn_messages),
        "model_calls": result.get("run_model_call_count"),
        "queries": queries,
    }

    structured = result.get("structured_response")
    if structured is None:
        return {
            **base,
            "status": "no_structured",
            "structured": None,
            "tables": [],
            "final_message": _last_ai_text(messages),
        }

    dumped = structured.model_dump(mode="json")
    # The tables the agent REPORTED, resolved UUID -> gcp_id. An unmapped UUID is kept as-is
    # so it fails to match the gold and surfaces the problem instead of hiding it.
    reported_tables = sorted(
        {
            uuid_to_gcp.get(source["table_id"], source["table_id"])
            for source in (dumped["data_sources"] or [])
        }
    )
    return {**base, "structured": dumped, "tables": reported_tables}


# =============================================================================
# Thread replay
# =============================================================================
def _run_config(
    thread_id: str, thread: dict, effort: str, repeat: int, turn_index: int, branch: str
) -> dict:
    """The per-turn LangGraph config: thread id for checkpointing, plus LangSmith labels."""
    return {
        "configurable": {"thread_id": thread_id},
        "run_name": f"{thread['id']}#{effort}-r{repeat}-t{turn_index}",
        "tags": [f"branch:{branch}", f"effort:{effort}", f"thread:{thread['id']}"],
        "metadata": {
            "eval_thread": thread["id"],
            "effort": effort,
            "repeat": repeat,
            "turn_index": turn_index,
            "branch": branch,
        },
    }


def _skipped_turn(turn_index: int, user: str) -> dict:
    """A placeholder for a turn that never ran because an earlier turn broke the thread."""
    return {"turn_index": turn_index, "user": user, "status": "skipped"}


async def replay_thread(
    agent: CompiledStateGraph, thread: dict, effort: str, repeat: int, branch: str
) -> dict:
    """Replay one thread turn-by-turn on a shared thread_id; return the transcript unit.

    An agent failure on a turn leaves the thread state possibly broken, so the remaining
    turns are marked skipped; an extraction failure on an otherwise-successful run is
    recorded but the thread continues (its state is intact).

    Args:
        agent: The compiled agent (already at the target effort).
        thread: The gold thread (its `id` and `turns`).
        effort: The reasoning effort in effect (for labels and the unit record).
        repeat: The repeat index.
        branch: The checked-out git branch (for labels).

    Returns:
        The unit: `{thread, effort, repeat, turns, tables, uuid_to_gcp}`.
    """
    thread_id = f"{thread['id']}-{effort}-{repeat}"
    context = AgentContext(
        thread_id=thread_id, user_id=EVAL_USER_ID, language=DEFAULT_LANGUAGE
    )
    turn_records: list[dict] = []
    uuid_to_gcp: dict[str, str] = {}
    table_details: dict[str, dict] = {}

    for turn_index, turn in enumerate(thread["turns"]):
        config = _run_config(thread_id, thread, effort, repeat, turn_index, branch)
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": turn["user"]}]},
                config=config,
                context=context,
            )
        except Exception as exc:
            turn_records.append(
                {
                    "turn_index": turn_index,
                    "user": turn["user"],
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            turn_records.extend(
                _skipped_turn(j, thread["turns"][j]["user"])
                for j in range(turn_index + 1, len(thread["turns"]))
            )
            break

        # The whole-thread scan is cumulative, so it subsumes earlier turns' metadata.
        uuid_to_gcp, table_details = _collect_table_metadata(result["messages"])
        try:
            record = extract_turn(result, uuid_to_gcp)
        except Exception as exc:
            record = {
                "status": "error",
                "error": f"extract_turn: {type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        record["turn_index"], record["user"] = turn_index, turn["user"]
        turn_records.append(record)

    return {
        "thread": thread["id"],
        "effort": effort,
        "repeat": repeat,
        "turns": turn_records,
        "tables": table_details,
        "uuid_to_gcp": uuid_to_gcp,
    }


def _unit_progress(unit: dict) -> str:
    """A one-line progress summary of a finished unit: each turn's status/kind."""
    marks = []
    for record in unit["turns"]:
        status = record["status"]
        if status == "ok":
            marks.append(
                f"t{record['turn_index']}:{'query' if record['is_query'] else 'no-query'}"
            )
        else:
            marks.append(f"t{record['turn_index']}:{status.upper()}")
    return "  ".join(marks)


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default=str(EVAL_DIR / "eval_gold.yaml"))
    parser.add_argument("--repeats", type=int, default=5, help="Replays per thread")
    parser.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        help=f"Reasoning effort for this run (default {DEFAULT_EFFORT!r}); compare by running twice",
    )
    parser.add_argument(
        "--thread", action="append", help="Only this thread id (repeatable)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Parallel thread replays"
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--langsmith-project",
        default=None,
        help="LangSmith project for these traces (default: <settings project>-eval)",
    )
    parser.add_argument(
        "--no-trace", action="store_true", help="Disable LangSmith tracing"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan and exit without calling the agent",
    )
    return parser.parse_args()


def configure_tracing(args: argparse.Namespace) -> tuple[str | None, bool]:
    """Forward LangSmith config from `settings` into `os.environ` (the tracer reads env).

    Isolates eval traces in their own project by default. Must run before the agent does.
    """
    ls_project = args.langsmith_project or f"{settings.LANGSMITH_PROJECT}-eval"
    tracing = settings.LANGSMITH_TRACING and not args.no_trace
    os.environ["LANGSMITH_TRACING"] = "true" if tracing else "false"
    if tracing:
        os.environ["LANGSMITH_PROJECT"] = ls_project
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    return (ls_project if tracing else None), tracing


async def main() -> None:
    """Replay every gold thread K times at one effort and write the transcript."""
    args = parse_args()
    effort = args.effort
    threads = [
        thread
        for thread in gold.load_threads(args.gold)
        if not args.thread or thread["id"] in args.thread
    ]

    branch = current_branch()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.out or str(
        EVAL_DIR / f"transcript_{branch.replace('/', '-')}_{effort}_{timestamp}.json"
    )
    ls_project, tracing = configure_tracing(args)

    print(
        f"\nbranch={branch!r}  model={settings.MODEL_URI!r}  effort={effort!r}\n"
        f"repeats={args.repeats}  threads={[thread['id'] for thread in threads]}\n"
        f"langsmith: {f'project={ls_project!r}' if tracing else 'disabled'}\n"
    )

    total_turns = sum(len(thread["turns"]) for thread in threads)
    total_runs = total_turns * args.repeats
    if args.dry_run:
        print("DRY RUN — no agent calls will be made.")
        for thread in threads:
            print(f"  {thread['id']:<16}: {len(thread['turns'])} turn(s)")
        print(
            f"\n  {len(threads)} thread(s) x {args.repeats} repeat(s) at effort {effort!r}"
            f"\n  total agent runs: {total_runs} (each = one live multi-step invocation)"
            f"\n  output would be:  {out_path}"
        )
        return

    agent = build_agent(effort)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_unit(thread: dict, repeat: int) -> dict:
        async with semaphore:
            unit = await replay_thread(agent, thread, effort, repeat, branch)
            print(f"[{thread['id']:<16} {effort} #{repeat}] {_unit_progress(unit)}")
            return unit

    units = await asyncio.gather(
        *(
            run_unit(thread, repeat)
            for thread in threads
            for repeat in range(args.repeats)
        )
    )

    transcript = {
        "branch": branch,
        "model": settings.MODEL_URI,
        "effort": effort,
        "repeats": args.repeats,
        "timestamp": timestamp,
        "gold_path": args.gold,
        "langsmith_project": ls_project,
        "system_prompt": SYSTEM_PROMPT,
        "units": list(units),
    }
    with open(out_path, "w") as file:
        json.dump(transcript, file, ensure_ascii=False, indent=2)
    print(f"\nwrote {out_path} ({len(units)} units)")


if __name__ == "__main__":
    asyncio.run(main())
