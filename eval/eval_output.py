"""Multi-turn agent eval: replay conversation threads and score deterministic dimensions.

Replays each thread in `eval_gold.yaml` turn-by-turn on a shared `thread_id`
(InMemorySaver, so follow-ups see prior context), K times per thread, for the
currently checked-out branch + given settings. Scores five deterministic
dimensions per turn from the agent's structured response fields (the purpose
of this eval is to check those fields are filled with the expected values):

  action: query-vs-clarify, from whether the agent filled the sql_queries field
  source: data_sources (its table UUIDs resolved to gcp_ids) match the gold
  period: temporal_coverage matches the gold rule (any/latest/range/match_previous/exact)
  format_ok: the prose carries no Markdown ATX headers (the one hard-forbidden format rule)
  completed: the turn finished without errors

It writes a rich JSON (full per-turn records: structured response, response text,
model-call count, and the agent's OWN executed queries + their result rows) so the
later judge eval and the A-vs-B pass can score the SAME transcripts without re-running
the agent. Persisting the query results lets the judge check grounding against what the
agent actually retrieved — not only against the gold reference query.

On the free-text baseline (e.g. the main branch, no structured output), pass
--no-structured: the agent is built without a response_format, the answer is read from
the final message, and only action/format_ok/completed are scored (source/period need the
structured fields). The resulting transcript is still judgeable by eval_quality.

Pipeline: this is step 1 — the ONLY script that runs the agent. It writes the transcript
JSON that the three post-hoc scorers read (no agent; run them in any order):
  eval_quality.py       answer quality vs the gold — LLM judge + live reference_sql
  eval_faithfulness.py  structured output's self-consistency — gold-free, no LLM/BQ
  eval_queries.py       source/period from the executed SQL vs the gold — no LLM/BQ

Example Usage:
    uv run eval/eval_output.py --repeats 10 --temperature 0.0
    uv run eval/eval_output.py --thread <thread-id> --dry-run
    uv run eval/eval_output.py --no-structured   # baseline: free-text agent (e.g. main)

Each turn runs the real agent (live BQ + LLM) — mind the cost (sum of turns x K).
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import yaml
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
)
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

# Make the repo root importable so `app` resolves whether this file is run as a
# module (python -m eval.eval_output) or directly (python eval/eval_output.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.prompts import SYSTEM_PROMPT  # noqa: E402
from app.agent.tools import BDToolkit  # noqa: E402
from app.settings import settings  # noqa: E402

# Structured output only exists on the structured-response branch. On main (no structured
# output) this import fails; run with --no-structured, which builds the agent without a
# response_format and reads the answer from the final message instead.
try:
    from app.agent.schemas import StructuredResponse  # noqa: E402
except ImportError:
    StructuredResponse = None

# This script's folder — gold input and result files default here, so the eval
# works regardless of the current working directory.
EVAL_DIR = Path(__file__).resolve().parent

# Agent middleware tunables (kept in step with production config).
SUMMARIZE_TRIGGER_TOKENS = 500_000
SUMMARIZE_KEEP_TOKENS = 250_000
MODEL_CALL_RUN_LIMIT = 20

# Deterministic dimensions scored per turn, in display order (matches _mark and score_turn).
SCORE_DIMENSIONS = ("action", "source", "period", "format_ok", "completed")

# format_ok: the one prose-format rule the system prompt hard-forbids (no Markdown ATX
# headers) is mechanical, so it's scored here rather than spending a judge call on it.
_CODE_FENCE_RE = re.compile(r"^\s*```")
_MD_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}(?:\s|$)")

# Cap rows persisted per executed query, so the transcript (and the judge's view of the
# agent's own data) stays bounded; the true row count is kept alongside the capped rows.
QUERY_ROWS_CAP = 50


# =============================================================================
# Git and Agent setup
# =============================================================================
def current_branch() -> str:
    """The current git branch name.

    Returns:
        str: The abbreviated branch name, or "unknown" if git can't be queried.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def build_agent(
    temperature: float,
    checkpointer: InMemorySaver,
    system_prompt: str,
    structured: bool = True,
) -> CompiledStateGraph:
    """Build the agent under test, mirroring the production configuration.

    Args:
        temperature (float): Sampling temperature for the model.
        checkpointer (InMemorySaver): Per-thread conversation state store.
        system_prompt (str): The (already formatted) system prompt.
        structured (bool): When True, attach the StructuredResponse response_format;
            when False (the free-text baseline), the agent answers in its final message.

    Returns:
        CompiledStateGraph: The compiled agent graph ready for `ainvoke`.
    """
    model = init_chat_model(
        model=settings.MODEL_URI,
        temperature=temperature,
        credentials=settings.GOOGLE_CREDENTIALS,
        thinking_level=settings.THINKING_LEVEL,
        include_thoughts=True,
    )
    middleware = [
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", SUMMARIZE_TRIGGER_TOKENS),
            keep=("tokens", SUMMARIZE_KEEP_TOKENS),
        ),
        ModelCallLimitMiddleware(
            run_limit=MODEL_CALL_RUN_LIMIT,
            exit_behavior="end",
        ),
    ]
    # No response_format on the free-text baseline (--no-structured); the agent answers
    # in the final message instead of a StructuredResponse.
    response_format = {"response_format": StructuredResponse} if structured else {}
    return create_agent(
        model=model,
        tools=BDToolkit.get_tools(),
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=checkpointer,
        **response_format,
    )


# =============================================================================
# Per-turn extraction from the agent's final state
# =============================================================================
def _empty_record(status: str, **extra) -> dict:
    """Base per-turn record for the non-`ok` cases (no_structured / error).

    The scoring fields are zeroed so downstream code can read them uniformly.

    Args:
        status (str): The turn status ("no_structured" or "error").
        **extra: Fields to override or add (e.g. an `error` message, a recovered
            `response_text`).

    Returns:
        dict: The per-turn record with default (empty) fields plus `extra`.
    """
    return {
        "status": status,
        "is_query": False,
        "tables": [],
        "table_period_ends": {},
        "structured": None,
        "response_text": None,
        "model_calls": None,
        "tools_used": [],
        "queries": [],
        **extra,
    }


def _last_ai_text(messages: list[AnyMessage]) -> str | None:
    """The text of the last AI message (the free-text answer).

    Args:
        messages (list[AnyMessage]): The full thread's messages.

    Returns:
        str | None: The most recent AI message's text, or None if there is none.
    """
    for message in reversed(messages):
        if message.type == "ai":
            return message.text
    return None


def _tools_used(messages: list[AnyMessage]) -> set[str]:
    """Distinct tool names the agent called in THIS turn.

    The checkpointer returns the whole thread, so this slices to the messages after
    the last human turn.

    Args:
        messages (list[AnyMessage]): The full thread's messages.

    Returns:
        set[str]: The distinct tool names invoked this turn.
    """
    last_human_index = max(
        (i for i, message in enumerate(messages) if message.type == "human"), default=-1
    )
    return {
        tool_call["name"]
        for message in messages[last_human_index + 1 :]
        for tool_call in getattr(message, "tool_calls", None) or []
    }


def _tool_metadata(messages: list[AnyMessage]) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve table identity/coverage metadata from the agent's exploration calls.

    From the agent's get_table_details / get_dataset_details outputs, build
    (uuid -> gcp_id) to resolve data_sources, and (gcp_id -> period_end) for the
    `latest` period rule. A failed tool call serializes to a ToolError object
    ({"status": "error", ...}) rather than the expected payload, so skip those.

    Args:
        messages (list[AnyMessage]): The full thread's messages.

    Returns:
        tuple[dict[str, str], dict[str, str]]: (uuid_to_gcp, gcp_id_to_period_end).
    """
    uuid_to_gcp: dict[str, str] = {}
    period_end: dict[str, str] = {}
    for message in messages:
        if message.type != "tool" or message.name not in (
            "get_table_details",
            "get_dataset_details",
        ):
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        # A failed tool call serializes to a ToolError object ({"status": "error", ...}),
        # which lacks the id/tables keys — skip it. A success payload has every key.
        if payload.get("status") == "error":
            continue
        if message.name == "get_table_details":
            uuid_to_gcp[payload["id"]] = payload["gcp_id"]
            if payload["period_end"] is not None:
                period_end[payload["gcp_id"]] = payload["period_end"]
        else:  # get_dataset_details
            for table in payload["tables"]:
                uuid_to_gcp[table["id"]] = table["gcp_id"]
    return uuid_to_gcp, period_end


def _truncate_rows(content: str) -> tuple[int | None, list | None, str | None]:
    """Parse an execute_bigquery_sql result body into (row_count, capped_rows, message).

    A JSON array -> (total count, first QUERY_ROWS_CAP rows, None). Anything else — the
    '0 rows' notice, a plain error string, or a JSON object (a serialized ToolError from a
    failed query) -> (None, None, the raw string), so a failed query is kept as a message,
    not a crash.

    Args:
        content (str): The raw tool-message content.

    Returns:
        tuple[int | None, list | None, str | None]: (row_count, capped_rows, message).
    """
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None, None, content
    if not isinstance(parsed, list):
        return None, None, content
    return len(parsed), parsed[:QUERY_ROWS_CAP], None


def _executed_queries(messages: list[AnyMessage]) -> list[dict]:
    """The execute_bigquery_sql calls made in THIS turn, each paired with its result rows.

    This is the agent's OWN retrieved data. It lets the judge score `grounded` against
    what the agent actually queried (not only the gold reference). Sliced to the messages
    after the last human turn, like _tools_used.

    Args:
        messages (list[AnyMessage]): The full thread's messages.

    Returns:
        list[dict]: One {sql, status, row_count, rows, message} per execute_bigquery_sql
            call made this turn.
    """
    last_human_index = max(
        (i for i, message in enumerate(messages) if message.type == "human"), default=-1
    )
    turn_messages = messages[last_human_index + 1 :]
    sql_by_id = {
        tool_call["id"]: tool_call["args"].get("sql_query")
        for message in turn_messages
        for tool_call in getattr(message, "tool_calls", None) or []
        if tool_call["name"] == "execute_bigquery_sql"
    }
    queries = []
    for tool_message in turn_messages:
        if tool_message.type == "tool" and tool_message.name == "execute_bigquery_sql":
            row_count, rows, message = _truncate_rows(tool_message.content)
            queries.append(
                {
                    "sql": sql_by_id.get(tool_message.tool_call_id),
                    "status": tool_message.status,
                    "row_count": row_count,
                    "rows": rows,
                    "message": message,
                }
            )
    return queries


def extract_turn(result: dict, structured_mode: bool = True) -> dict:
    """Build a per-turn record from the agent's final state.

    Args:
        result (dict): The agent's `ainvoke` result (its `messages` and, in structured
            mode, its `structured_response`).
        structured_mode (bool): When False (the free-text baseline), read the answer from
            the final message and leave source/period unscored; when True, read the
            structured fields.

    Returns:
        dict: The per-turn record (status "ok", or "no_structured" when the structured
            agent produced no structured response).
    """
    messages = result["messages"]

    # --no-structured (e.g. the main branch): the answer is the final AI message and there
    # are no structured fields. is_query is derived from the tool trace (did it run a query
    # this turn); source/period are left unscored (no data_sources / temporal_coverage).
    if not structured_mode:
        queries = _executed_queries(messages)
        # period_end (per gcp_id) still comes from the trace's get_table_details outputs,
        # so a trace-based eval (eval_queries) can do the `latest` year check for main too.
        _, table_period_ends = _tool_metadata(messages)
        return {
            "status": "ok",
            "is_query": bool(queries),
            "tables": [],
            "table_period_ends": table_period_ends,
            "structured": None,
            "response_text": _last_ai_text(messages),
            "model_calls": result.get("run_model_call_count"),
            "tools_used": sorted(_tools_used(messages)),
            "queries": queries,
        }

    structured = result.get("structured_response")

    # This eval targets the structured-output agent: a missing structured response
    # is a hard fail for the turn (not a crash, not a skip).
    if structured is None:
        return _empty_record(
            "no_structured",
            response_text=_last_ai_text(messages),
            model_calls=result.get("run_model_call_count"),
        )

    uuid_to_gcp, table_period_ends = _tool_metadata(messages)
    # Tables the agent REPORTED in data_sources, UUID -> gcp_id (unmapped UUIDs are
    # kept as-is so they fail to match the gold and surface the problem).
    data_source_tables = sorted(
        {
            uuid_to_gcp.get(source.table_id, source.table_id)
            for source in (structured.data_sources or [])
        }
    )
    return {
        "status": "ok",
        # query vs clarify, from whether the agent filled the sql_queries field.
        "is_query": bool(structured.sql_queries),
        "tables": data_source_tables,
        "table_period_ends": table_period_ends,
        "structured": structured.model_dump(mode="json"),
        "response_text": structured.response,
        "model_calls": result.get("run_model_call_count"),
        "tools_used": sorted(_tools_used(messages)),
        "queries": _executed_queries(messages),
    }


# =============================================================================
# Deterministic scoring
# =============================================================================
def _lead_year(value) -> int | None:
    """The leading 4-digit year of a period value ('2015', '2015-03-01', '2026-05').

    Args:
        value: The period value (str or None).

    Returns:
        int | None: The leading year, or None if the value has no leading 4 digits.
    """
    match = re.match(r"\s*(\d{4})", str(value or ""))
    return int(match.group(1)) if match else None


def _tc_years(temporal_coverage: dict | None) -> tuple[int | None, int | None]:
    """The (start_year, end_year) of a temporal_coverage dict.

    Uses the leading 4-digit year of period_start/period_end (handles '2015' and
    '2015-03-01' alike).

    Args:
        temporal_coverage (dict | None): The reported temporal_coverage.

    Returns:
        tuple[int | None, int | None]: (start_year, end_year), each None when absent.
    """
    if not temporal_coverage:
        return None, None
    return (
        _lead_year(temporal_coverage.get("period_start")),
        _lead_year(temporal_coverage.get("period_end")),
    )


def _parse_period(period) -> tuple[str, str] | None:
    """Split a period value into (granularity, value) in its granularity's own format.

    'YYYY' -> ('year', ...), 'YYYY-MM' -> ('month', ...), 'YYYY-MM-DD' -> ('day', ...).
    Strict (re.fullmatch): a malformed value like '2026-5' returns None -> a miss, not
    something to silently repair.

    Args:
        period: The period value (str or None).

    Returns:
        tuple[str, str] | None: (granularity, value), or None when malformed.
    """
    period = str(period or "").strip().strip("'").strip('"')
    for granularity, pattern in (
        ("day", r"\d{4}-\d{2}-\d{2}"),
        ("month", r"\d{4}-\d{2}"),
        ("year", r"\d{4}"),
    ):
        if re.fullmatch(pattern, period):
            return granularity, period
    return None


def _score_period(record: dict, gold: dict, prev_query: dict | None):
    """Score the period from the agent's reported `temporal_coverage`.

    Normalized and independent of SQL surface form / temporal-column name.

    Args:
        record (dict): The per-turn record (its `structured`, `tables`, `table_period_ends`).
        gold (dict): The gold turn (its `action` and `temporal` rule).
        prev_query (dict | None): The previous query turn's record, for `match_previous`.

    Returns:
        float | None: Score in [0, 1], or None when the rule doesn't apply / can't be checked.
    """
    rule = gold.get("temporal")
    if gold["action"] != "query" or rule in (None, "none", "any", "", {}):
        return None
    start_year, end_year = _tc_years(record["structured"]["temporal_coverage"])
    if start_year is None or end_year is None:
        return 0.0  # queried but reported no usable temporal_coverage
    if rule == "match_previous":
        if prev_query is None:
            return None
        prev_start_year, prev_end_year = _tc_years(
            prev_query["structured"]["temporal_coverage"]
        )
        if prev_start_year is None or prev_end_year is None:
            return 0.0
        # The follow-up must stay WITHIN the previous turn's period. A justified
        # narrowing (e.g., the requested metric only exists for part of the range)
        # is fine; only drifting OUTSIDE the established window fails.
        return (
            1.0 if prev_start_year <= start_year and end_year <= prev_end_year else 0.0
        )
    if rule == "range":
        return 1.0 if end_year > start_year else 0.0
    if rule == "latest":
        # Expected = the table's period_end, matched EXACTLY at its own granularity:
        # '2026-05' must be reported as month 2026-05 (not year 2026), a full date to
        # the day. The model must emit the value in its granularity's format.
        queried_tables = set(record["tables"])
        candidates = [
            period_value
            for gcp_id, period_value in record["table_period_ends"].items()
            if gcp_id in queried_tables
        ]
        candidates = candidates or list(record["table_period_ends"].values())
        targets = [
            parsed
            for period_value in candidates
            if (parsed := _parse_period(period_value)) is not None
        ]
        if not targets:
            return None  # can't determine the table's period_end from the trace
        target = max(targets, key=lambda parsed: parsed[1])  # latest period_end
        temporal_coverage = record["structured"]["temporal_coverage"] or {}
        start, end = (
            _parse_period(temporal_coverage.get("period_start")),
            _parse_period(temporal_coverage.get("period_end")),
        )
        matches = (
            start == target
            and end == target
            and temporal_coverage.get("granularity") == target[0]
        )
        return 1.0 if matches else 0.0
    span = rule.get("exact", rule) if isinstance(rule, dict) else None
    if isinstance(span, dict) and "start" in span and "end" in span:
        return (
            1.0
            if start_year == int(span["start"]) and end_year == int(span["end"])
            else 0.0
        )
    return None


def _prose_format_ok(text: str | None) -> bool:
    """Whether the prose obeys the one hard-forbidden format rule: no Markdown ATX headers.

    Header lines (`#`..`######`) inside fenced code blocks are ignored (a `#` comment in a
    code sample is not a section title).

    Args:
        text (str | None): The prose answer.

    Returns:
        bool: True if the prose contains no Markdown ATX header outside a code fence.
    """
    in_code_fence = False
    for line in (text or "").splitlines():
        if _CODE_FENCE_RE.match(line):
            in_code_fence = not in_code_fence
            continue
        if not in_code_fence and _MD_HEADER_RE.match(line):
            return False
    return True


def score_turn(record: dict, gold: dict, prev_query: dict | None) -> dict:
    """Score one turn's deterministic dimensions against the gold expectation.

    Args:
        record (dict): The per-turn record from extract_turn / _empty_record.
        gold (dict): The gold turn (its `action`, `tables`, `temporal`).
        prev_query (dict | None): The previous query turn's record, for `match_previous`.

    Returns:
        dict: The per-dimension scores (True/False/float/None) for action, source, period,
            format_ok, completed.
    """
    if record["status"] == "no_structured":
        # agent returned no structured response -> the structured contract failed
        return {
            "action": False,
            "source": None,
            "period": None,
            "format_ok": None,
            "completed": False,
        }

    # --no-structured mode: an ok turn with no structured fields (structured is None only
    # here; a structured-mode failure has status "no_structured"). Only action (from the
    # tool trace), format_ok (prose) and completed are scoreable; source/period need the
    # data_sources / temporal_coverage fields the agent doesn't produce.
    if record["structured"] is None:
        return {
            "action": ("query" if record["is_query"] else "clarify") == gold["action"],
            "source": None,
            "period": None,
            "format_ok": _prose_format_ok(record["response_text"]),
            "completed": record["status"] == "ok",
        }

    observed = "query" if record["is_query"] else "clarify"

    if gold["action"] == "query" and gold.get("tables"):
        expected_tables, observed_tables = set(gold["tables"]), set(record["tables"])
        source = (
            1.0
            if observed_tables == expected_tables
            else (0.5 if observed_tables & expected_tables else 0.0)
        )
    else:
        source = None  # clarify turn, or gcp_ids not filled in yet
    return {
        "action": observed == gold["action"],
        "source": source,
        "period": _score_period(record, gold, prev_query),
        "format_ok": _prose_format_ok(record["structured"]["response"]),
        "completed": record["status"] == "ok",
    }


# =============================================================================
# Thread replay
# =============================================================================
def _turn_run_config(
    base: dict,
    *,
    thread_id: str,
    thread: dict,
    repeat: int,
    turn_index: int,
    turn: dict,
    branch: str,
    temperature: float,
) -> dict:
    """Build the per-turn LangGraph config.

    Labels the run so it's findable in LangSmith; `thread_id` in metadata makes LangSmith
    group a replay's turns into one conversation.

    Args:
        base (dict): The shared base run config (e.g. an optional recursion_limit).
        thread_id (str): The per-replay thread id (thread + repeat).
        thread (dict): The gold thread being replayed.
        repeat (int): The repeat index.
        turn_index (int): The zero-based turn index within the thread.
        turn (dict): The gold turn (for the expected_action / expected_temporal tags).
        branch (str): The checked-out git branch.
        temperature (float): The sampling temperature in effect.

    Returns:
        dict: The LangGraph run config for this turn.
    """
    return {
        **base,
        "configurable": {"thread_id": thread_id},
        "run_name": f"{thread['id']}#{repeat}-t{turn_index}",
        "tags": [
            f"branch:{branch}",
            f"temp:{temperature}",
            f"thread:{thread['id']}",
        ],
        "metadata": {
            "thread_id": thread_id,
            "eval_thread": thread["id"],
            "repeat": repeat,
            "turn_index": turn_index,
            "branch": branch,
            "temperature": temperature,
            "expected_action": turn["action"],
            "expected_temporal": turn.get("temporal"),
        },
    }


def _skipped_record(turn_index: int, user: str) -> dict:
    """A placeholder record for a turn that never ran (the thread was truncated earlier).

    Args:
        turn_index (int): The zero-based turn index.
        user (str): The user message for the skipped turn.

    Returns:
        dict: The skipped-turn record.
    """
    return {"turn_index": turn_index, "user": user, "status": "skipped", "scores": {}}


async def replay_thread(
    agent: CompiledStateGraph,
    thread: dict,
    repeat: int,
    run_config: dict,
    branch: str,
    temperature: float,
    max_turns: int | None = None,
    structured: bool = True,
) -> list:
    """Replay one thread turn-by-turn on a shared thread_id and score each turn.

    Args:
        agent (CompiledStateGraph): The compiled agent to invoke.
        thread (dict): The gold thread (its `id` and `turns`).
        repeat (int): The repeat index for this replay.
        run_config (dict): The shared base run config.
        branch (str): The checked-out git branch (for run labels).
        temperature (float): The sampling temperature in effect.
        max_turns (int | None): Run only the first N turns (a prefix), or all when None.
        structured (bool): Whether the agent produces structured output.

    Returns:
        list: One record per turn (ok / no_structured / error / skipped).
    """
    thread_id = f"{thread['id']}-{repeat}"
    records, prev_query = [], None
    turns = thread["turns"] if max_turns is None else thread["turns"][:max_turns]

    for turn_index, turn in enumerate(turns):
        turn_config = _turn_run_config(
            run_config,
            thread_id=thread_id,
            thread=thread,
            repeat=repeat,
            turn_index=turn_index,
            turn=turn,
            branch=branch,
            temperature=temperature,
        )

        # Separate the two failure modes: an ainvoke failure means the agent/thread state
        # may be broken (skip the rest of the thread), whereas an extract_turn failure is
        # OUR parsing bug on an otherwise-successful run (record it, but keep going —
        # the thread state is intact, so follow-ups can still run).
        agent_failed = False
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": turn["user"]}]},
                config=turn_config,
            )
        except Exception as exc:
            agent_failed = True
            record = _empty_record(
                "error",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
        else:
            try:
                record = extract_turn(result, structured)
            except Exception as exc:
                record = _empty_record(
                    "error",
                    error=f"extract_turn: {type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )

        record["turn_index"], record["user"] = turn_index, turn["user"]
        record["scores"] = (
            score_turn(record, turn, prev_query) if record["status"] != "error" else {}
        )
        records.append(record)

        if agent_failed:  # broken agent state -> skip the rest of the thread
            records.extend(
                _skipped_record(j, turns[j]["user"])
                for j in range(turn_index + 1, len(turns))
            )
            break

        if record["is_query"]:
            prev_query = record

    return records


# =============================================================================
# Aggregation & Reporting
# =============================================================================
def _mark(scores: dict) -> str:
    """Render a turn's scores as a compact one-line marks string (a:… s:… …).

    Args:
        scores (dict): The per-dimension scores for one turn.

    Returns:
        str: The marks line, e.g. "a:✓ s:~ p:✗ f:✓ c:✓".
    """

    def symbol(value):
        """Map a single score value to its display symbol.

        Args:
            value: True/False/float/None score.

        Returns:
            str: "·" (n/a), "✓" (pass), "~" (partial), or "✗" (fail).
        """
        return (
            "·"
            if value is None
            else (
                "✓" if value is True or value == 1.0 else ("~" if value == 0.5 else "✗")
            )
        )

    return (
        f"a:{symbol(scores.get('action'))} "
        f"s:{symbol(scores.get('source'))} "
        f"p:{symbol(scores.get('period'))} "
        f"f:{symbol(scores.get('format_ok'))} "
        f"c:{symbol(scores.get('completed'))}"
    )


def _unit_line(turn_records: list) -> str:
    """A one-line per-thread progress summary: each turn's marks, or its status.

    Args:
        turn_records (list): The records for one replayed thread.

    Returns:
        str: The joined per-turn summary.
    """
    return "  ".join(
        f"t{record['turn_index']}:"
        + (
            _mark(record["scores"])
            if record["status"] == "ok"
            else record["status"].upper()
        )
        for record in turn_records
    )


def aggregate(units: list) -> dict:
    """Compute the pass-rate per dimension over all scored turns.

    None-valued scores (not applicable / skipped) are excluded from the rate.

    Args:
        units (list): The per-thread replay units (each with its `turns`).

    Returns:
        dict: Per-dimension {"rate", "n"} plus `_errors` / `_skipped` / `_no_structured` counts.
    """
    per_dimension = defaultdict(lambda: {"hit": 0.0, "n": 0})
    errors = skipped = no_structured = 0
    for unit in units:
        for record in unit["turns"]:
            if record["status"] == "skipped":
                skipped += 1
                continue
            if record["status"] == "error":
                errors += 1
            elif record["status"] == "no_structured":
                no_structured += 1
            for dimension, value in (record.get("scores") or {}).items():
                if value is None:
                    continue
                per_dimension[dimension]["hit"] += float(value)
                per_dimension[dimension]["n"] += 1
    summary = {
        dimension: {"rate": round(counts["hit"] / counts["n"], 3), "n": counts["n"]}
        for dimension, counts in per_dimension.items()
    }
    summary["_errors"], summary["_skipped"], summary["_no_structured"] = (
        errors,
        skipped,
        no_structured,
    )
    return summary


def print_scorecard(summary: dict) -> None:
    """Print the deterministic scorecard (pass-rate per dimension + status counts).

    Args:
        summary (dict): The aggregate() result.

    Returns:
        None
    """
    print("\n=== Deterministic scorecard (pass-rate over applicable turns) ===")
    for dimension in SCORE_DIMENSIONS:
        if dimension in summary:
            print(
                f"  {dimension:<10} {summary[dimension]['rate']:.0%}  (n={summary[dimension]['n']})"
            )
    print(
        f"  errors={summary['_errors']}  skipped={summary['_skipped']}  "
        f"no_structured={summary['_no_structured']}"
    )


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default=str(EVAL_DIR / "eval_gold.yaml"))
    parser.add_argument("--repeats", type=int, default=3, help="Replays per thread")
    parser.add_argument(
        "--thread", action="append", help="Only this thread id (repeatable)"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Run only the first N turns of each thread (a prefix — follow-ups need prior context)",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=None,
        help="Omit to match production (LangGraph default 25)",
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
        "--no-structured",
        action="store_true",
        help="Eval a free-text agent (e.g. the main branch): build without response_format, "
        "read the answer from the final message. source/period are left unscored.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run plan (threads/turns/repeats, total agent runs) and exit without calling the agent",
    )
    return parser.parse_args()


def configure_tracing(args: argparse.Namespace) -> tuple[str, bool]:
    """Forward LangSmith config from `settings` into os.environ.

    LangChain's tracer reads the env, but pydantic-settings only populates `settings`.
    This isolates eval runs in their own project by default. Must run before the agent does.

    Args:
        args (argparse.Namespace): The parsed CLI args (its langsmith_project / no_trace).

    Returns:
        tuple[str, bool]: (langsmith_project, tracing_enabled).
    """
    ls_project = args.langsmith_project or f"{settings.LANGSMITH_PROJECT}-eval"
    tracing = settings.LANGSMITH_TRACING and not args.no_trace
    os.environ["LANGSMITH_TRACING"] = "true" if tracing else "false"
    if tracing:
        os.environ["LANGSMITH_PROJECT"] = ls_project
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    return ls_project, tracing


async def main() -> None:
    """Parse args, replay every gold thread K times, score each turn, and write the report."""
    args = parse_args()

    structured_mode = not args.no_structured
    if structured_mode and StructuredResponse is None:
        raise SystemExit(
            "No StructuredResponse importable (are you on the main branch?). "
            "Re-run with --no-structured to eval a free-text agent."
        )

    with open(args.gold) as f:
        gold = yaml.safe_load(f)
    threads = [
        thread for thread in gold if not args.thread or thread["id"] in args.thread
    ]

    temperature = (
        args.temperature if args.temperature is not None else settings.MODEL_TEMPERATURE
    )
    run_config: dict = {}
    if args.recursion_limit is not None:
        run_config["recursion_limit"] = args.recursion_limit

    branch = current_branch()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.out or str(
        EVAL_DIR
        / f"thread_eval_{branch.replace('/', '-')}_temp{temperature}_{timestamp}.json"
    )

    ls_project, tracing = configure_tracing(args)

    print(
        f"\nbranch={branch!r}  model={settings.MODEL_URI!r}  structured_output={structured_mode}\n"
        f"temperature={temperature}  repeats={args.repeats}  threads={[thread['id'] for thread in threads]}\n"
        f"langsmith: {f'project={ls_project!r}' if tracing else 'disabled'}\n"
    )

    if args.dry_run:
        max_turns = args.max_turns
        turn_counts = {
            thread["id"]: min(len(thread["turns"]), max_turns)
            if max_turns
            else len(thread["turns"])
            for thread in threads
        }
        total = args.repeats * sum(turn_counts.values())
        print("DRY RUN — no agent calls will be made.")
        for thread_id, count in turn_counts.items():
            print(f"  {thread_id:<16} {count} turn(s) x {args.repeats} repeat(s)")
        print(
            f"\n  total agent runs: {total} (each = one live multi-step agent invocation)"
        )
        print(f"  output would be:  {out_path}")
        return

    system_prompt = SYSTEM_PROMPT.format(current_date=date.today().isoformat())
    agent = build_agent(temperature, InMemorySaver(), system_prompt, structured_mode)
    semaphore = asyncio.Semaphore(args.concurrency)
    work = [(thread, repeat) for thread in threads for repeat in range(args.repeats)]

    async def run_unit(thread, repeat):
        async with semaphore:
            turn_records = await replay_thread(
                agent,
                thread,
                repeat,
                run_config,
                branch,
                temperature,
                args.max_turns,
                structured_mode,
            )
            print(f"[{thread['id']:<16} #{repeat}] {_unit_line(turn_records)}")
            return {"thread": thread["id"], "repeat": repeat, "turns": turn_records}

    units = await asyncio.gather(*(run_unit(thread, repeat) for thread, repeat in work))
    summary = aggregate(units)
    print_scorecard(summary)

    report = {
        "branch": branch,
        "structured_output": structured_mode,
        "langsmith_project": ls_project if tracing else None,
        "model": settings.MODEL_URI,
        "temperature": temperature,
        "recursion_limit": args.recursion_limit,
        "repeats": args.repeats,
        "timestamp": timestamp,
        "system_prompt": system_prompt,
        "summary": summary,
        "units": units,
    }

    with open(out_path, "w") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
