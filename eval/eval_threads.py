"""Multi-turn agent eval: replay conversation threads and score deterministic dimensions.

Replays each thread in `eval_gold.yaml` turn-by-turn on a shared `thread_id`
(InMemorySaver, so follow-ups see prior context), K times per thread, for the
currently checked-out branch + given settings. Scores four deterministic
dimensions per turn from the agent's structured response fields (the purpose
of this eval is to check those fields are filled with the expected values):

  action: query-vs-clarify, from whether the agent filled the sql_queries field
  source: data_sources (its table UUIDs resolved to gcp_ids) match the gold
  period: temporal_coverage matches the gold rule (any/latest/range/match_previous/exact)
  completed: the turn finished without errors

It writes a rich JSON (full per-turn records: structured response, response text,
model-call count) so the later judge eval and the A-vs-B pass can score
the SAME transcripts without re-running the agent.

Example Usage:
    uv run python eval_threads.py --repeats 10 --temperature 0.0
    uv run python eval_threads.py --repeats 10 --threads <thread-id> --temperature 0.0

Each turn runs the real agent (live BQ + LLM) — mind the cost (sum of turns x K).
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
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

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import BDToolkit
from app.settings import settings

try:
    from app.agent.schemas import StructuredResponse
except ImportError as exc:
    raise NotImplementedError(
        "This eval only supports the structured-response agent"
    ) from exc

# Agent middleware tunables (kept in step with production config).
SUMMARIZE_TRIGGER_TOKENS = 500_000
SUMMARIZE_KEEP_TOKENS = 250_000
MODEL_CALL_RUN_LIMIT = 20

# Deterministic dimensions scored per turn, in display order.
SCORE_DIMENSIONS = ("action", "source", "period", "completed")


# =============================================================================
# Git and Agent setup
# =============================================================================
def current_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def build_agent(
    temperature: float, checkpointer: InMemorySaver, system_prompt: str
) -> CompiledStateGraph:
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
    return create_agent(
        model=model,
        tools=BDToolkit.get_tools(),
        system_prompt=system_prompt,
        middleware=middleware,
        checkpointer=checkpointer,
        response_format=StructuredResponse,
    )


# =============================================================================
# per-turn extraction from the agent's final state
# =============================================================================
def _empty_record(status: str, **extra) -> dict:
    """Base per-turn record for the non-`ok` cases (no_structured / error), with the
    scoring fields zeroed so downstream code can read them uniformly. `extra` overrides
    or adds fields (e.g. an `error` message, a recovered `response_text`)."""
    return {
        "status": status,
        "is_query": False,
        "tables": [],
        "table_period_ends": {},
        "structured": None,
        "response_text": None,
        "model_calls": None,
        "tools_used": [],
        **extra,
    }


def _last_ai_text(messages: list[AnyMessage]) -> str | None:
    for m in reversed(messages):
        if m.type == "ai":
            return m.text
    return None


def _tools_used(messages: list[AnyMessage]) -> set[str]:
    """Distinct tool names the agent called in THIS turn. The checkpointer returns
    the whole thread, so slice to the messages after the last human turn."""
    last_human = max(
        (i for i, m in enumerate(messages) if m.type == "human"), default=-1
    )
    return {
        tc["name"]
        for m in messages[last_human + 1 :]
        for tc in getattr(m, "tool_calls", None) or []
    }


def _tool_metadata(messages: list[AnyMessage]) -> tuple[dict[str, str], dict[str, str]]:
    """From the agent's get_table_details / get_dataset_details outputs, build
    (uuid -> gcp_id) to resolve data_sources, and (gcp_id -> period_end) for the
    `latest` period rule. Tool payloads are pydantic-serialized JSON so keys are
    guaranteed; a non-JSON body means that tool call errored, so skip it."""
    uuid_to_gcp: dict[str, str] = {}
    period_end: dict[str, str] = {}
    for m in messages:
        if m.type != "tool" or m.name not in (
            "get_table_details",
            "get_dataset_details",
        ):
            continue
        try:
            d = json.loads(m.content)
        except json.JSONDecodeError:
            continue
        if m.name == "get_table_details":
            uuid_to_gcp[d["id"]] = d["gcp_id"]
            if d["period_end"] is not None:
                period_end[d["gcp_id"]] = d["period_end"]
        else:  # get_dataset_details
            for t in d["tables"]:
                uuid_to_gcp[t["id"]] = t["gcp_id"]
    return uuid_to_gcp, period_end


def extract_turn(result: dict) -> dict:
    messages = result["messages"]
    structured: StructuredResponse | None = result.get("structured_response")

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
            uuid_to_gcp.get(d.table_id, d.table_id)
            for d in (structured.data_sources or [])
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
    }


# =============================================================================
# Deterministic scoring
# =============================================================================
def _lead_year(s) -> int | None:
    """Leading 4-digit year of a period value ('2015', '2015-03-01', '2026-05')."""
    m = re.match(r"\s*(\d{4})", str(s or ""))
    return int(m.group(1)) if m else None


def _tc_years(tc: dict | None) -> tuple[int | None, int | None]:
    """(start_year, end_year) from a temporal_coverage dict — the leading 4-digit
    year of period_start/period_end (handles '2015' and '2015-03-01' alike)."""
    if not tc:
        return None, None
    return _lead_year(tc.get("period_start")), _lead_year(tc.get("period_end"))


def _parse_period(s) -> tuple[str, str] | None:
    """(granularity, value) for a period string in its granularity's own format:
    'YYYY' -> ('year', ...), 'YYYY-MM' -> ('month', ...), 'YYYY-MM-DD' -> ('day', ...).
    Strict (re.fullmatch): a malformed value like '2026-5' returns None -> a miss, not
    something to silently repair."""
    s = str(s or "").strip().strip("'").strip('"')
    for gran, pat in (
        ("day", r"\d{4}-\d{2}-\d{2}"),
        ("month", r"\d{4}-\d{2}"),
        ("year", r"\d{4}"),
    ):
        if re.fullmatch(pat, s):
            return gran, s
    return None


def _score_period(rec: dict, gold: dict, prev_query: dict | None):
    """Score the period from the agent's reported `temporal_coverage` — normalized
    and independent of SQL surface form / temporal-column name. Returns [0,1] or None.
    """
    rule = gold.get("temporal")
    if gold["action"] != "query" or rule in (None, "none", "any", "", {}):
        return None
    ps, pe = _tc_years(rec["structured"]["temporal_coverage"])
    if ps is None or pe is None:
        return 0.0  # queried but reported no usable temporal_coverage
    if rule == "match_previous":
        if prev_query is None:
            return None
        pps, ppe = _tc_years(prev_query["structured"]["temporal_coverage"])
        if pps is None or ppe is None:
            return 0.0
        # The follow-up must stay WITHIN the previous turn's period. A justified
        # narrowing (e.g., the requested metric only exists for part of the range)
        # is fine; only drifting OUTSIDE the established window fails.
        return 1.0 if pps <= ps and pe <= ppe else 0.0
    if rule == "range":
        return 1.0 if pe > ps else 0.0
    if rule == "latest":
        # Expected = the table's period_end, matched EXACTLY at its own granularity:
        # '2026-05' must be reported as month 2026-05 (not year 2026), a full date to
        # the day. The model must emit the value in its granularity's format.
        queried = set(rec["tables"])
        cands = [v for k, v in rec["table_period_ends"].items() if k in queried]
        cands = cands or list(rec["table_period_ends"].values())
        targets = [p for v in cands if (p := _parse_period(v)) is not None]
        if not targets:
            return None  # can't determine the table's period_end from the trace
        target = max(targets, key=lambda p: p[1])  # latest period_end
        tc = rec["structured"]["temporal_coverage"] or {}
        start, end = (
            _parse_period(tc.get("period_start")),
            _parse_period(tc.get("period_end")),
        )
        matches = (
            start == target and end == target and tc.get("granularity") == target[0]
        )
        return 1.0 if matches else 0.0
    span = rule.get("exact", rule) if isinstance(rule, dict) else None
    if isinstance(span, dict) and "start" in span and "end" in span:
        return 1.0 if ps == int(span["start"]) and pe == int(span["end"]) else 0.0
    return None


def score_turn(rec: dict, gold: dict, prev_query: dict | None) -> dict:
    if rec["status"] == "no_structured":
        # agent returned no structured response -> the structured contract failed
        return {"action": False, "source": None, "period": None, "completed": False}

    observed = "query" if rec["is_query"] else "clarify"

    if gold["action"] == "query" and gold.get("tables"):
        exp, obs = set(gold["tables"]), set(rec["tables"])
        source = 1.0 if obs == exp else (0.5 if obs & exp else 0.0)
    else:
        source = None  # clarify turn, or gcp_ids not filled in yet
    return {
        "action": observed == gold["action"],
        "source": source,
        "period": _score_period(rec, gold, prev_query),
        "completed": rec["status"] == "ok",
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
    """Per-turn LangGraph config. Labels the run so it's findable in LangSmith;
    `thread_id` in metadata makes LangSmith group a replay's turns into one
    conversation."""
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
    return {"turn_index": turn_index, "user": user, "status": "skipped", "scores": {}}


async def replay_thread(
    agent: CompiledStateGraph,
    thread: dict,
    repeat: int,
    run_config: dict,
    branch: str,
    temperature: float,
    max_turns: int | None = None,
) -> list:
    thread_id = f"{thread['id']}-{repeat}"
    records, prev_query = [], None
    turns = thread["turns"] if max_turns is None else thread["turns"][:max_turns]

    for i, turn in enumerate(turns):
        cfg = _turn_run_config(
            run_config,
            thread_id=thread_id,
            thread=thread,
            repeat=repeat,
            turn_index=i,
            turn=turn,
            branch=branch,
            temperature=temperature,
        )

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": turn["user"]}]}, config=cfg
            )
            rec = extract_turn(result)
        except Exception as exc:
            rec = _empty_record(
                "error",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )

        rec["turn_index"], rec["user"] = i, turn["user"]
        rec["scores"] = (
            score_turn(rec, turn, prev_query) if rec["status"] != "error" else {}
        )
        records.append(rec)

        if rec["status"] == "error":  # broken state -> skip the rest of the thread
            records.extend(
                _skipped_record(j, turns[j]["user"]) for j in range(i + 1, len(turns))
            )
            break

        if rec["is_query"]:
            prev_query = rec

    return records


# =============================================================================
# Aggregation & Reporting
# =============================================================================
def _mark(scores: dict) -> str:
    def m(v):
        return (
            "·"
            if v is None
            else ("✓" if v is True or v == 1.0 else ("~" if v == 0.5 else "✗"))
        )

    return (
        f"a:{m(scores.get('action'))} "
        f"s:{m(scores.get('source'))} "
        f"p:{m(scores.get('period'))} "
        f"c:{m(scores.get('completed'))}"
    )


def _unit_line(turn_records: list) -> str:
    """One-line per-thread progress summary: each turn's marks, or its status."""
    return "  ".join(
        f"t{tr['turn_index']}:"
        + (_mark(tr["scores"]) if tr["status"] == "ok" else tr["status"].upper())
        for tr in turn_records
    )


def aggregate(units: list) -> dict:
    """Pass-rate per dimension over all scored turns (None = not applicable, skipped)."""
    agg = defaultdict(lambda: {"hit": 0.0, "n": 0})
    errors = skipped = no_structured = 0
    for u in units:
        for rec in u["turns"]:
            if rec["status"] == "skipped":
                skipped += 1
                continue
            if rec["status"] == "error":
                errors += 1
            elif rec["status"] == "no_structured":
                no_structured += 1
            for dim, val in (rec.get("scores") or {}).items():
                if val is None:
                    continue
                agg[dim]["hit"] += float(val)
                agg[dim]["n"] += 1
    summary = {
        d: {"rate": round(v["hit"] / v["n"], 3), "n": v["n"]} for d, v in agg.items()
    }
    summary["_errors"], summary["_skipped"], summary["_no_structured"] = (
        errors,
        skipped,
        no_structured,
    )
    return summary


def print_scorecard(summary: dict) -> None:
    print("\n=== Deterministic scorecard (pass-rate over applicable turns) ===")
    for dim in SCORE_DIMENSIONS:
        if dim in summary:
            print(f"  {dim:<10} {summary[dim]['rate']:.0%}  (n={summary[dim]['n']})")
    print(
        f"  errors={summary['_errors']}  skipped={summary['_skipped']}  "
        f"no_structured={summary['_no_structured']}"
    )


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="eval_gold.yaml")
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
    return parser.parse_args()


def configure_tracing(args: argparse.Namespace) -> tuple[str, bool]:
    """Forward LangSmith config from `settings` into os.environ (LangChain's tracer
    reads the env, but pydantic-settings only populates `settings`), isolating eval
    runs in their own project by default. Returns (project, enabled); must run before
    the agent does."""
    ls_project = args.langsmith_project or f"{settings.LANGSMITH_PROJECT}-eval"
    tracing = settings.LANGSMITH_TRACING and not args.no_trace
    os.environ["LANGSMITH_TRACING"] = "true" if tracing else "false"
    if tracing:
        os.environ["LANGSMITH_PROJECT"] = ls_project
        os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    return ls_project, tracing


async def main() -> None:
    args = parse_args()

    with open(args.gold) as f:
        gold = yaml.safe_load(f)
    threads = [t for t in gold if not args.thread or t["id"] in args.thread]

    temperature = (
        args.temperature if args.temperature is not None else settings.MODEL_TEMPERATURE
    )
    run_config: dict = {}
    if args.recursion_limit is not None:
        run_config["recursion_limit"] = args.recursion_limit

    branch = current_branch()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = args.out or str(
        Path(__file__).resolve().parent
        / f"thread_eval_{branch.replace('/', '-')}_temp{temperature}_{ts}.json"
    )

    ls_project, tracing = configure_tracing(args)

    print(
        f"\nbranch={branch!r}  model={settings.MODEL_URI!r}\n"
        f"temperature={temperature}  repeats={args.repeats}  threads={[t['id'] for t in threads]}\n"
        f"langsmith: project={ls_project!r}\n"
        if tracing
        else "langsmith: disabled\n"
    )

    system_prompt = SYSTEM_PROMPT.format(current_date=date.today().isoformat())
    agent = build_agent(temperature, InMemorySaver(), system_prompt)
    sem = asyncio.Semaphore(args.concurrency)
    work = [(t, r) for t in threads for r in range(args.repeats)]

    async def run_unit(thread, repeat):
        async with sem:
            turn_records = await replay_thread(
                agent, thread, repeat, run_config, branch, temperature, args.max_turns
            )
            print(f"[{thread['id']:<16} #{repeat}] {_unit_line(turn_records)}")
            return {"thread": thread["id"], "repeat": repeat, "turns": turn_records}

    units = await asyncio.gather(*(run_unit(t, r) for t, r in work))
    summary = aggregate(units)
    print_scorecard(summary)

    report = {
        "branch": branch,
        "langsmith_project": ls_project if tracing else None,
        "model": settings.MODEL_URI,
        "temperature": temperature,
        "recursion_limit": args.recursion_limit,
        "repeats": args.repeats,
        "timestamp": ts,
        "system_prompt": system_prompt,
        "summary": summary,
        "units": units,
    }

    with open(out_path, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
