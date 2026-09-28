"""The LLM judge shared by `eval_quality.py` and the LangSmith evaluators.

One judge call per answered turn. The verdict keeps separate scores (see
:class:`JudgeVerdict`): `correct` is anchored on the gold `reference_sql` result,
`grounded` on the data the agent itself retrieved, and the rest are rubric.

Unlike the other `eval.lib` modules, this one calls an LLM (the judge) and BigQuery
(to run a `reference_sql`).
"""

import hashlib
import json

from google.cloud import bigquery as bq
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent.tools.bigquery import MAX_BYTES_BILLED, _bq_client
from app.settings import settings
from eval.lib import gold

# The judge should be a stronger or different-family model than the agent (less
# self-preference).
DEFAULT_JUDGE_MODEL = "google_genai:gemini-3.1-pro-preview"

SCORE_FIELDS = (
    "correct",
    "grounded",
    "answers_question",
    "stated_assumption",
)
MAX_REFERENCE_ROWS = 50  # cap rows shown to the judge to bound tokens


# =============================================================================
# Judge schema & prompt
# =============================================================================
class JudgeVerdict(BaseModel):
    correct: bool | None = Field(
        default=None,
        description="(query turns WITH a reference result) Are the main figures/entities/ranking CONSISTENT with the REFERENCE RESULT? A different-but-valid slice/grouping/period the user didn't specify, extra metrics, rounding, or language differences are OK — fail ONLY on a direct contradiction (a different value for the same quantity, wrong entity, wrong ordering). null on ask turns AND when no reference result is provided (correctness can't be verified).",
    )
    grounded: bool = Field(
        description="Does every quantitative/factual claim trace to data the assistant ACTUALLY RETRIEVED (this turn OR an earlier turn of the conversation)? Check the prose against the ASSISTANT'S RETRIEVED DATA section (its own query results), NOT the reference — a value present there is grounded even if absent from the reference. Inventing numbers/trends/datasets, or asserting a value with no supporting query, fails. (ask: backed by real exploration, no assumed value)."
    )
    answers_question: bool = Field(
        description="Does the answer address what the user needed this turn? (query: answers the question; ask: does the right clarification/guidance per the note)."
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

If the type is `ask` (it should NOT query data; per the note, it should ask for the missing detail, OR explore the catalog and guide, OR report that the data is not available):
- correct: null (there is no data answer to verify).
- grounded: the answer does NOT invent datasets/tables/values. If it describes available data, that must be backed by real exploration — check the tools used (search_datasets/get_dataset_details/get_table_details). If it assumes a value the user did not provide (e.g., a specific município), grounded=false.
- answers_question: it does the right clarification/guidance per the note (e.g., asks which município; or describes what exists and suggests specific refinements), without having queried data.
- stated_assumption: null.

Be strict but fair: wording/rounding differences, extra valid metrics, and a differently-but-validly-scoped query are acceptable; a wrong value for the same quantity, wrong entities, claims absent from the assistant's retrieved data (fabrications), unrequested assumptions, or not doing what the turn required are failures. Give a one- to two-sentence rationale."""


# =============================================================================
# Reference SQL
# =============================================================================
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
        ref_block = "N/A — `ask` turn; there is no data answer to verify."
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
- follow_up_prompts: {structured.get("follow_up_prompts")}

# ASSISTANT'S RETRIEVED DATA (its OWN queries + results, this turn AND earlier turns of the conversation — anchor for `grounded`)
{agent_data_block}

# REFERENCE RESULT (gold, from the trusted query — anchor for `correct`)
{ref_block}"""


def build_tasks(
    transcript: dict, gold_index: dict, max_repeats: int | None
) -> list[dict]:
    """Build one judge task per answered (`ok`) turn that has a gold expectation.

    Args:
        transcript (dict): A runner transcript (only its `units` are read).
        gold_index (dict): (thread_id, turn_index) -> gold turn, from gold.index_turns().
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
            gold_turn = gold_index.get((unit["thread"], turn["turn_index"]))
            # Judge any turn the agent actually answered (query or ask) for which
            # we have a gold expectation. Query turns carry a reference_sql to anchor
            # correctness; ask turns are judged on the rubric only.
            if gold_turn is None or turn["status"] != "ok":
                continue
            tasks.append(
                {
                    "thread": unit["thread"],
                    "repeat": unit["repeat"],
                    "turn_index": turn["turn_index"],
                    # The older gold file still says `clarify`; the rubric names it `ask`.
                    "turn_type": gold.ACTION_QUERY
                    if gold_turn["action"] == gold.ACTION_QUERY
                    else gold.ACTION_ASK,
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


async def ajudge(judge, task: dict) -> dict:
    """Judge one task and return its verdict record; a failure is recorded, not raised.

    Args:
        judge: The structured-output judge model from :func:`build_judge`.
        task: A task from :func:`build_tasks`, with `reference_rows` set.

    Returns:
        `{"verdict": dict | None, "input_tokens", "output_tokens"}`, plus `"error"` when
        the call failed or the verdict did not parse.
    """
    try:
        judge_output = await judge.ainvoke(
            [SystemMessage(JUDGE_SYSTEM), HumanMessage(render_turn(task))]
        )
    except Exception as exc:
        return {"verdict": None, "error": f"{type(exc).__name__}: {exc}"}
    verdict: JudgeVerdict | None = judge_output["parsed"]
    usage = getattr(judge_output["raw"], "usage_metadata", None) or {}
    record = {
        "verdict": verdict.model_dump() if verdict is not None else None,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }
    if verdict is None:
        record["error"] = f"parse: {judge_output.get('parsing_error')}"
    return record
