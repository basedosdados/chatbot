"""LangSmith online code rule: gold-free checks on one production agent trace.

Paste this whole file into a LangSmith code evaluator rule on the production project.
The rule sandbox cannot import this repo, so the file uses the standard library only
and copies its logic from the offline harness:

    online_tool_error_rate     runner._tool_calls + score_tool_error_rate (lower is better)
    online_within_call_limit   runner._hit_call_limit
    online_query_has_sources   score_deterministic.score_answer
    online_response_nonempty   score_deterministic.score_answer
    online_prose_no_leak       score_deterministic.score_answer (with sql.gcp_refs_in_text)
    online_followups_3         score_deterministic.score_answer

`tests/eval/test_online_code_evaluator.py` checks that each copy agrees with the repo.
After a change to one side, change the other, then paste the file again.

The rule reads the root run `outputs`: `messages` (the whole thread, so it scores only
the messages after the last human message) and `structured_response`. A check that does
not apply to the turn is left out.
"""

import json
import re

# The text ModelCallLimitMiddleware puts in the AI message it adds when it stops a run.
_CALL_LIMIT_PREFIX = "Model call limits exceeded"

# A `project.dataset.table` reference in prose (from `eval/lib/sql.py`).
_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_-]*"
_GCP_REF_WORD_RE = re.compile(rf"\b{_IDENTIFIER}\.{_IDENTIFIER}\.{_IDENTIFIER}\b")


def _text(message: dict) -> str:
    """The text of a serialized message, like LangChain's `message.text`."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else block["text"]
        for block in content or []
        if isinstance(block, str)
        or (block.get("type") == "text" and isinstance(block.get("text"), str))
    )


def _turn_messages(messages: list[dict]) -> list[dict]:
    """The messages after the last human message: the turn this trace ran."""
    last_human = max(
        (i for i, message in enumerate(messages) if message.get("type") == "human"),
        default=-1,
    )
    return messages[last_human + 1 :]


def _tool_failed(message: dict) -> bool:
    """Whether a tool call failed: a raised error, or a returned `ToolError` JSON."""
    if message.get("status") == "error":
        return True
    try:
        payload = json.loads(message.get("content"))
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "error"


def _gcp_refs_in_text(text: str) -> set[str]:
    return set(_GCP_REF_WORD_RE.findall((text or "").replace("`", "")))


def perform_eval(run, example=None):
    """Score one root run; LangSmith calls this function once per sampled trace."""
    outputs = run.get("outputs") or {}
    turn = _turn_messages(outputs.get("messages") or [])
    if not turn:
        return {}

    scores = {}
    tools = [message for message in turn if message.get("type") == "tool"]
    if tools:
        failed = sum(_tool_failed(message) for message in tools)
        scores["online_tool_error_rate"] = failed / len(tools)
    scores["online_within_call_limit"] = int(
        not any(
            message.get("type") == "ai"
            and _text(message).startswith(_CALL_LIMIT_PREFIX)
            for message in turn
        )
    )

    structured = outputs.get("structured_response")
    if any(message.get("name") == "execute_bigquery_sql" for message in tools):
        scores["online_query_has_sources"] = int(
            bool(structured and structured.get("data_sources"))
        )
    if structured is not None:
        response = structured.get("response") or ""
        scores["online_response_nonempty"] = int(bool(response.strip()))
        scores["online_prose_no_leak"] = int(
            not ("```sql" in response.lower() or bool(_gcp_refs_in_text(response)))
        )
        follow_ups = structured.get("follow_up_prompts")
        scores["online_followups_3"] = int(
            isinstance(follow_ups, list)
            and len(follow_ups) == 3
            and all((prompt or "").strip() for prompt in follow_ups)
        )
    return scores
