"""The online code rule must agree with the offline checks it copies."""

import json
from pathlib import Path

import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from eval import runner
from eval import score_deterministic as scorer
from eval.online import code_evaluator

FIXTURE = Path(__file__).parent / "fixtures" / "online_run.json"

_MESSAGE_TYPES = {"human": HumanMessage, "ai": AIMessage, "tool": ToolMessage}


def _recorded_outputs() -> dict:
    return json.loads(FIXTURE.read_text())["outputs"]


def _messages(serialized: list[dict]) -> list:
    return [_MESSAGE_TYPES[message["type"]](**message) for message in serialized]


def _repo_scores(outputs: dict) -> dict:
    """The same checks, computed with the repo's runner and scorer."""
    turn_messages = runner._messages_since_last_human(_messages(outputs["messages"]))
    if not turn_messages:
        return {}
    calls = runner._tool_calls(turn_messages)
    turn = {"status": "ok", "tool_calls": calls}
    scores = {
        "tool_error_rate": scorer.score_tool_error_rate(turn),
        "within_call_limit": not runner._hit_call_limit(turn_messages),
    }
    structured = outputs.get("structured_response")
    if structured is not None:
        # The trace drops a None field; the offline record always has every field.
        structured = {
            "data_sources": None,
            "follow_up_prompts": None,
            "response": None,
            **structured,
        }
    is_query = bool(runner._executed_queries(turn_messages))
    scores.update(scorer.score_answer(structured, is_query))
    return {
        f"online_{key}": float(value)
        for key, value in scores.items()
        if value is not None
    }


def _run(messages: list, structured: dict | None) -> dict:
    return {
        "outputs": {
            "messages": [message.model_dump() for message in messages],
            "structured_response": structured,
        }
    }


def _ai(*calls, content=""):
    return AIMessage(
        content=content,
        tool_calls=[
            {"id": call_id, "name": name, "args": args} for call_id, name, args in calls
        ],
    )


GOOD_ANSWER = {
    "response": "Em 2020 foram 10 milhões.",
    "data_sources": [{"dataset_id": "d", "table_id": "t", "name": "n"}],
    "follow_up_prompts": ["a", "b", "c"],
}

CASES = {
    "query with sources": _run(
        [
            HumanMessage("q"),
            _ai(("c1", "execute_bigquery_sql", {"sql_query": "SELECT 1"})),
            ToolMessage(
                json.dumps({"row_count": 1, "rows": [{"x": 1}]}),
                tool_call_id="c1",
                name="execute_bigquery_sql",
            ),
            _ai(content="done"),
        ],
        GOOD_ANSWER,
    ),
    "query without sources and a leaked table": _run(
        [
            HumanMessage("q"),
            _ai(("c1", "execute_bigquery_sql", {"sql_query": "SELECT 1"})),
            ToolMessage(
                json.dumps({"status": "error", "message": "bad"}),
                tool_call_id="c1",
                name="execute_bigquery_sql",
            ),
        ],
        {
            "response": "Veja `basedosdados.br_x.a`.",
            "data_sources": None,
            "follow_up_prompts": ["a", ""],
        },
    ),
    "raised tool error and call limit stop": _run(
        [
            HumanMessage("earlier"),
            AIMessage("earlier answer"),
            HumanMessage("q"),
            _ai(("c1", "search_datasets", {"query": "x"})),
            ToolMessage(
                "boom", tool_call_id="c1", name="search_datasets", status="error"
            ),
            AIMessage("Model call limits exceeded: run limit (1/1)"),
        ],
        None,
    ),
    "ask turn with no tools": _run(
        [HumanMessage("q"), _ai(content="Qual município?")],
        {"response": "", "data_sources": None, "follow_up_prompts": ["a", "b", "c"]},
    ),
    "recorded run": {"outputs": _recorded_outputs()},
}


@pytest.mark.parametrize("run", CASES.values(), ids=CASES.keys())
def test_rule_agrees_with_the_offline_checks(run):
    assert code_evaluator.perform_eval(run, None) == _repo_scores(run["outputs"])


def test_rule_scores_the_recorded_run():
    scores = code_evaluator.perform_eval({"outputs": _recorded_outputs()}, None)

    assert scores == {
        "online_tool_error_rate": pytest.approx(2 / 11),
        "online_within_call_limit": 1,
        "online_query_has_sources": 1,
        "online_response_nonempty": 1,
        "online_prose_no_leak": 1,
        "online_followups_3": 1,
    }


def test_rule_returns_nothing_for_a_run_without_outputs():
    assert code_evaluator.perform_eval({"outputs": None}, None) == {}


def test_rule_uses_only_the_standard_library():
    source = Path(code_evaluator.__file__).read_text()
    imports = {
        line.split()[1].split(".")[0]
        for line in source.splitlines()
        if line.startswith(("import ", "from "))
    }

    assert imports <= {"json", "re"}
