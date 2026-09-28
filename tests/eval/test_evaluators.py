from types import SimpleNamespace

from eval import evaluators
from eval.lib.judge import JudgeVerdict

TABLE_A = "basedosdados.br_x.a"
TABLE_B = "basedosdados.br_x.b"
SQL_A = f"SELECT ano, COUNT(*) AS n FROM `{TABLE_A}` WHERE ano = 2024 GROUP BY ano"
SQL_B = f"SELECT ano FROM `{TABLE_B}` WHERE ano = 2024"

REFERENCE = {
    "turns": [
        {"user": "q0", "action": "query", "sources": [TABLE_A], "period": 2024},
        {"user": "q1", "action": "query", "sources": [TABLE_B], "period": 2024},
        {"user": "q2", "action": "ask"},
    ]
}


def _call(name, step, status="success", gcp_id=None, sql=None):
    return {"name": name, "status": status, "step": step, "gcp_id": gcp_id, "sql": sql}


def _unit():
    """A replay of a three-turn thread.

    t0 reads the details of A, then queries A, and answers in full.
    t1 fetches the details of B in parallel with its query, one query fails, and the
    call limit stops the turn before a structured answer.
    t2 asks back without tools.
    """
    return {
        "thread": "demo",
        "effort": "medium",
        "repeat": 0,
        "tables": {},
        "uuid_to_gcp": {},
        "turns": [
            {
                "turn_index": 0,
                "user": "q0",
                "status": "ok",
                "is_query": True,
                "tools_used": ["execute_bigquery_sql", "get_table_details"],
                "tool_calls": [
                    _call("get_table_details", 1, gcp_id=TABLE_A),
                    _call("execute_bigquery_sql", 2, sql=SQL_A),
                ],
                "model_calls": 3,
                "call_limit_hit": False,
                "queries": [
                    {
                        "sql": SQL_A,
                        "status": "success",
                        "row_count": 1,
                        "rows": [{"ano": 2024, "n": 7}],
                        "error": None,
                    }
                ],
                "structured": {
                    "response": "Foram 7 registros em 2024.",
                    "data_sources": [{"table_id": "uuid-a", "name": "A"}],
                    "follow_up_prompts": ["a", "b", "c"],
                },
                "tables": [TABLE_A],
            },
            {
                "turn_index": 1,
                "user": "q1",
                "status": "no_structured",
                "is_query": True,
                "tools_used": ["execute_bigquery_sql", "get_table_details"],
                "tool_calls": [
                    _call("get_table_details", 1, gcp_id=TABLE_B),
                    _call("execute_bigquery_sql", 1, sql=SQL_B),
                    _call("execute_bigquery_sql", 2, status="error", sql="SELECT"),
                ],
                "model_calls": 20,
                "call_limit_hit": True,
                "queries": [
                    {
                        "sql": SQL_B,
                        "status": "success",
                        "row_count": 1,
                        "rows": [{"ano": 2024}],
                        "error": None,
                    }
                ],
                "structured": None,
                "tables": [],
                "final_message": "Model call limits exceeded: run limit (20/20)",
            },
            {
                "turn_index": 2,
                "user": "q2",
                "status": "ok",
                "is_query": False,
                "tools_used": [],
                "tool_calls": [],
                "model_calls": 1,
                "call_limit_hit": False,
                "queries": [],
                "structured": {
                    "response": "Qual município?",
                    "data_sources": None,
                    "follow_up_prompts": ["a", "b", "c"],
                },
                "tables": [],
            },
        ],
    }


def _by_key(results):
    return {result["key"]: result for result in results["results"]}


def test_deterministic_folds_turn_checks_into_per_key_means():
    results = _by_key(evaluators.deterministic(_unit(), REFERENCE))

    assert results["action"]["score"] == 1.0
    assert results["completed"]["score"] == 2 / 3
    assert results["completed"]["comment"] == "failing turns: t1"
    assert results["source_executed"]["score"] == 1.0
    assert results["period"]["score"] == 1.0
    # Only t0 and t2 produced a structured answer.
    assert results["response_nonempty"]["comment"] == "passed on 2 turn(s)"
    assert results["query_has_sources"]["score"] == 0.5


def test_deterministic_leaves_out_a_check_no_turn_applies_to():
    results = _by_key(evaluators.deterministic(_unit(), REFERENCE))

    # No table metadata was recorded, so no partitioned table is known.
    assert "partition_filtered" not in results
    assert all(result["score"] is not None for result in results.values())


def test_tool_calling_flags_a_query_without_prior_table_details():
    results = _by_key(evaluators.tool_calling(_unit()))

    details = results["details_before_query"]
    assert details["score"] == 0.5
    assert details["comment"] == "failing turns: t1"
    assert results["within_call_limit"]["score"] == 2 / 3
    assert results["within_call_limit"]["comment"] == "failing turns: t1"
    assert results["model_calls"]["score"] == 8.0
    assert results["model_calls"]["comment"] == "t0=3 t1=20 t2=1"


def test_details_from_an_earlier_turn_count_for_a_later_query():
    unit = _unit()
    unit["turns"][1]["tool_calls"] = [_call("execute_bigquery_sql", 1, sql=SQL_A)]

    results = _by_key(evaluators.tool_calling(unit))

    assert results["details_before_query"]["score"] == 1.0


def test_tool_correctness_scores_the_share_of_failed_tool_calls():
    results = _by_key(evaluators.tool_correctness(_unit()))

    error_rate = results["tool_error_rate"]
    # t0: 0 of 2 failed; t1: 1 of 3 failed; t2 made no tool call.
    assert error_rate["score"] == (0 + 1 / 3) / 2
    assert error_rate["comment"] == "failing turns: t1"


def test_skipped_and_errored_turns_are_not_scored():
    unit = _unit()
    unit["turns"][1] = {"turn_index": 1, "user": "q1", "status": "error", "error": "x"}
    unit["turns"][2] = {"turn_index": 2, "user": "q2", "status": "skipped"}

    calling = _by_key(evaluators.tool_calling(unit))
    correctness = _by_key(evaluators.tool_correctness(unit))

    assert calling["within_call_limit"]["comment"] == "passed on 1 turn(s)"
    assert calling["model_calls"]["comment"] == "t0=3"
    assert correctness["tool_error_rate"]["score"] == 0.0


class _FakeJudge:
    """A judge model stub: grounded fails on t2, and no turn has a reference."""

    def __init__(self):
        self.prompts = []

    async def ainvoke(self, messages):
        prompt = messages[1].content
        self.prompts.append(prompt)
        is_ask = "# Expected turn type: ask" in prompt
        verdict = JudgeVerdict(
            grounded=not is_ask,
            answers_question=True,
            stated_assumption=None,
            rationale="inventou um dado" if is_ask else "ok",
        )
        usage = {"input_tokens": 10, "output_tokens": 2}
        return {"parsed": verdict, "raw": SimpleNamespace(usage_metadata=usage)}


async def test_judge_scores_each_answered_turn_once():
    fake = _FakeJudge()

    results = _by_key(await evaluators.judge_evaluator(fake)(_unit(), REFERENCE))

    # t1 has no structured answer, so only t0 and t2 are judged.
    assert len(fake.prompts) == 2
    assert results["answers_question"]["score"] == 1.0
    assert results["grounded"]["score"] == 0.5
    assert results["grounded"]["comment"] == "failing turns: t2\nt2: inventou um dado"
    # No reference_sql and no ambiguity: both keys have no applicable turn.
    assert "correct" not in results
    assert "stated_assumption" not in results
    assert "judge_errors" not in results


async def test_judge_prompt_uses_the_current_response_fields():
    fake = _FakeJudge()

    await evaluators.judge_evaluator(fake)(_unit(), REFERENCE)

    assert "follow_up_prompts: ['a', 'b', 'c']" in fake.prompts[0]
    assert "temporal_coverage" not in fake.prompts[0]


async def test_judge_reports_a_failed_call_as_judge_errors():
    class _BrokenJudge:
        async def ainvoke(self, messages):
            raise RuntimeError("quota")

    results = _by_key(
        await evaluators.judge_evaluator(_BrokenJudge())(_unit(), REFERENCE)
    )

    assert results["judge_errors"]["score"] == 2
    assert "RuntimeError: quota" in results["judge_errors"]["comment"]
    assert "grounded" not in results
