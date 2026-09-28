"""LangSmith evaluators over one replayed gold thread.

A dataset example is one gold thread (see `ls_dataset.py`). The experiment target
replays it and returns the runner's transcript unit (`runner.replay_thread`). Each
evaluator scores that unit and returns `{"results": [...]}`, one feedback key per check:

    deterministic     every core `score_turn` check: routing, sources, period, SQL rules,
                      and the structured-answer checks
    tool_calling      details_before_query, within_call_limit, model_calls
    tool_correctness  tool_error_rate (lower is better)
    judge             grounded, answers_question, stated_assumption, correct

A key's score is the mean over the thread's turns where the check applies; a turn where
the check is None does not count, and a key with no such turn is left out. The comment
names the failing turns. None of these evaluators needs BigQuery, except the judge on a
turn with a `reference_sql`.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from eval import score_deterministic as scorer
from eval.lib import gold
from eval.lib.judge import (
    SCORE_FIELDS,
    ajudge,
    build_tasks,
    dedup_tasks,
    run_reference_sql,
)

Results = dict[str, list[dict[str, Any]]]

# Reference SQL result rows, by SQL text, shared by every judge call in the process so
# each distinct `reference_sql` runs once per experiment.
_REFERENCE_CACHE: dict[str, list] = {}


def _gold_index(thread_id: str, reference_outputs: dict) -> dict:
    """Rebuild the `(thread, turn_index) -> gold turn` map that the scorers read."""
    return gold.index_turns([{"id": thread_id, "turns": reference_outputs["turns"]}])


def _rate(
    key: str,
    per_turn: dict[int, Any],
    failed: Callable[[Any], bool] = lambda value: value != 1.0,
    notes: dict[int, str] | None = None,
) -> dict[str, Any] | None:
    """Fold per-turn values into one feedback result: the mean over applicable turns.

    Args:
        key: The feedback key.
        per_turn: `turn_index -> value`; a None value does not count.
        failed: Whether a value marks its turn as failing, for the comment.
        notes: Optional per-turn text added to the comment for a failing turn.

    Returns:
        `{key, score, comment}`, or None when no turn applies.
    """
    scored = {index: value for index, value in per_turn.items() if value is not None}
    if not scored:
        return None
    failing = sorted(index for index, value in scored.items() if failed(value))
    if failing:
        comment = "failing turns: " + ", ".join(f"t{index}" for index in failing)
        comment += "".join(
            f"\nt{index}: {notes[index]}"
            for index in failing
            if notes and notes.get(index)
        )
    else:
        comment = f"passed on {len(scored)} turn(s)"
    return {
        "key": key,
        "score": sum(float(value) for value in scored.values()) / len(scored),
        "comment": comment,
    }


def _mean(key: str, per_turn: dict[int, Any]) -> dict[str, Any] | None:
    """Like :func:`_rate`, for a count: the comment lists every turn's value."""
    scored = {index: value for index, value in per_turn.items() if value is not None}
    if not scored:
        return None
    return {
        "key": key,
        "score": sum(scored.values()) / len(scored),
        "comment": " ".join(f"t{index}={value}" for index, value in scored.items()),
    }


def _results(*results: dict[str, Any] | None) -> Results:
    return {"results": [result for result in results if result is not None]}


def deterministic(outputs: dict, reference_outputs: dict) -> Results:
    """Every core `score_turn` check, one key each (see `score_deterministic.CORE_CHECKS`)."""
    rows = scorer.score_unit(outputs, _gold_index(outputs["thread"], reference_outputs))
    return _results(
        *(
            _rate(check, {row["turn_index"]: row[check] for row in rows})
            for check in scorer.CORE_CHECKS
        )
    )


def tool_calling(outputs: dict) -> Results:
    """Whether the agent read table details before its SQL and stayed under the call limit."""
    turns = outputs["turns"]
    return _results(
        _rate("details_before_query", scorer.score_details_before_query(outputs)),
        _rate(
            "within_call_limit",
            {
                turn["turn_index"]: scorer.score_within_call_limit(turn)
                for turn in turns
            },
        ),
        _mean(
            "model_calls",
            {
                turn["turn_index"]: turn["model_calls"]
                for turn in turns
                if scorer.turn_ran(turn)
            },
        ),
    )


def tool_correctness(outputs: dict) -> Results:
    """The share of tool calls that failed, per turn (lower is better)."""
    return _results(
        _rate(
            "tool_error_rate",
            {
                turn["turn_index"]: scorer.score_tool_error_rate(turn)
                for turn in outputs["turns"]
            },
            failed=lambda value: value > 0,
        )
    )


def judge_evaluator(
    judge_model: Any,
) -> Callable[[dict, dict], Awaitable[Results]]:
    """Build the async judge evaluator around one judge model.

    Args:
        judge_model: The structured-output judge from `eval.lib.judge.build_judge`.

    Returns:
        An evaluator that judges every answered turn once and returns one key per judge
        score, plus `judge_errors` (the count of failed judge calls) when a call fails.
    """

    async def judge(outputs: dict, reference_outputs: dict) -> Results:
        gold_index = _gold_index(outputs["thread"], reference_outputs)
        tasks = dedup_tasks(build_tasks({"units": [outputs]}, gold_index, None))
        for task in tasks:
            task["reference_rows"] = (
                await asyncio.to_thread(
                    run_reference_sql, task["reference_sql"], _REFERENCE_CACHE
                )
                if task["reference_sql"]
                else None
            )
        records = await asyncio.gather(*(ajudge(judge_model, task) for task in tasks))
        verdicts = {
            task["turn_index"]: record["verdict"]
            for task, record in zip(tasks, records)
        }
        rationales = {
            index: verdict["rationale"]
            for index, verdict in verdicts.items()
            if verdict is not None
        }
        errors = {
            task["turn_index"]: record["error"]
            for task, record in zip(tasks, records)
            if "error" in record
        }
        return _results(
            *(
                _rate(
                    field,
                    {
                        index: verdict[field] if verdict is not None else None
                        for index, verdict in verdicts.items()
                    },
                    notes=rationales,
                )
                for field in SCORE_FIELDS
            ),
            {
                "key": "judge_errors",
                "score": len(errors),
                "comment": "\n".join(f"t{index}: {e}" for index, e in errors.items()),
            }
            if errors
            else None,
        )

    return judge
