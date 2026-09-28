from types import SimpleNamespace

import pytest

from eval import compare, evaluators
from tests.eval.test_evaluators import REFERENCE, _FakeJudge, _unit


def _feedback(key, score):
    return SimpleNamespace(key=key, score=score)


def test_mean_scores_average_each_key_and_skip_none():
    feedback = [
        _feedback("action", 1.0),
        _feedback("action", 0.0),
        _feedback("action", None),
        _feedback("model_calls", 3),
    ]

    assert compare.mean_scores(feedback) == {
        "action": (0.5, 2),
        "model_calls": (3.0, 1),
    }


def test_fall_follows_the_better_direction_of_each_key():
    assert compare.fall("action", 0.9, 0.7) == pytest.approx(20.0)
    assert compare.fall("action", 0.7, 0.9) == pytest.approx(-20.0)
    assert compare.fall("tool_error_rate", 0.1, 0.3) == pytest.approx(20.0)
    assert compare.fall("model_calls", 4.0, 5.0) == pytest.approx(25.0)
    assert compare.fall("judge_errors", 0.0, 1.0) == float("inf")
    assert compare.fall("judge_errors", 0.0, 0.0) == 0.0


def test_compare_fails_only_the_keys_past_their_tolerance():
    baseline = {
        "action": (1.0, 8),
        "period": (0.9, 4),
        "select_shape_ok": (1.0, 4),
        "model_calls": (4.0, 8),
    }
    candidate = {
        "action": (0.95, 8),
        "period": (0.7, 4),
        "select_shape_ok": (0.75, 4),
        "model_calls": (4.2, 8),
        "new_check": (1.0, 4),
    }

    rows = {
        row["key"]: row for row in compare.compare(baseline, candidate, min_samples=1)
    }

    assert not rows["action"]["failed"]
    assert rows["period"]["failed"]
    assert rows["period"]["reason"] == "fell more than the tolerance"
    assert rows["select_shape_ok"]["failed"]
    assert not rows["model_calls"]["failed"]
    assert rows["new_check"]["baseline"] is None
    assert not rows["new_check"]["failed"]


def test_compare_fails_a_baseline_key_missing_from_the_candidate():
    rows = compare.compare(
        {"action": (1.0, 8), "grounded": (0.8, 2)}, {"action": (1.0, 8)}
    )
    grounded = {row["key"]: row for row in rows}["grounded"]

    # Few samples do not excuse a key the candidate lost.
    assert grounded["failed"]
    assert grounded["reason"] == "missing in candidate"
    assert "FAIL: missing in candidate" in compare.format_row(grounded)


def test_compare_passes_the_same_scores():
    scores = {"action": (1.0, 8), "select_shape_ok": (1.0, 4)}

    assert not any(row["failed"] for row in compare.compare(scores, scores))


def test_format_row_marks_a_failed_key_and_a_missing_side():
    rows = compare.compare(
        {"period": (0.9, 4)}, {"period": (0.5, 4), "new": (1.0, 1)}, min_samples=1
    )
    lines = {row["key"]: compare.format_row(row) for row in rows}

    assert "FAIL: fell more than the tolerance" in lines["period"]
    assert "only in candidate" in lines["new"]
    assert "FAIL" not in lines["new"]


def test_compare_does_not_gate_a_key_with_few_samples():
    rows = compare.compare({"period": (1.0, 2)}, {"period": (0.5, 8)}, min_samples=5)

    assert rows[0]["few_samples"]
    assert not rows[0]["failed"]
    assert "few samples" in compare.format_row(rows[0])


POPULATION = {"dataset_id": "d1", "dataset_tag": "v1", "example_selection": "s1"}


@pytest.mark.parametrize("side", ["baseline", "candidate"])
def test_gate_fails_an_experiment_without_scored_feedback(side):
    scores = {"action": (1.0, 8)}
    sides = {"baseline": scores, "candidate": scores, side: {}}

    problems = compare.gate_problems(
        sides["baseline"], sides["candidate"], POPULATION, POPULATION
    )

    assert problems == [f"the {side} has no scored feedback"]


def test_gate_fails_feedback_with_only_none_scores():
    empty = compare.mean_scores([_feedback("action", None)])

    assert compare.gate_problems(
        {"action": (1.0, 8)}, empty, POPULATION, POPULATION
    ) == ["the candidate has no scored feedback"]


def test_gate_passes_the_same_population():
    scores = {"action": (1.0, 8)}

    assert compare.gate_problems(scores, scores, POPULATION, dict(POPULATION)) == []


@pytest.mark.parametrize(
    ("key", "mismatch"),
    [
        ("dataset_id", "different datasets"),
        ("dataset_tag", "different dataset versions"),
        ("example_selection", "different example selections"),
    ],
)
def test_gate_fails_a_different_population(key, mismatch):
    scores = {"action": (1.0, 8)}
    candidate = {**POPULATION, key: "other"}

    assert compare.gate_problems(scores, scores, POPULATION, candidate) == [
        f"the experiments ran over {mismatch} ({key} {POPULATION[key]!r} and 'other')"
    ]


@pytest.mark.parametrize("key", ["dataset_id", "dataset_tag", "example_selection"])
def test_gate_fails_an_experiment_without_population_metadata(key):
    scores = {"action": (1.0, 8)}
    baseline = {name: value for name, value in POPULATION.items() if name != key}

    assert compare.gate_problems(scores, scores, baseline, POPULATION) == [
        f"an experiment has no {key} metadata"
    ]


class _BrokenJudge:
    async def ainvoke(self, messages):
        raise RuntimeError("quota")


async def _judge_feedback(judges):
    """The feedback records the judge evaluator logs, one replay per judge."""
    return [
        SimpleNamespace(**result)
        for judge in judges
        for result in (await evaluators.judge_evaluator(judge)(_unit(), REFERENCE))[
            "results"
        ]
    ]


async def test_compare_fails_when_every_judge_call_fails():
    baseline = compare.mean_scores(await _judge_feedback([_FakeJudge()] * 5))
    candidate = compare.mean_scores(await _judge_feedback([_BrokenJudge()] * 5))

    rows = {row["key"]: row for row in compare.compare(baseline, candidate)}

    assert rows["grounded"]["reason"] == "missing in candidate"
    assert rows["answers_question"]["reason"] == "missing in candidate"
    assert rows["judge_errors"]["reason"] == "judge calls failed"


async def test_compare_fails_when_some_judge_calls_fail():
    baseline = compare.mean_scores(await _judge_feedback([_FakeJudge()] * 5))
    candidate = compare.mean_scores(
        await _judge_feedback([_FakeJudge()] * 5 + [_BrokenJudge()])
    )

    rows = {row["key"]: row for row in compare.compare(baseline, candidate)}

    # The scores that remain match the baseline, but a turn lost its judge score.
    assert not rows["grounded"]["failed"]
    assert rows["judge_errors"]["failed"]
    assert rows["judge_errors"]["baseline"] is None


def test_compare_fails_judge_errors_in_the_baseline():
    rows = compare.compare(
        {"grounded": (1.0, 8), "judge_errors": (1.0, 1)}, {"grounded": (1.0, 8)}
    )

    assert {row["key"]: row["reason"] for row in rows} == {
        "grounded": None,
        "judge_errors": "judge calls failed",
    }


class _FakeClient:
    """A LangSmith client stub with one root run per experiment."""

    def __init__(self, experiments):
        self.experiments = experiments

    def read_project(self, project_name=None, project_id=None):
        population, _ = self.experiments[project_name]
        return SimpleNamespace(id=project_name, name=project_name, metadata=population)

    def list_runs(self, project_id, is_root):
        return [SimpleNamespace(id=project_id)]

    def list_feedback(self, run_ids):
        return [record for run_id in run_ids for record in self.experiments[run_id][1]]


def _run_main(monkeypatch, experiments):
    monkeypatch.setattr(compare, "Client", lambda api_key: _FakeClient(experiments))
    monkeypatch.setattr(
        "sys.argv",
        ["compare", "--baseline", "base", "--candidate", "cand", "--min-samples", "1"],
    )
    compare.main()


def test_main_exits_non_zero_on_empty_candidate_feedback(monkeypatch, capsys):
    experiments = {
        "base": (POPULATION, [_feedback("action", 1.0)]),
        "cand": (POPULATION, []),
    }

    with pytest.raises(SystemExit) as exit_info:
        _run_main(monkeypatch, experiments)

    assert "the candidate has no scored feedback" in str(exit_info.value.code)
    assert "missing in candidate" in str(exit_info.value.code)
    assert "no key fell" not in capsys.readouterr().out


def test_main_exits_non_zero_on_a_different_example_selection(monkeypatch):
    records = [_feedback("action", 1.0)]
    subset = {**POPULATION, "example_selection": "s2"}
    experiments = {"base": (POPULATION, records), "cand": (subset, records)}

    with pytest.raises(SystemExit) as exit_info:
        _run_main(monkeypatch, experiments)

    assert "different example selections" in str(exit_info.value.code)


def test_main_exits_non_zero_without_population_metadata(monkeypatch):
    records = [_feedback("action", 1.0)]
    experiments = {"base": ({"dataset_tag": "v1"}, records), "cand": ({}, records)}

    with pytest.raises(SystemExit) as exit_info:
        _run_main(monkeypatch, experiments)

    assert "an experiment has no dataset_id metadata" in str(exit_info.value.code)


def test_main_passes_comparable_experiments(monkeypatch, capsys):
    records = [_feedback("action", 1.0)]
    experiments = {"base": (POPULATION, records), "cand": (POPULATION, records)}

    _run_main(monkeypatch, experiments)

    assert "no key fell more than its tolerance" in capsys.readouterr().out
