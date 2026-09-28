from types import SimpleNamespace

import pytest

from eval import compare


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
        "grounded": (0.8, 4),
    }
    candidate = {
        "action": (0.95, 8),
        "period": (0.7, 4),
        "select_shape_ok": (0.75, 4),
        "model_calls": (4.2, 8),
        "judge_errors": (1.0, 1),
    }

    rows = {
        row["key"]: row for row in compare.compare(baseline, candidate, min_samples=1)
    }

    assert not rows["action"]["failed"]
    assert rows["period"]["failed"]
    assert rows["select_shape_ok"]["failed"]
    assert not rows["model_calls"]["failed"]
    assert rows["grounded"]["fall"] is None and not rows["grounded"]["failed"]
    assert rows["judge_errors"]["baseline"] is None
    assert not rows["judge_errors"]["failed"]


def test_compare_passes_the_same_scores():
    scores = {"action": (1.0, 8), "select_shape_ok": (1.0, 4)}

    assert not any(row["failed"] for row in compare.compare(scores, scores))


def test_format_row_marks_a_failed_key_and_a_missing_side():
    rows = compare.compare(
        {"period": (0.9, 4)}, {"period": (0.5, 4), "new": (1.0, 1)}, min_samples=1
    )
    lines = {row["key"]: compare.format_row(row) for row in rows}

    assert "FAIL" in lines["period"]
    assert "only in candidate" in lines["new"]


def test_compare_does_not_gate_a_key_with_few_samples():
    rows = compare.compare({"period": (1.0, 2)}, {"period": (0.5, 8)}, min_samples=5)

    assert rows[0]["few_samples"]
    assert not rows[0]["failed"]
    assert "few samples" in compare.format_row(rows[0])
