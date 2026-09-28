import pytest

from eval.lib import gold

LEAN_GOLD = gold.load_threads("eval/eval_gold_lean.yaml")


def _thread(thread_id="t", **turn):
    return {"id": thread_id, "turns": [{"user": "q", "action": "ask", **turn}]}


def test_lean_gold_is_valid():
    assert gold.validate(LEAN_GOLD) is LEAN_GOLD


@pytest.mark.parametrize(
    "period",
    [
        None,
        "none",
        "any",
        "latest",
        2025,
        "2026-05",
        "2026-05-01",
        {"last_years": 5},
        {"start": 2015, "end": 2020},
        {"start": "2015-01", "end": 2020},
    ],
)
def test_validate_accepts_known_period_rules(period):
    gold.validate([_thread(action="query", period=period)])


@pytest.mark.parametrize(
    "period",
    [
        "match_previous",
        "2025/05",
        True,
        {"last_years": 0},
        {"last_years": "1"},
        {"start": 2015},
        {"start": 2015, "end": "later"},
        {"latest": True},
    ],
)
def test_validate_rejects_unknown_period_rules(period):
    with pytest.raises(ValueError, match="unknown period"):
        gold.validate([_thread(action="query", period=period)])


def test_validate_rejects_unknown_action():
    with pytest.raises(ValueError, match="unknown action 'clarify'"):
        gold.validate([_thread(action="clarify")])


def test_validate_rejects_duplicate_thread_ids():
    with pytest.raises(ValueError, match="dup: duplicate thread id"):
        gold.validate([_thread("dup"), _thread("dup")])


def test_validate_reports_every_problem_at_once():
    with pytest.raises(ValueError) as error:
        gold.validate([_thread("a", action="clarify", period="soon")])
    assert "unknown action" in str(error.value)
    assert "unknown period" in str(error.value)


@pytest.mark.parametrize("turns", [[], None])
def test_validate_rejects_a_thread_without_turns(turns):
    with pytest.raises(ValueError, match="empty: a thread needs at least one turn"):
        gold.validate([{"id": "empty", "turns": turns}])


@pytest.mark.parametrize("user", [None, "", "   "])
def test_validate_rejects_a_turn_without_a_user_message(user):
    with pytest.raises(ValueError, match=r"t\[0\]: missing user message"):
        gold.validate([_thread(user=user)])
