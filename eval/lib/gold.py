"""Loading and indexing the eval gold set (`eval_gold.yaml`).

The gold is a list of multi-turn *threads*; a single-turn case is simply a one-turn
thread. Each turn declares the expected `action`, the `sources` its SQL should
reference, its `period` rule, and an optional `reference_sql`. This module loads and
indexes that file and defines the small `action` / `period` vocabulary the scorers
branch on — so those values live in one place, not scattered as string literals.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from eval.lib import period

# ---- action vocabulary -------------------------------------------------------
# The trace only proves query-vs-not, so that's the whole taxonomy: `query` ran
# execute_bigquery_sql; `ask` ran none (whether it explored the catalog first, asked
# directly, described available data, or reported it can't help is the judge's to weigh).
ACTION_QUERY = "query"  # ran execute_bigquery_sql to answer a specific question
ACTION_ASK = (
    "ask"  # ran no SQL — clarify, describe the catalog, or report it can't help
)
ACTIONS = frozenset({ACTION_QUERY, ACTION_ASK})

# ---- period vocabulary -------------------------------------------------------
# A gold turn's `period` is one of:
#   none / any          -> not period-scored (ask turns, or a query left ungraded)
#   latest              -> run-relative STOCK anchor: the queried period equals the table's
#                          period_end, at its own granularity (from the run's metadata)
#   {last_years: N}     -> run-relative FLOW anchor: the N most-recent-COMPLETE-year window
#   <point>             -> a concrete year/month/day pinned by the question
#                          (2025, or 'YYYY-MM[-DD]' read at that granularity)
#   {start: ., end: .}  -> a concrete range, each endpoint a point
# A same-period follow-up just repeats the anchor (e.g. `latest` again), so there is no
# separate "match previous" rule.
PERIOD_NONE = "none"
PERIOD_ANY = "any"
PERIOD_LATEST = "latest"

_UNSCORED_PERIOD = (None, PERIOD_NONE, PERIOD_ANY, "", {})

GoldTurn = dict
ThreadKey = tuple[str, int]  # (thread_id, turn_index)


def load_threads(path: str | Path) -> list[dict]:
    """Load the raw gold threads from `eval_gold.yaml`, in file order.

    Args:
        path: Path to the gold YAML file.

    Returns:
        The threads, each a `{"id": ..., "turns": [...]}` dict.
    """
    with open(path) as file:
        return yaml.safe_load(file)


def validate(threads: list[dict]) -> list[dict]:
    """Reject a malformed gold set before any agent call spends money on it.

    Checks that thread ids are unique (a dataset example and the turn index key on
    them), that every thread has at least one turn and every turn a `user` message,
    that every `action` is in :data:`ACTIONS`, and that every `period` is a rule the
    scorer knows (see the period vocabulary above).

    Args:
        threads: The threads from :func:`load_threads`.

    Returns:
        The same threads, unchanged, so a caller can chain it.

    Raises:
        ValueError: One message per problem, all problems in one error.
    """
    problems = []
    seen: set[str] = set()
    for thread in threads:
        thread_id = thread["id"]
        if thread_id in seen:
            problems.append(f"{thread_id}: duplicate thread id")
        seen.add(thread_id)
        turns = thread.get("turns")
        if not isinstance(turns, list) or not turns:
            problems.append(f"{thread_id}: a thread needs at least one turn")
            continue
        for turn_index, turn in enumerate(turns):
            where = f"{thread_id}[{turn_index}]"
            user = turn.get("user")
            if not isinstance(user, str) or not user.strip():
                problems.append(f"{where}: missing user message")
            if turn.get("action") not in ACTIONS:
                problems.append(f"{where}: unknown action {turn.get('action')!r}")
            if not _is_known_period(turn.get("period")):
                problems.append(f"{where}: unknown period {turn.get('period')!r}")
    if problems:
        raise ValueError("invalid gold set:\n" + "\n".join(problems))
    return threads


def _is_known_period(rule: object) -> bool:
    """Whether `rule` is one of the period forms the scorer reads."""
    if rule in _UNSCORED_PERIOD or rule == PERIOD_LATEST:
        return True
    if isinstance(rule, dict):
        if rule.keys() == {"last_years"}:
            years = rule["last_years"]
            return isinstance(years, int) and not isinstance(years, bool) and years > 0
        if rule.keys() == {"start", "end"}:
            return all(_is_point(rule[key]) for key in ("start", "end"))
        return False
    return _is_point(rule)


def _is_point(value: object) -> bool:
    """Whether `value` is a concrete year / month / day (`2025`, `"2026-05[-01]"`)."""
    return (
        isinstance(value, (int, str))
        and not isinstance(value, bool)
        and (period.parse_period(value) is not None)
    )


def index_turns(threads: list[dict]) -> dict[ThreadKey, GoldTurn]:
    """Index a thread list by `(thread_id, turn_index)` for turn-level lookup.

    Args:
        threads: The threads from :func:`load_threads`.

    Returns:
        A `(thread_id, turn_index) -> gold turn` mapping.
    """
    return {
        (thread["id"], turn_index): turn
        for thread in threads
        for turn_index, turn in enumerate(thread["turns"])
    }


def load_indexed(path: str | Path) -> dict[ThreadKey, GoldTurn]:
    """Load and index the gold in one step: `(thread_id, turn_index) -> gold turn`."""
    return index_turns(load_threads(path))


def scores_period(turn: GoldTurn) -> bool:
    """Whether this turn expects a graded period.

    True only for a query turn carrying a real `period` rule; ask turns and
    query turns marked `none`/`any` are excluded.

    Args:
        turn: A gold turn.

    Returns:
        True when the turn's period should be scored.
    """
    # `action` is required on every gold turn (fail fast on a malformed one); `period` is
    # optional — a turn that declares no rule (or `none`/`any`) simply isn't period-scored.
    return turn["action"] == ACTION_QUERY and turn.get("period") not in _UNSCORED_PERIOD
