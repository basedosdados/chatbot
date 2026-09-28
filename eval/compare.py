"""Offline drift gate: compare the feedback of a candidate experiment with a baseline.

Both experiments come from `experiment.py`. For each feedback key, the score is the mean
over every feedback record of that key on the experiment's root runs, so repetitions
count as more samples. A key *falls* when its score moves the wrong way: down for most
keys, up for the keys in `evaluators.LOWER_IS_BETTER`.

The tolerance of a rate (a 0-1 score) is in points: the default 10 means a rate may
fall by up to 0.10. The tolerance of a count (`evaluators.COUNT_KEYS`) is relative: 10
means the count may rise by up to 10%. `select_shape_ok` has no tolerance. The script
exits non-zero when any key falls more than its tolerance. A key found in only one
experiment, or with fewer than `--min-samples` samples on either side, is printed but
does not fail the gate: on a few samples, one turn moves the mean by many points.

    uv run python -m eval.compare --baseline "<experiment>" --candidate "<experiment>"
    uv run python -m eval.compare --baseline "<experiment>" --candidate "<experiment>" --tolerance 5

Pin one experiment per released `agent_config_id` as the baseline, and run a candidate
before each change to the prompt, tools, model, or effort. Use the LangSmith comparison
view for the per-thread detail.
"""

import argparse
import sys
import uuid
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from langsmith import Client

from app.settings import settings
from eval.evaluators import COUNT_KEYS, LOWER_IS_BETTER

DEFAULT_TOLERANCE = 10.0

# The fewest samples on each side for a key to count in the gate.
DEFAULT_MIN_SAMPLES = 5

# Keys with their own tolerance, in the same units as the default.
KEY_TOLERANCE = {"select_shape_ok": 0.0}

# Float noise below this is not a fall.
_EPSILON = 1e-9

# Run ids per `list_feedback` request, to keep the request URL short.
_RUN_ID_BATCH = 100


def mean_scores(feedback: Iterable[Any]) -> dict[str, tuple[float, int]]:
    """The mean score and the sample count of each feedback key.

    Args:
        feedback: Feedback records with `key` and `score`; a None score does not count.

    Returns:
        `key -> (mean, n)`.
    """
    scores: defaultdict[str, list[float]] = defaultdict(list)
    for record in feedback:
        if record.score is not None:
            scores[record.key].append(float(record.score))
    return {
        key: (sum(values) / len(values), len(values)) for key, values in scores.items()
    }


def fall(key: str, baseline: float, candidate: float) -> float:
    """How far a key fell from the baseline to the candidate, in tolerance units.

    Args:
        key: The feedback key.
        baseline: The baseline score.
        candidate: The candidate score.

    Returns:
        Points (x100) for a rate, percent of the baseline for a count. A negative value
        is a gain. A count that rises from zero falls by infinity.
    """
    change = candidate - baseline
    if key not in LOWER_IS_BETTER:
        change = -change
    if key not in COUNT_KEYS:
        return change * 100
    if baseline == 0:
        return float("inf") if change > 0 else 0.0
    return change / baseline * 100


def compare(
    baseline: dict[str, tuple[float, int]],
    candidate: dict[str, tuple[float, int]],
    tolerance: float = DEFAULT_TOLERANCE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> list[dict[str, Any]]:
    """One row per feedback key found in either experiment, sorted by key.

    Args:
        baseline: The baseline `mean_scores`.
        candidate: The candidate `mean_scores`.
        tolerance: The default tolerance, for keys not in `KEY_TOLERANCE`.
        min_samples: The fewest samples on each side for a key to fail the gate.

    Returns:
        Rows of `{key, baseline, candidate, fall, tolerance, few_samples, failed}`.
        `baseline` and `candidate` are `(mean, n)` or None; `fall` is None when a side
        is missing.
    """
    rows = []
    for key in sorted(baseline.keys() | candidate.keys()):
        key_tolerance = KEY_TOLERANCE.get(key, tolerance)
        before, after = baseline.get(key), candidate.get(key)
        key_fall = (
            fall(key, before[0], after[0])
            if before is not None and after is not None
            else None
        )
        few_samples = key_fall is not None and min(before[1], after[1]) < min_samples
        rows.append(
            {
                "key": key,
                "baseline": before,
                "candidate": after,
                "fall": key_fall,
                "tolerance": key_tolerance,
                "few_samples": few_samples,
                "failed": key_fall is not None
                and not few_samples
                and key_fall > key_tolerance + _EPSILON,
            }
        )
    return rows


def _format_side(side: tuple[float, int] | None) -> str:
    return "-" if side is None else f"{side[0]:.3f} (n={side[1]})"


def format_row(row: dict[str, Any]) -> str:
    """One printed line for a `compare` row."""
    unit = "%" if row["key"] in COUNT_KEYS else "pt"
    if row["fall"] is None:
        verdict = "only in " + ("candidate" if row["baseline"] is None else "baseline")
    else:
        # Adding 0.0 turns a -0.0 fall into 0.0 for printing.
        verdict = (
            f"fall {row['fall'] + 0.0:+.1f}{unit} (tol {row['tolerance']:g}{unit})"
        )
        if row["few_samples"]:
            verdict += "  few samples"
        if row["failed"]:
            verdict += "  FAIL"
    return (
        f"  {row['key']:<24} {_format_side(row['baseline']):>16} "
        f"-> {_format_side(row['candidate']):>16}  {verdict}"
    )


def _read_experiment(client: Client, experiment: str) -> Any:
    """Read an experiment by id or by name."""
    try:
        return client.read_project(project_id=str(uuid.UUID(experiment)))
    except ValueError:
        return client.read_project(project_name=experiment)


def experiment_scores(client: Client, project: Any) -> dict[str, tuple[float, int]]:
    """The `mean_scores` of every feedback record on an experiment's root runs."""
    run_ids = [run.id for run in client.list_runs(project_id=project.id, is_root=True)]
    feedback = [
        record
        for start in range(0, len(run_ids), _RUN_ID_BATCH)
        for record in client.list_feedback(
            run_ids=run_ids[start : start + _RUN_ID_BATCH]
        )
    ]
    return mean_scores(feedback)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--baseline", required=True, help="Experiment name or id")
    parser.add_argument("--candidate", required=True, help="Experiment name or id")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Allowed fall: points for a rate, percent for a count (default {DEFAULT_TOLERANCE:g})",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help=f"Fewest samples per side for a key to gate (default {DEFAULT_MIN_SAMPLES})",
    )
    return parser.parse_args()


def main() -> None:
    """Print one delta per key and exit non-zero when a key falls too far."""
    args = parse_args()
    client = Client(api_key=settings.LANGSMITH_API_KEY)
    projects = {
        side: _read_experiment(client, name)
        for side, name in (("baseline", args.baseline), ("candidate", args.candidate))
    }
    for side, project in projects.items():
        metadata = project.metadata or {}
        print(
            f"{side:<9} {project.name!r}  effort={metadata.get('reasoning_effort')!r}"
            f"  dataset={metadata.get('dataset_tag')!r}"
        )
    tags = {
        (project.metadata or {}).get("dataset_tag") for project in projects.values()
    }
    if len(tags) > 1:
        print("warning: the experiments ran over different dataset versions")

    rows = compare(
        experiment_scores(client, projects["baseline"]),
        experiment_scores(client, projects["candidate"]),
        args.tolerance,
        args.min_samples,
    )
    print()
    for row in rows:
        print(format_row(row))
    failed = [row["key"] for row in rows if row["failed"]]
    if failed:
        sys.exit(f"\ndrift: {', '.join(failed)} fell more than the tolerance")
    print("\nno key fell more than its tolerance")


if __name__ == "__main__":
    main()
