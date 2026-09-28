"""Run a LangSmith experiment: replay the gold dataset and score every thread.

The experiment runs over the dataset `ls_dataset.py` syncs, pinned to the version tag of
the gold file (`ls_dataset.dataset_tag`), so a baseline and a candidate run over the
same examples. The target replays one example with `runner.replay_thread` and returns
the transcript unit. The evaluators in `evaluators.py` score that unit.

The experiment name starts with the `agent_config_label` of the config under test, and
the experiment metadata holds its full identity (`agent_config_id`, hashes, effort),
plus `branch`, `eval_run`, `models`, and `judge_model`. It also identifies the examples
the experiment ran over (`population_metadata`), so `compare.py` gates only experiments
over the same examples. Every agent trace carries the same metadata and the production
tags, so it joins the production traces of that config.

    uv run python -m eval.experiment --effort medium --repetitions 1 --split ask --no-judge
    uv run python -m eval.experiment --effort medium --repetitions 3
    uv run python -m eval.experiment --thread comex-stat-exports --repetitions 1

Every turn is a live multi-step agent run: cost is (threads x turns x repetitions).
"""

import argparse
import asyncio
import hashlib
import itertools
import sys
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime

from langgraph.graph.state import CompiledStateGraph
from langsmith import Client, aevaluate
from langsmith.schemas import Example

from app.settings import settings
from eval import evaluators, ls_dataset, runner
from eval.lib.judge import DEFAULT_JUDGE_MODEL, build_judge


def thread_from_inputs(inputs: dict) -> dict:
    """Rebuild the gold thread `replay_thread` reads from a dataset example's inputs.

    Args:
        inputs: The example inputs (`{"thread_id", "turns": [user message, ...]}`).

    Returns:
        The thread as `{"id", "turns": [{"user": ...}, ...]}`.
    """
    return {
        "id": inputs["thread_id"],
        "turns": [{"user": user} for user in inputs["turns"]],
    }


def population_metadata(examples: Sequence[Example]) -> dict:
    """The experiment metadata that identifies the examples it runs over.

    `--dataset`, `--split`, and `--thread` change the examples, and the dataset version
    tag does not show that. So the metadata keeps the dataset id and a fingerprint of the
    selected example ids. The order of the examples does not change the fingerprint.

    Args:
        examples: The selected dataset examples, all from one dataset.

    Returns:
        `{dataset_id, example_count, example_selection}`; `example_selection` is the
        SHA-256 hex digest of the sorted example ids.
    """
    dataset_ids = {str(example.dataset_id) for example in examples}
    if len(dataset_ids) != 1:
        raise ValueError(f"examples from {len(dataset_ids)} datasets, expected 1")
    example_ids = sorted(str(example.id) for example in examples)
    return {
        "dataset_id": dataset_ids.pop(),
        "example_count": len(example_ids),
        "example_selection": hashlib.sha256(
            "\n".join(example_ids).encode()
        ).hexdigest(),
    }


def make_target(
    agent: CompiledStateGraph, effort: str, metadata: dict
) -> Callable[[dict], Awaitable[dict]]:
    """Build the experiment target: replay one example and return its transcript unit.

    The agent and its in-memory checkpointer are shared by every call, and LangSmith does
    not tell the target which repetition it runs. So each call takes the next repeat
    index of its thread, which gives each replay its own checkpoint `thread_id`.

    Args:
        agent: The compiled agent at the effort under test.
        effort: The reasoning effort under test.
        metadata: The run's `runner.trace_metadata`.

    Returns:
        The async target for `aevaluate`.
    """
    repeats: defaultdict[str, itertools.count] = defaultdict(itertools.count)

    async def target(inputs: dict) -> dict:
        thread = thread_from_inputs(inputs)
        repeat = next(repeats[thread["id"]])
        unit = await runner.replay_thread(agent, thread, effort, repeat, metadata)
        print(f"[{thread['id']:<16} {effort} #{repeat}] {runner.unit_progress(unit)}")
        return unit

    return target


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--effort",
        default=runner.DEFAULT_EFFORT,
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        help=f"Reasoning effort under test (default {runner.DEFAULT_EFFORT!r})",
    )
    parser.add_argument(
        "--repetitions", type=int, default=1, help="Replays per example"
    )
    parser.add_argument(
        "--split", choices=["ask", "query"], default=None, help="Only this split"
    )
    parser.add_argument(
        "--thread", action="append", help="Only this thread id (repeatable)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Parallel example replays"
    )
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM judge")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument(
        "--gold",
        default=str(ls_dataset.DEFAULT_GOLD),
        help="Gold file whose dataset version tag the experiment pins",
    )
    parser.add_argument("--dataset", default=ls_dataset.DEFAULT_DATASET)
    parser.add_argument(
        "--langsmith-project",
        default=None,
        help="LangSmith project for traces outside the experiment (default: <settings project>-eval)",
    )
    # An experiment always traces; `runner.configure_tracing` reads this flag.
    parser.set_defaults(no_trace=False)
    return parser.parse_args()


async def main() -> None:
    """Run one experiment over the pinned dataset version."""
    args = parse_args()
    _, tracing = runner.configure_tracing(args)
    if not tracing:
        sys.exit("an experiment needs LANGSMITH_TRACING=true")

    client = Client(api_key=settings.LANGSMITH_API_KEY)
    tag = ls_dataset.dataset_tag(args.gold)
    examples = [
        example
        for example in client.list_examples(
            dataset_name=args.dataset,
            as_of=tag,
            splits=[args.split] if args.split else None,
        )
        if not args.thread or (example.metadata or {}).get("thread") in args.thread
    ]
    if not examples:
        sys.exit(
            f"no examples in {args.dataset!r} at {tag!r}; "
            "run `uv run python -m eval.ls_dataset sync` first"
        )

    effort = args.effort
    branch = runner.current_branch()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_run = f"{branch.replace('/', '-')}_{effort}_{timestamp}"
    metadata = runner.trace_metadata(effort, branch, eval_run)
    judge_model = None if args.no_judge else args.judge_model

    scorers = [
        evaluators.deterministic,
        evaluators.tool_calling,
        evaluators.tool_correctness,
    ]
    if judge_model:
        scorers.append(evaluators.judge_evaluator(build_judge(judge_model)))

    print(
        f"\n{metadata['agent_config_label']}  effort={effort!r}  branch={branch!r}\n"
        f"dataset={args.dataset!r} at {tag!r}: {len(examples)} example(s)"
        f" x {args.repetitions} repetition(s)\n"
        f"judge: {judge_model or 'disabled'}\n"
    )

    agent = runner.build_agent(effort)
    results = await aevaluate(
        make_target(agent, effort, metadata),
        data=examples,
        evaluators=scorers,
        experiment_prefix=metadata["agent_config_label"],
        description=f"effort {effort} on {branch}, dataset {tag}",
        metadata={
            **metadata,
            **population_metadata(examples),
            "judge_model": judge_model,
            "dataset_tag": tag,
        },
        num_repetitions=args.repetitions,
        max_concurrency=args.concurrency,
        client=client,
    )
    print(f"\nexperiment {results.experiment_name!r} done")


if __name__ == "__main__":
    asyncio.run(main())
