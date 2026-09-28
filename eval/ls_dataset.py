"""Sync the gold threads into the LangSmith dataset the experiments run over.

One dataset example is one gold thread:

    inputs   = {"thread_id": id, "turns": [user message, ...]}
    outputs  = {"turns": [gold turn dict, ...]}
    metadata = {"thread": id}
    split    = the first turn's action (`ask` or `query`)

Examples are matched by `metadata.thread`, so a re-sync updates an example in place
instead of adding a copy, and an unchanged thread is left alone. A thread removed from
the gold file is deleted from the dataset; LangSmith keeps the old dataset versions, so
an older experiment can still read it with `as_of`.

After a sync, the current dataset version gets the tag `gold-<hash>`, where the hash is
the gold file's content hash. An experiment pins that tag, so a baseline and a candidate
run over the same examples.

    uv run python -m eval.ls_dataset sync --dry-run
    uv run python -m eval.ls_dataset sync
"""

import argparse
import hashlib
from pathlib import Path
from typing import Any

from langsmith import Client

from app.settings import settings
from eval.lib import gold

EVAL_DIR = Path(__file__).resolve().parent

DEFAULT_GOLD = EVAL_DIR / "eval_gold_lean.yaml"
DEFAULT_DATASET = "bd-chatbot-gold"

# LangSmith keeps an example's split in its metadata under this key, as a list.
_SPLIT_KEY = "dataset_split"


def dataset_tag(gold_path: str | Path) -> str:
    """The dataset version tag for a gold file: `gold-` plus its content hash prefix.

    Args:
        gold_path: Path to the gold YAML file.

    Returns:
        The tag name, for example `gold-3f9a1c0b2d4e`.
    """
    return f"gold-{hashlib.sha256(Path(gold_path).read_bytes()).hexdigest()[:12]}"


def build_example(thread: dict) -> dict[str, Any]:
    """Map one gold thread to one dataset example.

    Args:
        thread: A gold thread (`{"id": ..., "turns": [...]}`).

    Returns:
        The example as an `inputs` / `outputs` / `metadata` / `split` dict.
    """
    return {
        "inputs": {
            "thread_id": thread["id"],
            "turns": [turn["user"] for turn in thread["turns"]],
        },
        "outputs": {"turns": thread["turns"]},
        "metadata": {"thread": thread["id"]},
        "split": thread["turns"][0]["action"],
    }


def _is_current(example: Any, wanted: dict[str, Any]) -> bool:
    """Whether a stored example already holds the wanted content."""
    metadata = example.metadata or {}
    return (
        example.inputs == wanted["inputs"]
        and example.outputs == wanted["outputs"]
        and metadata.get("thread") == wanted["metadata"]["thread"]
        and metadata.get(_SPLIT_KEY) == [wanted["split"]]
    )


def plan_sync(
    threads: list[dict], stored: list[Any]
) -> tuple[list[dict], list[dict], list[Any]]:
    """Split the gold threads into examples to create, to update, and to delete.

    Args:
        threads: The validated gold threads.
        stored: The examples the dataset holds now.

    Returns:
        `(to_create, to_update, to_delete)`. An update dict carries the stored `id`.
        `to_delete` holds stored examples whose thread left the gold file, and every
        copy after the first when the dataset holds one thread more than once.
    """
    by_thread: dict[str | None, Any] = {}
    duplicates = []
    for example in stored:
        thread_id = (example.metadata or {}).get("thread")
        if thread_id in by_thread:
            duplicates.append(example)
        else:
            by_thread[thread_id] = example
    to_create, to_update = [], []
    for thread in threads:
        wanted = build_example(thread)
        example = by_thread.pop(thread["id"], None)
        if example is None:
            to_create.append(wanted)
        elif not _is_current(example, wanted):
            to_update.append({"id": example.id, **wanted})
    return to_create, to_update, list(by_thread.values()) + duplicates


def sync(
    client: Client, gold_path: str | Path, dataset_name: str, dry_run: bool
) -> None:
    """Upsert every gold thread into the dataset, then tag the dataset version.

    Args:
        client: The LangSmith client.
        gold_path: Path to the gold YAML file.
        dataset_name: The LangSmith dataset name.
        dry_run: Print the plan and write nothing.
    """
    threads = gold.validate(gold.load_threads(gold_path))
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        stored = list(client.list_examples(dataset_id=dataset.id))
    else:
        dataset, stored = None, []

    to_create, to_update, to_delete = plan_sync(threads, stored)
    tag = dataset_tag(gold_path)
    print(f"dataset {dataset_name!r} from {gold_path}")
    print(
        f"  create {len(to_create)}, update {len(to_update)}, delete {len(to_delete)}"
    )
    print(f"  unchanged {len(threads) - len(to_create) - len(to_update)}, tag {tag!r}")
    if dry_run:
        return

    if dataset is None:
        dataset = client.create_dataset(
            dataset_name,
            description="Base dos Dados chatbot gold threads (one example per thread).",
        )
    if to_create:
        client.create_examples(dataset_id=dataset.id, examples=to_create)
    if to_update:
        client.update_examples(dataset_id=dataset.id, updates=to_update)
    if to_delete:
        client.delete_examples([example.id for example in to_delete])

    # Tag the newest version as the server records it, not the local clock, so the
    # tag cannot point to a version before the writes above.
    latest = max(
        client.list_dataset_versions(dataset_id=dataset.id),
        key=lambda version: version.as_of,
    )
    client.update_dataset_tag(dataset_id=dataset.id, as_of=latest.as_of, tag=tag)
    print(f"  tagged version {latest.as_of.isoformat()} as {tag!r}")


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["sync"])
    parser.add_argument("--gold", default=str(DEFAULT_GOLD))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan and write nothing"
    )
    return parser.parse_args()


def main() -> None:
    """Run the chosen command."""
    args = parse_args()
    client = Client(api_key=settings.LANGSMITH_API_KEY)
    sync(client, args.gold, args.dataset, args.dry_run)


if __name__ == "__main__":
    main()
