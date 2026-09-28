from types import SimpleNamespace

from eval import ls_dataset

THREADS = [
    {"id": "ask-one", "turns": [{"user": "Economia", "action": "ask"}]},
    {
        "id": "query-two",
        "turns": [
            {"user": "q1", "action": "query", "period": "latest"},
            {"user": "q2", "action": "ask"},
        ],
    },
]


def _stored(example_id, thread):
    """A stored example as LangSmith returns it, split kept in metadata."""
    wanted = ls_dataset.build_example(thread)
    return SimpleNamespace(
        id=example_id,
        inputs=wanted["inputs"],
        outputs=wanted["outputs"],
        metadata={**wanted["metadata"], "dataset_split": [wanted["split"]]},
    )


def test_build_example_maps_thread_to_inputs_outputs_and_split():
    example = ls_dataset.build_example(THREADS[1])

    assert example == {
        "inputs": {"thread_id": "query-two", "turns": ["q1", "q2"]},
        "outputs": {"turns": THREADS[1]["turns"]},
        "metadata": {"thread": "query-two"},
        "split": "query",
    }


def test_plan_sync_creates_every_thread_on_an_empty_dataset():
    to_create, to_update, to_delete = ls_dataset.plan_sync(THREADS, [])

    assert [example["metadata"]["thread"] for example in to_create] == [
        "ask-one",
        "query-two",
    ]
    assert to_update == []
    assert to_delete == []


def test_plan_sync_is_a_no_op_when_the_dataset_matches():
    stored = [_stored("id-1", THREADS[0]), _stored("id-2", THREADS[1])]

    assert ls_dataset.plan_sync(THREADS, stored) == ([], [], [])


def test_plan_sync_updates_a_changed_thread_in_place():
    stored = [_stored("id-1", THREADS[0]), _stored("id-2", THREADS[1])]
    changed = [THREADS[0], {**THREADS[1], "turns": [{"user": "q1", "action": "ask"}]}]

    to_create, to_update, to_delete = ls_dataset.plan_sync(changed, stored)

    assert to_create == []
    assert [update["id"] for update in to_update] == ["id-2"]
    assert to_update[0]["split"] == "ask"
    assert to_delete == []


def test_plan_sync_updates_an_example_whose_split_is_missing():
    stored = [_stored("id-1", THREADS[0])]
    stored[0].metadata = {"thread": "ask-one"}

    _, to_update, _ = ls_dataset.plan_sync(THREADS[:1], stored)

    assert [update["id"] for update in to_update] == ["id-1"]


def test_plan_sync_deletes_a_thread_removed_from_the_gold():
    stored = [_stored("id-1", THREADS[0]), _stored("id-2", THREADS[1])]

    _, _, to_delete = ls_dataset.plan_sync(THREADS[:1], stored)

    assert [example.id for example in to_delete] == ["id-2"]


def test_dataset_tag_follows_the_gold_file_content(tmp_path):
    gold_file = tmp_path / "gold.yaml"
    gold_file.write_text("- id: a\n")
    first = ls_dataset.dataset_tag(gold_file)

    assert first.startswith("gold-") and len(first) == len("gold-") + 12
    assert ls_dataset.dataset_tag(gold_file) == first
    gold_file.write_text("- id: b\n")
    assert ls_dataset.dataset_tag(gold_file) != first


def test_plan_sync_deletes_duplicate_copies_of_one_thread():
    stored = [
        _stored("id-1", THREADS[0]),
        _stored("id-1-copy", THREADS[0]),
        _stored("id-2", THREADS[1]),
    ]

    to_create, to_update, to_delete = ls_dataset.plan_sync(THREADS, stored)

    assert to_create == []
    assert to_update == []
    assert [example.id for example in to_delete] == ["id-1-copy"]
