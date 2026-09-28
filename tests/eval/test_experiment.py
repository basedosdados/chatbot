import asyncio
from types import SimpleNamespace

import pytest
from google.auth.credentials import AnonymousCredentials
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI

from eval import experiment, runner
from eval.lib import judge


def test_thread_from_inputs_rebuilds_the_gold_thread_shape():
    inputs = {"thread_id": "t", "turns": ["first", "second"]}

    assert experiment.thread_from_inputs(inputs) == {
        "id": "t",
        "turns": [{"user": "first"}, {"user": "second"}],
    }


def _example(example_id, dataset_id="d1"):
    return SimpleNamespace(id=example_id, dataset_id=dataset_id)


def test_population_metadata_fingerprints_the_selected_examples():
    everything = [_example("a"), _example("b"), _example("c")]

    metadata = experiment.population_metadata(everything)

    assert metadata["dataset_id"] == "d1"
    assert metadata["example_count"] == 3
    # The order of the examples does not change the fingerprint.
    assert experiment.population_metadata(everything[::-1]) == metadata
    subset = experiment.population_metadata(everything[:2])
    assert subset["example_selection"] != metadata["example_selection"]
    other_dataset = experiment.population_metadata([_example("a", "d2")])
    assert other_dataset["dataset_id"] == "d2"


def test_population_metadata_rejects_examples_from_two_datasets():
    with pytest.raises(ValueError, match="2 datasets"):
        experiment.population_metadata([_example("a"), _example("b", "d2")])


def test_target_gives_each_replay_of_a_thread_its_own_repeat(monkeypatch):
    calls = []

    async def fake_replay(agent, thread, effort, repeat, metadata):
        calls.append((thread["id"], repeat))
        return {"thread": thread["id"], "repeat": repeat, "turns": []}

    monkeypatch.setattr(runner, "replay_thread", fake_replay)
    target = experiment.make_target(agent=None, effort="medium", metadata={})

    async def run_all():
        for thread_id in ["a", "a", "b", "a"]:
            await target({"thread_id": thread_id, "turns": ["q"]})

    asyncio.run(run_all())

    assert calls == [("a", 0), ("a", 1), ("b", 0), ("a", 2)]


def test_default_judge_builds_without_a_model_call(monkeypatch):
    for name in (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        monkeypatch.delenv(name, raising=False)

    class FakeServiceAccount(AnonymousCredentials):
        project_id = "fake-project"

    monkeypatch.setattr(
        judge, "settings", SimpleNamespace(GOOGLE_CREDENTIALS=FakeServiceAccount())
    )
    built = []

    def spy_init_chat_model(*args, **kwargs):
        built.append(init_chat_model(*args, **kwargs))
        return built[-1]

    def no_model_call(*args, **kwargs):
        raise AssertionError("building the judge must not call the model")

    monkeypatch.setattr(judge, "init_chat_model", spy_init_chat_model)
    monkeypatch.setattr(ChatGoogleGenerativeAI, "_generate", no_model_call)
    monkeypatch.setattr(ChatGoogleGenerativeAI, "_agenerate", no_model_call)

    structured = judge.build_judge(experiment.DEFAULT_JUDGE_MODEL)

    (model,) = built
    assert isinstance(model, ChatGoogleGenerativeAI)
    assert f"google_genai:{model.model}" == experiment.DEFAULT_JUDGE_MODEL
    assert model.project == "fake-project"
    assert model.temperature == 0
    assert structured is not None
