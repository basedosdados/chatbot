import asyncio

from eval import experiment, runner


def test_thread_from_inputs_rebuilds_the_gold_thread_shape():
    inputs = {"thread_id": "t", "turns": ["first", "second"]}

    assert experiment.thread_from_inputs(inputs) == {
        "id": "t",
        "turns": [{"user": "first"}, {"user": "second"}],
    }


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
