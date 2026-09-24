from datetime import datetime
from types import SimpleNamespace

from app.agent.observability import build_observability_metadata
from app.agent.tools import BDToolkit


def _tools():
    return [
        SimpleNamespace(
            name="alpha",
            description="alpha tool",
            args={"type": "object", "properties": {"value": {"type": "string"}}},
            response_format="content",
        ),
        SimpleNamespace(
            name="beta",
            description="beta tool",
            args={"type": "object", "properties": {}},
            response_format="content_and_artifact",
        ),
    ]


def test_build_observability_metadata_includes_redacted_hashes():
    metadata = build_observability_metadata("pt")

    assert metadata["environment"]
    assert metadata["model_config_recorded_at"].endswith("+00:00")
    assert metadata["provider"] == "openai"
    assert metadata["model"]
    assert metadata["reasoning_effort"]
    assert len(metadata["prompt_hash"]) == 64
    assert len(metadata["prompt_rendering_id"]) == 64
    assert len(metadata["tool_set_hash"]) == 64
    assert len(metadata["agent_config_id"]) == 64
    assert "You are the research assistant" not in str(metadata)


def test_prompt_rendering_identity_varies_by_language():
    assert build_observability_metadata("pt")["prompt_rendering_id"] != (
        build_observability_metadata("en")["prompt_rendering_id"]
    )


def test_tools_have_exact_redacted_fields_and_shared_utc_snapshot(monkeypatch):
    monkeypatch.setattr(BDToolkit, "get_tools", staticmethod(_tools))
    metadata = build_observability_metadata("pt")

    assert set(metadata) == {
        "environment",
        "model_config_recorded_at",
        "provider",
        "model",
        "reasoning_effort",
        "prompt_hash",
        "prompt_rendering_id",
        "tool_set_hash",
        "tools",
        "agent_config_id",
    }
    assert len(metadata["tools"]) == 2
    timestamps = {tool["created_at"] for tool in metadata["tools"]}
    assert len(timestamps) == 1
    assert datetime.fromisoformat(next(iter(timestamps))).isoformat().endswith("+00:00")
    for tool in metadata["tools"]:
        assert set(tool) == {
            "name",
            "docstring_hash",
            "input_schema_hash",
            "output_schema_hash",
            "id_tool",
            "docstring",
            "created_at",
        }
        assert tool["docstring"] in {"alpha tool", "beta tool"}
        assert "updated_at" not in tool
        assert all(
            len(tool[key]) == 64
            for key in tool
            if key.endswith("hash") or key == "id_tool"
        )


def test_tool_identity_is_stable_and_changes_with_definition(monkeypatch):
    tools = _tools()
    monkeypatch.setattr(BDToolkit, "get_tools", staticmethod(lambda: tools))
    first = build_observability_metadata("pt")
    second = build_observability_metadata("pt")
    assert [tool["id_tool"] for tool in first["tools"]] == [
        tool["id_tool"] for tool in second["tools"]
    ]
    assert first["tool_set_hash"] == second["tool_set_hash"]
    tools[0].description = "changed"
    changed_metadata = build_observability_metadata("pt")
    assert changed_metadata["tools"][0]["docstring"] == "changed"
    assert changed_metadata["tools"][0]["id_tool"] != first["tools"][0]["id_tool"]
    tools[0].description = "alpha tool"
    tools[0].name = "renamed"
    assert build_observability_metadata("pt")["tools"][0]["id_tool"] != first["tools"][0]["id_tool"]
