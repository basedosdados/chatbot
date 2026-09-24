from datetime import datetime
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from app.agent import observability, runtime_config
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
        "code_hash",
        "model_config_recorded_at",
        "provider",
        "model",
        "reasoning_effort",
        "reasoning_summary",
        "prompt_hash",
        "prompt_rendering_id",
        "tool_set_hash",
        "response_schema_hash",
        "summarization",
        "model_call_limit",
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
            "response_format_hash",
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


def test_response_format_changes_tool_identity(monkeypatch):
    tools = _tools()
    monkeypatch.setattr(BDToolkit, "get_tools", staticmethod(lambda: tools))
    first = build_observability_metadata("pt")["tools"][0]
    tools[0].response_format = "content_and_artifact"
    changed = build_observability_metadata("pt")["tools"][0]
    assert changed["response_format_hash"] != first["response_format_hash"]
    assert changed["id_tool"] != first["id_tool"]


class _OtherResponse(BaseModel):
    answer: str


@pytest.mark.parametrize(
    ("target", "name", "value"),
    [
        (runtime_config, "REASONING_SUMMARY", "detailed"),
        (runtime_config, "SUMMARIZATION_TRIGGER", ("tokens", 400_000)),
        (runtime_config, "SUMMARIZATION_KEEP", ("tokens", 50_000)),
        (runtime_config, "SUMMARIZATION_TRIM_TOKENS", 4_000),
        (runtime_config, "MODEL_CALL_RUN_LIMIT", 10),
        (runtime_config, "MODEL_CALL_EXIT_BEHAVIOR", "error"),
        (observability, "StructuredResponse", _OtherResponse),
    ],
)
def test_agent_config_id_changes_with_covered_runtime_setting(
    monkeypatch, target, name, value
):
    first_id = build_observability_metadata("pt")["agent_config_id"]
    assert build_observability_metadata("pt")["agent_config_id"] == first_id
    monkeypatch.setattr(target, name, value)
    assert build_observability_metadata("pt")["agent_config_id"] != first_id


def test_agent_config_id_ignores_environment(monkeypatch):
    first = build_observability_metadata("pt")
    other_environment = "staging" if first["environment"] != "staging" else "production"
    monkeypatch.setattr(
        observability,
        "settings",
        observability.settings.model_copy(update={"ENVIRONMENT": other_environment}),
    )
    changed = build_observability_metadata("pt")
    assert changed["environment"] == other_environment
    assert changed["agent_config_id"] == first["agent_config_id"]


def test_metadata_exposes_runtime_settings_without_response_schema():
    metadata = build_observability_metadata("pt")
    assert metadata["reasoning_summary"] == runtime_config.REASONING_SUMMARY
    assert metadata["summarization"]["trigger"] == runtime_config.SUMMARIZATION_TRIGGER
    limit = metadata["model_call_limit"]
    assert limit["run_limit"] == runtime_config.MODEL_CALL_RUN_LIMIT
    assert len(metadata["response_schema_hash"]) == 64
    assert "follow_up_prompts" not in str(metadata)


def test_code_hash_is_present_and_stable():
    metadata = build_observability_metadata("pt")
    assert len(metadata["code_hash"]) == 64
    assert build_observability_metadata("pt")["code_hash"] == metadata["code_hash"]


def test_code_hash_is_excluded_from_agent_config_id(monkeypatch):
    first = build_observability_metadata("pt")
    monkeypatch.setattr(observability, "_CODE_HASH", "0" * 64)
    changed = build_observability_metadata("pt")
    assert changed["code_hash"] == "0" * 64
    assert changed["agent_config_id"] == first["agent_config_id"]


def test_code_hash_changes_with_source_content(tmp_path):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    main_file = tmp_path / "main.py"
    (agent_dir / "observability.py").write_text("VALUE = 1\n")
    main_file.write_text("VALUE = 1\n")

    first = observability._hash_code_identity(agent_dir, main_file)
    assert first == observability._hash_code_identity(agent_dir, main_file)

    (agent_dir / "observability.py").write_text("VALUE = 2\n")
    assert observability._hash_code_identity(agent_dir, main_file) != first
