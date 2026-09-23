from app.agent.observability import build_observability_metadata


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
