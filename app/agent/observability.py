"""Redacted, deterministic identities for LangSmith trace metadata."""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from app.agent import runtime_config
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.schemas import StructuredResponse
from app.agent.tools import BDToolkit
from app.settings import settings


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _tool_identity(tool: Any) -> dict[str, str]:
    # LangChain's `args` is already the JSON-schema-shaped tool input exposed
    # to the model. `args_schema.model_json_schema()` may fail because the
    # internal Pydantic model also includes injected runtime parameters.
    schema_json = getattr(tool, "args", {})
    identity = {
        "name": tool.name,
        "docstring_hash": _hash(tool.description or ""),
        "input_schema_hash": _hash(schema_json),
        "response_format_hash": _hash(getattr(tool, "response_format", "content")),
    }
    identity["id_tool"] = _hash(identity)
    return identity


def build_observability_metadata(language: str) -> dict[str, Any]:
    """Build trace-safe identities; never includes prompt or tool content."""
    snapshot_created_at = datetime.now(timezone.utc).isoformat()
    tools = sorted(BDToolkit.get_tools(), key=lambda tool: tool.name)
    tool_snapshots = []
    for tool in tools:
        snapshot = _tool_identity(tool)
        snapshot["docstring"] = tool.description or ""
        snapshot["created_at"] = snapshot_created_at
        tool_snapshots.append(snapshot)
    prompt_hash = _hash(SYSTEM_PROMPT)
    tool_set_hash = _hash(
        [{key: value for key, value in tool.items() if key != "created_at"} for tool in tool_snapshots]
    )
    config = {
        "provider": "openai",
        "model": settings.MODEL_URI,
        "reasoning_effort": settings.REASONING_EFFORT,
        "reasoning_summary": runtime_config.REASONING_SUMMARY,
        "prompt_hash": prompt_hash,
        "prompt_rendering_id": _hash({"prompt_hash": prompt_hash, "language": language}),
        "tool_set_hash": tool_set_hash,
        "response_schema_hash": _hash(StructuredResponse.model_json_schema()),
        "summarization": {
            "trigger": runtime_config.SUMMARIZATION_TRIGGER,
            "keep": runtime_config.SUMMARIZATION_KEEP,
            "trim_tokens_to_summarize": runtime_config.SUMMARIZATION_TRIM_TOKENS,
        },
        "model_call_limit": {
            "run_limit": runtime_config.MODEL_CALL_RUN_LIMIT,
            "exit_behavior": runtime_config.MODEL_CALL_EXIT_BEHAVIOR,
        },
    }
    return {
        "environment": settings.ENVIRONMENT,
        "model_config_recorded_at": datetime.now(timezone.utc).isoformat(),
        **config,
        "tools": tool_snapshots,
        "agent_config_id": _hash(config),
    }
