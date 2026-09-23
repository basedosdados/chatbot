"""Redacted, deterministic identities for LangSmith trace metadata."""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from app.agent.prompts import SYSTEM_PROMPT
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
    return {
        "name": tool.name,
        "docstring_hash": _hash(tool.description or ""),
        "input_schema_hash": _hash(schema_json),
        "output_schema_hash": _hash(getattr(tool, "response_format", "content")),
    }


def build_observability_metadata(language: str) -> dict[str, str]:
    """Build trace-safe identities; never includes prompt or tool content."""
    tools = sorted((_tool_identity(tool) for tool in BDToolkit.get_tools()), key=lambda x: x["name"])
    prompt_hash = _hash(SYSTEM_PROMPT)
    tool_set_hash = _hash(tools)
    config = {
        "provider": "openai",
        "model": settings.MODEL_URI,
        "reasoning_effort": settings.REASONING_EFFORT,
        "prompt_hash": prompt_hash,
        "prompt_rendering_id": _hash({"prompt_hash": prompt_hash, "language": language}),
        "tool_set_hash": tool_set_hash,
    }
    return {
        "environment": settings.ENVIRONMENT,
        "model_config_recorded_at": datetime.now(timezone.utc).isoformat(),
        "provider": config["provider"],
        "model": config["model"],
        "reasoning_effort": config["reasoning_effort"],
        "prompt_hash": prompt_hash,
        "prompt_rendering_id": config["prompt_rendering_id"],
        "tool_set_hash": tool_set_hash,
        "agent_config_id": _hash(config),
    }
