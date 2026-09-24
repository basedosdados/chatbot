"""Redacted, deterministic identities for LangSmith trace metadata."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.agent import runtime_config
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.schemas import StructuredResponse
from app.agent.tools import BDToolkit
from app.settings import settings

# The two source locations that define agent behavior. Hashed once at import
# time: the container never changes these files while the process is running.
_AGENT_DIR = Path(__file__).resolve().parent
_MAIN_FILE = _AGENT_DIR.parent / "main.py"


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _hash_code_identity(agent_dir: Path, main_file: Path) -> str:
    """Content hash of the agent's own source, computed in Python.

    Unlike a Git SHA, this needs no value injected from outside the running
    process — `.dockerignore` excludes `.git/`, so nothing inside the container
    can read the real commit SHA. Hashing the source content that is already
    here gives an equivalent identity: it changes whenever this code changes,
    computed the same way `prompt_hash`/`tool_set_hash` already are.
    """
    files = sorted(agent_dir.rglob("*.py")) + [main_file]
    payload = {
        str(path.relative_to(agent_dir.parent)): path.read_text() for path in files
    }
    return _hash(payload)


_CODE_HASH = _hash_code_identity(_AGENT_DIR, _MAIN_FILE)

# Process start, computed once. `model`/`reasoning_effort`/`provider` come from
# env vars fixed for the process lifetime, so this timestamp approximates
# "since when has this configuration been in effect" — a restart is how a new
# value takes effect. No CI/build injection needed, same as `_CODE_HASH`.
_CONFIG_EFFECTIVE_SINCE = datetime.now(timezone.utc).isoformat()


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
        # `code_hash` identifies the code, not the configuration, so it stays
        # out of `agent_config_id`.
        "code_hash": _CODE_HASH,
        # Unlike `model_config_recorded_at` (recomputed per trace), this is
        # fixed once at process start — it identifies the deploy, not the trace.
        "config_effective_since": _CONFIG_EFFECTIVE_SINCE,
        "model_config_recorded_at": datetime.now(timezone.utc).isoformat(),
        **config,
        "tools": tool_snapshots,
        "agent_config_id": (agent_config_id := _hash(config)),
        # Chart label only; derived from `config`, so it stays out of
        # `agent_config_id` itself. Lets a LangSmith chart grouped by this key
        # show the model name instead of a bare hash.
        "agent_config_label": f"{config['model']} · {agent_config_id[:8]}",
    }


def build_observability_tags(metadata: dict[str, Any]) -> list[str]:
    """LangSmith native run tags, derived from an already-built metadata dict.

    Tags are a true multi-value list (unlike a Git tag name), so each fact gets
    its own `key:value` entry — searchable and filterable independently in the
    LangSmith UI, without writing `metadata.<field>:<value>` queries.
    """
    return [
        f"provider:{metadata['provider']}",
        f"model:{metadata['model']}",
        # Day granularity: process restarts within the same day (e.g. replica
        # rollout) still land in one release group.
        f"release_date:{metadata['config_effective_since'][:10]}",
        f"toolset:{metadata['tool_set_hash'][:8]}",
    ]
