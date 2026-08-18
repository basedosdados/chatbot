from dataclasses import dataclass

from app.i18n import LanguageCode


@dataclass
class AgentContext:
    """Per-run context for an agent run.

    Read by:
      - The system-prompt middleware (`language`)
      - The tools (`language` for localized metadata; `thread_id` and `user_id` for BigQuery job labels).

    `thread_id` is *also* kept in `config["configurable"]` because the langgraph
    checkpointer keys persistence on it there; on this context it is app metadata.
    """

    thread_id: str
    user_id: str
    language: LanguageCode
