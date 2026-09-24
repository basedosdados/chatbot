"""Fixed agent runtime parameters shared by the agent setup and trace identities."""

from typing import Final

REASONING_SUMMARY: Final = "auto"

# Once the running context passes the trigger, summarize: older turns
# collapse into one summary while the most recent tokens are kept verbatim,
# and the summary is built from the full discarded history (no trimming).
SUMMARIZATION_TRIGGER: Final = ("tokens", 500_000)
SUMMARIZATION_KEEP: Final = ("tokens", 100_000)
SUMMARIZATION_TRIM_TOKENS: Final = None

MODEL_CALL_RUN_LIMIT: Final = 20
MODEL_CALL_EXIT_BEHAVIOR: Final = "end"
