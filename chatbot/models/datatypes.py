from enum import Enum


class ModelURI(str, Enum):
    gpt_4o = "openai/gpt-4o"
    gpt_4o_mini = "openai/gpt-4o-mini"
    gemini_1_5_flash = "google/gemini-1.5-flash-001"

    @classmethod
    def values(cls) -> list[str]:
        return [member.value for member in cls]
