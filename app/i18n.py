from enum import StrEnum
from typing import Literal, get_args

LanguageCode = Literal["pt", "en", "es"]

_LANGUAGES: tuple[LanguageCode, ...] = get_args(LanguageCode)

_LANGUAGE_NAMES: dict[LanguageCode, str] = {
    "pt": "Portuguese",
    "en": "English",
    "es": "Spanish",
}

_LANGUAGE_FIELD_SUFFIX: dict[LanguageCode, str] = {
    "pt": "Pt",
    "en": "En",
    "es": "Es",
}

DEFAULT_LANGUAGE: LanguageCode = "pt"


def normalize_language(value: str | None) -> LanguageCode:
    """Coerce an arbitrary language value to a supported code, falling back to the default.

    Args:
        value (str | None): A raw language value from a request.

    Returns:
        LanguageCode: One of the supported language codes.
    """
    normalized = (value or DEFAULT_LANGUAGE).lower()
    return normalized if normalized in _LANGUAGES else DEFAULT_LANGUAGE


def localized_field(node: dict, field: str, language: LanguageCode) -> str | None:
    """Pick a GraphQL node's localized `field` for `language`, pt as fallback.

    Args:
        node (dict): A GraphQL node exposing the fields.
        field (str): The base field name, e.g. "name" or "description".
        language (LanguageCode): A supported language code.

    Returns:
        str | None: The requested language's value, the pt value when it is empty
            (coverage is partial), or None when neither is set.
    """
    suffix = _LANGUAGE_FIELD_SUFFIX[language]
    return node.get(f"{field}{suffix}") or node.get(f"{field}Pt")


def language_directive(language: LanguageCode) -> str:
    """Build the instruction that sets the site's language as the response default.

    Args:
        language (LanguageCode): A supported language code.

    Returns:
        str: A one-line directive.
    """
    name = _LANGUAGE_NAMES[language]

    return (
        f"The interface language is {name}. Respond in {name} unless the user clearly "
        f"writes in another language, in which case respond in that language."
    )


# ===========================================================================
# ==                    Server-emitted user-facing text                    ==
# ===========================================================================
class MessageKey(StrEnum):
    """Keys for server-emitted, user-facing strings (see `translate`)."""

    ERROR_INTERRUPTED = "error_interrupted"
    ERROR_MODEL_CALL_LIMIT = "error_model_call_limit"
    ERROR_UNEXPECTED = "error_unexpected"
    RESULTS_EXPIRED = "results_expired"
    RESULTS_TOO_LARGE = "results_too_large"
    DEFAULT_EXPORT_FILENAME = "default_export_filename"


# Keep every key populated for all of _LANGUAGES (enforced by tests).
_MESSAGES = {
    MessageKey.ERROR_INTERRUPTED: {
        "pt": "A conexão com o servidor foi interrompida. Por favor, tente novamente.",
        "en": "The connection to the server was interrupted. Please try again.",
        "es": "Se interrumpió la conexión con el servidor. Por favor, inténtalo de nuevo.",
    },
    MessageKey.ERROR_MODEL_CALL_LIMIT: {
        "pt": (
            "Essa pergunta gerou um raciocínio muito longo e não consegui chegar a uma "
            "conclusão. Por favor, tente ser mais específico ou divida sua pergunta em "
            "partes menores."
        ),
        "en": (
            "This question led to a very long chain of reasoning and I couldn't reach a "
            "conclusion. Please try to be more specific or break your question into smaller "
            "parts."
        ),
        "es": (
            "Esta pregunta generó un razonamiento demasiado largo y no pude llegar a una "
            "conclusión. Intenta ser más específico o divide tu pregunta en partes más "
            "pequeñas."
        ),
    },
    MessageKey.ERROR_UNEXPECTED: {
        "pt": "Ocorreu um erro inesperado. Por favor, tente novamente. Se o problema persistir, avise-nos.",
        "en": "An unexpected error occurred. Please try again. If the problem persists, let us know.",
        "es": "Ocurrió un error inesperado. Inténtalo de nuevo. Si el problema persiste, avísanos.",
    },
    MessageKey.RESULTS_EXPIRED: {
        "pt": "Estes resultados não estão mais disponíveis para download.",
        "en": "These results are no longer available for download.",
        "es": "Estos resultados ya no están disponibles para descargar.",
    },
    MessageKey.RESULTS_TOO_LARGE: {
        "pt": "Estes resultados são grandes demais para baixar em um único arquivo.",
        "en": "These results are too large to download in a single file.",
        "es": "Estos resultados son demasiado grandes para descargar en un solo archivo.",
    },
    MessageKey.DEFAULT_EXPORT_FILENAME: {
        "pt": "resultados",
        "en": "results",
        "es": "resultados",
    },
}


def translate(key: MessageKey, language: LanguageCode) -> str:
    """Return the localized string for `key` in `language`.

    Args:
        key (MessageKey): The message to localize.
        language (LanguageCode): A supported language code.

    Returns:
        str: The localized string.
    """
    return _MESSAGES[key][language]
