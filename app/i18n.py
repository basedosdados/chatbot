"""Localization for user-facing backend strings and per-run language steering.

The chatbot serves three locales, one per Base dos Dados / Data Basis domain:
`pt` (basedosdados.org), `en` (data-basis.org), `es` (basedelosdatos.org). A thread's
language is captured at creation from the site the user is on (see `Thread.language`) and
threaded into each run. The model's system prompt already asks it to answer in the user's
language; `language_directive` gives it the site default while still honoring a user who
writes in another language.

Only strings the server itself emits live here — agent errors and download details. Agent
answers are localized by the model, and step labels are rendered by the frontend.
"""

from typing import Literal

LanguageCode = Literal["pt", "en", "es"]

LANGUAGES: tuple[str, ...] = ("pt", "en", "es")
DEFAULT_LANGUAGE: str = "pt"

# Endonyms would read better in-product, but the directive is an instruction to the model,
# so English names keep it unambiguous regardless of the target language.
LANGUAGE_NAMES: dict[str, str] = {
    "pt": "Portuguese",
    "en": "English",
    "es": "Spanish",
}


def normalize_language(value: str | None) -> str:
    """Coerce an arbitrary language value to a supported code, falling back to the default.

    Args:
        value (str | None): A raw language value (e.g. from a request or a stored row).

    Returns:
        str: One of `LANGUAGES`.
    """
    normalized = (value or DEFAULT_LANGUAGE).lower()
    return normalized if normalized in LANGUAGES else DEFAULT_LANGUAGE


def language_directive(language: str) -> str:
    """Build the per-run instruction that sets the site's language as the default.

    "Domain default, honor the user": the model answers in the site's language unless the
    user clearly writes in another language, in which case it matches the user.

    Args:
        language (str): A supported language code (unsupported values fall back to default).

    Returns:
        str: A one-line directive to prepend to the user's turn.
    """
    name = LANGUAGE_NAMES[normalize_language(language)]
    return (
        f"[Interface language: {name}. Respond in {name} unless the user clearly writes in a "
        f"different language, in which case respond in that language. Do not mention this note.]"
    )


# ==============================================================================
# ==                       Server-emitted user-facing text                    ==
# ==============================================================================
# key -> {language: text}. Keep every key populated for all of LANGUAGES.
_MESSAGES: dict[str, dict[str, str]] = {
    "error_interrupted": {
        "pt": "A conexão com o servidor foi interrompida. Por favor, tente novamente.",
        "en": "The connection to the server was interrupted. Please try again.",
        "es": "Se interrumpió la conexión con el servidor. Por favor, inténtalo de nuevo.",
    },
    "error_model_call_limit": {
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
    "error_unexpected": {
        "pt": (
            "Ocorreu um erro inesperado. Por favor, tente novamente. Se o problema "
            "persistir, avise-nos."
        ),
        "en": (
            "An unexpected error occurred. Please try again. If the problem persists, let "
            "us know."
        ),
        "es": (
            "Ocurrió un error inesperado. Inténtalo de nuevo. Si el problema persiste, "
            "avísanos."
        ),
    },
    "results_expired": {
        "pt": "Estes resultados não estão mais disponíveis para download.",
        "en": "These results are no longer available for download.",
        "es": "Estos resultados ya no están disponibles para descargar.",
    },
    "results_too_large": {
        "pt": "Estes resultados são grandes demais para baixar em um único arquivo.",
        "en": "These results are too large to download in a single file.",
        "es": "Estos resultados son demasiado grandes para descargar en un solo archivo.",
    },
    "default_export_filename": {
        "pt": "resultados",
        "en": "results",
        "es": "resultados",
    },
}


def t(key: str, language: str) -> str:
    """Return the localized string for `key` in `language`.

    Args:
        key (str): A key present in `_MESSAGES`.
        language (str): A language code (unsupported values fall back to the default).

    Returns:
        str: The localized string.
    """
    return _MESSAGES[key][normalize_language(language)]


# GraphQL modeltranslation columns are exposed as `{field}Pt`/`{field}En`/`{field}Es`.
_LANG_FIELD_SUFFIX: dict[str, str] = {"pt": "Pt", "en": "En", "es": "Es"}


def localized_field(node: dict, field: str, language: str) -> str | None:
    """Pick a GraphQL node's localized `field` for `language`, pt as fallback.

    modeltranslation exposes per-language columns as `{field}Pt`/`{field}En`/
    `{field}Es` (e.g. `name` -> `namePt`/`nameEn`/`nameEs`, `description` ->
    `descriptionPt`/...). The unqualified accessor (`name`) is deliberately not
    used: it returns the server's active-language value, ambiguous for a
    headless request.

    Args:
        node (dict): A GraphQL node exposing the modeltranslation columns.
        field (str): The base field name, e.g. "name" or "description".
        language (str): A language code; unsupported values fall back to default.

    Returns:
        str | None: The requested language's value, the pt value when it is empty
            (coverage is partial), or None when neither is set.
    """
    suffix = _LANG_FIELD_SUFFIX[normalize_language(language)]
    return node.get(f"{field}{suffix}") or node.get(f"{field}Pt")
