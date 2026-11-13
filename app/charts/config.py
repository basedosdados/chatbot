from typing import Any

from app.i18n import LanguageCode

# Number formats per language, set on every chart (see `_server_config`).
_NUMBER_LOCALES: dict[LanguageCode, dict[str, Any]] = {
    "pt": {"decimal": ",", "thousands": ".", "grouping": [3], "currency": ["R$ ", ""]},
    "es": {"decimal": ",", "thousands": ".", "grouping": [3], "currency": ["$", ""]},
    "en": {"decimal": ".", "thousands": ",", "grouping": [3], "currency": ["$", ""]},
}

# Datetime formats per language, set on every chart (see `_server_config`).
_TIME_LOCALES: dict[LanguageCode, dict[str, Any]] = {
    "pt": {
        "dateTime": "%A, %e de %B de %Y. %X",
        "date": "%d/%m/%Y",
        "time": "%H:%M:%S",
        "periods": ["AM", "PM"],
        "days": ["Domingo", "Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado"],
        "shortDays": ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sáb"],
        "months": [
            "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
            "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro",
        ],
        "shortMonths": [
            "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
            "Jul", "Ago", "Set", "Out", "Nov", "Dez",
        ],
    },
    "es": {
        "dateTime": "%A, %e de %B de %Y, %X",
        "date": "%d/%m/%Y",
        "time": "%H:%M:%S",
        "periods": ["AM", "PM"],
        "days": ["domingo", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado"],
        "shortDays": ["dom", "lun", "mar", "mié", "jue", "vie", "sáb"],
        "months": [
            "enero", "febrero", "marzo", "abril", "mayo", "junio",
            "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
        ],
        "shortMonths": [
            "ene", "feb", "mar", "abr", "may", "jun",
            "jul", "ago", "sep", "oct", "nov", "dic",
        ],
    },
}  # fmt: skip

# Marks that show a tooltip of their encoded fields unless the spec sets its own. Lines
# and areas are left out: one mark spans many rows, so its tooltip would show only one.
_TOOLTIP_MARKS = ("bar", "point", "circle", "square", "rect", "arc", "geoshape")


def _server_config(language: LanguageCode) -> dict[str, Any]:
    """Build the Vega-Lite `config` the server sets on every chart.

    It holds what does not depend on the chart and the model should not have to get
    right: number and date formats in the user's language, and a tooltip on marks that
    stand for one row. Colors, fonts and sizes stay with the website, which merges its
    own config under the spec's.

    Args:
        language (LanguageCode): The language of the chart's text.

    Returns:
        dict[str, Any]: The config.
    """
    locale: dict[str, Any] = {"number": _NUMBER_LOCALES[language]}
    if language in _TIME_LOCALES:
        locale["time"] = _TIME_LOCALES[language]
    tooltips = {mark: {"tooltip": {"content": "encoding"}} for mark in _TOOLTIP_MARKS}
    return {"locale": locale, **tooltips}


def _merge_config(base: Any, override: Any) -> Any:
    """Merge two config objects key by key; `override` wins where both set a value.

    Args:
        base (Any): The config to start from.
        override (Any): The config whose values take precedence.

    Returns:
        Any: The merged config.
    """
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _merge_config(base[key], value) if key in base else value
    return merged


def apply_server_config(spec: dict[str, Any], language: LanguageCode) -> dict[str, Any]:
    """Set the server's config on a spec, over any value the model gave the same keys.

    Args:
        spec (dict[str, Any]): A sanitized spec.
        language (LanguageCode): The language of the chart's text.

    Returns:
        dict[str, Any]: The spec with the server's config merged into its own.
    """
    config = _merge_config(spec.get("config") or {}, _server_config(language))
    return {**spec, "config": config}
