from typing import get_args

import pytest

from app.i18n import (
    DEFAULT_LANGUAGE,
    LanguageCode,
    MessageKey,
    localized_field,
    normalize_language,
    translate,
)


class TestNormalizeLanguage:
    @pytest.mark.parametrize("value", ["pt", "en", "es"])
    def test_supported_codes_pass_through(self, value: str):
        assert normalize_language(value) == value

    @pytest.mark.parametrize("value", ["PT", "En", "ES"])
    def test_case_is_normalized(self, value: str):
        assert normalize_language(value) == value.lower()

    @pytest.mark.parametrize("value", [None, "", "fr", "de", "unknown"])
    def test_unsupported_falls_back_to_default(self, value):
        assert normalize_language(value) == DEFAULT_LANGUAGE


class TestTranslate:
    def test_every_key_resolves_in_every_language(self):
        # Guards against a key that's missing a language (translate would KeyError)
        # or has empty text — the common failure when adding a message or a locale.
        for key in MessageKey:
            for language in get_args(LanguageCode):
                text = translate(key, language)
                assert text and text.strip(), f"{key} / {language} is empty"

    def test_returns_language_specific_text(self):
        expired = MessageKey.RESULTS_EXPIRED
        assert translate(expired, "en") != translate(expired, "pt")
        assert translate(expired, "en") != translate(expired, "es")

    def test_does_not_fall_back_on_unsupported_language(self):
        # Normalization is the caller's responsibility (see normalize_language);
        # translate trusts it receives a valid LanguageCode.
        with pytest.raises(KeyError):
            translate(MessageKey.ERROR_UNEXPECTED, "fr")


class TestLocalizedField:
    """The metadata-localization primitive: pick `{field}{Suffix}`, fall back to pt."""

    def test_returns_requested_language_value(self):
        node = {"namePt": "Município", "nameEn": "Municipality", "nameEs": "Municipio"}
        assert localized_field(node, "name", "en") == "Municipality"
        assert localized_field(node, "name", "es") == "Municipio"
        assert localized_field(node, "name", "pt") == "Município"

    def test_falls_back_to_pt_when_requested_language_missing(self):
        # Partial translation coverage: only pt is populated for this node.
        node = {"namePt": "Município"}
        assert localized_field(node, "name", "en") == "Município"

    def test_falls_back_to_pt_when_requested_language_is_empty(self):
        # An empty localized value is treated as "not translated" (the `or` branch).
        node = {"namePt": "Município", "nameEn": ""}
        assert localized_field(node, "name", "en") == "Município"

    def test_returns_none_when_neither_language_present(self):
        assert localized_field({}, "name", "en") is None
