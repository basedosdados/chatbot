import pytest

from app.i18n import (
    DEFAULT_LANGUAGE,
    LANGUAGES,
    _MESSAGES,
    language_directive,
    normalize_language,
    t,
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
    def test_every_key_covers_every_language(self):
        for key, translations in _MESSAGES.items():
            assert set(translations) == set(LANGUAGES), f"'{key}' is missing a language"
            assert all(v.strip() for v in translations.values()), f"'{key}' has empty text"

    def test_returns_language_specific_text(self):
        assert t("results_expired", "en") != t("results_expired", "pt")
        assert t("results_expired", "en") != t("results_expired", "es")

    def test_unsupported_language_falls_back_to_default(self):
        assert t("error_unexpected", "fr") == t("error_unexpected", DEFAULT_LANGUAGE)


class TestLanguageDirective:
    @pytest.mark.parametrize(
        ("language", "expected_name"),
        [("pt", "Portuguese"), ("en", "English"), ("es", "Spanish")],
    )
    def test_names_the_target_language(self, language: str, expected_name: str):
        assert expected_name in language_directive(language)

    def test_unsupported_language_falls_back_to_default_name(self):
        assert "Portuguese" in language_directive("fr")
