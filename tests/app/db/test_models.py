import pytest

from app.db.models import ThreadPayload


class TestThreadLanguageNormalization:
    """`ThreadPayload.language` is the single normalization boundary: a `before` validator
    coerces raw input to a valid LanguageCode so everything downstream can trust it."""

    def test_normalizes_case(self):
        assert ThreadPayload(title="t", language="EN").language == "en"

    @pytest.mark.parametrize("value", ["fr", "de", "unknown", ""])
    def test_coerces_unsupported_to_default_without_raising(self, value: str):
        # The Literal field would reject these, but the `before` validator coerces first —
        # this guards against someone dropping mode="before" (which would 422 real requests).
        assert ThreadPayload(title="t", language=value).language == "pt"

    def test_explicit_null_falls_back_to_default(self):
        assert ThreadPayload(title="t", language=None).language == "pt"

    def test_omitted_uses_default(self):
        assert ThreadPayload(title="t").language == "pt"
