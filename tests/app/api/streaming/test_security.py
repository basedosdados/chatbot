import pytest
from pytest_mock import MockerFixture

from app.api.streaming.security import _is_allowed_url, sanitize_markdown_links


class TestIsAllowedUrl:
    """Tests for _is_allowed_url function."""

    @pytest.mark.parametrize(
        "url",
        [
            "https://basedosdados.org",
            "https://basedosdados.org/dataset/123",
            "https://basedosdados.org/dataset/123?table=456",
            "https://console.cloud.google.com/bigquery",
        ],
    )
    def test_allowed_urls(self, url: str):
        """Test that URLs with allowed schemes and domains pass."""
        assert _is_allowed_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com",
            "https://example.com/phishing",
            "https://notbasedosdados.org",
            "https://evil.basedosdados.org",
            "https://basedosdados.org.evil.com",
        ],
    )
    def test_disallowed_domains(self, url: str):
        """Test that URLs with non-allowlisted domains are rejected."""
        assert _is_allowed_url(url) is False

    @pytest.mark.parametrize(
        "url",
        [
            "blob:https://basedosdados.org/some-uuid",
            "data:text/html,<script>alert(1)</script>",
            "file:///etc/passwd",
            "ftp://basedosdados.org",
            "http://basedosdados.org",
            "javascript:alert(1)",
            "mailto:user@basedosdados.org",
            "resource://basedosdados.org",
            "ssh://basedosdados.org",
            "tel:+5511999999999",
            "urn:isbn:0451450523",
            "vbscript:msgbox",
            "view-source:https://basedosdados.org",
            "ws://basedosdados.org",
            "wss://basedosdados.org",
        ],
    )
    def test_disallowed_schemes(self, url: str):
        """Test that URLs with non-HTTPS schemes are rejected."""
        assert _is_allowed_url(url) is False

    @pytest.mark.parametrize("url", ["", "not-a-url", "://missing-scheme"])
    def test_malformed_urls(self, url: str):
        """Test that malformed or empty URLs are rejected."""
        assert _is_allowed_url(url) is False

    def test_urlparse_exception_returns_false(self, mocker: MockerFixture):
        """Test that urlparse exceptions are caught and treated as disallowed."""
        mocker.patch(
            "app.api.streaming.security.urlparse", side_effect=ValueError("Invalid URL")
        )
        assert _is_allowed_url("https://basedosdados.org") is False


class TestSanitizeMarkdownLinks:
    """Tests for sanitize_markdown_links function."""

    def test_valid_inline_link_preserved(self):
        """Test that inline links to allowed domains are preserved."""
        md = "[Dataset 123](https://basedosdados.org/dataset/123)"
        assert sanitize_markdown_links(md) == md

    def test_invalid_inline_link_gets_noop(self):
        """Test that inline links to disallowed domains are replaced with '#'."""
        result = sanitize_markdown_links("[click here](https://example.com)")
        assert result == "[click here](#)"

    def test_valid_image_preserved(self):
        """Test that image links to allowed domains are preserved."""
        md = "![BD](https://basedosdados.org/img/logo.png)"
        assert sanitize_markdown_links(md) == md

    def test_invalid_image_gets_noop(self):
        """Test that image links to disallowed domains are replaced with '#'."""
        result = sanitize_markdown_links("![tracker](https://example.com/pixel.png)")
        assert result == "![tracker](#)"

    def test_valid_autolink_preserved(self):
        """Test that autolinks to allowed domains are preserved."""
        md = "Visit <https://basedosdados.org> now"
        assert sanitize_markdown_links(md) == md

    def test_invalid_autolink_gets_noop(self):
        """Test that autolinks to disallowed domains are replaced with '#'."""
        result = sanitize_markdown_links("Visit <https://example.com> now")
        assert result == "Visit <#> now"

    def test_valid_reference_link_preserved(self):
        """Test that reference-style links to allowed domains are preserved."""
        md = "See [data][ref]\n\n[ref]: https://basedosdados.org/dataset/123"
        assert sanitize_markdown_links(md) == md

    def test_invalid_reference_link_gets_noop(self):
        """Test that reference-style links to disallowed domains are replaced with '#'."""
        md = "See [data][ref]\n\n[ref]: https://example.com"
        result = sanitize_markdown_links(md)
        assert result == "See [data](#)\n\n[ref]: #"

    def test_mixed_valid_and_invalid_links(self):
        """Test that only disallowed links are replaced when mixed with allowed ones."""
        md = "[data](https://basedosdados.org/dataset/123) and [evil](https://example.com)"
        result = sanitize_markdown_links(md)
        assert result == "[data](https://basedosdados.org/dataset/123) and [evil](#)"

    def test_non_https_scheme_gets_noop(self):
        """Test that links with non-HTTPS schemes are replaced with '#'."""
        result = sanitize_markdown_links("[xss](javascript:alert(1))")
        assert result == "[xss](#)"

    def test_code_block_preserved(self):
        """Test that URLs inside fenced code blocks are not sanitized."""
        md = "```sql\nSELECT * FROM [table](https://example.com)\n```"
        assert sanitize_markdown_links(md) == md

    def test_inline_code_preserved(self):
        """Test that URLs inside inline code spans are not sanitized."""
        md = "Use `[link](https://example.com)` syntax"
        assert sanitize_markdown_links(md) == md

    def test_plain_text_preserved(self):
        """Test that plain text without links passes through unchanged."""
        md = "No links here, just **bold** and *italic* text."
        assert sanitize_markdown_links(md) == md

    def test_empty_string_preserved(self):
        """Test that an empty string passes through unchanged."""
        assert sanitize_markdown_links("") == ""

    def test_link_with_title_preserved(self):
        """Test that links with title attributes to allowed domains are preserved."""
        md = '[data](https://basedosdados.org/dataset/123 "Dataset")'
        assert sanitize_markdown_links(md) == md

    def test_invalid_link_with_title_gets_noop(self):
        """Test that disallowed links with title attributes are replaced with '#'."""
        result = sanitize_markdown_links('[evil](https://example.com "Evil Site")')
        assert result == '[evil](# "Evil Site")'

    def test_table_with_links(self):
        """Test that links inside markdown tables are sanitized correctly."""
        md = (
            "| Name | Link                           |\n"
            "| ---- | ------------------------------ |\n"
            "| Good | [BD](https://basedosdados.org) |\n"
            "| Bad  | [Evil](https://example.com)       |\n"
        )
        result = sanitize_markdown_links(md)

        assert result == (
            "| Name | Link                           |\n"
            "| ---- | ------------------------------ |\n"
            "| Good | [BD](https://basedosdados.org) |\n"
            "| Bad  | [Evil](#)       |"
        )
