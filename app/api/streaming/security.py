from urllib.parse import urlparse

from marko import Markdown, block, inline
from marko.md_renderer import MarkdownRenderer

_ALLOWED_SCHEMES = {"https"}
_ALLOWED_DOMAINS = {"basedosdados.org", "console.cloud.google.com"}


def _is_allowed_url(url: str) -> bool:
    """Check if a URL has an allowed scheme and domain.

    Args:
        url (str): The URL to check.

    Returns:
        bool: Wether it's allowed or not.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    return parsed.scheme in _ALLOWED_SCHEMES and parsed.hostname in _ALLOWED_DOMAINS


class _MarkdownSanitizer(MarkdownRenderer):
    """Markdown renderer that strips links pointing to disallowed domains.

    Handles all markdown link constructs (inline, reference-style, autolinks).
    Invalid inline and reference-style links are replaced with their display text with a no-op link.
    Invalid images and autolinks are removed entirely.
    """

    def render_link(self, element: inline.Link) -> str:
        if not _is_allowed_url(element.dest):
            element.dest = "#"
        return super().render_link(element)

    def render_auto_link(self, element: inline.AutoLink) -> str:
        if not _is_allowed_url(element.dest):
            return ""
        return super().render_auto_link(element)

    def render_link_ref_def(self, element: block.LinkRefDef) -> str:
        if not _is_allowed_url(element.dest):
            element.dest = "#"
        return super().render_link_ref_def(element)

    def render_image(self, element: inline.Image) -> str:
        if not _is_allowed_url(element.dest):
            return ""
        return super().render_image(element)


_sanitizer = Markdown(renderer=_MarkdownSanitizer)


def sanitize_markdown_links(content: str) -> str:
    """Parse markdown into an AST, strip disallowed links, and render back to markdown.

    Args:
        content (str): The original text content.

    Returns:
        str: The sanitized content.
    """
    return _sanitizer(content).strip()
