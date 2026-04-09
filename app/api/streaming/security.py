from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

from marko import Markdown
from marko.md_renderer import MarkdownRenderer

_ALLOWED_SCHEMES = {"https"}
_ALLOWED_DOMAINS = {"basedosdados.org", "console.cloud.google.com"}


def _is_allowed_url(url: str) -> bool:
    """Check if a URL has an allowed scheme and domain.

    Args:
        url (str): The URL to check.

    Returns:
        bool: Whether it's allowed or not.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    return parsed.scheme in _ALLOWED_SCHEMES and parsed.hostname in _ALLOWED_DOMAINS


def sanitize_markdown_links(text: str) -> str:
    """Sanitize markdown text by replacing disallowed URLs with '#'.

    Uses marko to parse the markdown AST, finds all URLs (covering inline
    links, images, autolinks, and reference definitions), then replaces
    disallowed URLs with a no-op link.

    Args:
        text (str): The original markdown text.

    Returns:
        str: The sanitized markdown text.
    """
    md = Markdown(renderer=MarkdownRenderer)

    doc = md.parse(text)

    def walk_and_sanitize(node: Any):
        if hasattr(node, "dest"):
            if not _is_allowed_url(node.dest):
                node.dest = "#"
        if (
            hasattr(node, "children")
            and isinstance(node.children, Sequence)
            and not isinstance(node.children, str)
        ):
            for child in node.children:
                walk_and_sanitize(child)

    walk_and_sanitize(doc)

    return md.render(doc).strip()
