from app.api.streaming import _sanitize_markdown_links

safe_markdown = _sanitize_markdown_links(
    "Check [this](https://basedosdados.com) and [that](https://basedosdados.org)"
)

print(safe_markdown)
