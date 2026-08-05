import asyncio
import json
from typing import Any

import httpx
from loguru import logger

from app.i18n import DEFAULT_LANGUAGE, normalize_language
from app.settings import settings

# Minimal GraphQL query to resolve a table's name and its dataset's name from the
# table UUID. The localized name fields (modeltranslation) are fetched alongside
# the default pt `name` so the display name can match the thread's language.
TABLE_NAME_QUERY = """
query getTableName($id: ID!) {
    allTable(id: $id, first: 1) {
        edges {
            node {
                name
                nameEn
                nameEs
                dataset {
                    name
                    nameEn
                    nameEs
                }
            }
        }
    }
}
"""

# Base dos Dados GraphQL endpoint, used to resolve data source display names.
_GRAPHQL_URL = f"{settings.BASEDOSDADOS_BASE_URL}/graphql"

# Dedicated HTTP client + cache for resolving data source display names from table
# UUIDs. The cache is keyed by (language, table_id) because the resolved name is
# localized — the same table has a different display name per language.
_http_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0, read=60.0))
_TABLE_NAME_CACHE: dict[tuple[str, str], str] = {}


def _pick_localized_name(node: dict, language: str) -> str | None:
    """Pick a node's display name for `language`, falling back to the pt `name`.

    Args:
        node (dict): A GraphQL node exposing `name` (pt) plus `nameEn`/`nameEs`.
        language (str): One of "pt", "en", "es".

    Returns:
        str | None: The localized name, the pt `name` when the localized field is
            empty (coverage is partial), or None if the node has no name at all.
    """
    localized = {"en": node.get("nameEn"), "es": node.get("nameEs")}.get(language)
    return localized or node.get("name")


async def _resolve_table_name(table_id: str, language: str) -> str | None:
    """Resolve a table UUID to a human-readable "{dataset_name} — {table_name}",
    localized to `language`, and cache results per (language, table_id).

    Args:
        table_id (str): A Table UUID.
        language (str): The thread's language ("pt", "en", "es"); the resolved
            name uses the matching localized fields, falling back to pt.

    Returns:
        str | None: "{dataset_name} — {table_name}", or None if the table can't
            be resolved (unknown UUID, missing names, network error, etc.).
    """
    cache_key = (language, table_id)
    if cache_key in _TABLE_NAME_CACHE:
        return _TABLE_NAME_CACHE[cache_key]

    try:
        response = await _http_client.post(
            url=_GRAPHQL_URL,
            json={"query": TABLE_NAME_QUERY, "variables": {"id": table_id}},
        )
        response.raise_for_status()
        data: dict = response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to resolve name for table '{table_id}': {e!r}")
        return None

    all_tables = data.get("data", {}).get("allTable") or {}
    edges = all_tables.get("edges", [])

    if not edges:
        return None

    node = edges[0]["node"]
    table_name = _pick_localized_name(node, language)
    dataset_name = _pick_localized_name(node.get("dataset") or {}, language)

    if not table_name or not dataset_name:
        return None

    name = f"{dataset_name} — {table_name}"
    _TABLE_NAME_CACHE[cache_key] = name
    return name


async def resolve_data_source_names(
    structured_response: dict[str, Any], language: str = DEFAULT_LANGUAGE
) -> None:
    """Overwrite each data source's display name with a deterministic
    "{dataset_name} — {table_name}" resolved from its table UUID, localized to
    `language` (falling back to the pt name where a translation is missing).

    The resolved name is authoritative; the model-provided `name` (see
    `app.agent.schemas.DataSource`) is kept as a fallback for any source whose
    UUID can't be resolved (unknown UUID, missing names, network error, etc.).
    Written onto the structured response in place.

    Args:
        structured_response (dict[str, Any]): The dumped StructuredResponse whose
            `data_sources` entries are enriched in place.
        language (str): The thread's language; selects the localized name fields.
    """
    language = normalize_language(language)
    data_sources = structured_response.get("data_sources") or []
    resolvable = [source for source in data_sources if source["table_id"]]

    if not resolvable:
        return

    names = await asyncio.gather(
        *(_resolve_table_name(source["table_id"], language) for source in resolvable),
        return_exceptions=True,
    )

    for source, name in zip(resolvable, names):
        if isinstance(name, str):
            source["name"] = name
