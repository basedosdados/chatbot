import asyncio
import json
from typing import Any

import httpx
from loguru import logger

from app.settings import settings

# Minimal GraphQL query to resolve a table's name and its dataset's name from the table UUID.
TABLE_NAME_QUERY = """
query getTableName($id: ID!) {
    allTable(id: $id, first: 1) {
        edges {
            node {
                name
                dataset {
                    name
                }
            }
        }
    }
}
"""

# Base dos Dados GraphQL endpoint, used to resolve data source display names.
_GRAPHQL_URL = f"{settings.BASEDOSDADOS_BASE_URL}/graphql"

# Dedicated HTTP client + cache for resolving data source display names from table UUIDs.
_http_client = httpx.AsyncClient(timeout=httpx.Timeout(5.0, read=60.0))
_TABLE_NAME_CACHE: dict[str, str] = {}


async def _resolve_table_name(table_id: str) -> str | None:
    """Resolve a table UUID to a human-readable "{dataset_name} — {table_name}" and caches results.

    Args:
        table_id (str): A Table UUID.

    Returns:
        str | None: "{dataset_name} — {table_name}", or None if the table can't
            be resolved (unknown UUID, missing names, network error, etc.).
    """
    if table_id in _TABLE_NAME_CACHE:
        return _TABLE_NAME_CACHE[table_id]

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
    table_name = node.get("name")
    dataset_name = (node.get("dataset") or {}).get("name")

    if not table_name or not dataset_name:
        return None

    name = f"{dataset_name} — {table_name}"
    _TABLE_NAME_CACHE[table_id] = name
    return name


async def resolve_data_source_names(structured_response: dict[str, Any]) -> None:
    """Overwrite each data source's display name with a deterministic
    "{dataset_name} — {table_name}" resolved from its table UUID.

    The resolved name is authoritative; the model-provided `name` (see
    `app.agent.schemas.DataSource`) is kept as a fallback for any source whose
    UUID can't be resolved (unknown UUID, missing names, network error, etc.).
    Written onto the structured response in place.

    Args:
        structured_response (dict[str, Any]): The dumped StructuredResponse whose
            `data_sources` entries are enriched in place.
    """
    data_sources = structured_response.get("data_sources") or []
    resolvable = [source for source in data_sources if source["table_id"]]

    if not resolvable:
        return

    names = await asyncio.gather(
        *(_resolve_table_name(source["table_id"]) for source in resolvable),
        return_exceptions=True,
    )

    for source, name in zip(resolvable, names):
        if isinstance(name, str):
            source["name"] = name
