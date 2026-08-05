import json

import httpx
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from app.agent.context import AgentContext
from app.agent.tools.exceptions import handle_tool_errors
from app.agent.tools.models import (
    Column,
    Dataset,
    DatasetOverview,
    Table,
    TableOverview,
)
from app.agent.tools.queries import DATASET_DETAILS_QUERY, TABLE_DETAILS_QUERY
from app.i18n import DEFAULT_LANGUAGE, LanguageCode, localized_field
from app.settings import settings

# httpx default timeout
TIMEOUT = 5.0

# httpx read timeout
READ_TIMEOUT = 60.0

# maximum number of datasets returned on search
PAGE_SIZE = 10

# directory datasets to skip
SKIP_DIRECTORY_DATASETS = {"br_bd_diretorios_data_tempo"}

# URL for searching datasets
SEARCH_URL = f"{settings.BASEDOSDADOS_BASE_URL}/search/"

# URL for fetching dataset details
GRAPHQL_URL = f"{settings.BASEDOSDADOS_BASE_URL}/graphql"

# Base URL for fetching usage guides; the language subpath (pt/en/es) is appended
# per request, falling back to pt when a localized guide does not exist yet.
BASE_USAGE_GUIDE_URL = "https://raw.githubusercontent.com/basedosdados/website/refs/heads/main/next/content/userGuide"

_client = httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT, read=READ_TIMEOUT))


async def _fetch_usage_guide(gcp_dataset_id: str, language: LanguageCode) -> str | None:
    """Fetch a dataset's usage-guide markdown for `language`, falling back to the default.

    Localized guides may not exist yet (only the default language is populated today), so
    a missing localized file falls back to the default-language guide.

    Args:
        gcp_dataset_id (str): The BigQuery dataset id.
        language (LanguageCode): The thread's language.

    Returns:
        str | None: The guide markdown, or None if no guide exists in any language.
    """
    filename = gcp_dataset_id.replace("_", "-")

    # Try the requested language, then the default; `fromkeys` preserves order
    # and drops the duplicate when `language` is already the default.
    locales = dict.fromkeys((language, DEFAULT_LANGUAGE))

    for locale in locales:
        response = await _client.get(f"{BASE_USAGE_GUIDE_URL}/{locale}/{filename}.md")
        if response.status_code == httpx.codes.OK:
            return response.text.strip()

    return None


@tool
@handle_tool_errors
async def search_datasets(query: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Search for datasets in Base dos Dados using keywords.

    CRITICAL: Use individual KEYWORDS only, not full sentences. The search engine uses Elasticsearch.

    Args:
        query (str): 2-3 keywords maximum. Use Portuguese terms, organization names, or dataset names.
            Good Examples: "censo", "rais", "ibge", "inep", "educacao", "saude".
            Avoid: "Brazilian population data by municipality".

    Returns:
        str: JSON array of datasets. If empty/irrelevant results, try different keywords.

    Strategy: hierarchical funnel — ALWAYS start with a SINGLE keyword and broaden a level only if it returns nothing:
        1. Dataset name ("censo", "rais", "enem") or organization ("ibge", "inep", "tse").
        2. Core theme ("educacao", "saude", "economia", "emprego").
        3. English term ("health", "education").
        4. A 2-3 word combination only if the levels above fail ("saude ms", "censo municipio").

    Next step: Use `get_dataset_details()` with returned dataset IDs.
    """
    response = await _client.get(
        url=SEARCH_URL,
        params={
            "contains": "tables",
            "q": query,
            "page_size": PAGE_SIZE,
            "locale": runtime.context.language,
        },
    )

    response.raise_for_status()
    data: dict = response.json()

    datasets = data.get("results", [])

    overviews = []

    for dataset in datasets:
        dataset_overview = DatasetOverview(
            id=dataset["id"],
            name=dataset["name"],
            description=dataset.get("description"),
            organizations=[org["name"] for org in dataset.get("organizations", [])],
            tags=[tag["name"] for tag in dataset.get("tags", [])],
            themes=[theme["name"] for theme in dataset.get("themes", [])],
        )
        overviews.append(dataset_overview.model_dump())

    return json.dumps(overviews, ensure_ascii=False)


@tool
@handle_tool_errors
async def get_dataset_details(
    dataset_id: str, runtime: ToolRuntime[AgentContext]
) -> str:
    """Get comprehensive details about a specific dataset including all its tables.

    Use AFTER `search_datasets()` to understand data structure before writing queries.

    Args:
        dataset_id (str): Dataset ID obtained from `search_datasets()`.
            This is a UUID-like string, not the human-readable name.

    Returns:
        str: JSON object with complete dataset information, including:
            - Basic metadata (name, description, tags, themes, organizations).
            - tables: Array of all tables in the dataset with:
                - gcp_id: Full BigQuery table reference (`project.dataset.table`).
                - table descriptions explaining what each table contains.
            - usage_guide: Provide key information and best practices for using the dataset.

    Next step: Use `get_table_details()` with returned table IDs.
    """
    response = await _client.post(
        url=GRAPHQL_URL,
        json={
            "query": DATASET_DETAILS_QUERY,
            "variables": {"id": dataset_id},
        },
    )

    response.raise_for_status()
    data: dict[str, dict[str, dict]] = response.json()

    all_datasets = data.get("data", {}).get("allDataset") or {}
    dataset_edges = all_datasets.get("edges", [])

    if not dataset_edges:
        raise ValueError(
            f"Dataset '{dataset_id}' not found. Verify the dataset ID from search_datasets results."
        )

    dataset = dataset_edges[0]["node"]

    language = runtime.context.language

    dataset_id = dataset["id"].split("DatasetNode:")[-1]
    dataset_name = localized_field(dataset, "name", language)
    dataset_description = localized_field(dataset, "description", language)

    # Tags
    dataset_tags = []

    for edge in dataset.get("tags", {}).get("edges", []):
        if tag := localized_field(edge.get("node", {}), "name", language):
            dataset_tags.append(tag)

    # Themes
    dataset_themes = []

    for edge in dataset.get("themes", {}).get("edges", []):
        if theme := localized_field(edge.get("node", {}), "name", language):
            dataset_themes.append(theme)

    # Organizations
    dataset_organizations = []

    for edge in dataset.get("organizations", {}).get("edges", []):
        if org := localized_field(edge.get("node", {}), "name", language):
            dataset_organizations.append(org)

    # Tables
    dataset_tables = []
    gcp_dataset_id = None

    for edge in dataset.get("tables", {}).get("edges", []):
        table = edge["node"]

        table_id = table["id"].split("TableNode:")[-1]
        table_name = localized_field(table, "name", language)
        table_description = localized_field(table, "description", language)

        cloud_table_edges = table["cloudTables"]["edges"]
        if cloud_table_edges:
            cloud_table = cloud_table_edges[0]["node"]
            gcp_project_id = cloud_table["gcpProjectId"]
            gcp_dataset_id = gcp_dataset_id or cloud_table["gcpDatasetId"]
            gcp_table_id = cloud_table["gcpTableId"]
            table_gcp_id = f"{gcp_project_id}.{gcp_dataset_id}.{gcp_table_id}"
        else:
            table_gcp_id = None

        dataset_tables.append(
            TableOverview(
                id=table_id,
                dataset_id=dataset_id,
                gcp_id=table_gcp_id,
                name=table_name,
                description=table_description,
            )
        )

    # Fetch usage guide (localized, pt fallback)
    usage_guide = None

    if gcp_dataset_id is not None:
        usage_guide = await _fetch_usage_guide(gcp_dataset_id, language)

    result = Dataset(
        id=dataset_id,
        name=dataset_name,
        description=dataset_description,
        tags=dataset_tags,
        themes=dataset_themes,
        organizations=dataset_organizations,
        tables=dataset_tables,
        usage_guide=usage_guide,
    )

    return result.model_dump_json()


@tool
@handle_tool_errors
async def get_table_details(table_id: str, runtime: ToolRuntime[AgentContext]) -> str:
    """Get comprehensive details about a specific table including all its columns.

    Use AFTER `get_dataset_details()` to understand table structure before writing queries.

    Args:
        table_id (str): Table ID obtained from `get_dataset_details()`.
            This is typically a UUID-like string, not the human-readable name.

    Returns:
        str: JSON object with complete table information, including:
            - Basic metadata (name, description).
            - gcp_id: Full BigQuery table reference (`project.dataset.table`).
            - columns: All column names, types, and descriptions, including
                `needs_decoding` and `reference_table_id` for coded columns.
            - partitioned_by: Columns to filter on for cost control.
            - period_start / period_end: First and last period covered by the table.
                Format varies (`2024`, `'2026-04-12'`, etc.) — use the value verbatim,
                matched to the appropriate temporal column (`ano`, `data`, etc.).
    """
    response = await _client.post(
        url=GRAPHQL_URL,
        json={
            "query": TABLE_DETAILS_QUERY,
            "variables": {"id": table_id},
        },
    )

    response.raise_for_status()
    data: dict[str, dict[str, dict]] = response.json()

    all_tables = data.get("data", {}).get("allTable") or {}
    table_edges = all_tables.get("edges", [])

    if not table_edges:
        raise ValueError(
            f"Table '{table_id}' not found. Verify the table ID from get_dataset_details results."
        )

    table = table_edges[0]["node"]

    language = runtime.context.language

    table_id = table["id"].split("TableNode:")[-1]
    table_name = localized_field(table, "name", language)
    table_description = localized_field(table, "description", language)
    table_temporal_coverage = table.get("temporalCoverage") or {}

    cloud_table_edges = table["cloudTables"]["edges"]
    if cloud_table_edges:
        cloud_table = cloud_table_edges[0]["node"]
        gcp_project_id = cloud_table["gcpProjectId"]
        gcp_dataset_id = cloud_table["gcpDatasetId"]
        gcp_table_id = cloud_table["gcpTableId"]
        table_gcp_id = f"{gcp_project_id}.{gcp_dataset_id}.{gcp_table_id}"
    else:
        table_gcp_id = None

    table_columns = []
    partitioned_by = []

    for edge in table["columns"]["edges"]:
        column = edge["node"]

        if column["isPartition"]:
            partitioned_by.append(column["name"])

        directory_primary_key = column["directoryPrimaryKey"]

        if directory_primary_key is not None:
            directory_table = directory_primary_key["table"]
            directory_cloud_table = directory_table["cloudTables"]["edges"][0]["node"]
            if directory_cloud_table["gcpDatasetId"] in SKIP_DIRECTORY_DATASETS:
                directory_table_id = None
            else:
                directory_table_id = directory_table["id"].split("TableNode:")[-1]
        else:
            directory_table_id = None

        table_columns.append(
            Column(
                name=column["name"],
                type=column["bigqueryType"]["name"],
                description=localized_field(column, "description", language),
                unit=column.get("measurementUnit"),
                reference_table_id=directory_table_id,
                needs_decoding=column["coveredByDictionary"],
            )
        )

    dataset_id = table["dataset"]["id"].split("DatasetNode:")[-1]

    result = Table(
        id=table_id,
        dataset_id=dataset_id,
        gcp_id=table_gcp_id,
        name=table_name,
        description=table_description,
        columns=table_columns,
        partitioned_by=partitioned_by,
        period_start=table_temporal_coverage.get("start"),
        period_end=table_temporal_coverage.get("end"),
    )

    return result.model_dump_json()
