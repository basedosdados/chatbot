import json

import httpx
from langchain_core.tools import tool

from app.agent.tools.exceptions import handle_tool_errors
from app.agent.tools.models import (
    Column,
    Dataset,
    DatasetOverview,
    Table,
    TableOverview,
)
from app.agent.tools.queries import DATASET_DETAILS_QUERY, TABLE_DETAILS_QUERY
from app.settings import settings

# httpx default timeout
TIMEOUT = 5.0

# httpx read timeout
READ_TIMEOUT = 60.0

# maximum number of datasets returned on search
PAGE_SIZE = 10

# url for searching datasets
SEARCH_URL = f"{settings.BASEDOSDADOS_BASE_URL}/search/"

# URL for fetching dataset details
GRAPHQL_URL = f"{settings.BASEDOSDADOS_BASE_URL}/graphql"

# URL for fetching usage guides
BASE_USAGE_GUIDE_URL = "https://raw.githubusercontent.com/basedosdados/website/refs/heads/main/next/content/userGuide/pt"

_client = httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT, read=READ_TIMEOUT))


@tool
@handle_tool_errors
async def search_datasets(query: str) -> str:
    """Search for datasets in Base dos Dados using keywords.

    CRITICAL: Use individual KEYWORDS only, not full sentences. The search engine uses Elasticsearch.

    Args:
        query (str): 2-3 keywords maximum. Use Portuguese terms, organization acronyms, or dataset acronyms.
            Good Examples: "censo", "educacao", "ibge", "inep", "rais", "saude"
            Avoid: "Brazilian population data by municipality"

    Returns:
        str: JSON array of datasets. If empty/irrelevant results, try different keywords.

    Strategy: Start with broad terms like "censo", "ibge", "inep", "rais", then get specific if needed.
    Next step: Use `get_dataset_details()` with returned dataset IDs.
    """
    response = await _client.get(
        url=SEARCH_URL,
        params={"contains": "tables", "q": query, "page_size": PAGE_SIZE},
    )

    response.raise_for_status()
    data: dict = response.json()

    datasets = data.get("results", [])

    overviews = []

    for dataset in datasets:
        dataset_overview = DatasetOverview(
            id=dataset["id"],
            name=dataset["name"],
            slug=dataset.get("slug"),
            description=dataset.get("description"),
            tags=[tag["name"] for tag in dataset.get("tags", [])],
            themes=[theme["name"] for theme in dataset.get("themes", [])],
            organizations=[org["name"] for org in dataset.get("organizations", [])],
        )
        overviews.append(dataset_overview.model_dump())

    return json.dumps(overviews, ensure_ascii=False, indent=2)


@tool
@handle_tool_errors
async def get_dataset_details(dataset_id: str) -> str:
    """Get comprehensive details about a specific dataset including all its tables.

    Use AFTER `search_datasets()` to understand data structure before writing queries.

    Args:
        dataset_id (str): Dataset ID obtained from `search_datasets()`.
            This is typically a UUID-like string, not the human-readable name.

    Returns:
        str: JSON object with complete dataset information, including:
            - Basic metadata (name, description, tags, themes, organizations)
            - tables: Array of all tables in the dataset with:
                - gcp_id: Full BigQuery table reference (`project.dataset.table`)
                - temporal coverage: Authoritative temporal coverage for the table
                - table descriptions explaining what each table contains
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

    dataset_id = dataset["id"].split("DatasetNode:")[-1]
    dataset_name = dataset["name"]
    dataset_slug = dataset.get("slug")
    dataset_description = dataset.get("description")

    # Tags
    dataset_tags = []

    for edge in dataset.get("tags", {}).get("edges", []):
        if tag := edge.get("node", {}).get("name"):
            dataset_tags.append(tag)

    # Themes
    dataset_themes = []

    for edge in dataset.get("themes", {}).get("edges", []):
        if theme := edge.get("node", {}).get("name"):
            dataset_themes.append(theme)

    # Organizations
    dataset_organizations = []

    for edge in dataset.get("organizations", {}).get("edges", []):
        if org := edge.get("node", {}).get("name"):
            dataset_organizations.append(org)

    # Tables
    dataset_tables = []
    gcp_dataset_id = None

    for edge in dataset.get("tables", {}).get("edges", []):
        table = edge["node"]

        table_id = table["id"].split("TableNode:")[-1]
        table_name = table["name"]
        table_slug = table.get("slug")
        table_description = table.get("description")
        table_temporal_coverage = table.get("temporalCoverage")

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
                gcp_id=table_gcp_id,
                name=table_name,
                slug=table_slug,
                description=table_description,
                temporal_coverage=table_temporal_coverage,
            )
        )

    # Fetch usage guide
    usage_guide = None

    if gcp_dataset_id is not None:
        filename = gcp_dataset_id.replace("_", "-")

        response = await _client.get(f"{BASE_USAGE_GUIDE_URL}/{filename}.md")

        if response.status_code == httpx.codes.OK:
            usage_guide = response.text.strip()

    result = Dataset(
        id=dataset_id,
        name=dataset_name,
        slug=dataset_slug,
        description=dataset_description,
        tags=dataset_tags,
        themes=dataset_themes,
        organizations=dataset_organizations,
        tables=dataset_tables,
        usage_guide=usage_guide,
    )

    return result.model_dump_json(indent=2)


@tool
@handle_tool_errors
async def get_table_details(table_id: str) -> str:
    """Get comprehensive details about a specific table including all its columns.

    Use AFTER `get_dataset_details()` to understand table structure before writing queries.

    Args:
        table_id (str): Table ID obtained from `get_dataset_details()`.
            This is typically a UUID-like string, not the human-readable name.

    Returns:
        str: JSON object with complete table information, including:
            - Basic metadata (name, description, slug)
            - gcp_id: Full BigQuery table reference (`project.dataset.table`)
            - temporal coverage: Authoritative temporal coverage for the table
            - columns: All column names, types, and descriptions

    Next step: Use `execute_bigquery_sql()` to execute queries.
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

    table_id = table["id"].split("TableNode:")[-1]
    table_name = table["name"]
    table_slug = table.get("slug")
    table_description = table.get("description")
    table_temporal_coverage = table.get("temporalCoverage")

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
    for edge in table["columns"]["edges"]:
        column = edge["node"]

        directory_primary_key = column["directoryPrimaryKey"]

        if directory_primary_key is not None:
            directory_table = directory_primary_key["table"]
            directory_table_id = directory_table["id"].split("TableNode:")[-1]
        else:
            directory_table_id = None

        table_columns.append(
            Column(
                name=column["name"],
                type=column["bigqueryType"]["name"],
                description=column.get("description"),
                unit=column.get("measurementUnit"),
                reference_table_id=directory_table_id,
            )
        )

    result = Table(
        id=table_id,
        gcp_id=table_gcp_id,
        name=table_name,
        slug=table_slug,
        description=table_description,
        temporal_coverage=table_temporal_coverage,
        columns=table_columns,
    )

    return result.model_dump_json(indent=2)
