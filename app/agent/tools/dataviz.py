import asyncio
import json
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from app.agent.context import AgentContext
from app.agent.tools.exceptions import handle_tool_errors
from app.charts import build_chart_spec, generate_chart_spec, load_chart_source
from app.db.database import AsyncDatabase, sessionmaker
from app.exports import (
    OFFERED_EXPORT_FORMATS,
    ResultTableExpired,
    ResultTooLarge,
    is_result_expired,
    materialize_export,
    sanitize_export_filename,
)
from app.i18n import MessageKey, translate


@tool
@handle_tool_errors
async def list_query_results(runtime: ToolRuntime[AgentContext]) -> str:
    """List this conversation's query results, including ones from earlier turns.

    Returns:
        JSON array of {query_ref, description, ran_at, expired}, oldest first.
    """
    async with sessionmaker() as session:
        handles = await AsyncDatabase(session).get_query_handles_by_thread(
            runtime.context.thread_id
        )

    results = [
        {
            "query_ref": handle.query_ref,
            "description": handle.slug,
            "ran_at": handle.created_at.isoformat(),
            "expired": is_result_expired(handle.created_at),
        }
        for handle in handles
    ]

    return json.dumps(results, ensure_ascii=False, default=str)


@tool(response_format="content_and_artifact")
@handle_tool_errors(response_format="content_and_artifact")
async def export_query_result(
    query_ref: str, file_format: str, runtime: ToolRuntime[AgentContext]
) -> tuple[str, dict[str, Any]]:
    """Offer a query's result as a downloadable file in the requested format.

    Args:
        query_ref (str): The handle of the result to export.
        file_format (str): One of AVRO, CSV, JSONL, PARQUET.

    Returns:
        A confirmation that the download is ready, or an error describing what to fix.
    """
    file_format = file_format.upper()

    if file_format not in OFFERED_EXPORT_FORMATS:
        raise ValueError(
            f"Unsupported format '{file_format}'. "
            f"Available: {', '.join(OFFERED_EXPORT_FORMATS)}."
        )

    async with sessionmaker() as session:
        handle = await AsyncDatabase(session).get_query_handle_from_thread(
            query_ref, runtime.context.thread_id
        )

    if handle is None:
        raise ValueError(
            f"No query result found for '{query_ref}'. "
            "Call list_query_results to see the available results."
        )

    if is_result_expired(handle.created_at):
        raise ValueError(
            f"The result for '{query_ref}' has expired (results are kept ~24h). "
            "Re-run the query, then export the new result."
        )

    # The user asked for the file explicitly, so materialize it now rather than lazily on click.
    # This lets the card show the exact file size and the real download filename, and surfaces
    # an over-limit result up front instead of a download that 400s on click.
    try:
        exported = await asyncio.to_thread(
            materialize_export,
            query_ref=handle.query_ref,
            destination_table=handle.destination_table,
            file_format=file_format,
            filename=sanitize_export_filename(
                handle.slug,
                translate(MessageKey.DEFAULT_EXPORT_FILENAME, runtime.context.language),
            ),
            message_id=str(handle.message_id),
        )
    except ResultTableExpired as e:
        raise ValueError(
            f"The result for '{query_ref}' has expired (results are kept ~24h). "
            "Re-run the query, then export the new result."
        ) from e
    except ResultTooLarge as e:
        raise ValueError(
            f"The result for '{query_ref}' is too large to export as a single file. "
            "Aggregate or filter it in SQL first, then export the smaller result."
        ) from e

    # A client-facing affordance (not a `query_result` handle, so it is not redacted):
    artifact = {
        "type": "export",
        "query_ref": handle.query_ref,
        "format": file_format,
        "filename": exported.filename,
        "size_bytes": exported.size_bytes,
        "message_id": str(handle.message_id),
    }

    content = json.dumps(
        {
            "status": "ready",
            "query_ref": query_ref,
            "format": file_format,
            "filename": exported.filename,
            "size_bytes": exported.size_bytes,
        },
        ensure_ascii=False,
    )

    return content, artifact


@tool(response_format="content_and_artifact")
@handle_tool_errors(response_format="content_and_artifact")
async def chart_query_result(
    query_ref: str, instructions: str, runtime: ToolRuntime[AgentContext]
) -> tuple[str, dict[str, Any]]:
    """Render a chart from a query's result.

    Args:
        query_ref (str): The handle of the result to chart.
        instructions (str): A natural-language description of the chart.

    Returns:
        A confirmation that the chart was rendered, or an error describing what to fix.
    """
    handle, columns, rows = await load_chart_source(
        query_ref, runtime.context.thread_id
    )

    spec = await generate_chart_spec(columns, rows, instructions)

    # A client-facing artifact (not a `query_result` handle, so it is not redacted): the
    # interface renders it with Vega-Embed. The data is bound server-side from the exact
    # result rows, so the model cannot substitute the numbers.
    artifact = {
        "type": "chart",
        "query_ref": handle.query_ref,
        "spec": build_chart_spec(spec, rows),
    }

    content = json.dumps(
        {"status": "rendered", "query_ref": query_ref, "row_count": len(rows)},
        ensure_ascii=False,
    )

    return content, artifact
