import codecs
from typing import TypedDict

import sqlparse

from chatbot.agents.router_agent import RouterAgentState
from chatbot.agents.sql_agent import SQLAgentState
from chatbot.agents.structured_outputs import Visualization


class SQLAgentFormattedResponse(TypedDict):
    content: str
    sql_queries: list[str]|None

class RouterAgentFormattedResponse(TypedDict):
    content: str
    sql_queries: list[str]|None
    visualization: Visualization|None

def format_sql_agent_response(response: SQLAgentState) -> SQLAgentFormattedResponse:
    """Formats the response that will be presented to the user

    Args:
        response (dict[str, list]): The model's response

    Returns:
        SQLAgentFormattedResponse:
            A dictionary containing the formatted final answer
            and the formatted generated sql queries
    """
    answer = response["final_answer"]
    answer = codecs.escape_decode(answer)[0]
    answer = answer.decode("utf-8").strip()

    sql_queries = []

    for item in response.get("sql_queries", []):
        sql_query = sqlparse.format(
            item.content,
            reindent=True,
            keyword_case="upper"
        )
        sql_queries.append(sql_query)

    formatted_response = {
        "content": answer,
        "sql_queries": sql_queries or None,
    }

    return formatted_response

def format_router_agent_response(response: RouterAgentState) -> RouterAgentFormattedResponse:
    """Formats the response that will be presented to the user

    Args:
        response (dict[str, list]): The model's response

    Returns:
        RouterAgentFormattedResponse:
            A dictionary containing the formatted final answer,
            the formatted generated sql queries and a `Chart` object
    """
    answer = response["final_answer"]
    answer = codecs.escape_decode(answer)[0]
    answer = answer.decode("utf-8").strip()

    sql_queries = []

    for item in response.get("sql_queries", []):
        sql_query = sqlparse.format(
            item.content,
            reindent=True,
            keyword_case="upper"
        )
        sql_queries.append(sql_query)

    formatted_response = {
        "content": answer,
        "sql_queries": sql_queries or None,
        "visualization": response.get("visualization"),
    }

    return formatted_response
