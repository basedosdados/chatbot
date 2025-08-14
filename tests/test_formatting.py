from chatbot.agents.reducers import Item
from chatbot.agents.router_agent import RouterAgentState
from chatbot.agents.sql_agent import SQLAgentState
from chatbot.agents.structured_outputs import Visualization
from chatbot.assistants.formatting import (format_router_agent_response,
                                           format_sql_agent_response)


# ===============================================
# ==                 SQL Agent                 ==
# ===============================================
def test_format_sql_agent_response():
    response = SQLAgentState(
        question="mock question",
        final_answer="hello world!",
        sql_queries=[
            Item(content="select * from table_1"),
            Item(content="select * from table_2"),
        ],
        sql_queries_results=[],
        similar_examples=[],
        messages=[],
        is_last_step=False,
    )

    expected_formatted_response = {
        "content": "hello world!",
        "sql_queries": ["SELECT *\nFROM table_1", "SELECT *\nFROM table_2"],
    }

    formatted_response = format_sql_agent_response(response)

    assert formatted_response == expected_formatted_response

def test_format_sql_agent_response_with_special_characters():
    response = SQLAgentState(
        question="mock question",
        final_answer="\n\\xc4\\xa7&\\xc5\\x82\\xc5\\x82\\xc3\\xb8 w\\xc3\\xb8\\xc2\\xae\\xc5\\x82\\xc3\\xb0!\n",
        sql_queries=[
            Item(content="select * from table_1"),
            Item(content="select * from table_2"),
        ],
        sql_queries_results=[],
        similar_examples=[],
        messages=[],
        is_last_step=False,
    )

    expected_formatted_response = {
        "content": "ħ&łłø wø®łð!",
        "sql_queries": ["SELECT *\nFROM table_1", "SELECT *\nFROM table_2"],
    }

    formatted_response = format_sql_agent_response(response)

    assert formatted_response == expected_formatted_response

def test_format_sql_agent_response_with_no_sql_queries():
    response = SQLAgentState(
        question="mock question",
        final_answer="hello world!",
        sql_queries=[],
        sql_queries_results=[],
        similar_examples=[],
        messages=[],
        is_last_step=False,
    )

    expected_formatted_response = {
        "content": "hello world!",
        "sql_queries": None,
    }

    formatted_response = format_sql_agent_response(response)

    assert formatted_response == expected_formatted_response

# ===============================================
# ==               Router Agent                ==
# ===============================================
def test_format_router_agent_response():
    response = RouterAgentState(
        _previous="mock previous",
        _next="mock next",
        question="mock question",
        rewrite_query=False,
        sql_answer="mock sql agent answer",
        final_answer="hello world!",
        sql_queries=[
            Item(content="select * from table_1"),
            Item(content="select * from table_2")
        ],
        sql_queries_results=[],
        data_turn_ids=None,
        question_for_viz_agent=None,
        visualization=None,
        chat_history={},
    )

    expected_formatted_response = {
        "content": "hello world!",
        "sql_queries": ["SELECT *\nFROM table_1", "SELECT *\nFROM table_2"],
        "visualization": None
    }

    formatted_response = format_router_agent_response(response)

    assert formatted_response == expected_formatted_response

def test_format_router_agent_response_with_special_characters():
    response = RouterAgentState(
        _previous="mock previous",
        _next="mock next",
        question="mock question",
        rewrite_query=False,
        sql_answer="mock sql agent answer",
        final_answer="\n\\xc4\\xa7&\\xc5\\x82\\xc5\\x82\\xc3\\xb8 w\\xc3\\xb8\\xc2\\xae\\xc5\\x82\\xc3\\xb0!\n",
        sql_queries=[
            Item(content="select * from table_1"),
            Item(content="select * from table_2")
        ],
        sql_queries_results=[],
        data_turn_ids=None,
        question_for_viz_agent=None,
        visualization=None,
        chat_history={},
    )

    expected_formatted_response = {
        "content": "ħ&łłø wø®łð!",
        "sql_queries": ["SELECT *\nFROM table_1", "SELECT *\nFROM table_2"],
        "visualization": None,
    }

    formatted_response = format_router_agent_response(response)

    assert formatted_response == expected_formatted_response

def test_format_router_agent_response_with_no_sql_queries():
    response = RouterAgentState(
        _previous="mock previous",
        _next="mock next",
        question="mock question",
        rewrite_query=False,
        sql_answer="mock sql agent answer",
        final_answer="hello world!",
        sql_queries=[],
        sql_queries_results=[],
        data_turn_ids=None,
        question_for_viz_agent=None,
        visualization=None,
        chat_history={},
    )

    expected_formatted_response = {
        "content": "hello world!",
        "sql_queries": None,
        "visualization": None
    }

    formatted_response = format_router_agent_response(response)

    assert formatted_response == expected_formatted_response
