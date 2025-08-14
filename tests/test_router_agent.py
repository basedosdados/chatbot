import json

import pytest

from chatbot.agents.reducers import ChatTurn, ChatTurnRemove, Item
from chatbot.agents.router_agent import (RouterAgent, RouterAgentState,
                                         _format_input_message,
                                         _get_data_from_chat_turns,
                                         _normalize_data)
from chatbot.agents.structured_outputs import Visualization

QUESTION_LIMIT = 5

@pytest.fixture
def agent(monkeypatch) -> RouterAgent:
    def mock_agent_init(self, question_limit: int = QUESTION_LIMIT):
        self.question_limit = question_limit

    monkeypatch.setattr(RouterAgent, "__init__", mock_agent_init)

    return RouterAgent()

@pytest.fixture()
def chat_history() -> dict[int, ChatTurn]:
    return {
        1: ChatTurn(
            id=1,
            user_question="mock question",
            ai_response="mock response",
            data=[]
        )
    }

@pytest.fixture()
def full_chat_history() -> dict[int, ChatTurn]:
    return {
        i+1: ChatTurn(
            id=i+1,
            user_question="mock question",
            ai_response="mock response",
            data=[]
        ) for i in range(QUESTION_LIMIT)
    }

@pytest.fixture()
def visualization() -> Visualization:
    return Visualization(
        script="mock script",
        reasoning="mock reasoning",
        insights="mock insights",
        data = [],
        data_placeholder="INPUT_DATA",
    )

# =====================================================
# ==                  Prune History                  ==
# =====================================================
def test_prune_history_not_delete(agent: RouterAgent, chat_history: dict[int, ChatTurn]):
    response = agent._prune_history({"chat_history": chat_history})
    expected = {"chat_history": {}}
    assert response == expected

def test_prune_history_delete(agent: RouterAgent, full_chat_history: dict[int, ChatTurn]):
    response = agent._prune_history({"chat_history": full_chat_history})
    key = min(full_chat_history.keys())
    expected = {"chat_history": {key: ChatTurnRemove(id=key)}}
    assert response == expected

def test_prune_history_question_limit_none(agent: RouterAgent, full_chat_history: dict[int, ChatTurn]):
    agent.question_limit = None
    response = agent._prune_history({"chat_history": full_chat_history})
    expected = {"chat_history": {}}
    assert response == expected

# =====================================================
# ==                 Process Outputs                 ==
# =====================================================
def test_process_outputs_sql_then_viz(agent: RouterAgent, visualization: Visualization):
    state = RouterAgentState(
        _previous="sql_agent",
        _next="viz_agent",
        question="mock question",
        rewrite_query=False,
        sql_answer="mock sql answer",
        final_answer="",
        sql_queries=[],
        sql_queries_results=[],
        data_turn_ids=None,
        question_for_viz_agent=None,
        visualization=visualization,
        chat_history={},
    )
    response = agent._process_outputs(state)
    final_answer = f"{state['sql_answer']}\n\n{visualization.insights}"
    expected = {
        "final_answer": final_answer,
        "chat_history": {
            1: ChatTurn(
                id=1,
                user_question=state["question"],
                ai_response=final_answer,
                data=state["sql_queries_results"],
            )
        }
    }
    assert response == expected

def test_process_outputs_sql_then_viz_with_error(agent: RouterAgent):
    state = RouterAgentState(
        _previous="sql_agent",
        _next="viz_agent",
        question="mock question",
        rewrite_query=False,
        sql_answer="mock sql answer",
        final_answer="",
        sql_queries=[],
        sql_queries_results=[],
        data_turn_ids=None,
        question_for_viz_agent=None,
        visualization=None,
        chat_history={},
    )
    response = agent._process_outputs(state)
    final_answer = state["sql_answer"]
    expected = {
        "final_answer": final_answer,
        "chat_history": {
            1: ChatTurn(
                id=1,
                user_question=state["question"],
                ai_response=final_answer,
                data=state["sql_queries_results"],
            )
        }
    }
    assert response == expected

def test_process_outputs_only_viz(agent: RouterAgent, visualization: Visualization):
    state = RouterAgentState(
        _previous=None,
        _next="viz_agent",
        question="mock question",
        rewrite_query=False,
        sql_answer="mock sql answer",
        final_answer="",
        sql_queries=[],
        sql_queries_results=[],
        data_turn_ids=None,
        question_for_viz_agent=None,
        visualization=visualization,
        chat_history={},
    )
    response = agent._process_outputs(state)
    final_answer = visualization.insights
    expected = {
        "final_answer": final_answer,
        "chat_history": {
            1: ChatTurn(
                id=1,
                user_question=state["question"],
                ai_response=final_answer,
                data=state["sql_queries_results"],
            )
        }
    }
    assert response == expected

def test_process_outputs_only_viz_with_error(agent: RouterAgent):
    state = RouterAgentState(
        _previous=None,
        _next="viz_agent",
        question="mock question",
        rewrite_query=False,
        sql_answer="mock sql answer",
        final_answer="",
        sql_queries=[],
        sql_queries_results=[],
        data_turn_ids=None,
        question_for_viz_agent=None,
        visualization=None,
        chat_history={},
    )
    response = agent._process_outputs(state)
    final_answer = state["final_answer"]
    expected = {
        "final_answer": final_answer,
        "chat_history": {
            1: ChatTurn(
                id=1,
                user_question=state["question"],
                ai_response=final_answer,
                data=state["sql_queries_results"],
            )
        }
    }
    assert response == expected

def test_process_outputs_only_sql(agent: RouterAgent):
    state = RouterAgentState(
        _previous="sql_agent",
        _next="process_outputs",
        question="mock question",
        rewrite_query=False,
        sql_answer="mock sql answer",
        final_answer="",
        sql_queries=[],
        sql_queries_results=[],
        data_turn_ids=None,
        question_for_viz_agent=None,
        visualization=None,
        chat_history={},
    )
    response = agent._process_outputs(state)
    final_answer = state["sql_answer"]
    expected = {
        "final_answer": final_answer,
        "chat_history": {
            1: ChatTurn(
                id=1,
                user_question=state["question"],
                ai_response=final_answer,
                data=state["sql_queries_results"],
            )
        },
        "visualization": None
    }
    assert response == expected

def test_process_outputs_with_non_empty_chat_history(agent: RouterAgent):
    chat_history = {
        1: ChatTurn(id=1, user_question="q1", ai_response="r1", data=[]),
        2: ChatTurn(id=2, user_question="q2", ai_response="r2", data=[]),
    }
    state = RouterAgentState(
        _previous="sql_agent",
        _next="process_outputs",
        question="mock question",
        rewrite_query=False,
        sql_answer="mock sql answer",
        final_answer="",
        sql_queries=[],
        sql_queries_results=[],
        data_turn_ids=None,
        question_for_viz_agent=None,
        visualization=None,
        chat_history=chat_history,
    )

    response = agent._process_outputs(state)

    assert 3 in response["chat_history"]
    assert len(response["chat_history"]) == 1

# ====================================================
# ==                Helper Functions                ==
# ====================================================
def test_normalize_data():
    item_1_data = [{"k1": "v1"}, {"k1": "v2"}]
    item_2_data = [{"k2": "v1"}, {"k2": "v2"}]

    data = [
        Item(content=json.dumps(item_1_data)),
        Item(content=json.dumps(item_2_data))
    ]

    expected = item_1_data.copy()
    expected.extend(item_2_data)

    normalized_data = _normalize_data(data)

    assert normalized_data == expected

def test_normalize_data_json_decode_error():
    item_1_data = [{"k1": "v1"}, {"k1": "v2"}]

    data = [
        Item(content=json.dumps(item_1_data)),
        Item(content="[1, 2, ")
    ]

    normalized_data = _normalize_data(data)

    assert normalized_data == item_1_data

def test_get_data_from_chat_turns():
    chat_history = {
        1: ChatTurn(id=1, user_question="q1", ai_response="r1", data=[Item(content="a")]),
        2: ChatTurn(id=2, user_question="q2", ai_response="r2", data=[Item(content="b"), Item(content="c")]),
        3: ChatTurn(id=3, user_question="q2", ai_response="r2", data=[Item(content="d")])
    }

    expected = chat_history[2].data.copy()
    expected.extend(chat_history[3].data)

    data = _get_data_from_chat_turns([2, 3], chat_history)

    assert data == expected

def test_get_data_from_chat_turns_not_found():
    chat_history = {
        1: ChatTurn(id=1, user_question="q1", ai_response="r1", data=[Item(content="a")]),
        2: ChatTurn(id=2, user_question="q2", ai_response="r2", data=[Item(content="b"), Item(content="c")]),
        3: ChatTurn(id=3, user_question="q2", ai_response="r2", data=[Item(content="d")])
    }

    expected = chat_history[2].data

    data = _get_data_from_chat_turns([2, 4], chat_history)

    assert data == expected

def test_format_input_message():
    current_question = "mock questiÕn"

    chat_history = {
        1: ChatTurn(id=1, user_question="q1", ai_response="r1", data=[]),
        2: ChatTurn(id=2, user_question="q2", ai_response="r2", data=[]),
    }

    expected = json.dumps({
        "conversation_history": [
            {
                "turn_id": turn_id,
                "user_question": chat_turn.user_question,
                "ai_response": chat_turn.ai_response
            } for turn_id, chat_turn in chat_history.items()
        ],
        "current_question": current_question,
    }, ensure_ascii=False, indent=2)

    input_message = _format_input_message(current_question, chat_history)

    assert expected == input_message
