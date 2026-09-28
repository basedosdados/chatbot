import json

from langchain.messages import AIMessage, HumanMessage, ToolMessage

from eval import runner

TABLE = "basedosdados.br_x.a"


def _ai(*calls, content=""):
    return AIMessage(
        content=content,
        tool_calls=[
            {"id": call_id, "name": name, "args": args} for call_id, name, args in calls
        ],
    )


def test_tool_calls_record_order_step_status_table_and_sql():
    messages = [
        HumanMessage("q"),
        _ai(("c1", "get_table_details", {"table_id": "uuid-a"})),
        ToolMessage(
            json.dumps({"id": "uuid-a", "gcp_id": TABLE}),
            tool_call_id="c1",
            name="get_table_details",
        ),
        _ai(
            ("c2", "execute_bigquery_sql", {"sql_query": "SELECT 1"}),
            ("c3", "decode_table_values", {}),
        ),
        ToolMessage(
            json.dumps({"row_count": 1, "rows": [{"x": 1}]}),
            tool_call_id="c2",
            name="execute_bigquery_sql",
        ),
        ToolMessage(
            json.dumps({"status": "error", "message": "bad"}),
            tool_call_id="c3",
            name="decode_table_values",
        ),
        _ai(content="done"),
    ]

    assert runner._tool_calls(runner._messages_since_last_human(messages)) == [
        {
            "name": "get_table_details",
            "status": "success",
            "step": 1,
            "gcp_id": TABLE,
            "sql": None,
        },
        {
            "name": "execute_bigquery_sql",
            "status": "success",
            "step": 2,
            "gcp_id": None,
            "sql": "SELECT 1",
        },
        {
            "name": "decode_table_values",
            "status": "error",
            "step": 2,
            "gcp_id": None,
            "sql": None,
        },
    ]


def test_tool_calls_count_a_raised_tool_error_and_non_json_content():
    messages = [
        _ai(("c1", "get_table_details", {})),
        ToolMessage(
            "Traceback ...", tool_call_id="c1", name="get_table_details", status="error"
        ),
        _ai(("c2", "search_datasets", {})),
        ToolMessage("plain text", tool_call_id="c2", name="search_datasets"),
    ]

    calls = runner._tool_calls(messages)

    assert [call["status"] for call in calls] == ["error", "success"]
    assert calls[0]["gcp_id"] is None


def test_hit_call_limit_reads_the_middleware_stop_message():
    stopped = [_ai(content="Model call limits exceeded: run limit (20/20)")]
    finished = [_ai(content="A resposta é 7.")]

    assert runner._hit_call_limit(stopped) is True
    assert runner._hit_call_limit(finished) is False


def test_trace_metadata_keys_the_config_by_effort_without_docstrings():
    medium = runner.trace_metadata("medium", "main", "run-1")
    high = runner.trace_metadata("high", "main", "run-1")

    assert medium["agent_config_id"] != high["agent_config_id"]
    assert medium["reasoning_effort"] == "medium"
    assert medium["branch"] == "main"
    assert medium["eval_run"] == "run-1"
    assert medium["models"].startswith("openai:")
    assert medium["tools"]
    assert all("docstring" not in tool for tool in medium["tools"])
    assert all("docstring_hash" in tool for tool in medium["tools"])


def test_run_config_carries_the_trace_metadata_and_production_tags():
    metadata = runner.trace_metadata("medium", "main", "run-1")
    config = runner._run_config("t-medium-0", {"id": "t"}, "medium", 0, 2, metadata)

    assert config["configurable"] == {"thread_id": "t-medium-0"}
    assert config["metadata"]["agent_config_id"] == metadata["agent_config_id"]
    assert config["metadata"]["eval_run"] == "run-1"
    assert config["metadata"]["eval_thread"] == "t"
    assert config["metadata"]["turn_index"] == 2
    assert "branch:main" in config["tags"]
    assert f"model:{metadata['model']}" in config["tags"]
    assert f"toolset:{metadata['tool_set_hash'][:8]}" in config["tags"]


def test_model_calls_count_ai_messages_but_not_the_call_limit_stop():
    messages = [
        HumanMessage("q"),
        _ai(("c1", "get_table_details", {"table_id": "uuid-a"})),
        ToolMessage("{}", tool_call_id="c1", name="get_table_details"),
        _ai(content="answer"),
        AIMessage(content="Model call limits exceeded: run limit (1/1)"),
    ]

    assert runner._model_calls(messages) == 2
