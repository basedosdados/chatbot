from app.agent.tools import BDToolkit


class TestBDToolkit:
    """Tests for BDToolkit class."""

    def test_get_tools_returns_all_tools(self):
        """Test that get_tools returns all expected tools."""
        tools = BDToolkit.get_tools()

        assert len(tools) == 6

        tool_names = [tool.name for tool in tools]

        assert "search_datasets" in tool_names
        assert "get_dataset_details" in tool_names
        assert "get_table_details" in tool_names
        assert "execute_bigquery_sql" in tool_names
        assert "decode_table_values" in tool_names
        assert "export_query_results" in tool_names
