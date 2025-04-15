import json
import pytest

from chatbot.agents.structured_outputs import ChartData


@pytest.mark.parametrize(
    "input, output",
    [
        (
            [
                {"col1": "value1", "col2": "value1"},
                {"col1": "value2", "col2": "value2"},
                {"col1": "value3", "col2": "value3"},
                {"col1": "value4", "col2": "value4"}
            ],
            {
                "col1": ["value1", "value2", "value3", "value4"],
                "col2": ["value1", "value2", "value3", "value4"]
            }
        ),
        (
            [
                {"col1": "value1", "col2": "value1"},
                {"col1": "value2", "col2": "value2"},
                {"col1": "value3", "col2": "value3"},
                {"col1": "value4"}
            ],
            {
                "col1": ["value1", "value2", "value3", "value4"],
                "col2": ["value1", "value2", "value3", None]
            }
        ),
        (
            [
                {"col2": "value1"},
                {"col1": "value2", "col2": "value2"},
                {"col1": "value3", "col2": "value3"},
                {"col1": "value4", "col2": "value4"}
            ],
            {
                "col1": [None, "value2", "value3", "value4"],
                "col2": ["value1", "value2", "value3", "value4"]
            }
        ),
        (
            [
                {"col1": "value1", "col2": "value1"},
                {"col2": "value2"},
                {"col1": "value3"},
                {"col1": "value4", "col2": "value4"}
            ],
            {
                "col1": ["value1", None, "value3", "value4"],
                "col2": ["value1", "value2", None, "value4"]
            }
        ),
        (
            [
                {"col1": "value1"},
                {"col1": "value2", "col2": "value2"},
                {"col1": "value3", "col2": "value3"},
                {"col2": "value4"}
            ],
            {
                "col1": ["value1", "value2", "value3", None],
                "col2": [None, "value2", "value3", "value4"]
            }
        ),
    ]
)
def test_merge_query_results(input: list[dict], output: dict[list]):
    data = json.dumps(input)

    chart_data = ChartData(
        data=data,
        reasoning=""
    )

    assert chart_data.data == output
