from app.charts.prompts import _summarize_column, chart_spec_user_prompt, profile_result


class TestChartSpecUserPrompt:
    def test_carries_task_data_on_the_first_attempt(self):
        prompt = chart_spec_user_prompt(
            "a bar chart", "pt", "Rows: 1 ... ano ... total", None, []
        )

        assert "a bar chart" in prompt
        assert "Portuguese" in prompt
        assert "ano" in prompt and "total" in prompt
        assert "rejected" not in prompt  # no retry feedback on the first attempt

    def test_echoes_the_rejected_spec_and_errors_on_retry(self):
        prompt = chart_spec_user_prompt(
            "a bar chart", "en", "Rows: 1", {"mark": "nope"}, ["bad column"]
        )

        assert '"nope"' in prompt  # the model's own rejected spec, echoed back
        assert "bad column" in prompt  # the validator's reason
        assert "rejected" in prompt


class TestSummarizeColumn:
    def test_numbers_get_a_range(self):
        summary = _summarize_column("FLOAT", [96.25, None, 1731.84, 715.52])

        assert summary == {
            "type": "FLOAT",
            "distinct": 3,
            "null_pct": 25.0,
            "min": 96.25,
            "max": 1731.84,
        }

    def test_dates_get_a_range(self):
        summary = _summarize_column("DATE", ["2025-03-01", "2024-01-01"])

        assert summary["min"] == "2024-01-01" and summary["max"] == "2025-03-01"

    def test_low_cardinality_text_lists_every_value(self):
        summary = _summarize_column("STRING", ["SP", "MG", "SP"])

        assert summary["distinct"] == 2
        assert summary["values"] == ["MG", "SP"]

    def test_high_cardinality_text_shows_a_few_examples(self):
        summary = _summarize_column("STRING", [f"m{i:04d}" for i in range(5570)])

        assert summary["distinct"] == 5570
        assert summary["examples"] == ["m0000", "m0001", "m0002"]
        assert "values" not in summary

    def test_booleans_are_listed_not_ranged(self):
        summary = _summarize_column("BOOLEAN", [True, False, True])

        assert summary["values"] == [False, True]

    def test_all_null_column(self):
        assert _summarize_column("STRING", [None, None]) == {
            "type": "STRING",
            "distinct": 0,
            "null_pct": 100.0,
        }


class TestProfileResult:
    def test_carries_row_count_types_summary_and_sample(self):
        rows = [{"uf": "SP", "valor": 1}, {"uf": "MG", "valor": 3}]

        profile = profile_result({"uf": "STRING", "valor": "INTEGER"}, rows)

        assert "Rows: 2" in profile
        assert '"type": "STRING"' in profile and '"type": "INTEGER"' in profile
        assert '"min": 1, "max": 3' in profile
        assert '"MG"' in profile
