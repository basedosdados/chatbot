import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import SystemMessage

from app.charts import generation
from app.charts.config import apply_server_config
from app.charts.generation import (
    MAX_CHART_SPEC_ATTEMPTS,
    ChartSpec,
    ChartSpecInvalid,
    ChartSpecTimeout,
    generate_chart_spec,
)
from app.charts.prompts import (
    CHART_SPEC_INSTRUCTIONS,
)
from tests.app.charts.specs import ROWS, line_spec


def _structured_reply(spec: dict | None) -> dict:
    """A `with_structured_output(include_raw=True)` result: a parsed spec, or a parse miss."""
    parsed = ChartSpec(spec=spec) if spec is not None else None
    return {"raw": MagicMock(), "parsed": parsed, "parsing_error": None}


class TestGenerateChartSpec:
    def _model(self, monkeypatch, *messages):
        model = MagicMock()
        model.ainvoke = AsyncMock(side_effect=list(messages))
        monkeypatch.setattr(generation, "_chart_spec_model", lambda: model)
        return model

    async def test_retries_until_valid(self, monkeypatch):
        """A spec that fails validation is regenerated with the errors fed back."""
        bad = {"mark": "bar", "encoding": {"x": {"field": "nope"}}}
        good = {"mark": "bar", "encoding": {"x": {"field": "ano"}}}
        model = self._model(
            monkeypatch, _structured_reply(bad), _structured_reply(good)
        )
        monkeypatch.setattr(
            generation,
            "validate_chart_spec",
            lambda spec, columns, year_columns, rows: []
            if spec["encoding"] == good["encoding"]
            else ["bad column"],
        )

        result = await generate_chart_spec(
            {"ano": "INTEGER"}, [{"ano": 2025}], "a bar chart", "pt"
        )

        assert result["encoding"] == good["encoding"]
        assert model.ainvoke.await_count == 2

    async def test_year_columns_are_detected_from_all_rows(self, monkeypatch):
        """The year check sees every row, not just the sample shown to the model."""
        spec = {"mark": "bar", "encoding": {"x": {"field": "ano"}}}
        self._model(monkeypatch, _structured_reply(spec))
        seen = []
        monkeypatch.setattr(
            generation,
            "validate_chart_spec",
            lambda spec, columns, year_columns, rows: seen.append(year_columns) or [],
        )
        rows = [{"ano": 2000 + i, "n": 1} for i in range(10)] + [{"ano": 1, "n": 1}]

        await generate_chart_spec(
            {"ano": "INTEGER", "n": "INTEGER"}, rows, "a bar chart", "pt"
        )

        assert seen == [frozenset()]  # the 11th row is not a year, so `ano` is not one

    async def test_spec_is_validated_and_returned_with_the_server_config(
        self, monkeypatch
    ):
        """The chart is checked as it ships: with the server's config and the rows."""
        spec = {"mark": "bar", "encoding": {"x": {"field": "ano"}}}
        self._model(monkeypatch, _structured_reply(spec))
        seen = []
        monkeypatch.setattr(
            generation,
            "validate_chart_spec",
            lambda spec, columns, year_columns, rows: seen.append((spec, rows)) or [],
        )
        rows = [{"ano": 2025}]

        result = await generate_chart_spec(
            {"ano": "INTEGER"}, rows, "a bar chart", "pt"
        )

        assert result == apply_server_config(spec, "pt")
        assert seen == [(result, rows)]

    async def test_spec_drawing_nothing_with_the_rows_is_retried(self, monkeypatch):
        """End to end with the real validator: a bad scheme only fails on real rows."""
        good = line_spec()
        bad = line_spec(
            color={"field": "sigla_uf", "type": "nominal", "scale": {"scheme": "RdBu"}}
        )
        model = self._model(
            monkeypatch, _structured_reply(bad), _structured_reply(good)
        )
        columns = {"ano": "INTEGER", "sigla_uf": "STRING", "total": "FLOAT"}

        result = await generate_chart_spec(columns, ROWS, "a line chart", "pt")

        retry_prompt = model.ainvoke.await_args_list[1].args[0][1].content
        assert result["encoding"] == good["encoding"]
        assert "draws no marks at all" in retry_prompt
        assert "locale" not in retry_prompt  # the model is shown only what it wrote

    async def test_rejected_spec_is_fed_back_on_retry(self, monkeypatch):
        """The retry prompt carries the model's own rejected spec plus the errors."""
        bad = {"mark": "bar", "encoding": {"x": {"field": "nope"}}}
        good = {"mark": "bar", "encoding": {"x": {"field": "ano"}}}
        model = self._model(
            monkeypatch, _structured_reply(bad), _structured_reply(good)
        )
        monkeypatch.setattr(
            generation,
            "validate_chart_spec",
            lambda spec, columns, year_columns, rows: []
            if spec["encoding"] == good["encoding"]
            else ["column 'nope' is not in the result"],
        )

        await generate_chart_spec(
            {"ano": "INTEGER"}, [{"ano": 2025}], "a bar chart", "pt"
        )

        system_message, user_message = model.ainvoke.await_args_list[1].args[0]
        # The durable how-to is a system message; the retry data rides the user message.
        assert isinstance(system_message, SystemMessage)
        assert system_message.content == CHART_SPEC_INSTRUCTIONS
        assert '"nope"' in user_message.content  # its own rejected spec, echoed back
        assert "not in the result" in user_message.content  # the validator's reason

    async def test_prompt_carries_language_and_data_summary(self, monkeypatch):
        spec = {"mark": "bar", "encoding": {"x": {"field": "ano"}}}
        model = self._model(monkeypatch, _structured_reply(spec))
        monkeypatch.setattr(
            generation,
            "validate_chart_spec",
            lambda spec, columns, year_columns, rows: [],
        )

        await generate_chart_spec(
            {"ano": "INTEGER"}, [{"ano": 2024}, {"ano": 2025}], "a bar chart", "es"
        )

        _, user_message = model.ainvoke.await_args.args[0]
        assert "Spanish" in user_message.content
        assert "Rows: 2" in user_message.content
        assert '"min": 2024, "max": 2025' in user_message.content

    async def test_model_supplied_data_is_stripped(self, monkeypatch):
        """A model spec carrying data/url is sanitized (not rejected) before returning."""
        spec = {
            "mark": "bar",
            "encoding": {"x": {"field": "ano"}},
            "data": {"url": "https://evil.example/x.json"},
        }
        model = self._model(monkeypatch, _structured_reply(spec))
        monkeypatch.setattr(
            generation,
            "validate_chart_spec",
            lambda spec, columns, year_columns, rows: [],
        )

        result = await generate_chart_spec(
            {"ano": "INTEGER"}, [{"ano": 2025}], "a bar chart", "pt"
        )

        assert "data" not in result
        assert "url" not in json.dumps(result)
        assert model.ainvoke.await_count == 1

    async def test_missing_spec_is_retried(self, monkeypatch):
        """A reply the parser could not turn into a spec is treated as a failure and retried."""
        empty = _structured_reply(None)
        good = _structured_reply({"mark": "bar", "encoding": {"x": {"field": "ano"}}})
        model = self._model(monkeypatch, empty, good)
        monkeypatch.setattr(
            generation,
            "validate_chart_spec",
            lambda spec, columns, year_columns, rows: [],
        )

        result = await generate_chart_spec(
            {"ano": "INTEGER"}, [{"ano": 2025}], "a bar chart", "pt"
        )

        assert result["mark"] == "bar"
        assert model.ainvoke.await_count == 2

    async def test_raises_when_generation_exceeds_the_deadline(self, monkeypatch):
        """A hung model call is cut off instead of blocking the turn indefinitely."""
        monkeypatch.setattr(
            generation,
            "settings",
            generation.settings.model_copy(update={"CHART_SPEC_TIMEOUT_SECONDS": 0.05}),
        )

        async def hang(messages):
            await asyncio.sleep(10)

        model = MagicMock()
        model.ainvoke = hang
        monkeypatch.setattr(generation, "_chart_spec_model", lambda: model)

        with pytest.raises(ChartSpecTimeout):
            await generate_chart_spec(
                {"ano": "INTEGER"}, [{"ano": 2025}], "a bar chart", "pt"
            )

    async def test_raises_after_max_attempts(self, monkeypatch):
        spec = {"mark": "bar", "encoding": {}}
        messages = [_structured_reply(spec) for _ in range(MAX_CHART_SPEC_ATTEMPTS)]
        model = self._model(monkeypatch, *messages)
        monkeypatch.setattr(
            generation,
            "validate_chart_spec",
            lambda spec, columns, year_columns, rows: ["always bad"],
        )

        with pytest.raises(ChartSpecInvalid):
            await generate_chart_spec(
                {"ano": "INTEGER"}, [{"ano": 2025}], "a bar chart", "pt"
            )

        assert model.ainvoke.await_count == MAX_CHART_SPEC_ATTEMPTS
