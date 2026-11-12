import json
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.api_core.exceptions import NotFound

import app.charts.data as chart_data
from app.charts.data import (
    ChartHandleNotFound,
    ChartResultTooLarge,
    _fetch_rows,
    fetch_chart_data,
    spread_sample,
)
from app.charts.validation import _integer_year_columns
from app.db.models import QueryHandle
from app.exports import ResultTableExpired

DESTINATION = {"projectId": "p", "datasetId": "d", "tableId": "t"}


def _handle(query_ref="qr_1", slug="resultado", age=timedelta(0)):
    return QueryHandle(
        query_ref=query_ref,
        message_id=uuid.uuid4(),
        slug=slug,
        destination_table=DESTINATION,
        created_at=datetime.now(timezone.utc) - age,
    )


class TestFetchRows:
    def _client(self, rows, columns=("col1",)):
        client = MagicMock()
        client.get_table.return_value = SimpleNamespace(
            schema=[
                SimpleNamespace(name=name, field_type="STRING") for name in columns
            ],
        )
        client.list_rows.return_value = iter(rows)
        return client

    def test_reads_columns_and_rows(self, mocker):
        client = self._client([{"col1": "a"}, {"col1": "b"}])
        mocker.patch("app.charts.data._bq_client", return_value=client)

        columns, rows = _fetch_rows(DESTINATION)

        assert columns == {"col1": "STRING"}
        assert rows == [{"col1": "a"}, {"col1": "b"}]

    def test_allows_a_dense_result_within_budget(self, mocker):
        """A high-cardinality result (e.g. a municipal choropleth ~5.5k rows) is allowed."""
        rows = [{"id": i} for i in range(6000)]
        client = self._client(rows, columns=("id",))
        mocker.patch("app.charts.data._bq_client", return_value=client)

        _, got = _fetch_rows(DESTINATION)

        assert len(got) == 6000

    def test_raises_when_over_byte_budget(self, mocker, monkeypatch):
        """The bound data is rejected once it would exceed the payload budget."""
        monkeypatch.setattr(
            chart_data,
            "settings",
            chart_data.settings.model_copy(update={"CHART_MAX_BYTES": 32}),
        )
        client = self._client([{"col1": "x" * 100}, {"col1": "y" * 100}])
        mocker.patch("app.charts.data._bq_client", return_value=client)

        with pytest.raises(ChartResultTooLarge):
            _fetch_rows(DESTINATION)

    def test_coerces_non_json_types_to_json_native(self, mocker):
        """BigQuery date/datetime/Decimal values become JSON-serializable rows.

        A Decimal (NUMERIC/BIGNUMERIC) must become a float, not a string — a quantitative
        axis or colour scale binds these values, and a string breaks the numeric encoding.
        """
        rows = [
            {"data": date(2025, 12, 31), "temperatura_media": Decimal("24.79")},
        ]
        client = self._client(rows, columns=("data", "temperatura_media"))
        mocker.patch("app.charts.data._bq_client", return_value=client)

        _, got = _fetch_rows(DESTINATION)

        assert got == [{"data": "2025-12-31", "temperatura_media": 24.79}]
        assert isinstance(got[0]["temperatura_media"], float)
        # The bound rows must be plain JSON values, or spec serialization fails.
        json.dumps(got)

    def test_whole_number_decimal_becomes_int(self, mocker):
        """A whole-number Decimal becomes an int, not a float.

        A NUMERIC year (`Decimal("2019")`) must arrive as 2019, not 2019.0,
        or `_integer_year_columns` would not recognize the column as years.
        """
        rows = [{"ano": Decimal("2019"), "total": Decimal("10.00")}]
        client = self._client(rows, columns=("ano", "total"))
        mocker.patch("app.charts.data._bq_client", return_value=client)

        columns, got = _fetch_rows(DESTINATION)

        assert got == [{"ano": 2019, "total": 10}]
        assert all(isinstance(value, int) for value in got[0].values())
        assert _integer_year_columns(list(columns), got) == {"ano"}

    def test_missing_table_maps_to_expired(self, mocker):
        client = MagicMock()
        client.get_table.side_effect = NotFound("gone")
        mocker.patch("app.charts.data._bq_client", return_value=client)

        with pytest.raises(ResultTableExpired):
            _fetch_rows(DESTINATION)


class TestFetchChartData:
    def _patch_db(self, monkeypatch, handle):
        db = MagicMock()
        db.get_query_handle_from_thread = AsyncMock(return_value=handle)

        @asynccontextmanager
        async def mock_sessionmaker():
            yield None

        monkeypatch.setattr(chart_data, "sessionmaker", mock_sessionmaker)
        monkeypatch.setattr(chart_data, "AsyncDatabase", lambda session: db)
        return db

    async def test_returns_handle_columns_and_rows(self, monkeypatch):
        handle = _handle(age=timedelta(hours=1))
        self._patch_db(monkeypatch, handle)
        monkeypatch.setattr(
            chart_data,
            "_fetch_rows",
            lambda dest: ({"ano": "INTEGER"}, [{"ano": 2025}]),
        )

        got_handle, columns, rows = await fetch_chart_data("qr_1", "test-thread")

        assert got_handle is handle
        assert columns == {"ano": "INTEGER"}
        assert rows == [{"ano": 2025}]

    async def test_missing_handle_raises(self, monkeypatch):
        self._patch_db(monkeypatch, None)

        with pytest.raises(ChartHandleNotFound):
            await fetch_chart_data("qr_missing", "test-thread")

    async def test_expired_handle_raises(self, monkeypatch):
        self._patch_db(monkeypatch, _handle(age=timedelta(hours=48)))

        with pytest.raises(ResultTableExpired):
            await fetch_chart_data("qr_old", "test-thread")


class TestSpreadSample:
    def test_returns_every_row_when_few(self):
        rows = [{"i": i} for i in range(3)]

        assert spread_sample(rows, 10) == rows

    def test_spreads_across_the_whole_result(self):
        """A result sorted by group shows every group, not just the first one."""
        rows = [{"grupo": "pobres", "i": i} for i in range(50)] + [
            {"grupo": "ricos", "i": i} for i in range(50)
        ]

        sample = spread_sample(rows, 10)

        assert len(sample) == 10
        assert sample[0] == rows[0] and sample[-1] == rows[-1]
        assert {row["grupo"] for row in sample} == {"pobres", "ricos"}
