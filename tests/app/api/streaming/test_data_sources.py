from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from app.api.streaming.data_sources import (
    _GRAPHQL_URL,
    _TABLE_NAME_CACHE,
    _resolve_table_name,
    resolve_data_source_names,
)


class TestResolveTableName:
    """Tests for the _resolve_table_name GraphQL lookup + cache."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """Isolate tests from the process-lifetime name cache."""
        _TABLE_NAME_CACHE.clear()
        yield
        _TABLE_NAME_CACHE.clear()

    @staticmethod
    def _node_response(dataset_name: str | None, table_name: str | None):
        node = {"name": table_name, "dataset": {"name": dataset_name}}
        return {"data": {"allTable": {"edges": [{"node": node}]}}}

    @respx.mock
    async def test_builds_dataset_dash_table_name(self):
        """A resolved table yields '{dataset_name} — {table_name}'."""
        respx.post(_GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200, json=self._node_response("Diretórios Brasileiros", "Município")
            )
        )

        assert await _resolve_table_name("tb1") == "Diretórios Brasileiros — Município"

    @respx.mock
    async def test_caches_successful_resolution(self):
        """A second lookup for the same UUID hits the cache, not the network."""
        route = respx.post(_GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200, json=self._node_response("Diretórios Brasileiros", "Município")
            )
        )

        first = await _resolve_table_name("tb1")
        second = await _resolve_table_name("tb1")

        assert first == second == "Diretórios Brasileiros — Município"
        assert route.call_count == 1

    @respx.mock
    async def test_returns_none_when_not_found(self):
        """An unknown UUID (no edges) resolves to None and is not cached."""
        respx.post(_GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json={"data": {"allTable": {"edges": []}}})
        )

        assert await _resolve_table_name("missing") is None
        assert "missing" not in _TABLE_NAME_CACHE

    @respx.mock
    async def test_returns_none_on_missing_dataset_name(self):
        """A node missing the dataset name resolves to None."""
        respx.post(_GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200, json=self._node_response(None, "Município")
            )
        )

        assert await _resolve_table_name("tb1") is None

    @respx.mock
    async def test_returns_none_on_missing_table_name(self):
        """A node missing the table name resolves to None."""
        respx.post(_GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200, json=self._node_response("Diretórios Brasileiros", None)
            )
        )

        assert await _resolve_table_name("tb1") is None

    @respx.mock
    async def test_returns_none_on_http_error(self):
        """A backend error is swallowed and resolves to None."""
        respx.post(_GRAPHQL_URL).mock(return_value=httpx.Response(500))

        assert await _resolve_table_name("tb1") is None

    @respx.mock
    async def test_returns_none_on_json_decode_error(self):
        """A malformed JSON body is swallowed and resolves to None."""
        # `content=` sends the raw bytes verbatim; `json=` would serialize this
        # to a *valid* JSON string and never trigger a decode error.
        respx.post(_GRAPHQL_URL).mock(
            return_value=httpx.Response(200, content=b"{'malformed': 'json'")
        )

        assert await _resolve_table_name("tb1") is None


class TestResolveDataSourceNames:
    """Tests for the resolve_data_source_names enrichment entry point."""

    async def test_resolved_name_overwrites_model_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A successful resolution overwrites the model-provided fallback name."""
        resolve_name = AsyncMock(side_effect=lambda tid: f"Conjunto - {tid}")

        monkeypatch.setattr(
            "app.api.streaming.data_sources._resolve_table_name", resolve_name
        )

        structured = {
            "data_sources": [
                {"dataset_id": "ds1", "table_id": "tb1", "name": "model name 1"},
                {"dataset_id": "ds2", "table_id": "tb2", "name": "model name 2"},
            ]
        }

        await resolve_data_source_names(structured)

        assert structured["data_sources"] == [
            {"dataset_id": "ds1", "table_id": "tb1", "name": "Conjunto - tb1"},
            {"dataset_id": "ds2", "table_id": "tb2", "name": "Conjunto - tb2"},
        ]

    async def test_keeps_model_fallback_when_unresolvable(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A source whose UUID can't be resolved keeps the model's fallback name."""
        monkeypatch.setattr(
            "app.api.streaming.data_sources._resolve_table_name",
            AsyncMock(return_value=None),
        )

        structured = {
            "data_sources": [
                {"dataset_id": "ds1", "table_id": "tb1", "name": "model fallback"}
            ]
        }

        await resolve_data_source_names(structured)

        assert structured["data_sources"] == [
            {"dataset_id": "ds1", "table_id": "tb1", "name": "model fallback"}
        ]

    async def test_no_data_sources_skips_resolution(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """With no data sources the resolver is never called."""
        resolve_name = AsyncMock()

        monkeypatch.setattr(
            "app.api.streaming.data_sources._resolve_table_name", resolve_name
        )

        await resolve_data_source_names({"data_sources": None})
        await resolve_data_source_names({"data_sources": []})
        await resolve_data_source_names({})

        resolve_name.assert_not_awaited()
