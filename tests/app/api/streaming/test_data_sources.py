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
        node = {"namePt": table_name, "dataset": {"namePt": dataset_name}}
        return {"data": {"allTable": {"edges": [{"node": node}]}}}

    @staticmethod
    def _bilingual_response(pt: tuple[str, str], en: tuple[str, str]):
        node = {
            "namePt": pt[1],
            "nameEn": en[1],
            "dataset": {"namePt": pt[0], "nameEn": en[0]},
        }
        return {"data": {"allTable": {"edges": [{"node": node}]}}}

    @respx.mock
    async def test_resolves_in_requested_language(self):
        """The resolved name uses the requested language's fields."""
        respx.post(_GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200,
                json=self._bilingual_response(
                    pt=("Diretórios", "Município"), en=("Directories", "Municipality")
                ),
            )
        )

        assert await _resolve_table_name("tb1", "en") == "Directories — Municipality"

    @respx.mock
    async def test_cache_is_keyed_per_language(self):
        """The same table resolves independently per language — the cache must not collide."""
        route = respx.post(_GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200,
                json=self._bilingual_response(
                    pt=("Diretórios", "Município"), en=("Directories", "Municipality")
                ),
            )
        )

        pt_name = await _resolve_table_name("tb1", "pt")
        en_name = await _resolve_table_name("tb1", "en")

        assert pt_name == "Diretórios — Município"
        assert en_name == "Directories — Municipality"
        # Two distinct cache keys => two lookups (not one shared, wrong-language entry).
        assert route.call_count == 2

    @respx.mock
    async def test_builds_dataset_dash_table_name(self):
        """A resolved table yields '{dataset_name} — {table_name}'."""
        respx.post(_GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200, json=self._node_response("Diretórios Brasileiros", "Município")
            )
        )

        assert (
            await _resolve_table_name("tb1", "pt")
            == "Diretórios Brasileiros — Município"
        )

    @respx.mock
    async def test_caches_successful_resolution(self):
        """A second lookup for the same UUID hits the cache, not the network."""
        route = respx.post(_GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200, json=self._node_response("Diretórios Brasileiros", "Município")
            )
        )

        first = await _resolve_table_name("tb1", "pt")
        second = await _resolve_table_name("tb1", "pt")

        assert first == second == "Diretórios Brasileiros — Município"
        assert route.call_count == 1

    @respx.mock
    async def test_returns_none_when_not_found(self):
        """An unknown UUID (no edges) resolves to None and is not cached."""
        respx.post(_GRAPHQL_URL).mock(
            return_value=httpx.Response(200, json={"data": {"allTable": {"edges": []}}})
        )

        assert await _resolve_table_name("missing", "pt") is None
        # Cache is keyed (table_id, language); assert the real key
        # so this actually verifies that None results are not cached.
        assert ("missing", "pt") not in _TABLE_NAME_CACHE

    @respx.mock
    async def test_returns_none_on_missing_dataset_name(self):
        """A node missing the dataset name resolves to None."""
        respx.post(_GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200, json=self._node_response(None, "Município")
            )
        )

        assert await _resolve_table_name("tb1", "pt") is None

    @respx.mock
    async def test_returns_none_on_missing_table_name(self):
        """A node missing the table name resolves to None."""
        respx.post(_GRAPHQL_URL).mock(
            return_value=httpx.Response(
                200, json=self._node_response("Diretórios Brasileiros", None)
            )
        )

        assert await _resolve_table_name("tb1", "pt") is None

    @respx.mock
    async def test_returns_none_on_http_error(self):
        """A backend error is swallowed and resolves to None."""
        respx.post(_GRAPHQL_URL).mock(return_value=httpx.Response(500))

        assert await _resolve_table_name("tb1", "pt") is None

    @respx.mock
    async def test_returns_none_on_json_decode_error(self):
        """A malformed JSON body is swallowed and resolves to None."""
        # `content=` sends the raw bytes verbatim; `json=` would serialize this
        # to a *valid* JSON string and never trigger a decode error.
        respx.post(_GRAPHQL_URL).mock(
            return_value=httpx.Response(200, content=b"{'malformed': 'json'")
        )

        assert await _resolve_table_name("tb1", "pt") is None


class TestResolveDataSourceNames:
    """Tests for the resolve_data_source_names enrichment entry point."""

    async def test_resolved_name_overwrites_model_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A successful resolution overwrites the model-provided fallback name."""
        resolve_name = AsyncMock(side_effect=lambda tid, language: f"Conjunto - {tid}")

        monkeypatch.setattr(
            "app.api.streaming.data_sources._resolve_table_name", resolve_name
        )

        structured = {
            "data_sources": [
                {"dataset_id": "ds1", "table_id": "tb1", "name": "model name 1"},
                {"dataset_id": "ds2", "table_id": "tb2", "name": "model name 2"},
            ]
        }

        await resolve_data_source_names(structured, "pt")

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

        await resolve_data_source_names(structured, "pt")

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

        await resolve_data_source_names({"data_sources": None}, "pt")
        await resolve_data_source_names({"data_sources": []}, "pt")
        await resolve_data_source_names({}, "pt")

        resolve_name.assert_not_awaited()
