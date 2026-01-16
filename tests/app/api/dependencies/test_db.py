from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

from pytest_mock import MockerFixture

from app.api.dependencies.db import get_database, get_session
from app.db.database import AsyncDatabase


class TestGetSession:
    """Tests for get_session dependency."""

    async def test_yields_session(self, mocker: MockerFixture):
        """Test that get_session yields a session from the sessionmaker."""
        mock_session = AsyncMock()

        @asynccontextmanager
        async def mock_sessionmaker():
            yield mock_session

        mocker.patch("app.api.dependencies.db.sessionmaker", mock_sessionmaker)

        async for session in get_session():
            assert session is mock_session


class TestGetDatabase:
    """Tests for get_database dependency."""

    async def test_returns_async_database(self):
        """Test that get_database returns an AsyncDatabase instance."""
        mock_session = AsyncMock()

        result = await get_database(mock_session)

        assert isinstance(result, AsyncDatabase)
        assert result.session is mock_session
