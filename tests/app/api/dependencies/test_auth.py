import uuid

import jwt
import pytest
from fastapi import HTTPException, status

from app.api.dependencies.auth import get_user_id
from app.settings import settings


class TestGetUserId:
    """Tests for get_user_id dependency."""

    @pytest.fixture(autouse=True)
    def disable_auth_dev_mode(self, monkeypatch: pytest.MonkeyPatch):
        """Ensure auth dev mode is disabled for all tests in this class."""
        monkeypatch.setattr(
            "app.api.dependencies.auth.settings",
            settings.model_copy(update={"AUTH_DEV_MODE": False}),
        )

    async def test_valid_token(self):
        """Test decoding a valid JWT token."""
        user_id = str(uuid.uuid4())

        token = jwt.encode(
            {"uuid": user_id},
            key=settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )

        result = await get_user_id(token)

        assert result == user_id

    async def test_missing_user_id_in_payload(self):
        """Test token with missing user_id raises 401."""
        token = jwt.encode(
            {"key": "value"},
            key=settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )

        with pytest.raises(HTTPException) as e:
            await get_user_id(token)

        assert e.value.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_invalid_token(self):
        """Test invalid token raises 401."""
        with pytest.raises(HTTPException) as e:
            await get_user_id("invalidtoken")

        assert e.value.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_token_with_wrong_secret(self):
        """Test token signed with wrong secret raises 401."""
        token = jwt.encode(
            {"user_id": 1},
            key="wrong_secret_key",
            algorithm=settings.JWT_ALGORITHM,
        )

        with pytest.raises(HTTPException) as e:
            await get_user_id(token)

        assert e.value.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_missing_token_raises_401(self):
        """Test missing token raises 401."""
        with pytest.raises(HTTPException) as e:
            await get_user_id(None)

        assert e.value.status_code == status.HTTP_401_UNAUTHORIZED


class TestAuthDevMode:
    """Tests for AUTH_DEV_MODE functionality."""

    async def test_auth_dev_mode_works_with_token(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test dev mode bypasses JWT validation and returns configured user ID."""
        dev_user_id = 1

        monkeypatch.setattr(
            "app.api.dependencies.auth.settings",
            settings.model_copy(
                update={
                    "AUTH_DEV_MODE": True,
                    "AUTH_DEV_USER_ID": dev_user_id,
                    "ENVIRONMENT": "development",
                }
            ),
        )

        result = await get_user_id("any-token")

        assert result == dev_user_id

    async def test_auth_dev_mode_works_without_token(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test dev mode bypasses JWT validation even when no token is provided."""
        dev_user_id = 1

        monkeypatch.setattr(
            "app.api.dependencies.auth.settings",
            settings.model_copy(
                update={
                    "AUTH_DEV_MODE": True,
                    "AUTH_DEV_USER_ID": dev_user_id,
                    "ENVIRONMENT": "development",
                }
            ),
        )

        result = await get_user_id(None)

        assert result == dev_user_id

    async def test_auth_dev_mode_ignored_in_production(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that dev mode is ignored when ENVIRONMENT is 'production'."""
        monkeypatch.setattr(
            "app.api.dependencies.auth.settings",
            settings.model_copy(
                update={"AUTH_DEV_MODE": True, "ENVIRONMENT": "production"}
            ),
        )

        with pytest.raises(HTTPException) as e:
            await get_user_id("any-token")

        assert e.value.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_auth_dev_mode_ignored_in_staging(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that dev mode is ignored when ENVIRONMENT is 'staging'."""
        monkeypatch.setattr(
            "app.api.dependencies.auth.settings",
            settings.model_copy(
                update={"AUTH_DEV_MODE": True, "ENVIRONMENT": "staging"}
            ),
        )

        with pytest.raises(HTTPException) as e:
            await get_user_id("any-token")

        assert e.value.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_auth_dev_mode_ignored_when_not_development(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that dev mode is ignored when ENVIRONMENT is not 'development'."""
        monkeypatch.setattr(
            "app.api.dependencies.auth.settings",
            settings.model_copy(
                update={"AUTH_DEV_MODE": True, "ENVIRONMENT": "anything"}
            ),
        )

        with pytest.raises(HTTPException) as e:
            await get_user_id("any-token")

        assert e.value.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_auth_dev_mode_disabled_invalid_token(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that with dev mode disabled, invalid token raises 401."""
        monkeypatch.setattr(
            "app.api.dependencies.auth.settings",
            settings.model_copy(update={"AUTH_DEV_MODE": False}),
        )

        with pytest.raises(HTTPException) as e:
            await get_user_id("invalid-token")

        assert e.value.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_auth_dev_mode_disabled_missing_token(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that with dev mode disabled, missing token raises 401."""
        monkeypatch.setattr(
            "app.api.dependencies.auth.settings",
            settings.model_copy(update={"AUTH_DEV_MODE": False}),
        )

        with pytest.raises(HTTPException) as e:
            await get_user_id(None)

        assert e.value.status_code == status.HTTP_401_UNAUTHORIZED
