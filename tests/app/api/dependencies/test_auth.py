import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import jwt
import pytest
from fastapi import HTTPException, status

from app.api.dependencies.auth import _verify_token, get_user_id
from app.settings import settings


class TestVerifyToken:
    """Tests for _verify_token function."""

    def _mock_graphql_response(self, has_access: bool):
        """Create a mock response for the GraphQL endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {"verifyToken": {"payload": {"has_chatbot_access": has_access}}}
        }
        return mock_response

    async def test_returns_true_when_user_has_access(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test returns True when user has chatbot access."""
        mock_response = self._mock_graphql_response(has_access=True)
        monkeypatch.setattr(
            "app.api.dependencies.auth._http_client",
            MagicMock(post=AsyncMock(return_value=mock_response)),
        )

        result = await _verify_token("valid-token")

        assert result is True

    async def test_returns_false_when_user_lacks_access(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test returns False when user lacks chatbot access."""
        mock_response = self._mock_graphql_response(has_access=False)
        monkeypatch.setattr(
            "app.api.dependencies.auth._http_client",
            MagicMock(post=AsyncMock(return_value=mock_response)),
        )

        result = await _verify_token("valid-token")

        assert result is False

    async def test_raises_503_on_http_error(self, monkeypatch: pytest.MonkeyPatch):
        """Test raises 503 when GraphQL endpoint returns HTTP error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=httpx.Request("POST", "http://test"),
            response=mock_response,
        )
        monkeypatch.setattr(
            "app.api.dependencies.auth._http_client",
            MagicMock(post=AsyncMock(return_value=mock_response)),
        )

        with pytest.raises(HTTPException) as e:
            await _verify_token("valid-token")

        assert e.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    async def test_raises_503_on_connect_error(self, monkeypatch: pytest.MonkeyPatch):
        """Test raises 503 when GraphQL endpoint is unreachable."""
        monkeypatch.setattr(
            "app.api.dependencies.auth._http_client",
            MagicMock(
                post=AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            ),
        )

        with pytest.raises(HTTPException) as e:
            await _verify_token("valid-token")

        assert e.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE


class TestGetUserId:
    """Tests for get_user_id dependency."""

    @pytest.fixture(autouse=True)
    def disable_auth_dev_mode(self, monkeypatch: pytest.MonkeyPatch):
        """Ensure auth dev mode is disabled for all tests in this class."""
        monkeypatch.setattr(
            "app.api.dependencies.auth.settings",
            settings.model_copy(update={"AUTH_DEV_MODE": False}),
        )

    async def test_valid_token(self, monkeypatch: pytest.MonkeyPatch):
        """Test decoding a valid JWT token with chatbot access."""
        user_id = str(uuid.uuid4())

        async def mock_verify_token(token: str) -> bool:
            return True

        monkeypatch.setattr(
            "app.api.dependencies.auth._verify_token", mock_verify_token
        )

        token = jwt.encode(
            {"uuid": user_id},
            key=settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )

        result = await get_user_id(token)

        assert result == user_id

    async def test_valid_token_without_chatbot_access(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test valid JWT token but user lacks chatbot access raises 403."""
        user_id = str(uuid.uuid4())

        async def mock_verify_token(token: str) -> bool:
            return False

        monkeypatch.setattr(
            "app.api.dependencies.auth._verify_token", mock_verify_token
        )

        token = jwt.encode(
            {"uuid": user_id},
            key=settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )

        with pytest.raises(HTTPException) as e:
            await get_user_id(token)

        assert e.value.status_code == status.HTTP_403_FORBIDDEN

    async def test_verify_token_service_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """Test that 503 is raised when token verification service is unavailable."""
        user_id = str(uuid.uuid4())

        async def mock_verify_token(token: str) -> bool:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Unable to verify user access",
            )

        monkeypatch.setattr(
            "app.api.dependencies.auth._verify_token", mock_verify_token
        )

        token = jwt.encode(
            {"uuid": user_id},
            key=settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )

        with pytest.raises(HTTPException) as e:
            await get_user_id(token)

        assert e.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

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
        dev_user_id = str(uuid.uuid4())

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
        dev_user_id = str(uuid.uuid4())

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
