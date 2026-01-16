import jwt
import pytest
from fastapi import HTTPException, status

from app.api.dependencies.auth import get_user_id
from app.settings import settings


class TestGetUserId:
    """Tests for get_user_id dependency."""

    async def test_valid_token(self):
        """Test decoding a valid JWT token."""
        user_id = 1

        token = jwt.encode(
            {"user_id": user_id},
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
