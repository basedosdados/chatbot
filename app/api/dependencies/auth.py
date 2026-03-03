import time
from typing import Annotated

import httpx
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from loguru import logger

from app.settings import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


async def _verify_token(token: str) -> bool:
    query = """
        mutation verifyToken($token: String!) {
            verifyToken(token: $token) {
                payload
            }
        }
    """
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{settings.BASEDOSDADOS_BASE_URL}/graphql",
                json={"query": query, "variables": {"token": token}},
            )
        response.raise_for_status()
    except (httpx.HTTPStatusError, httpx.ConnectError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to verify user access",
        )
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"Token verification elapsed time: {elapsed:.4f}s")

    payload = response.json()["data"]["verifyToken"]["payload"]
    return payload["has_chatbot_access"]


async def get_user_id(token: Annotated[str | None, Depends(oauth2_scheme)]) -> int:
    if settings.AUTH_DEV_MODE and settings.ENVIRONMENT == "development":
        logger.warning(
            "AUTH DEV MODE ENABLED: bypassing JWT validation, "
            f"using user_id={settings.AUTH_DEV_USER_ID}",
        )
        return settings.AUTH_DEV_USER_ID

    if settings.AUTH_DEV_MODE and settings.ENVIRONMENT != "development":
        logger.warning(
            f"AUTH_DEV_MODE is enabled but ENVIRONMENT is '{settings.ENVIRONMENT}'. "
            "Auth dev mode will be ignored."
        )

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if token is None:
        raise credentials_exception

    try:
        payload: dict = jwt.decode(
            token,
            key=settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )

        user_id = payload.get("uuid")

        if user_id is None:
            raise credentials_exception

    except jwt.exceptions.InvalidTokenError:
        raise credentials_exception

    if not await _verify_token(token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have chatbot access",
        )

    return user_id


UserID = Annotated[str, Depends(get_user_id)]
