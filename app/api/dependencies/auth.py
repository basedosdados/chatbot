from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from loguru import logger

from app.settings import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)


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

    return user_id


UserID = Annotated[str, Depends(get_user_id)]
