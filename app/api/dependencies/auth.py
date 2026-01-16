from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.settings import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


async def get_user_id(token: Annotated[str, Depends(oauth2_scheme)]) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload: dict = jwt.decode(
            token,
            key=settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )

        user_id = payload.get("user_id")

        if user_id is None:
            raise credentials_exception

    except jwt.exceptions.InvalidTokenError:
        raise credentials_exception

    return user_id


UserID = Annotated[str, Depends(get_user_id)]
