from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import AsyncDatabase, sessionmaker


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async database session.

    Yields:
        AsyncSession: An async SQLAlchemy session.
    """
    async with sessionmaker() as session:
        yield session


async def get_database(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AsyncDatabase:
    """Provide an AsyncDatabase instance with an injected session.

    Args:
        session: The async session from get_session dependency.

    Returns:
        AsyncDatabase: A database repository instance.
    """
    return AsyncDatabase(session)


AsyncDB = Annotated[AsyncDatabase, Depends(get_database)]
