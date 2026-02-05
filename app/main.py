from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from loguru import logger
from psycopg_pool import AsyncConnectionPool

from app.agent import ReActAgent
from app.agent.hooks import trim_messages_before_agent
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import BDToolkit
from app.api.main import api_router
from app.db.database import engine, init_database
from app.logging import setup_logger
from app.settings import settings

setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):  # pragma: no cover
    if settings.AUTH_DEV_MODE and settings.ENVIRONMENT == "development":
        logger.warning(
            "AUTH DEV MODE ENABLED: JWT validation is bypassed, "
            f"all requests will use user_id={settings.AUTH_DEV_USER_ID}"
        )

    if settings.AUTH_DEV_MODE and settings.ENVIRONMENT != "development":
        logger.warning(
            f"AUTH_DEV_MODE is enabled but ENVIRONMENT is '{settings.ENVIRONMENT}'. "
            "Auth dev mode will be ignored."
        )

    await init_database(engine)

    # Connection kwargs defined according to:
    # https://github.com/langchain-ai/langgraph/issues/2887
    # https://langchain-ai.github.io/langgraph/how-tos/persistence_postgres
    conn_kwargs = {"autocommit": True, "prepare_threshold": 0}

    model = init_chat_model(
        model=settings.MODEL_URI,
        temperature=settings.MODEL_TEMPERATURE,
        credentials=settings.GOOGLE_CREDENTIALS,
    )

    async with AsyncConnectionPool(
        conninfo=settings.DB_URL, max_size=8, kwargs=conn_kwargs
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)

        await checkpointer.setup()

        agent = ReActAgent(
            model=model,
            tools=BDToolkit.get_tools(),
            start_hook=trim_messages_before_agent,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer,
        )

        app.state.agent = agent

        yield

    await engine.dispose()


app = FastAPI(lifespan=lifespan)

app.include_router(api_router)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
