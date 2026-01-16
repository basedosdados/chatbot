from fastapi import APIRouter

from app.api.routers import chatbot
from app.settings import settings

api_router = APIRouter(prefix=settings.API_PREFIX)

api_router.include_router(chatbot.router)
