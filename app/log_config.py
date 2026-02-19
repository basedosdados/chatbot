import logging
import sys

from loguru import logger

from app.settings import settings


class EndpointFilter(logging.Filter):
    def __init__(self, excluded_endpoints: list[str]):
        self.excluded_endpoints = excluded_endpoints

    def filter(self, record: logging.LogRecord):
        return not any(ep in record.getMessage() for ep in self.excluded_endpoints)


def _format(record):
    if "classname" in record["extra"]:
        keyname = record["extra"]["classname"]
    else:
        keyname = record["name"]

    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | "
        "<level>{level:<8}</> | "
        "<bold>PID: {process}</> | "
        "<cyan>%s:{function}:{line}</> - {message}\n{exception}" % keyname
    )


def setup_logging():
    logging.getLogger("uvicorn.access").addFilter(
        EndpointFilter(excluded_endpoints=["/health"])
    )

    logger.remove()

    logger.add(
        sink=sys.stdout,
        level=settings.LOG_LEVEL,
        format=_format,
        backtrace=settings.LOG_BACKTRACE,
        diagnose=settings.LOG_DIAGNOSE,
        enqueue=settings.LOG_ENQUEUE,
    )
