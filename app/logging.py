import sys

from loguru import logger

from app.settings import settings


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


def setup_logger():
    # Remove loguru default handler
    logger.remove()

    logger.add(
        sink=sys.stdout,
        level=settings.LOG_LEVEL,
        format=_format,
        backtrace=settings.LOG_BACKTRACE,
        diagnose=settings.LOG_DIAGNOSE,
        enqueue=settings.LOG_ENQUEUE,
    )
