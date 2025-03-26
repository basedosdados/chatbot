import copy
from pathlib import Path
from typing import Literal

from loguru import logger


# Remove all handlers
logger.remove()

# Formatting function
def _format(record):
    keyname = record["extra"]["classname"] if "classname" in record["extra"] else record["name"]
    return "<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | <lvl>{level:<8}</> | <c>%s:{function}:{line}</> - {message}\n{exception}" % keyname

def get_logger(
    classname: str|None = None,
    sink: str|Path = "/var/log/chatbot/chatbot.log",
    level: Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"] = "DEBUG",
    rotation: str|None = "1 day",
    retention: str|None = None,
    backtrace: bool = True,
    diagnose: bool = False,
    enqueue: bool = True
):
    """Creates a loguru logger

    Args:
        classname (str | None, optional): The name of the class which the logger belongs to. Defaults to None.
        sink (str | Path, optional): The file where the logger will write stuff. Defaults to _home_dir/".chatbot/log/chatbot.log".
        level (Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"], optional): The logger severity level. Defaults to "DEBUG".
        rotation (str | None, optional): Rotation time. Defaults to "1 day".
        retention (str | None, optional): Retention time. Defaults to None.
        backtrace (bool, optional): Whether to backtrace exceptions. Defaults to True.
        diagnose (bool, optional): Whether to diagnose exceptions. Defaults to True.
        enqueue (bool, optional): Whether to enqueue log messages, for async logging and multiprocess-safe sinks. Defaults to True.

    Returns:
        Logger: A loguru logger
    """
    logger_ = copy.deepcopy(logger)

    logger_.add(
        sink=sink,
        level=level,
        format=_format,
        rotation=rotation,
        retention=retention,
        backtrace=backtrace,
        diagnose=diagnose,
        enqueue=enqueue
    )

    if classname:
        logger_ = logger_.bind(classname=classname)

    return logger_
