"""Set Logger

The implementation is based on https://github.com/serend1p1ty/core-pytorch-utils/blob/main/cpu/logger.py
"""
import logging
import os
import sys
from copy import deepcopy
from typing import Optional

try:
    from termcolor import colored

    has_termcolor = True
except ImportError:
    has_termcolor = False

try:
    from rich.logging import RichHandler

    has_rich = True
except ImportError:
    has_rich = False

__all__ = [
    "set_logger",
]

logger_initialized = {}


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style="%"):
        super().__init__(fmt, datefmt, style)

    def formatMessage(self, record):
        record = deepcopy(record)  # deepcopy avoid change of original record, which may influencing other handler
        record.asctime = colored(record.asctime, "light_cyan")
        record.name = colored(record.name, "blue")
        # record.filename, record.funcName, record.lineno
        if record.levelno == logging.DEBUG:
            record.levelname = colored(record.levelname, "magenta")
        elif record.levelno == logging.INFO:
            record.levelname = colored(record.levelname, "green")
        elif record.levelno == logging.WARNING:
            record.levelname = colored(record.levelname, "yellow", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            record.levelname = colored(record.levelname, "red", attrs=["blink", "underline"])
        return super().formatMessage(record)


def set_logger(
    name: Optional[str] = None,
    output_dir: Optional[str] = None,
    rank: int = 0,
    log_level: int = logging.INFO,
    color: bool = True,
) -> logging.Logger:
    """Initialize the logger.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, only logger of the master
    process is added console handler. If ``output_dir`` is specified, all loggers
    will be added file handler.

    Args:
        name: Logger name. Defaults to None to set up root logger.
        output_dir: The directory to save log.
        rank: Process rank in the distributed training. Defaults to 0.
        log_level: Verbosity level of the logger. Defaults to ``logging.INFO``.
        color: If True, color the output. Defaults to True.

    Returns:
        logging.Logger: A initialized logger.
    """
    if name in logger_initialized:
        return logger_initialized[name]

    # get root logger if name is None
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    # the messages of this logger will not be propagated to its parent
    logger.propagate = False

    fmt = "%(asctime)s %(name)s %(levelname)s - %(message)s"
    datefmt = "[%Y-%m-%d %H:%M:%S]"

    # create console handler for master process
    if rank == 0:
        if color:
            if has_rich:
                console_handler = RichHandler(level=log_level, log_time_format=datefmt)
            elif has_termcolor:
                console_handler = logging.StreamHandler(stream=sys.stdout)
                console_handler.setLevel(log_level)
                console_handler.setFormatter(_ColorfulFormatter(fmt=fmt, datefmt=datefmt))
            else:
                raise NotImplementedError("If you want color, 'rich' or 'termcolor' has to be installed!")
        else:
            console_handler = logging.StreamHandler(stream=sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(console_handler)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(output_dir, f"rank{rank}.log"))
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(file_handler)

    logger_initialized[name] = logger
    return logger
