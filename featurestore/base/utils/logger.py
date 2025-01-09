"""
Module: logger

This module provides a singleton-based logging utility that integrates both Loguru and
Python's built-in logging module, enabling flexible and configurable logging across
the application.
"""
import logging as logging_logger
import sys

from loguru import logger as loguru_logger

from featurestore.base.utils.singleton import SingletonMeta


class Logger(metaclass=SingletonMeta):
    """
    A logging utility class that provides a configurable interface for using either
    Loguru or Python's built-in logging module.

    This class is implemented as a singleton to ensure consistent logging configuration
    across the application.

    Attributes:
        loguru_logger: Instance of the Loguru logger with predefined formatting and
            settings.
        logging_logger: Instance of Python's built-in `logging` module.
        is_loguru (bool): Determines whether to use Loguru as the default logger.

    Args:
        is_loguru (bool): If True, Loguru is used as the default logger.
            Otherwise, Python's logging module is used. Defaults to True.
        update (bool): Placeholder argument for potential future logic.
    """

    def __init__(self, is_loguru=True):
        self.loguru_logger = loguru_logger
        self.loguru_logger.remove()
        self.loguru_logger.add(
            sys.stderr,
            colorize=True,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}"
            "</green> | <level>{level: <8}</level>| <cyan>{name}"
            "</cyan>:<cyan>{function}</cyan>:<cyan>{line"
            "}</cyan>- <level>{message}</level>",
            level="INFO",
        )
        self.logging_logger = logging_logger
        self.logging_logger.basicConfig(level=logging_logger.INFO)

        self.is_loguru = is_loguru

    def get_logger(self):
        """
        Retrieves the appropriate logger instance.

        Returns:
            Logger: The active logger instance, either `loguru` logger
                or standard `logging` logger.
        """

        if self.is_loguru:
            return self.loguru_logger
        return self.logging_logger


logger = Logger().get_logger()
