"""
Module: resource

This module provides a singleton-based utility to retrieve system resource information,
such as the number of CPU cores and available working memory. It is designed to support
resource-aware operations by dynamically detecting or using environment-defined
resources.
"""
import os

import psutil

from featurestore.base.utils.singleton import SingletonMeta


class ResourceInfo(metaclass=SingletonMeta):
    """
    A singleton class for retrieving system resource information.

    This class provides details about the number of CPU cores and the available working
    memory, either from environment variables or by detecting them from the system.

    Attributes:
        num_cores (int): The number of CPU cores. This is fetched from the `NUM_CORES`
            environment variable if set, otherwise it detects the total number of
            CPU cores available.
        memory (float): The available working memory in gigabytes. This is fetched
            from the `WORKING_RAM` environment variable if set, otherwise it retrieves
            the available memory using `psutil.virtual_memory()`.
    """

    def __init__(self):
        if os.getenv("NUM_CORES") is None:
            self.num_cores = os.cpu_count()
        else:
            self.num_cores = int(os.getenv("NUM_CORES"))

        if os.getenv("WORKING_RAM") is None:
            self.memory = psutil.virtual_memory()[1] / 1e9
        else:
            self.memory = int(os.getenv("WORKING_RAM"))
