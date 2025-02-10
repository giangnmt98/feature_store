"""
Module: spark

This module provides utility classes and methods for managing PySpark operations,
offering functionality to initialize, configure, and manage Spark sessions.
It also supports efficient resource handling and temporary data cleanup.
"""
import atexit
import os
import random
import shutil
import threading
from pathlib import Path

from pyspark.sql import SparkSession

from featurestore.base.utils.logger import logger
from featurestore.base.utils.resource import ResourceInfo
from featurestore.base.utils.singleton import SingletonMeta


class AtomicCounter:
    """
    A thread-safe atomic counter.

    The `AtomicCounter` class provides a counter that can be safely incremented
    by multiple threads in a concurrent environment. It ensures atomicity
    using a threading lock, preventing race conditions during updates.

    Attributes:
        value (int): The current value of the counter, initialized to the specified
        value or 0 by default.
    """

    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value (default 0)."""
        self.value = initial
        self._lock = threading.Lock()

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value += num
            return self.value


class SparkOperations(metaclass=SingletonMeta):
    """
    Tập hợp funcs xử lý cho sparks
    """

    def __init__(self, config_spark=None):
        self.atom = AtomicCounter()
        if config_spark is None:
            cpu_factor = 0.5
            app_name = "spark-application"
            resource_info = ResourceInfo()
            num_cores = max(int((resource_info.num_cores - 1) * cpu_factor), 1)
            master = f"local[{num_cores}]"
            partitions = num_cores
            memory = resource_info.memory
            driver_memory = f"{int(0.8*  memory )}g"
            executor_memory = f"{int(0.8* memory)}g"
            auto_broadcast_join_threshold = 10485760
            base_checkpoint_dir = "/home/cuongit/BIGDATA/tmp/pyspark/"
            if os.getenv("PREFIX_CHECKPOINT_DIR") is not None:
                base_checkpoint_dir = (
                    os.getenv("PREFIX_CHECKPOINT_DIR") + base_checkpoint_dir
                )
            checkpoint_dir = (
                base_checkpoint_dir + f"tmp_{self.atom.increment()}"
                f"_{random.randint(10000, 100000)}"
                f"_{random.randint(100, 5000000)}"
            )
            logger.info(
                f"app_name={app_name} master={master} "
                f"partitions={partitions} driver_memory={driver_memory} "
                f"num_cores={num_cores} driver_memory={driver_memory} "
                f"executor_memory={executor_memory} checkpoint_dir={checkpoint_dir}"
            )
        else:
            app_name = config_spark.name
            master = config_spark.master
            params = config_spark.params
            partitions = params.sql_shuffle_partitions
            driver_memory = params.driver_memory
            num_cores = params.num_cores
            executor_memory = params.executor_memory
            auto_broadcast_join_threshold = params.auto_broadcast_join_threshold
            checkpoint_dir = params.checkpoint_dir
        self.checkpoint_dir = checkpoint_dir
        self.spark_config = (
            SparkSession.builder.appName(app_name)
            .master(master)
            .config("spark.sql.shuffle.partitions", partitions)
            .config("spark.driver.memory", driver_memory)
            .config("spark.executor.memory", executor_memory)
            .config("spark.sql.execution.arrow.pyspark.enabled", True)
            .config("spark.sql.files.ignoreCorruptFiles", True)
            .config("spark.sql.files.ignoreMissingFiles", True)
            .config("spark.executor.cores", num_cores)
            .config("spark.driver.cores", num_cores)
            .config("spark.driver.maxResultSize", "100g")
            .config(
                "spark.sql.autoBroadcastJoinThreshold",
                auto_broadcast_join_threshold,
            )
            .config("spark.local.dir", self.checkpoint_dir)
        )
        self.partitions = partitions
        # https://spark.apache.org/docs/latest/sql-data-sources-generic-options.html
        self.__spark = self.__init_spark_session()
        atexit.register(self.clean_tmp_data)

    def get_spark_session(self):
        """Init spark session, set log level to warn"""
        if self.__spark is None:
            self.__spark = self.__init_spark_session()
        return self.__spark

    def __init_spark_session(self):
        """Init spark session, set log level to warn"""
        spark = self.spark_config.getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        spark.catalog.clearCache()
        spark.sparkContext.setCheckpointDir(self.checkpoint_dir)
        spark.conf.set("spark.sql.session.timeZone", "UTC")
        return spark

    def clean_tmp_data(self):
        """
        Cleans temporary checkpoint data to release disk space.
        """
        if Path(self.checkpoint_dir).exists():
            logger.opt(depth=-1).info(
                "pyspark: cleaning all checkpoints to release disk cache"
            )
            try:
                shutil.rmtree(Path(self.checkpoint_dir))
            except Exception:
                logger.opt(depth=-1).info("checkpoint may not be removed clearly")
