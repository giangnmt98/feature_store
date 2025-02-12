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

    def __init__(self, spark_config=None):
        """
        Initialize SparkOperations with configuration.
        """

        def generate_checkpoint_dir(base_dir: str) -> str:
            """Generate a unique checkpoint directory."""
            return (
                base_dir + f"tmp_{self.atom.increment()}_{random.randint(10000,100000)}"
                f"_{random.randint(100, 5000000)}"
            )

        self.atom = AtomicCounter()
        resource_info = ResourceInfo()
        base_checkpoint_dir = os.getenv("PREFIX_CHECKPOINT_DIR", "")  # Common logic

        if spark_config is None:  # Default spark configuration
            default_cpu_factor = 0.7
            memory_percentage = 0.8
            default_broadcast_threshold = 10485760
            app_name = "spark-application"
            num_cores = max(int((resource_info.num_cores - 1) * default_cpu_factor), 1)
            master = f"local[{num_cores}]"
            memory = resource_info.memory

            config_params = {
                "partitions": num_cores,
                "driver_memory": f"{int(memory_percentage * memory)}g",
                "executor_memory": f"{int(memory_percentage * memory)}g",
                "auto_broadcast_join_threshold": default_broadcast_threshold,
            }
        else:  # Custom spark configuration
            app_name = spark_config["name"]
            master = spark_config["master"]
            params = spark_config["params"]
            num_cores = params["num_cores"]
            config_params = {
                "partitions": params["sql_shuffle_partitions"],
                "driver_memory": params["driver_memory"],
                "executor_memory": params["executor_memory"],
                "auto_broadcast_join_threshold": params[
                    "auto_broadcast_join_threshold"
                ],
            }

        # Assign base_checkpoint_dir and calculate checkpoint
        full_checkpoint_dir = generate_checkpoint_dir(base_checkpoint_dir)

        # Initialize Spark configurations
        self.spark_config = (
            SparkSession.builder.appName(app_name)
            .master(master)
            .config("spark.sql.shuffle.partitions", config_params["partitions"])
            .config("spark.driver.memory", config_params["driver_memory"])
            .config("spark.executor.memory", config_params["executor_memory"])
            .config("spark.sql.execution.arrow.pyspark.enabled", True)
            .config("spark.sql.files.ignoreCorruptFiles", True)
            .config("spark.sql.files.ignoreMissingFiles", True)
            .config("spark.executor.cores", num_cores)
            .config("spark.driver.cores", num_cores)
            .config(
                "spark.sql.autoBroadcastJoinThreshold",
                config_params["auto_broadcast_join_threshold"],
            )
            .config("spark.local.dir", full_checkpoint_dir)
            .config("spark.scheduler.listenerbus.eventqueue.capacity", "20000")
            .config("spark.sql.files.maxPartitionBytes", "64MB")
        )
        self.partitions = config_params["partitions"]
        logger.info(
            f"app_name={app_name} master={master} "
            f"partitions={config_params['partitions']} "
            f"driver_memory={config_params['driver_memory']} "
            f"executor_memory={config_params['executor_memory']} "
            f"checkpoint_dir={full_checkpoint_dir}"
        )
        self.checkpoint_dir = full_checkpoint_dir
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
