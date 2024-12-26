import os

import GPUtil
import numpy as np
import pandas as pd
import torch

from featurestore.base.utils.logger import logger
from featurestore.base.utils.singleton import SingletonMeta


class GpuLoading(metaclass=SingletonMeta):
    def __init__(self, max_memory=0.8):
        self.__device_id = self.__get_gpu_id(max_memory)
        self.__is_gpu_available = self.__device_id > -1
        logger.info(
            f"__is_gpu_available={self.__is_gpu_available},"
            f"detect device id  = {self.__device_id}"
        )
        self.__df_backbone = self.__get_pd_or_cudf()
        self.__scipy_backbone = self.__get_scipy_or_cuscipy()
        self.__np_backbone = self.__get_np_or_cupy()
        # self.__vector_search_backbone = self.__get_vector_search()
        self.set_gpu_use()

    def set_gpu_use(self):
        if self.is_gpu_available():
            import cupy

            cupy.cuda.Device(int(self.get_gpu_device_id())).use()
            torch.cuda.set_device(int(self.get_gpu_device_id()))
            logger.info(f"we have set device to {self.__device_id} id")

    def __get_gpu_id(self, max_memory=0.8):
        memory_free = 0
        available = -1
        if os.getenv("CUDA_VISIBLE_DEVICES") == "":
            return available
        for GPU in GPUtil.getGPUs():
            if GPU.memoryUtil > max_memory:
                continue
            if GPU.memoryFree >= memory_free:
                available = GPU.id
                # memory_free = GPU.memoryFree
                # todo fix issue
                # issue ilegial memory access if cudf on device !=0
                # currently, I don't know the reason, lets check later
                # break to auto select device 0
                break
        return available

    def is_gpu_available(
        self,
    ):
        return self.__is_gpu_available

    def get_gpu_device_id(self):
        return self.__device_id

    def __get_pd_or_cudf(self):
        """
        Get pandas or cudf depending on the environment
        """
        if self.is_gpu_available():
            try:
                import cudf

                return cudf
            except ImportError:
                logger.warning("cudf is not installed. Using pandas instead.")
                return pd
        else:
            return pd

    def __get_np_or_cupy(self):
        """
        Get numpy or cupy depending on the environment
        """
        if self.is_gpu_available():
            try:
                import cupy

                return cupy
            except ImportError:
                logger.warning("cupy is not installed. Using numpy instead.")
                return np
        else:
            return np

    def __get_scipy_or_cuscipy(self):
        """
        Get scipy or cupyx depending on the environment
        """
        import scipy

        if self.is_gpu_available():
            try:
                import cupyx.scipy

                return cupyx.scipy
            except ImportError:
                logger.warning("cupyx is not installed. Using scipy instead.")
                return scipy
        else:
            return scipy
