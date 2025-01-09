"""
Module: gpu

This module provides a singleton-based utility for managing GPU resources and
dynamically selecting the appropriate computation libraries based on the availability
of GPU hardware. It enables seamless integration of GPU-accelerated libraries with
fallback to CPU-based libraries when GPUs are not available.
"""
import os

import GPUtil
import torch

from featurestore.base.utils.logger import logger
from featurestore.base.utils.singleton import SingletonMeta


class GpuLoading(metaclass=SingletonMeta):
    """
    GPU resource management class.

    Attributes:
        __device_id (int): The ID of the GPU device to use.
        __is_gpu_available (bool): Whether GPU resources are available.
        __df_backbone: The data handling library to use (cuDF or Pandas).
        __scipy_backbone: The scientific computation library to use (cuSciPy or SciPy).
        __np_backbone: The array computation library to use (CuPy or NumPy).

    Args:
        max_memory (float): Maximum memory utilization of a GPU for it to be
            considered available.
    """

    def __init__(self, max_memory=0.8):
        self.__device_id = self.__get_gpu_id(max_memory)
        self.__is_gpu_available = self.__device_id > -1
        logger.info(
            f"__is_gpu_available={self.__is_gpu_available},"
            f"detect device id  = {self.__device_id}"
        )
        self.set_gpu_use()

    def set_gpu_use(self):
        """
        Configures the application to use a specified GPU device if available.

        This method checks for GPU availability and sets the GPU device for computation
        using libraries like CuPy and PyTorch. It ensures that the assigned device is
        used for all GPU-dependent operations and logs the selected device ID.
        """
        # pylint: disable=C0415
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
        for gpu in GPUtil.getGPUs():
            if gpu.memoryUtil > max_memory:
                continue
            if gpu.memoryFree >= memory_free:
                available = gpu.id
                # memory_free = GPU.memoryFree
                break
        return available

    def is_gpu_available(
        self,
    ):
        """
        Checks if a GPU is available.

        Returns:
            bool: True if a GPU is available, otherwise False.
        """
        return self.__is_gpu_available

    def get_gpu_device_id(self):
        """
        Gets the GPU device ID.

        Returns:
            int or None: The GPU device ID if a GPU is available; otherwise, None.
        """
        return self.__device_id
