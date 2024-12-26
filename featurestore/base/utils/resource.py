import os

import psutil

from featurestore.base.utils.singleton import SingletonMeta


class ResourceInfo(metaclass=SingletonMeta):
    def __init__(self):
        if os.getenv("NUM_CORES") is None:
            self.num_cores = os.cpu_count()
        else:
            self.num_cores = int(os.getenv("NUM_CORES"))

        if os.getenv("WORKING_RAM") is None:
            self.memory = psutil.virtual_memory()[1] / 1e9
        else:
            self.memory = int(os.getenv("WORKING_RAM"))
