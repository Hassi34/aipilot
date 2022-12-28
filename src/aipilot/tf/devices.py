from pathlib import Path
import sys
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import tensorflow as tf
from tensorflow.python.client import device_lib
from pckgLogger import logger

class Devices:
    def __init__(self):
        self.logger = logger 

    @property
    def built_with_cuda(self):
        if tf.test.is_built_with_cuda():
            self.logger.info("Tensorflow is built with cuda support ✅")
            return True 
        else:
            self.logger.error("Tensorflow is NOT built with cuda support, please install the CUDA and cuDNN first!")
            return False

    @property
    def gpu_device(self):
        if self.built_with_cuda:
            gpu_devices = tf.config.list_physical_devices('GPU')
            self.logger.info(gpu_devices)
            for device in  gpu_devices:
                details = tf.config.experimental.get_device_details(device)
                self.logger.info(details)
            
    @property
    def available_devices(self):
        all_devices = device_lib.list_local_devices()
        self.logger.info(all_devices)
        return all_devices