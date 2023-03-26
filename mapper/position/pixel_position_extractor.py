from abc import abstractmethod

import numpy as np
from open3d.cpu.pybind.camera import PinholeCameraIntrinsic

from mapper.model.sensor_recording import SensorRecording


class PixelPositionExtractor:
    @abstractmethod
    def get_position_of_pixel(
        self,
        x: int,
        y: int,
        depth: float,
        sensor_recording: SensorRecording,
        camera_intrinsic: PinholeCameraIntrinsic,
    ) -> np.ndarray:
        raise NotImplementedError
