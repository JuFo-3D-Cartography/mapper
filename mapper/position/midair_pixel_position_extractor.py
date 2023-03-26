import math

import numpy as np
from open3d.cpu.pybind.camera import PinholeCameraIntrinsic

from mapper.model.sensor_recording import SensorRecording
from mapper.position.pixel_position_extractor import PixelPositionExtractor


class MidAirPixelPositionExtractor(PixelPositionExtractor):
    def get_position_of_pixel(
        self,
        x: int,
        y: int,
        depth: float,
        sensor_recording: SensorRecording,
        camera_intrinsic: PinholeCameraIntrinsic,
    ) -> np.ndarray:
        focal_length: float = camera_intrinsic.width / 2
        ray_length: float = depth / math.sqrt(
            ((x - focal_length) ** 2)
            + ((y - focal_length) ** 2)
            + (focal_length**2)
        )
        camera_frame_position: np.ndarray = np.array(
            [
                ray_length * (x - focal_length),
                ray_length * (y - focal_length),
                ray_length * focal_length,
            ]
        )
        body_frame_position: np.ndarray = camera_frame_position[[2, 1, 0]]
        rotation_matrix: np.ndarray = (
            sensor_recording.rotation.get_rotation_matrix()
        )
        position: np.ndarray = rotation_matrix.dot(
            body_frame_position
        ) + np.array(
            [
                sensor_recording.position.x,
                sensor_recording.position.y,
                sensor_recording.position.z,
            ]
        )
        return position
