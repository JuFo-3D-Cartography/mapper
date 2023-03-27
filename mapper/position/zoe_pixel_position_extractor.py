import numpy as np
from open3d.cpu.pybind.camera import PinholeCameraIntrinsic

from mapper.model.sensor_recording import SensorRecording
from mapper.position.pixel_position_extractor import PixelPositionExtractor


class ZoePixelPositionExtractor(PixelPositionExtractor):
    def get_position_of_pixel(
        self,
        x: int,
        y: int,
        depth: float,
        sensor_recording: SensorRecording,
        camera_intrinsic: PinholeCameraIntrinsic,
    ) -> np.ndarray:
        intrinsic_matrix_inverse: np.ndarray = np.linalg.inv(
            camera_intrinsic.intrinsic_matrix
        )
        two_d_position: np.ndarray = np.array([x, y, 1]).reshape(3, 1)
        camera_position: np.ndarray = (
            depth * intrinsic_matrix_inverse @ two_d_position
        )
        body_position: np.ndarray = camera_position[[2, 1, 0], 0]
        rotation_matrix: np.ndarray = (
            sensor_recording.rotation.get_rotation_matrix()
        )
        position: np.ndarray = rotation_matrix @ body_position + np.array(
            [
                sensor_recording.position.x,
                sensor_recording.position.y,
                sensor_recording.position.z,
            ]
        )
        return position
