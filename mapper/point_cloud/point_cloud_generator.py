import math
from typing import Generator

import numpy as np
from open3d.cpu.pybind.camera import (
    PinholeCameraIntrinsic,
)
from open3d.cpu.pybind.geometry import PointCloud

from mapper.depth.depth_estimator import DepthEstimator
from mapper.model.image_frame import ImageFrame
from mapper.model.position import Position
from mapper.model.sensor_recording import SensorRecording


class PointCloudGenerator:
    DEFAULT_MIDAIR_INTRINSICS = PinholeCameraIntrinsic(
        width=1024,
        height=1024,
        fx=1024 / 2,
        fy=1024 / 2,
        cx=1024 / 2,
        cy=1024 / 2,
    )
    PIXEL_ITERATION_STEP = 10

    def __init__(self, depth_estimator: DepthEstimator) -> None:
        self._depth_estimator = depth_estimator

    def generate_point_cloud(
        self,
        image_frame_generator: Generator[ImageFrame, None, None],
        sensor_recording_generator: Generator[SensorRecording, None, None],
    ) -> PointCloud:
        point_cloud: PointCloud = PointCloud()
        for image_frame, sensor_recording in zip(
            image_frame_generator, sensor_recording_generator
        ):
            self._add_estimated_depth_map_to_image_frame_if_missing(image_frame)
            point_cloud += (
                self.generate_point_cloud_from_image_frame_and_sensor_recording(
                    image_frame, sensor_recording
                )
            )
            print("Iteration complete")
        return point_cloud

    def generate_point_cloud_from_image_frame_and_sensor_recording(
        self, image_frame: ImageFrame, sensor_recording: SensorRecording
    ) -> PointCloud:
        point_cloud: PointCloud = PointCloud()
        depth_map: np.ndarray = image_frame.depth_map
        image: np.ndarray = image_frame.image
        for x in range(0, depth_map.shape[0], self.PIXEL_ITERATION_STEP):
            for y in range(0, depth_map.shape[1], self.PIXEL_ITERATION_STEP):
                depth: float = depth_map[x, y]
                if depth == 0:
                    continue
                position: Position = self._get_position_of_midair_pixel(
                    x, y, depth, sensor_recording
                )
                position_array: np.ndarray = np.array(
                    [position.x, position.y, position.z]
                )
                point_cloud.points.append(position_array)
                point_cloud.colors.append(image[x, y] / 255)
        return point_cloud

    def _get_position_of_midair_pixel(
        self, x: int, y: int, depth: float, sensor_recording: SensorRecording
    ) -> Position:
        intrinsic: PinholeCameraIntrinsic = self.DEFAULT_MIDAIR_INTRINSICS
        focal_length: float = intrinsic.width / 2
        radius: float = depth / math.sqrt(
            ((x - focal_length) ** 2)
            + ((y - focal_length) ** 2)
            + (focal_length ** 2)
        )
        camera_frame_position: Position = Position(
            x=radius * (x - intrinsic.width / 2),
            y=radius * (y - intrinsic.height / 2),
            z=radius * focal_length,
        )
        body_frame_position: Position = Position(
            x=camera_frame_position.z,
            y=camera_frame_position.y,
            z=camera_frame_position.x,
        )
        rotation_matrix: np.ndarray = (
            sensor_recording.rotation.get_rotation_matrix()
        )
        position: Position = Position(
            x=rotation_matrix[0, 0] * body_frame_position.x
            + rotation_matrix[0, 1] * body_frame_position.y
            + rotation_matrix[0, 2] * body_frame_position.z
            + sensor_recording.position.x,
            y=rotation_matrix[1, 0] * body_frame_position.x
            + rotation_matrix[1, 1] * body_frame_position.y
            + rotation_matrix[1, 2] * body_frame_position.z
            + sensor_recording.position.y,
            z=rotation_matrix[2, 0] * body_frame_position.x
            + rotation_matrix[2, 1] * body_frame_position.y
            + rotation_matrix[2, 2] * body_frame_position.z
            + sensor_recording.position.z,
        )
        return position

    def _add_estimated_depth_map_to_image_frame_if_missing(
        self, image_frame: ImageFrame
    ) -> None:
        if image_frame.depth_map is None:
            image_frame.depth_map = self._depth_estimator.estimate_depth(
                image_frame.image
            )
