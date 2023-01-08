import math
from typing import Generator

import numpy as np
from open3d.cpu.pybind.camera import (
    PinholeCameraIntrinsic,
)
from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.utility import Vector3dVector
from tqdm import tqdm

from mapper.depth.depth_estimator import DepthEstimator
from mapper.model.image_frame import ImageFrame
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
    PIXEL_ITERATION_STEP = 20

    def __init__(self, depth_estimator: DepthEstimator) -> None:
        self._depth_estimator = depth_estimator

    def generate_point_cloud(
        self,
        image_frame_generator: Generator[ImageFrame, None, None],
        sensor_recording_generator: Generator[SensorRecording, None, None],
    ) -> PointCloud:
        point_cloud: PointCloud = PointCloud()
        for image_frame, sensor_recording in tqdm(
            zip(image_frame_generator, sensor_recording_generator)
        ):
            self._add_estimated_depth_map_to_image_frame_if_missing(image_frame)
            point_cloud += (
                self.generate_point_cloud_from_image_frame_and_sensor_recording(
                    image_frame, sensor_recording
                )
            )
        return point_cloud

    def generate_point_cloud_from_image_frame_and_sensor_recording(
        self, image_frame: ImageFrame, sensor_recording: SensorRecording
    ) -> PointCloud:
        depth_map: np.ndarray = image_frame.depth_map
        image: np.ndarray = image_frame.image

        x_range = range(0, depth_map.shape[0], self.PIXEL_ITERATION_STEP)
        y_range = range(0, depth_map.shape[1], self.PIXEL_ITERATION_STEP)
        x_coords, y_coords = np.meshgrid(x_range, y_range)
        x_coords: np.ndarray = x_coords.flatten()
        y_coords: np.ndarray = y_coords.flatten()

        number_of_points: int = len(x_range) * len(y_range)
        positions: np.ndarray = np.empty(
            (number_of_points, 3), dtype=np.float32
        )
        colors: np.ndarray = np.empty((number_of_points, 3), dtype=np.float32)

        valid_points: np.ndarray = depth_map[x_coords, y_coords] > 0
        x_coords: np.ndarray = x_coords[valid_points]
        y_coords: np.ndarray = y_coords[valid_points]

        positions[valid_points]: list[np.ndarray] = [
            self._get_position_of_midair_pixel(x, y, depth, sensor_recording)
            for x, y, depth in zip(
                x_coords, y_coords, depth_map[x_coords, y_coords]
            )
        ]
        colors[valid_points]: np.ndarray = image[x_coords, y_coords] / 255

        number_of_valid_points: int = valid_points.sum()
        positions: np.ndarray = positions[:number_of_valid_points]
        colors: np.ndarray = colors[:number_of_valid_points]

        point_cloud: PointCloud = PointCloud()
        point_cloud.points = Vector3dVector(positions)
        point_cloud.colors = Vector3dVector(colors)
        return point_cloud

    def _get_position_of_midair_pixel(
        self, x: int, y: int, depth: float, sensor_recording: SensorRecording
    ) -> np.ndarray:
        intrinsic: PinholeCameraIntrinsic = self.DEFAULT_MIDAIR_INTRINSICS
        focal_length: float = intrinsic.height / 2
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

    def _add_estimated_depth_map_to_image_frame_if_missing(
        self, image_frame: ImageFrame
    ) -> None:
        if image_frame.depth_map is None:
            image_frame.depth_map = self._depth_estimator.estimate_depth(
                image_frame.image
            )
