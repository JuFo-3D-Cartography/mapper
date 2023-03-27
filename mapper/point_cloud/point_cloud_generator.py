from itertools import islice
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
from mapper.position.pixel_position_extractor import PixelPositionExtractor


class PointCloudGenerator:
    IMAGE_WIDTH: int = 1024
    IMAGE_HEIGHT: int = 1024
    DEFAULT_MIDAIR_INTRINSICS = PinholeCameraIntrinsic(
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        fx=IMAGE_WIDTH / 2,
        fy=IMAGE_HEIGHT / 2,
        cx=IMAGE_WIDTH / 2,
        cy=IMAGE_HEIGHT / 2,
    )

    def __init__(
        self,
        pixel_iteration_step: int,
        depth_estimator: DepthEstimator,
        pixel_position_extractor: PixelPositionExtractor,
    ) -> None:
        self._pixel_iteration_step: int = pixel_iteration_step
        self._depth_estimator = depth_estimator
        self._pixel_position_extractor = pixel_position_extractor

    def generate_point_cloud(
        self,
        image_frame_generator: Generator[ImageFrame, None, None],
        sensor_recording_generator: Generator[SensorRecording, None, None],
        max_number_of_frames: int,
    ) -> PointCloud:
        point_cloud: PointCloud = PointCloud()
        for image_frame, sensor_recording in tqdm(
            islice(
                zip(image_frame_generator, sensor_recording_generator),
                max_number_of_frames,
            ),
            total=max_number_of_frames,
            desc="Generating point cloud",
        ):
            self._add_estimated_depth_map_to_image_frame_if_missing(image_frame)
            point_cloud += (
                self.generate_point_cloud_from_image_frame_and_sensor_recording(
                    image_frame, sensor_recording
                )
            )
        rotation_matrix: np.ndarray = (
            point_cloud.get_rotation_matrix_from_axis_angle((1.5, 1.5, -0.9))
        )
        point_cloud.rotate(
            rotation_matrix,
            center=(0, 0, 0),
        )
        return point_cloud

    def generate_point_cloud_from_image_frame_and_sensor_recording(
        self, image_frame: ImageFrame, sensor_recording: SensorRecording
    ) -> PointCloud:
        depth_map: np.ndarray = image_frame.depth_map
        image: np.ndarray = image_frame.image
        x_range: range = range(
            0, depth_map.shape[0], self._pixel_iteration_step
        )
        y_range: range = range(
            0, depth_map.shape[1], self._pixel_iteration_step
        )
        x_coords, y_coords = np.meshgrid(x_range, y_range)
        valid_points: np.ndarray = depth_map[x_coords, y_coords] < 100
        x_coords: np.ndarray = x_coords[valid_points]
        y_coords: np.ndarray = y_coords[valid_points]
        positions: list[np.ndarray] = [
            self._pixel_position_extractor.get_position_of_pixel(
                x, y, depth, sensor_recording, self.DEFAULT_MIDAIR_INTRINSICS
            )
            for x, y, depth in zip(
                x_coords, y_coords, depth_map[x_coords, y_coords]
            )
        ]
        colors: np.ndarray = image[x_coords, y_coords] / 255
        point_cloud: PointCloud = PointCloud()
        point_cloud.points = Vector3dVector(positions)
        point_cloud.colors = Vector3dVector(colors)
        return point_cloud

    def _add_estimated_depth_map_to_image_frame_if_missing(
        self, image_frame: ImageFrame
    ) -> None:
        if image_frame.depth_map is None:
            image_frame.depth_map = self._depth_estimator.estimate_depth(
                image_frame.image
            )
