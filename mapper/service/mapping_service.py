import time
from pathlib import Path
from typing import Generator

from open3d.cpu.pybind.geometry import PointCloud
from open3d.cpu.pybind.io import write_point_cloud
from open3d.cpu.pybind.visualization import draw_geometries

from mapper.model.image_frame import ImageFrame
from mapper.model.sensor_recording import SensorRecording
from mapper.point_cloud.point_cloud_generator import PointCloudGenerator
from mapper.reader.image_frames_reader import ImageFramesReader
from mapper.reader.sensor_recordings_reader import SensorRecordingsReader


class MappingService:
    def __init__(
        self,
        point_cloud_generator: PointCloudGenerator,
    ) -> None:
        self._point_cloud_generator = point_cloud_generator

    def generate_and_save_point_cloud_from_midair_data(
        self,
        sensor_recordings_path: Path,
        sensor_recordings_trajectory: str,
        image_frames_path: Path,
        depth_map_frames_path: Path,
        point_cloud_save_path: Path,
        max_number_of_frames: int,
    ) -> None:
        image_frame_generator: Generator[
            ImageFrame, None, None
        ] = ImageFramesReader.read_image_frames(
            image_frames_path, depth_map_frames_path
        )
        sensor_recordings_generator: Generator[
            SensorRecording, None, None
        ] = SensorRecordingsReader.read_sensor_recordings(
            sensor_recordings_path,
            sensor_recordings_trajectory,
        )

        start_time: float = time.time()
        point_cloud: PointCloud = (
            self._point_cloud_generator.generate_point_cloud(
                image_frame_generator,
                sensor_recordings_generator,
                max_number_of_frames,
            )
        )
        end_time: float = time.time()
        print(f"Time to generate point cloud: {end_time - start_time} seconds")

        write_point_cloud(str(point_cloud_save_path), point_cloud)
        draw_geometries([point_cloud])
