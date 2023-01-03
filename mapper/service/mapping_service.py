from pathlib import Path
from typing import Generator
from open3d.cpu.pybind.geometry import PointCloud

from mapper.model.image_frame import ImageFrame
from mapper.model.sensor_recording import SensorRecording
from mapper.point_cloud.point_cloud_generator import PointCloudGenerator
from mapper.reader.image_frames_reader import ImageFramesReader
from mapper.reader.sensor_recordings_reader import SensorRecordingsReader


class MappingService:
    def __init__(
        self,
        image_frame_reader: ImageFramesReader,
        sensor_recording_reader: SensorRecordingsReader,
        point_cloud_generator: PointCloudGenerator,
    ) -> None:
        self._image_frames_reader = image_frame_reader
        self._sensor_recordings_reader = sensor_recording_reader
        self._point_cloud_generator = point_cloud_generator

    def generate_point_cloud_from_midair_data(
        self,
        sensor_recordings_path: Path,
        sensor_recordings_trajectory: str,
        image_frames_path: Path,
        point_cloud_save_path: Path,
    ) -> None:
        image_frame_generator: Generator[
            ImageFrame, None, None
        ] = self._image_frames_reader.read_image_frames(image_frames_path)
        sensor_recordings_generator: Generator[
            SensorRecording, None, None
        ] = self._sensor_recordings_reader.read_sensor_recordings(
            sensor_recordings_path, sensor_recordings_trajectory
        )

        point_cloud: PointCloud = (
            self._point_cloud_generator.generate_point_cloud(
                image_frame_generator, sensor_recordings_generator
            )
        )

        point_cloud.save(point_cloud_save_path)
