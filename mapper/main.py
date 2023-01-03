from pathlib import Path

import typer

from mapper.depth.depth_estimator import DepthEstimator
from mapper.point_cloud.point_cloud_generator import PointCloudGenerator
from mapper.reader.image_frames_reader import ImageFramesReader
from mapper.reader.sensor_recordings_reader import SensorRecordingsReader
from mapper.service.mapping_service import MappingService
from mapper.validator.cli_path_validator import CliPathValidator


def validate_path_input(
    sensor_recordings_path: Path,
    image_frames_path: Path,
    point_cloud_save_path: Path,
) -> None:
    CliPathValidator.validate_file(sensor_recordings_path)
    CliPathValidator.validate_directory(image_frames_path)
    CliPathValidator.validate_file(point_cloud_save_path)


def main(
    sensor_recordings_path: Path = typer.Argument(
        ...,
        help="Path to sensor recordings, the data must in sync with the image "
        "frames (same number of frames, etc.)",
    ),
    sensor_recordings_trajectory: str = typer.Option(
        "trajectory_5000", help="Name of the trajectory to use"
    ),
    image_frames_path: Path = typer.Argument(
        ...,
        help="Path to image frames, the data must in sync with the sensor "
        "recordings (same number of frames, etc.)",
    ),
    point_cloud_save_path: Path = typer.Argument(
        ...,
        help="Path to save the generated point cloud, the file extension "
        "determines the file format",
    ),
) -> None:
    validate_path_input(
        sensor_recordings_path,
        image_frames_path,
        point_cloud_save_path,
    )

    image_frames_reader: ImageFramesReader = ImageFramesReader()
    sensor_recordings_reader: SensorRecordingsReader = SensorRecordingsReader()
    depth_estimator: DepthEstimator = DepthEstimator()
    point_cloud_generator: PointCloudGenerator = PointCloudGenerator(
        depth_estimator
    )
    mapping_service: MappingService = MappingService(
        image_frames_reader, sensor_recordings_reader, point_cloud_generator
    )

    mapping_service.generate_point_cloud_from_midair_data(
        sensor_recordings_path,
        sensor_recordings_trajectory,
        image_frames_path,
        point_cloud_save_path,
    )


if __name__ == "__main__":
    typer.run(main)
