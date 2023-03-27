from pathlib import Path

import typer

from mapper.depth.midas_depth_estimator import MidasDepthEstimator
from mapper.depth.zoe_depth_estimator import ZoeDepthEstimator
from mapper.point_cloud.point_cloud_generator import PointCloudGenerator
from mapper.position.midair_pixel_position_extractor import (
    MidAirPixelPositionExtractor,
)
from mapper.position.zoe_pixel_position_extractor import (
    ZoePixelPositionExtractor,
)
from mapper.service.mapping_service import MappingService
from mapper.validator.cli_input_validator import CliInputValidator

app = typer.Typer()


def validate_midair_mapping_input(
    pixel_iteration_step: int,
    sensor_recordings_path: Path,
    image_frames_path: Path,
    depth_maps_path: Path,
    point_cloud_save_path: Path,
) -> None:
    CliInputValidator.validate_pixel_iteration_step(pixel_iteration_step)
    CliInputValidator.validate_file(sensor_recordings_path)
    CliInputValidator.validate_directory(image_frames_path)
    if depth_maps_path is not None:
        CliInputValidator.validate_directory(depth_maps_path)
    CliInputValidator.validate_file(point_cloud_save_path)


def validate_single_frame_mapping_input(
    pixel_iteration_step: int,
    image_frame_path: Path,
    point_cloud_save_path: Path,
) -> None:
    CliInputValidator.validate_pixel_iteration_step(pixel_iteration_step)
    CliInputValidator.validate_file(image_frame_path)
    CliInputValidator.validate_file(point_cloud_save_path)


@app.command()
def map_midair(
    sensor_recordings_trajectory: str = typer.Option(
        None, help="Name of the trajectory to use"
    ),
    depth_map_frames_path: Path = typer.Option(
        None,
        help="Path to depth maps, if not provided, the depth maps will be "
        "generated from the image image_frames",
    ),
    pixel_iteration_step: int = typer.Option(
        1,
        help="Step to iterate over the pixels of the image frames and depth "
        "maps when generating the point cloud",
    ),
    max_number_of_frames: int = typer.Option(
        None,
        help="Maximum number of frames to use when generating the point cloud",
    ),
    sensor_recordings_path: Path = typer.Argument(
        ...,
        help="Path to sensor recordings, the data must in sync with the image "
        "image_frames (same number of image_frames, etc.)",
    ),
    image_frames_path: Path = typer.Argument(
        ...,
        help="Path to image image_frames, the data must in sync with the sensor "
        "recordings (same number of image_frames, etc.)",
    ),
    point_cloud_save_path: Path = typer.Argument(
        ...,
        help="Path to save the generated point cloud, the file extension "
        "determines the file format (xyz, xyzn, xyzrgb, pts, ply, pcd)",
    ),
) -> None:
    validate_midair_mapping_input(
        pixel_iteration_step,
        sensor_recordings_path,
        image_frames_path,
        depth_map_frames_path,
        point_cloud_save_path,
    )

    depth_estimator: MidasDepthEstimator = MidasDepthEstimator()
    midair_pixel_position_extractor: MidAirPixelPositionExtractor = (
        MidAirPixelPositionExtractor()
    )
    point_cloud_generator: PointCloudGenerator = PointCloudGenerator(
        pixel_iteration_step, depth_estimator, midair_pixel_position_extractor
    )
    mapping_service: MappingService = MappingService(point_cloud_generator)

    mapping_service.generate_and_save_point_cloud_from_midair_data(
        sensor_recordings_path,
        sensor_recordings_trajectory,
        image_frames_path,
        depth_map_frames_path,
        point_cloud_save_path,
        max_number_of_frames,
    )


@app.command()
def map_single_frame(
    pixel_iteration_step: int = typer.Option(
        1,
        help="Step to iterate over the pixels of the image frames and depth "
        "maps when generating the point cloud",
    ),
    image_frame_path: Path = typer.Argument(
        ...,
        help="Path to the single image frame",
    ),
    point_cloud_save_path: Path = typer.Argument(
        ...,
        help="Path to save the generated point cloud, the file extension "
        "determines the file format (xyz, xyzn, xyzrgb, pts, ply, pcd)",
    ),
) -> None:
    validate_single_frame_mapping_input(
        pixel_iteration_step, image_frame_path, point_cloud_save_path
    )

    depth_estimator: ZoeDepthEstimator = ZoeDepthEstimator()
    zoe_pixel_position_extractor: ZoePixelPositionExtractor = (
        ZoePixelPositionExtractor()
    )
    point_cloud_generator: PointCloudGenerator = PointCloudGenerator(
        2, depth_estimator, zoe_pixel_position_extractor
    )
    mapping_service: MappingService = MappingService(point_cloud_generator)

    mapping_service.generate_and_save_point_cloud_from_single_frame(
        image_frame_path, point_cloud_save_path
    )


if __name__ == "__main__":
    app()
