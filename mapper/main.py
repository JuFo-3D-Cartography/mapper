from pathlib import Path

import typer

from mapper.depth.depth_estimator import DepthEstimator
from mapper.point_cloud.point_cloud_generator import PointCloudGenerator
from mapper.service.mapping_service import MappingService
from mapper.validator.cli_input_validator import CliInputValidator

app = typer.Typer()


def validate_input(
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


@app.command()
def main(
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
    validate_input(
        pixel_iteration_step,
        sensor_recordings_path,
        image_frames_path,
        depth_map_frames_path,
        point_cloud_save_path,
    )

    depth_estimator: DepthEstimator = DepthEstimator()
    point_cloud_generator: PointCloudGenerator = PointCloudGenerator(
        pixel_iteration_step, depth_estimator
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


if __name__ == "__main__":
    app()
