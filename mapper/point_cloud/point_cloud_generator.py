from typing import Generator

from open3d.cpu.pybind.camera import PinholeCameraIntrinsic
from open3d.cpu.pybind.geometry import PointCloud, RGBDImage, Image
from open3d.cpu.pybind.visualization import draw_geometries

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
    MINAS_MAX_DEPTH = 50

    def __init__(self, depth_estimator: DepthEstimator) -> None:
        self._depth_estimator = depth_estimator

    def generate_point_cloud(
        self,
        image_frame_generator: Generator[ImageFrame, None, None],
        sensor_recording_generator: Generator[SensorRecording, None, None],
    ) -> PointCloud:
        point_cloud = PointCloud()
        for image_frame, sensor_recording in zip(
            image_frame_generator, sensor_recording_generator
        ):
            if image_frame.depth_map is None:
                image_frame.depth_map = self._depth_estimator.estimate_depth(
                    image_frame.image
                )
            rgb_image = Image(image_frame.image)
            depth_image = Image(
                (
                    image_frame.depth_map * (255 / self.MINAS_MAX_DEPTH)
                ).astype("uint8")
            )
            rgbd_image = RGBDImage.create_from_color_and_depth(
                rgb_image, depth_image, depth_scale=1.0, depth_trunc=50
            )
            draw_geometries([rgbd_image], width=600, height=600)
            # point_cloud += PointCloud.create_from_rgbd_image(
            #     rgbd_image, self.DEFAULT_MIDAIR_INTRINSICS
            # )
            # draw_geometries([point_cloud], width=600, height=600)
        return point_cloud
