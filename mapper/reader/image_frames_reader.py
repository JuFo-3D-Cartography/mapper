from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from PIL import Image

from mapper.model.image_frame import ImageFrame


class ImageFramesReader:
    @staticmethod
    def read_image_frames(
        image_frames_path: Path,
        depth_map_frames_path: Path,
    ) -> Generator[ImageFrame, None, None]:
        def read_depth_map(depth_map_path: Path) -> np.ndarray:
            depth_map_image: Image = Image.open(depth_map_path)
            depth_map: np.ndarray = np.asarray(depth_map_image, np.uint16)
            depth_map.dtype = np.float16
            return depth_map

        return (
            ImageFrame(
                image=cv2.cvtColor(
                    cv2.imread(str(image_frame_path)), cv2.COLOR_BGR2RGB
                ),
                depth_map=read_depth_map(depth_map_frame_path),
            )
            for image_frame_path, depth_map_frame_path in zip(
                sorted(image_frames_path.iterdir()),
                sorted(depth_map_frames_path.iterdir()),
            )
        )
