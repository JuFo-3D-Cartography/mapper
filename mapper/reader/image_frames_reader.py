from pathlib import Path
from typing import Generator

import cv2
from mapper.model.image_frame import ImageFrame


class ImageFramesReader:
    @staticmethod
    def read_image_frames(
        image_frames_path: Path,
    ) -> Generator[ImageFrame, None, None]:
        return (
            ImageFrame(
                image=cv2.cvtColor(
                    cv2.imread(str(image_frame_path)), cv2.COLOR_BGR2RGB
                ),
                depth_map=None,
            )
            for image_frame_path in image_frames_path.iterdir()
        )
