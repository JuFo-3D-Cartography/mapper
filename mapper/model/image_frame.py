from pathlib import Path
from typing import Optional

import numpy as np
import cv2


class ImageFrame:
    def __init__(
        self,
        image: np.ndarray,
        depth_map: Optional[np.ndarray] = None,
    ) -> None:
        self.image = image
        self.depth_map = depth_map

    def save(self, path: Path) -> None:
        cv2.imwrite(str(path), self.image)

    def __repr__(self) -> str:
        return f"ImageFrame(image={self.image}, depth_map={self.depth_map})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImageFrame):
            return NotImplemented
        return np.array_equal(self.image, other.image) and np.array_equal(
            self.depth_map, other.depth_map
        )

    def __hash__(self) -> int:
        return hash((self.image, self.depth_map))
