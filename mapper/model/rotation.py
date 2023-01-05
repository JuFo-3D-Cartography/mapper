import numpy as np
from open3d.cpu.pybind.geometry import get_rotation_matrix_from_quaternion


class Rotation:
    def __init__(self, x: float, y: float, z: float, w: float):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def get_rotation_matrix(self) -> np.ndarray:
        return get_rotation_matrix_from_quaternion(
            [self.x, self.y, self.z, self.w]
        )

    def __repr__(self) -> str:
        return f"Rotation(x={self.x}, y={self.y}, z={self.z}, w={self.w})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rotation):
            return NotImplemented
        return (
            self.x == other.x
            and self.y == other.y
            and self.z == other.z
            and self.w == other.w
        )

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z, self.w))
