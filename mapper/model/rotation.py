import numpy as np


class Rotation:
    def __init__(self, x: float, y: float, z: float, w: float) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def get_rotation_matrix(self) -> np.ndarray:
        return np.array(
            [
                [
                    1 - 2 * (self.y**2 + self.z**2),
                    2 * (self.x * self.y - self.z * self.w),
                    2 * (self.x * self.z + self.y * self.w),
                ],
                [
                    2 * (self.x * self.y + self.z * self.w),
                    1 - 2 * (self.x**2 + self.z**2),
                    2 * (self.y * self.z - self.x * self.w),
                ],
                [
                    2 * (self.x * self.z - self.y * self.w),
                    2 * (self.y * self.z + self.x * self.w),
                    1 - 2 * (self.x**2 + self.y**2),
                ],
            ]
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
