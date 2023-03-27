from mapper.model.position import Position
from mapper.model.rotation import Rotation


class SensorRecording:
    def __init__(
        self,
        position: Position,
        rotation: Rotation,
    ) -> None:
        self.position = position
        self.rotation = rotation

    def __repr__(self) -> str:
        return (
            f"SensorRecording("
            f"position={self.position}, "
            f"rotation={self.rotation})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SensorRecording):
            return NotImplemented
        return (
            self.position == other.position and self.rotation == other.rotation
        )

    def __hash__(self) -> int:
        return hash((self.position, self.rotation))
