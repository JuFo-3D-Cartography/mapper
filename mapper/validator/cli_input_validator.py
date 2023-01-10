from pathlib import Path


class CliInputValidator:
    @staticmethod
    def validate_pixel_iteration_step(pixel_iteration_step: int) -> None:
        if pixel_iteration_step < 1:
            raise ValueError(
                f"Pixel iteration step must be greater than 0, "
                f"got {pixel_iteration_step}"
            )

    @staticmethod
    def validate_directory(path: Path) -> None:
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory")

    @staticmethod
    def validate_file(path: Path) -> None:
        if not path.is_file():
            raise ValueError(f"Path {path} is not a file")
