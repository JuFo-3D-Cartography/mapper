from pathlib import Path


class CliPathValidator:
    @staticmethod
    def validate_directory(path: Path) -> None:
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory")

    @staticmethod
    def validate_file(path: Path) -> None:
        if not path.is_file():
            raise ValueError(f"Path {path} is not a file")
