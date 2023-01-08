from pathlib import Path
from typing import Generator

import h5py

from mapper.model.position import Position
from mapper.model.rotation import Rotation
from mapper.model.sensor_recording import SensorRecording


class SensorRecordingsReader:
    @staticmethod
    def read_sensor_recordings(
            sensor_recordings_path: Path,
            sensor_recordings_trajectory: str,
    ) -> Generator[SensorRecording, None, None]:
        sensor_recordings_file = h5py.File(sensor_recordings_path, "r")
        trajectory: h5py.Group = sensor_recordings_file[
            sensor_recordings_trajectory
        ]
        ground_truth: h5py.Group = trajectory["groundtruth"]
        position: h5py.Dataset = ground_truth["position"]
        attitude: h5py.Dataset = ground_truth["attitude"]
        return (
            SensorRecording(
                Position(
                    x=position[timestamp][0],
                    y=position[timestamp][1],
                    z=position[timestamp][2],
                ),
                Rotation(
                    w=attitude[timestamp][0],
                    x=attitude[timestamp][1],
                    y=attitude[timestamp][2],
                    z=attitude[timestamp][3],
                ),
            )
            for timestamp in range(len(position))
        )
