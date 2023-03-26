from abc import abstractmethod

import numpy as np


class DepthEstimator:
    @abstractmethod
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError
