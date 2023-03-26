import torch
import numpy as np
from PIL import Image

from mapper.depth.depth_estimator import DepthEstimator


class ZoeDepthEstimator(DepthEstimator):
    MODEL_TYPE: str = "ZoeD_N"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self) -> None:
        torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)
        self._zoedepth: torch.nn.Module = (
            torch.hub.load("isl-org/ZoeDepth", self.MODEL_TYPE, pretrained=True)
            .to(self.DEVICE)
            .eval()
        )

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        image: Image = Image.fromarray(image)
        depth_map: np.ndarray = self._zoedepth.infer_pil(image)
        return depth_map
