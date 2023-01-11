import numpy as np
import torch


class DepthEstimator:
    MODEL_TYPE = "DPT_Large"

    def __init__(self) -> None:
        self._midas: torch.nn.Module = torch.hub.load(
            "intel-isl/MiDaS", self.MODEL_TYPE
        )
        self._device: torch.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self._midas.to(self._device)
        self._midas.eval()
        midas_transforms: torch.nn.Module = torch.hub.load(
            "intel-isl/MiDaS", "transforms"
        )
        self._transform: torch.nn.Module = (
            midas_transforms.dpt_transform
            if self.MODEL_TYPE in ["DPT_Hybrid", "DPT_Large"]
            else midas_transforms.small_transform
        )

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        input_batch: torch.Tensor = self._transform(image).to(self._device)
        with torch.no_grad():
            prediction: torch.Tensor = self._midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        return prediction.cpu().numpy()
