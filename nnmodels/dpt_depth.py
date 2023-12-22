import cv2
import os
import numpy as np
import torch
from torchvision.transforms import Compose
from DPT.dpt.models import DPTDepthModel
from DPT.dpt.midas_net import MidasNet_large
from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet

DEPTH_IMAGE_WIDTH = 160
DEPTH_IMAGE_HEIGHT = 128
DEFAULT_MODEL_TYPE = "dpt_large"
DEFAULT_MODEL_PATH = os.path.abspath("./") + "/DPT/weights/dpt_large-midas-2f21e586.pt"

class DPTDepth:
    def __init__(
        self,
        device,
        model_type=DEFAULT_MODEL_TYPE,
        model_path=DEFAULT_MODEL_PATH,
        optimize=True,
    ):
        self.optimize = optimize
        self.THRESHOLD = torch.tensor(np.finfo("float").eps).to(device)
        self.device = device
        self.depth_image_width = DEPTH_IMAGE_WIDTH
        self.depth_image_height = DEPTH_IMAGE_HEIGHT

        # Initialize parameters based on model type
        self._init_model_params(model_type, model_path)

        # Compose transformations
        self.transform = Compose(
            [
                Resize(
                    self.net_w,
                    self.net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method=self.resize_mode,
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                self.normalization,
                PrepareForNet(),
            ]
        )

        self.model.eval()
        self._set_optimization(optimize)

    def _init_model_params(self, model_type, model_path):
        if model_type in ["dpt_large", "dpt_hybrid"]:
            self._set_common_params(model_path, model_type)
        elif model_type == "dpt_hybrid_kitti":
            self._set_kitti_params(model_path)
        elif model_type == "dpt_hybrid_nyu":
            self._set_nyu_params(model_path)
        elif model_type in ["midas_v21", "midas_v21_small"]:
            self._set_midas_params(model_type, model_path)
        else:
            assert (False), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

    def _set_common_params(self, model_path, model_type):
        self.net_w = self.net_h = 384
        self.resize_mode = "minimal"
        backbone = "vitl16_384" if model_type == "dpt_large" else "vitb_rn50_384"
        self.model = DPTDepthModel(
            path=model_path,
            backbone=backbone,
            non_negative=True,
            enable_attention_hooks=False,
        )
        self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def _set_kitti_params(self, model_path):
        self.net_w, self.net_h = 1216, 352
        self.resize_mode = "minimal"
        self.model = DPTDepthModel(
            path=model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def _set_nyu_params(self, model_path):
        self.net_w, self.net_h = 640, 480
        self.resize_mode = "minimal"
        self.model = DPTDepthModel(
            path=model_path,
            scale=0.000305,
            shift=0.1378,
            invert=True,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        self.normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def _set_midas_params(self, model_type, model_path):
        self.net_w = self.net_h = 384 if model_type == "midas_v21" else 256
        self.resize_mode = "upper_bound"
        self.model = MidasNet_large(model_path, non_negative=True)
        self.normalization = NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def _set_optimization(self, optimize):
        if optimize and self.device == torch.device("cuda"):
            self.model = self.model.to(memory_format=torch.channels_last)
            self.model = self.model.half()
        self.model.to(self.device)
    
    def run(self, rgb_img):
        sample = self._prepare_sample(rgb_img)
        prediction = self._predict_depth(sample)
        normalized_prediction = self._normalize_prediction(prediction)
        return normalized_prediction

    def _prepare_sample(self, rgb_img):
        img_input = self.transform({"image": rgb_img})["image"]
        sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
        if self.optimize and self.device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last).half()
        return sample

    def _predict_depth(self, sample):
        with torch.no_grad():
            prediction = self.model.forward(sample)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(0),
                size=(DEPTH_IMAGE_HEIGHT, DEPTH_IMAGE_WIDTH),
                mode="bicubic",
                align_corners=False,
            ).squeeze().cpu().numpy()
        return prediction

    def _normalize_prediction(self, prediction):
        depth_min, depth_max = prediction.min(), prediction.max()
        if depth_max - depth_min > self.THRESHOLD:
            return (prediction - depth_min) / (depth_max - depth_min)
        else:
            return np.zeros(prediction.shape, dtype=prediction.dtype)