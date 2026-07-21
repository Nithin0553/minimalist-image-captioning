from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as nn_functional
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from .config import ModelConfig

NoiseKind: TypeAlias = Literal["none", "gaussian", "salt_pepper"]
ImageOutput: TypeAlias = Literal["model", "display"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True, kw_only=True)
class NoiseSpec:
    kind: NoiseKind = "none"
    level: float = 0.0

    def validate(self) -> None:
        if self.kind == "none" and self.level != 0.0:
            raise ValueError("noise level must be zero when kind='none'")
        if self.kind == "gaussian" and self.level <= 0.0:
            raise ValueError("Gaussian noise standard deviation must be positive")
        if self.kind == "salt_pepper" and not 0.0 < self.level < 1.0:
            raise ValueError("salt-and-pepper amount must lie between 0 and 1")
        if self.kind not in {"none", "gaussian", "salt_pepper"}:
            raise ValueError(f"unsupported noise kind: {self.kind}")

    @property
    def label(self) -> str:
        return "clean" if self.kind == "none" else f"{self.kind}_{self.level:g}"


class DIPPreprocessor:
    """Resize, inject controlled noise, apply DIP filtering, and normalize an RGB image."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)

    def __call__(
        self,
        image: Image.Image,
        *,
        noise: NoiseSpec | None = None,
        seed: int = 0,
    ) -> Tensor:
        return self.apply(image, noise=noise, seed=seed, output="model")

    def apply(
        self,
        image: Image.Image,
        *,
        noise: NoiseSpec | None = None,
        seed: int = 0,
        output: ImageOutput = "model",
    ) -> Tensor:
        if output not in {"model", "display"}:
            raise ValueError("output must be 'model' or 'display'")
        spec = noise or NoiseSpec()
        spec.validate()
        rgb = image.convert("RGB")
        tensor = vision_functional.pil_to_tensor(rgb).to(dtype=torch.float32).div_(255.0)
        tensor = vision_functional.resize(
            tensor,
            self.config.resize_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = vision_functional.center_crop(
            tensor, [self.config.image_size, self.config.image_size]
        )
        tensor = self._inject_noise(tensor, spec=spec, seed=seed)
        tensor = self._apply_dip(tensor).clamp(0.0, 1.0)
        if output == "display":
            return tensor
        return (tensor - self._mean) / self._std

    def _inject_noise(self, image: Tensor, *, spec: NoiseSpec, seed: int) -> Tensor:
        if spec.kind == "none":
            return image
        generator = torch.Generator(device="cpu").manual_seed(seed)
        if spec.kind == "gaussian":
            noise = torch.randn(image.shape, generator=generator, dtype=image.dtype)
            return (image + noise * spec.level).clamp(0.0, 1.0)
        random_values = torch.rand(
            (1, image.shape[1], image.shape[2]), generator=generator, dtype=image.dtype
        )
        half = spec.level / 2.0
        result = image.clone()
        result[:, random_values[0] < half] = 0.0
        result[:, random_values[0] > 1.0 - half] = 1.0
        return result

    def _apply_dip(self, image: Tensor) -> Tensor:
        if self.config.dip_mode == "none":
            return image
        if self.config.dip_mode == "gaussian":
            return vision_functional.gaussian_blur(
                image,
                kernel_size=[self.config.gaussian_kernel_size] * 2,
                sigma=[self.config.gaussian_sigma] * 2,
            )
        return self._sobel_blend(image)

    def _sobel_blend(self, image: Tensor) -> Tensor:
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=image.dtype,
        )
        kernel_y = kernel_x.transpose(0, 1)
        weight_x = kernel_x.reshape(1, 1, 3, 3).repeat(3, 1, 1, 1)
        weight_y = kernel_y.reshape(1, 1, 3, 3).repeat(3, 1, 1, 1)
        batched = nn_functional.pad(image.unsqueeze(0), (1, 1, 1, 1), mode="reflect")
        gradient_x = nn_functional.conv2d(batched, weight_x, groups=3)
        gradient_y = nn_functional.conv2d(batched, weight_y, groups=3)
        magnitude = torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-12).squeeze(0)
        scale = magnitude.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        edges = magnitude / scale
        alpha = self.config.sobel_blend
        return image * (1.0 - alpha) + edges * alpha
