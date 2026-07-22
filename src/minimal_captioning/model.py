from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


@dataclass(frozen=True, kw_only=True)
class ArchitectureInfo:
    encoder_output_dim: int
    projection_input_dim: int
    latent_dim: int
    gru_hidden_dim: int
    gru_layers: int
    encoder_trainable_parameters: int
    bottleneck_parameters: int
    trainable_parameters: int
    total_parameters: int


class FrozenResNet18(nn.Module):
    """A fixed ResNet-18 whose classification head is removed to expose 512-D features."""

    output_dim = 512

    def __init__(self, *, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        # TorchVision types ``fc`` as Linear even though replacing the
        # classifier with any Module is its supported feature-extraction API.
        backbone.fc = cast(nn.Linear, nn.Identity())
        backbone.requires_grad_(False)
        backbone.eval()
        self.backbone = backbone

    def train(self, mode: bool = True) -> FrozenResNet18:
        super().train(False)
        self.backbone.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        with torch.no_grad():
            features = self.backbone(images)
        if features.ndim != 2 or features.shape[1] != self.output_dim:
            raise RuntimeError(f"ResNet-18 must return [batch, 512], got {tuple(features.shape)}")
        return features


class SixDimensionalBottleneck(nn.Module):
    """The professor-required single direct affine map from 512 dimensions to six."""

    input_dim = 512
    output_dim = 6

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, features: Tensor) -> Tensor:
        return self.projection(features)


class CaptionGRU(nn.Module):
    """A one-layer GRU whose hidden state is exactly the six-dimensional image latent."""

    hidden_dim = 6

    def __init__(self, *, vocab_size: int, embedding_dim: int, pad_index: int) -> None:
        super().__init__()
        if vocab_size < 4:
            raise ValueError("vocabulary must include at least the four special tokens")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input_tokens: Tensor, latent: Tensor) -> Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.hidden_dim:
            raise ValueError(f"latent must have shape [batch, 6], got {tuple(latent.shape)}")
        embedded = self.embedding(input_tokens)
        initial_hidden = latent.unsqueeze(0)
        decoded, _ = self.gru(embedded, initial_hidden)
        return self.output(decoded)

    def generate(
        self,
        latent: Tensor,
        *,
        bos_index: int,
        eos_index: int,
        max_length: int,
    ) -> Tensor:
        if max_length < 1:
            raise ValueError("max_length must be positive")
        if latent.ndim != 2 or latent.shape[1] != self.hidden_dim:
            raise ValueError(f"latent must have shape [batch, 6], got {tuple(latent.shape)}")
        batch_size = latent.shape[0]
        hidden = latent.unsqueeze(0)
        current = torch.full((batch_size,), bos_index, dtype=torch.long, device=latent.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=latent.device)
        generated: list[Tensor] = []
        for _ in range(max_length):
            embedded = self.embedding(current).unsqueeze(1)
            decoded, hidden = self.gru(embedded, hidden)
            current = self.output(decoded[:, 0, :]).argmax(dim=-1)
            generated.append(current)
            finished |= current.eq(eos_index)
            if bool(finished.all()):
                break
        return torch.stack(generated, dim=1)


class MinimalCaptioningModel(nn.Module):
    """Frozen ResNet-18 -> one 512x6 projection -> six-hidden-unit GRU."""

    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        pad_index: int,
        pretrained_resnet: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = FrozenResNet18(pretrained=pretrained_resnet)
        self.bottleneck = SixDimensionalBottleneck()
        self.decoder = CaptionGRU(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            pad_index=pad_index,
        )

    def encode(self, images: Tensor) -> Tensor:
        return self.bottleneck(self.encoder(images))

    def forward(self, images: Tensor, input_tokens: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(images)
        return self.decoder(input_tokens, latent), latent

    def generate(
        self,
        images: Tensor,
        *,
        bos_index: int,
        eos_index: int,
        max_length: int,
    ) -> tuple[Tensor, Tensor]:
        latent = self.encode(images)
        tokens = self.decoder.generate(
            latent,
            bos_index=bos_index,
            eos_index=eos_index,
            max_length=max_length,
        )
        return tokens, latent

    def architecture_info(self) -> ArchitectureInfo:
        all_parameters = tuple(self.parameters())
        encoder_parameters = tuple(self.encoder.parameters())
        bottleneck_parameters = tuple(self.bottleneck.parameters())
        return ArchitectureInfo(
            encoder_output_dim=self.encoder.output_dim,
            projection_input_dim=self.bottleneck.input_dim,
            latent_dim=self.bottleneck.output_dim,
            gru_hidden_dim=self.decoder.gru.hidden_size,
            gru_layers=self.decoder.gru.num_layers,
            encoder_trainable_parameters=sum(
                parameter.numel() for parameter in encoder_parameters if parameter.requires_grad
            ),
            bottleneck_parameters=sum(parameter.numel() for parameter in bottleneck_parameters),
            trainable_parameters=sum(
                parameter.numel() for parameter in all_parameters if parameter.requires_grad
            ),
            total_parameters=sum(parameter.numel() for parameter in all_parameters),
        )

    def assert_required_architecture(self) -> None:
        info = self.architecture_info()
        expected = (512, 512, 6, 6, 1, 0, 3078)
        actual = (
            info.encoder_output_dim,
            info.projection_input_dim,
            info.latent_dim,
            info.gru_hidden_dim,
            info.gru_layers,
            info.encoder_trainable_parameters,
            info.bottleneck_parameters,
        )
        if actual != expected:
            raise RuntimeError(f"required architecture changed: expected {expected}, got {actual}")
