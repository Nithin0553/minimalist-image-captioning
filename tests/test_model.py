from __future__ import annotations

import torch

from minimal_captioning.model import MinimalCaptioningModel


def build_test_model() -> MinimalCaptioningModel:
    return MinimalCaptioningModel(
        vocab_size=12,
        embedding_dim=16,
        pad_index=0,
        pretrained_resnet=False,
    )


def test_exact_professor_architecture() -> None:
    model = build_test_model()
    model.assert_required_architecture()
    info = model.architecture_info()
    assert info.encoder_output_dim == 512
    assert info.bottleneck_parameters == 3078
    assert info.latent_dim == info.gru_hidden_dim == 6
    assert info.gru_layers == 1
    assert info.encoder_trainable_parameters == 0


def test_forward_and_generation_shapes() -> None:
    model = build_test_model().eval()
    images = torch.rand(2, 3, 64, 64)
    inputs = torch.tensor([[2, 4, 5], [2, 6, 0]])
    logits, latent = model(images, inputs)
    generated, generated_latent = model.generate(images, bos_index=2, eos_index=3, max_length=5)
    assert logits.shape == (2, 3, 12)
    assert latent.shape == generated_latent.shape == (2, 6)
    assert generated.shape[0] == 2
    assert generated.shape[1] <= 5


def test_encoder_remains_in_eval_mode_during_training() -> None:
    model = build_test_model()
    model.train()
    assert model.training is True
    assert model.encoder.training is False
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
