#!/usr/bin/env python3
"""Focused CPU tests for the paper-faithful Sub-JEPA regularizer."""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lewm.models import MultiSubspaceSIGReg


def _make_regularizer() -> MultiSubspaceSIGReg:
    torch.manual_seed(16001)
    return MultiSubspaceSIGReg(
        embed_dim=12, num_subspaces=3, num_projections=11)


def _official_reference(reg: MultiSubspaceSIGReg, embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.dim() == 2:
        embeddings = embeddings.unsqueeze(1)
    projected = torch.einsum(
        'btd,ked->btke', embeddings.float(), reg.projection_matrices.float())
    projected = projected.permute(2, 1, 0, 3).contiguous()
    directions = torch.randn(
        reg.num_subspaces, reg.subspace_dim, reg.num_projections,
        device=projected.device, dtype=torch.float32)
    directions.div_(directions.norm(p=2, dim=1, keepdim=True))
    samples = torch.einsum(
        'ktbd,kdn->ktbn', projected, directions).unsqueeze(-1) * reg.t
    error = (
        (samples.cos().mean(dim=2) - reg.phi).square()
        + samples.sin().mean(dim=2).square()
    )
    return ((error @ reg.weights) * projected.size(2)).mean()


def test_frozen_row_orthonormal_geometry() -> None:
    reg = _make_regularizer()
    assert reg.subspace_dim == 4
    assert tuple(reg.projection_matrices.shape) == (3, 4, 12)
    assert len(list(reg.parameters())) == 0
    assert 'projection_matrices' in reg.state_dict()
    gram = reg.projection_matrices @ reg.projection_matrices.transpose(-1, -2)
    expected = torch.eye(4).expand(3, -1, -1)
    torch.testing.assert_close(gram, expected, atol=2e-6, rtol=2e-6)
    assert reg.t.numel() == 17
    torch.testing.assert_close(reg.t[[0, -1]], torch.tensor([0.0, 3.0]))


def test_matches_official_multisubspace_statistic() -> None:
    reg = _make_regularizer()
    embeddings = torch.randn(5, 4, 12)
    torch.manual_seed(16002)
    expected = _official_reference(reg, embeddings)
    torch.manual_seed(16002)
    actual = reg(embeddings)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_two_and_three_dimensional_inputs_agree() -> None:
    reg = _make_regularizer()
    embeddings = torch.randn(7, 12)
    torch.manual_seed(16003)
    flat_loss = reg(embeddings)
    torch.manual_seed(16003)
    sequence_loss = reg(embeddings.unsqueeze(1))
    torch.testing.assert_close(flat_loss, sequence_loss, atol=0.0, rtol=0.0)
    assert tuple(reg.project(embeddings).shape) == (3, 1, 7, 4)


def test_directions_are_resampled_per_forward() -> None:
    reg = _make_regularizer()
    embeddings = torch.randn(9, 2, 12)
    torch.manual_seed(16004)
    first = reg(embeddings)
    second = reg(embeddings)
    assert not torch.equal(first, second)
    torch.manual_seed(16004)
    replay = reg(embeddings)
    torch.testing.assert_close(first, replay, atol=0.0, rtol=0.0)


def test_fp32_autocast_and_gradients() -> None:
    reg = _make_regularizer()
    embeddings = torch.randn(8, 3, 12, requires_grad=True)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        loss = reg(embeddings)
    assert loss.dtype == torch.float32
    assert loss.isfinite()
    loss.backward()
    assert embeddings.grad is not None
    assert embeddings.grad.dtype == torch.float32
    assert bool(torch.isfinite(embeddings.grad).all())
    assert float(embeddings.grad.abs().sum()) > 0.0


def test_configuration_and_shape_errors() -> None:
    for factory in (
        lambda: MultiSubspaceSIGReg(embed_dim=0),
        lambda: MultiSubspaceSIGReg(embed_dim=12, num_subspaces=0),
        lambda: MultiSubspaceSIGReg(embed_dim=10, num_subspaces=3),
        lambda: MultiSubspaceSIGReg(embed_dim=12, num_projections=0),
    ):
        try:
            factory()
        except ValueError:
            pass
        else:
            raise AssertionError('invalid regularizer configuration was accepted')

    reg = _make_regularizer()
    for bad in (torch.randn(12), torch.randn(2, 3, 4, 12), torch.randn(2, 11)):
        try:
            reg(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid input shape {tuple(bad.shape)} was accepted')
    try:
        reg(torch.ones(2, 12, dtype=torch.int64))
    except TypeError:
        pass
    else:
        raise AssertionError('integer embeddings were accepted')


def main() -> None:
    tests = (
        test_frozen_row_orthonormal_geometry,
        test_matches_official_multisubspace_statistic,
        test_two_and_three_dimensional_inputs_agree,
        test_directions_are_resampled_per_forward,
        test_fp32_autocast_and_gradients,
        test_configuration_and_shape_errors,
    )
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
    print(f'PASS all {len(tests)} MultiSubspaceSIGReg tests')


if __name__ == '__main__':
    main()
