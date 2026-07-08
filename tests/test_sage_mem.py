"""Focused CPU contracts for the isolated SAGE-Mem carrier."""

from __future__ import annotations

import math

import pytest
import torch

from lewm.models.sage_mem import (
    DEFAULT_HALF_LIVES,
    SAGE_MEM_API_VERSION,
    SAGE_MEM_VARIANTS,
    SAGEMem,
    SAGEMemState,
    build_sage_mem_v1,
    sage_mem_parameter_count,
)


D, A, B, L = 16, 3, 2, 6


def _inputs(*, patches: int | None = None, seed: int = 0,
            dtype: torch.dtype = torch.float32
            ) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    if patches is None:
        shape = (B, L, D)
    else:
        shape = (B, L, patches, D)
    z = torch.randn(shape, generator=generator).to(dtype)
    actions = torch.randn(
        B, L - 1, A, generator=generator).to(dtype)
    return z, actions


def _activate_read(carrier: SAGEMem, seed: int = 1) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        carrier.read_scale.copy_(0.05 * torch.randn(
            carrier.read_scale.shape, generator=generator))
        carrier.action_projection.weight.copy_(0.02 * torch.randn(
            carrier.action_projection.weight.shape, generator=generator))


def test_exact_official_parameter_targets_and_trainable_tensor_ledger() -> None:
    expected = {(192, 10): 76_032, (384, 10): 299_520}
    for (embed_dim, action_dim), target in expected.items():
        carrier = SAGEMem(embed_dim, action_dim)
        assert sage_mem_parameter_count(embed_dim, action_dim) == target
        assert carrier.parameter_count() == target
        assert carrier.describe()["parameter_target"] == target

    carrier = SAGEMem(D, A)
    assert set(dict(carrier.named_parameters())) == {
        "gate_threshold",
        "read_scale",
        "w_x.weight",
        "gate_projection.weight",
        "action_projection.weight",
    }
    assert set(dict(carrier.named_buffers())) == {
        "half_lives", "decay", "gate_slope", "maturity_decay",
    }


@pytest.mark.parametrize("embed_dim,action_dim", [(192, 10), (384, 10)])
@pytest.mark.parametrize("tokens", [1, 196])
def test_compute_matched_revision_stays_within_registered_flop_margin(
        embed_dim: int, action_dim: int, tokens: int) -> None:
    carrier = SAGEMem(embed_dim, action_dim)
    candidate = carrier.estimate_flops(
        batch_size=1, timesteps=20, tokens=tokens)
    matched_baseline = (carrier.parameter_count() * 2 * 20 * tokens)
    assert abs(candidate - matched_baseline) / matched_baseline <= 0.10
    assert carrier.describe()["compute_revision"] == \
        "two-dense-plus-diagonal-read-v1.1"


@pytest.mark.parametrize("variant", SAGE_MEM_VARIANTS)
def test_versioned_builder_registers_each_control_variant(variant: str) -> None:
    carrier = build_sage_mem_v1(
        embed_dim=D, action_dim=A, variant=f"sage_mem_{variant}",
        config={"sage_mem": {"maturity_scale": 4.0}})
    description = carrier.describe()
    assert description["api_version"] == SAGE_MEM_API_VERSION
    assert description["variant"] == variant
    assert carrier.parameter_count() == sage_mem_parameter_count(D, A)
    assert carrier.persistent_state_floats() == D + 1
    assert carrier.estimate_flops(
        batch_size=2, timesteps=5, tokens=196) > 0


@pytest.mark.parametrize("patches", [None, 196])
def test_zero_initialized_read_is_exact_no_state_host(
        patches: int | None) -> None:
    z, actions = _inputs(patches=patches, seed=2, dtype=torch.float16)
    output = SAGEMem(D, A)(z, actions)
    assert torch.equal(output.z_tilde, z)
    assert torch.equal(output.prior_read, torch.zeros_like(z))
    assert output.z_tilde.dtype == z.dtype


@pytest.mark.parametrize("patches", [None, 196])
def test_versioned_forward_sequence_preserves_carrier_contract(
        patches: int | None) -> None:
    carrier = SAGEMem(D, A)
    _activate_read(carrier, seed=21)
    z, actions = _inputs(patches=patches, seed=22)
    legacy = carrier(z, actions)
    protocol = carrier.forward_sequence(z, actions)
    assert set(protocol) == {
        "fused", "prior", "posterior", "exposure", "diagnostics",
    }
    torch.testing.assert_close(protocol["fused"], legacy.z_tilde)
    torch.testing.assert_close(protocol["prior"], legacy.prior_read)
    assert protocol["posterior"].shape == z.shape
    assert protocol["posterior"].dtype == torch.float32
    torch.testing.assert_close(
        protocol["fused"], z + protocol["exposure"])


def test_registered_control_variants_have_explicit_behavior() -> None:
    z, actions = _inputs(seed=23)
    no_exposure = SAGEMem(D, A, variant="no_exposure")
    _activate_read(no_exposure, seed=24)
    blocked = no_exposure.forward_sequence(z, actions)
    assert torch.equal(blocked["fused"], z)
    assert torch.equal(blocked["exposure"], torch.zeros_like(z))
    assert torch.count_nonzero(blocked["posterior"]) > 0

    exposure_only = SAGEMem(D, A, variant="exposure_only")
    _activate_read(exposure_only, seed=25)
    changed = z.clone()
    changed[:, 0] += 100.0
    reference = exposure_only.forward_sequence(z, actions)
    intervention = exposure_only.forward_sequence(changed, actions)
    torch.testing.assert_close(
        reference["fused"][:, 1:], intervention["fused"][:, 1:])


def test_reset_mask_clears_only_declared_batch_histories() -> None:
    carrier = SAGEMem(D, A)
    _activate_read(carrier, seed=26)
    z, _ = _inputs(seed=27)
    actions = torch.zeros(B, L - 1, A)
    reset_mask = torch.zeros(B, L, dtype=torch.bool)
    reset_mask[0, 3] = True
    reset = carrier.forward_sequence(z, actions, reset_mask=reset_mask)
    uninterrupted = carrier.forward_sequence(z, actions)
    suffix = carrier.forward_sequence(z[0:1, 3:], actions[0:1, 3:])
    torch.testing.assert_close(
        reset["fused"][0:1, 3:], suffix["fused"])
    torch.testing.assert_close(
        reset["prior"][0:1, 3:], suffix["prior"])
    torch.testing.assert_close(
        reset["fused"][1], uninterrupted["fused"][1])


def test_fixed_multiscale_half_lives_and_decays() -> None:
    carrier = SAGEMem(D, A)
    expected_half_lives = torch.tensor(
        [DEFAULT_HALF_LIVES[index % len(DEFAULT_HALF_LIVES)]
         for index in range(D)])
    torch.testing.assert_close(carrier.half_lives, expected_half_lives)
    torch.testing.assert_close(
        carrier.decay, torch.exp2(-1.0 / expected_half_lives))
    assert carrier.half_lives.requires_grad is False
    assert carrier.decay.requires_grad is False


def test_surprise_gate_writes_large_innovations_more_strongly() -> None:
    carrier = SAGEMem(D, A)
    actions = torch.empty(1, 0, A)
    quiet = carrier(torch.zeros(1, 1, D), actions)
    surprising = carrier(torch.full((1, 1, D), 10.0), actions)
    quiet_gate = quiet.telemetry["write_gate_mean"].item()
    surprising_gate = surprising.telemetry["write_gate_mean"].item()
    assert quiet_gate < 0.1
    assert surprising_gate > 0.99
    assert surprising_gate > quiet_gate


@pytest.mark.parametrize("patches", [None, 196])
def test_batched_forward_matches_streaming_observation_path(
        patches: int | None) -> None:
    torch.manual_seed(3)
    carrier = SAGEMem(D, A).eval()
    _activate_read(carrier)
    z, actions = _inputs(patches=patches, seed=4)
    with torch.no_grad():
        sequence = carrier(z, actions)
        step = carrier.initialize(z[:, 0])
        state = step.state
        fused = [step.fused_z]
        priors = [step.prior_read]
        for time in range(1, L):
            step = carrier.observe(
                state, z[:, time], actions[:, time - 1])
            state = step.state
            fused.append(step.fused_z)
            priors.append(step.prior_read)
    torch.testing.assert_close(
        sequence.z_tilde, torch.stack(fused, dim=1))
    torch.testing.assert_close(
        sequence.prior_read, torch.stack(priors, dim=1))


def test_future_observations_are_causal_and_prior_is_pre_observation() -> None:
    carrier = SAGEMem(D, A).eval()
    _activate_read(carrier, seed=5)
    z, actions = _inputs(seed=6)
    cut = 3
    with torch.no_grad():
        reference = carrier(z, actions)

        changed_current = z.clone()
        changed_current[:, cut] += 50.0
        current = carrier(changed_current, actions)
        torch.testing.assert_close(
            reference.prior_read[:, cut], current.prior_read[:, cut])

        changed_future = z.clone()
        changed_future[:, cut + 1:] -= 70.0
        future = carrier(changed_future, actions)
        torch.testing.assert_close(
            reference.z_tilde[:, :cut + 1],
            future.z_tilde[:, :cut + 1])
        torch.testing.assert_close(
            reference.prior_read[:, :cut + 1],
            future.prior_read[:, :cut + 1])


def test_local_regional_global_aggregation_on_native_dino_grid() -> None:
    carrier = SAGEMem(D, A)
    patch_values = torch.arange(196, dtype=torch.float32).reshape(
        1, 196, 1).expand(1, 196, D)
    local, regional, global_memory = carrier.aggregation_components(
        patch_values)

    assert torch.equal(local, patch_values)
    # First 2x2 region in a 14x14 grid: mean(0, 1, 14, 15) = 7.5.
    for patch in (0, 1, 14, 15):
        torch.testing.assert_close(
            regional[0, patch], torch.full((D,), 7.5))
    torch.testing.assert_close(
        global_memory[0, 0], torch.full((D,), 97.5))

    aggregated = carrier.aggregate(patch_values)
    expected_first = 0.5 * 0.0 + 0.25 * 7.5 + 0.25 * 97.5
    torch.testing.assert_close(
        aggregated[0, 0], torch.full((D,), expected_first))
    assert torch.equal(aggregated, carrier.aggregate(patch_values))

    single = torch.randn(2, 1, D)
    assert torch.equal(carrier.aggregate(single), single)
    vector = single[:, 0]
    assert torch.equal(carrier.aggregate(vector), vector)


def test_reset_maturity_starts_suppressed_and_accumulates_with_writes() -> None:
    carrier = SAGEMem(D, A)
    _activate_read(carrier, seed=7)
    z = torch.stack([
        torch.full((1, D), 10.0 if time % 2 == 0 else -10.0)
        for time in range(8)
    ], dim=1)
    actions = torch.zeros(1, 7, A)
    output = carrier(z, actions)
    maturity = output.telemetry["maturity_mean"][0]

    assert 0.0 < maturity[0] < maturity[-1] < 1.0
    assert torch.all(maturity[1:] > maturity[:-1])
    # A true reset has no pre-observation evidence and therefore no prior read.
    reset = carrier.initialize(z[:, 0])
    assert torch.equal(reset.prior_read, torch.zeros_like(reset.prior_read))

    imagined = carrier.imagine(reset.state, torch.zeros(1, A))
    assert isinstance(imagined.state, SAGEMemState)
    assert torch.all(
        imagined.state.write_mass < reset.state.write_mass)


def test_clone_repeat_imagine_and_spatial_shapes() -> None:
    carrier = SAGEMem(D, A)
    _activate_read(carrier, seed=8)
    z0 = torch.randn(B, 196, D)
    state = carrier.initialize(z0).state
    assert isinstance(state, SAGEMemState)

    cloned = carrier.clone_state(state)
    for source, copied in (
            (state.memory, cloned.memory),
            (state.write_mass, cloned.write_mass)):
        assert torch.equal(source, copied)
        assert source.data_ptr() != copied.data_ptr()

    repeated = carrier.repeat_state(state, 3)
    assert repeated.memory.shape == (B * 3, 196, D)
    assert repeated.write_mass.shape == (B * 3, 196, 1)
    imagined = carrier.imagine(repeated, torch.randn(B * 3, A))
    assert imagined.fused_z is None
    assert imagined.prior_read.shape == (B * 3, 196, D)
    assert torch.isfinite(imagined.prior_read).all()


def test_internal_state_and_telemetry_remain_finite_fp32() -> None:
    carrier = SAGEMem(D, A)
    _activate_read(carrier, seed=9)
    z, actions = _inputs(seed=10, dtype=torch.float16)
    output = carrier(z, actions)
    assert output.z_tilde.dtype == torch.float16
    assert output.prior_read.dtype == torch.float16
    assert all(value.dtype == torch.float32
               for value in output.telemetry.values())
    assert all(torch.isfinite(value).all()
               for value in output.telemetry.values())

    state = carrier.initialize(z[:, 0]).state
    assert state.memory.dtype == torch.float32
    assert state.write_mass.dtype == torch.float32


def test_gradients_are_nonzero_and_finite_after_read_is_active() -> None:
    carrier = SAGEMem(D, A)
    _activate_read(carrier, seed=11)
    z, actions = _inputs(seed=12)
    output = carrier(z, actions)
    loss = output.z_tilde.square().mean() + output.prior_read.square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in carrier.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(float(gradient.abs().sum()) > 0.0 for gradient in gradients)


def test_actionable_shape_and_state_validation() -> None:
    carrier = SAGEMem(D, A)
    z, actions = _inputs(seed=13)
    with pytest.raises(ValueError, match="z0"):
        carrier.initialize(z[:, 0, :-1])
    state = carrier.initialize(z[:, 0]).state
    with pytest.raises(ValueError, match="a_prev"):
        carrier.observe(state, z[:, 1], actions[:, 0, :-1])
    with pytest.raises(ValueError, match="shapes must match"):
        spatial_state = carrier.initialize(
            torch.randn(B, 196, D)).state
        carrier.observe(spatial_state, z[:, 1], actions[:, 0])
    with pytest.raises(ValueError, match="positive integer"):
        carrier.repeat_state(state, 0)


@pytest.mark.parametrize("invalid", [(), (0.0,), (4.0, math.inf)])
def test_invalid_half_lives_are_rejected(invalid: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="half_lives"):
        SAGEMem(D, A, half_lives=invalid)
