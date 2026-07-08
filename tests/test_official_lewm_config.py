import gc
import json
from pathlib import Path

import pytest
import torch

from lewm.models.official_lewm import build_official_reacher_model
from lewm.models.official_lewm_config import (
    OfficialLeWMCheckpointError,
    OfficialLeWMConfig,
    OfficialLeWMConfigError,
    build_official_lewm_model,
    load_official_lewm_checkpoint,
    load_official_state_dict,
    preprocess_frames_for_config,
    sha256_file,
)
from lewm.models.official_lewm_pusht import (
    OFFICIAL_PUSHT_CHECKPOINT,
    load_official_pusht_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
PUSHT_CONFIG = ROOT / "tests/fixtures/official_lewm_pusht_config.json"
REACHER_BUNDLE = (
    ROOT / "outputs/paper_a_expansion/pretrained/lewm-reacher"
)


def _pusht_mapping() -> dict:
    return json.loads(PUSHT_CONFIG.read_text(encoding="utf-8"))


def test_parse_published_pusht_config_schema() -> None:
    config = OfficialLeWMConfig.from_json(PUSHT_CONFIG)

    assert config.encoder.size == "tiny"
    assert config.image_size == 224
    assert config.encoder.patch_size == 14
    assert config.encoder.hidden_size == 192
    assert config.history == 3
    assert config.action_dim == 10
    assert config.latent_dim == 192
    assert config.action_encoder.smoothed_dim == 10
    assert config.action_encoder.mlp_scale == 4


def test_config_drives_action_dimension_without_reacher_constants() -> None:
    raw = _pusht_mapping()
    raw["action_encoder"]["input_dim"] = 25
    config = OfficialLeWMConfig.from_mapping(raw, source="synthetic-cube")
    model = build_official_lewm_model(config)

    assert model.action_encoder.patch_embed.in_channels == 25
    assert model.official_config.action_dim == 25


def test_config_rejects_unknown_fields_and_dimension_drift() -> None:
    unknown = _pusht_mapping()
    unknown["encoder"]["silent_override"] = 1
    with pytest.raises(OfficialLeWMConfigError, match="unsupported fields"):
        OfficialLeWMConfig.from_mapping(unknown)

    mismatch = _pusht_mapping()
    mismatch["action_encoder"]["emb_dim"] = 191
    with pytest.raises(OfficialLeWMConfigError,
                       match="action embedding=191"):
        OfficialLeWMConfig.from_mapping(mismatch)


def test_generic_topology_preserves_reacher_loader_compatibility() -> None:
    config = OfficialLeWMConfig.from_json(PUSHT_CONFIG)
    generic = build_official_lewm_model(config)
    generic_shapes = {
        key: (tuple(value.shape), value.dtype)
        for key, value in generic.state_dict().items()
    }
    del generic
    gc.collect()

    legacy = build_official_reacher_model()
    legacy_shapes = {
        key: (tuple(value.shape), value.dtype)
        for key, value in legacy.state_dict().items()
    }
    assert generic_shapes == legacy_shapes


def test_strict_state_preflight_rejects_missing_shape_and_dtype() -> None:
    model = build_official_lewm_model(
        OfficialLeWMConfig.from_json(PUSHT_CONFIG))
    state = dict(model.state_dict())

    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(OfficialLeWMCheckpointError, match="missing="):
        load_official_state_dict(model, missing)

    shape = dict(state)
    action_key = "action_encoder.patch_embed.weight"
    shape[action_key] = shape[action_key][..., :-1]
    with pytest.raises(OfficialLeWMCheckpointError,
                       match="shape_mismatch="):
        load_official_state_dict(model, shape)

    dtype = dict(state)
    float_key = next(key for key, value in dtype.items()
                     if value.dtype == torch.float32)
    dtype[float_key] = dtype[float_key].to(torch.float64)
    with pytest.raises(OfficialLeWMCheckpointError,
                       match="dtype_mismatch="):
        load_official_state_dict(model, dtype)


def test_pusht_identity_rejects_environment_swapped_or_truncated_weights(
        tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    weights_path = tmp_path / "weights.pt"
    config_path.write_bytes(PUSHT_CONFIG.read_bytes())
    weights_path.write_bytes(b"not an official checkpoint")

    expected_config_hash = OFFICIAL_PUSHT_CHECKPOINT.config_sha256
    assert sha256_file(config_path) == expected_config_hash
    with pytest.raises(OfficialLeWMCheckpointError,
                       match="weights size mismatch"):
        load_official_pusht_checkpoint(tmp_path, device="cpu")


def test_configured_preprocess_keeps_published_pusht_size() -> None:
    config = OfficialLeWMConfig.from_json(PUSHT_CONFIG)
    frames = torch.zeros(2, 3, 80, 96, dtype=torch.uint8)
    output = preprocess_frames_for_config(frames, config)
    assert output.shape == (2, 3, 224, 224)


@pytest.mark.skipif(
    not (REACHER_BUNDLE / "config.json").is_file()
    or not (REACHER_BUNDLE / "weights.pt").is_file(),
    reason="released Reacher bundle is not present in this checkout",
)
def test_pusht_identity_rejects_shape_compatible_reacher_bundle() -> None:
    with pytest.raises(OfficialLeWMCheckpointError,
                       match="weights size mismatch"):
        load_official_pusht_checkpoint(REACHER_BUNDLE, device="cpu")


@pytest.mark.skipif(
    not (REACHER_BUNDLE / "config.json").is_file()
    or not (REACHER_BUNDLE / "weights.pt").is_file(),
    reason="released Reacher bundle is not present in this checkout",
)
def test_generic_loader_strictly_loads_existing_official_reacher_on_cpu() -> None:
    model = load_official_lewm_checkpoint(
        REACHER_BUNDLE / "config.json",
        REACHER_BUNDLE / "weights.pt",
        device="cpu",
    )
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert next(model.parameters()).device.type == "cpu"
    assert model.official_config.action_dim == 10
