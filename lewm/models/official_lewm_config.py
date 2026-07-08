"""Config-driven construction and strict loading for official LeWM models.

The official Hugging Face model repositories store a small Hydra-style
``config.json`` next to a raw PyTorch state dict.  This module parses the
published schema without importing Hydra or the upstream training package,
constructs the equivalent local :class:`OfficialLeWM`, and rejects checkpoint
or configuration drift before mutating model parameters.

The existing Reacher-specific helpers in :mod:`lewm.models.official_lewm`
remain the compatibility API for the completed Reacher study.  New official
environments should use this module and, where available, an environment-
specific identity manifest such as :mod:`official_lewm_pusht`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from transformers import ViTConfig, ViTModel

from lewm.models.official_lewm import (
    Embedder,
    MLP,
    OfficialLeWM,
    Predictor,
    preprocess_frames,
)


class OfficialLeWMConfigError(ValueError):
    """Raised when a published configuration cannot be represented exactly."""


class OfficialLeWMCheckpointError(RuntimeError):
    """Raised when checkpoint identity or state-dict validation fails."""


# ``stable_pretraining.backbone.utils.vit_hf`` follows the standard ViT
# dimensions.  Only sizes represented here can be reconstructed exactly.
_VIT_SIZES: dict[str, tuple[int, int, int, int]] = {
    # hidden size, layers, attention heads, MLP intermediate size
    "tiny": (192, 12, 3, 768),
    "small": (384, 12, 6, 1536),
    "base": (768, 12, 12, 3072),
}

_ROOT_TARGET = "stable_worldmodel.wm.lewm.LeWM"
_ENCODER_TARGET = "stable_pretraining.backbone.utils.vit_hf"
_PREDICTOR_TARGET = "stable_worldmodel.wm.lewm.module.Predictor"
_EMBEDDER_TARGET = "stable_worldmodel.wm.lewm.module.Embedder"
_MLP_TARGET = "stable_worldmodel.wm.lewm.module.MLP"
_BATCH_NORM_TARGET = "torch.nn.BatchNorm1d"


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OfficialLeWMConfigError(f"{path} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise OfficialLeWMConfigError(f"{path} keys must be strings")
    return value


def _check_keys(raw: Mapping[str, Any], *, path: str,
                allowed: set[str], required: set[str]) -> None:
    unknown = sorted(set(raw) - allowed)
    missing = sorted(required - set(raw))
    if unknown:
        raise OfficialLeWMConfigError(
            f"{path} has unsupported fields: {', '.join(unknown)}")
    if missing:
        raise OfficialLeWMConfigError(
            f"{path} is missing fields: {', '.join(missing)}")


def _target(raw: Mapping[str, Any], *, path: str,
            expected: str) -> None:
    actual = raw.get("_target_")
    if actual is not None and actual != expected:
        raise OfficialLeWMConfigError(
            f"{path}._target_ must be {expected!r}, got {actual!r}")


def _positive_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OfficialLeWMConfigError(f"{path} must be a positive integer")
    return value


def _nonnegative_float(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OfficialLeWMConfigError(f"{path} must be a non-negative number")
    result = float(value)
    if result < 0:
        raise OfficialLeWMConfigError(f"{path} must be a non-negative number")
    return result


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise OfficialLeWMConfigError(f"{path} must be boolean")
    return value


@dataclass(frozen=True)
class EncoderConfig:
    size: str
    patch_size: int
    image_size: int
    pretrained: bool = False
    use_mask_token: bool = False
    num_channels: int = 3

    @property
    def hidden_size(self) -> int:
        return _VIT_SIZES[self.size][0]

    @property
    def num_hidden_layers(self) -> int:
        return _VIT_SIZES[self.size][1]

    @property
    def num_attention_heads(self) -> int:
        return _VIT_SIZES[self.size][2]

    @property
    def intermediate_size(self) -> int:
        return _VIT_SIZES[self.size][3]

    @classmethod
    def from_mapping(cls, value: object) -> "EncoderConfig":
        raw = _mapping(value, "encoder")
        _check_keys(
            raw, path="encoder",
            allowed={"_target_", "size", "scale", "patch_size",
                     "image_size", "pretrained", "use_mask_token",
                     "hidden_size", "num_channels"},
            required={"patch_size", "image_size"},
        )
        _target(raw, path="encoder", expected=_ENCODER_TARGET)
        size = raw.get("size", raw.get("scale"))
        if not isinstance(size, str) or size not in _VIT_SIZES:
            supported = ", ".join(sorted(_VIT_SIZES))
            raise OfficialLeWMConfigError(
                f"encoder.size must be one of {supported}, got {size!r}")
        if "size" in raw and "scale" in raw and raw["size"] != raw["scale"]:
            raise OfficialLeWMConfigError(
                "encoder.size and encoder.scale disagree")
        patch_size = _positive_int(raw["patch_size"], "encoder.patch_size")
        image_size = _positive_int(raw["image_size"], "encoder.image_size")
        if image_size % patch_size:
            raise OfficialLeWMConfigError(
                "encoder.image_size must be divisible by encoder.patch_size")
        expected_hidden = _VIT_SIZES[size][0]
        if "hidden_size" in raw and _positive_int(
                raw["hidden_size"], "encoder.hidden_size") != expected_hidden:
            raise OfficialLeWMConfigError(
                f"encoder.hidden_size does not match {size!r} ViT size "
                f"({expected_hidden})")
        pretrained = _boolean(raw.get("pretrained", False),
                              "encoder.pretrained")
        use_mask_token = _boolean(raw.get("use_mask_token", False),
                                  "encoder.use_mask_token")
        if pretrained:
            raise OfficialLeWMConfigError(
                "encoder.pretrained=true is unsupported for strict official "
                "checkpoint reconstruction")
        if use_mask_token:
            raise OfficialLeWMConfigError(
                "encoder.use_mask_token=true is unsupported")
        return cls(
            size=size,
            patch_size=patch_size,
            image_size=image_size,
            pretrained=pretrained,
            use_mask_token=use_mask_token,
            num_channels=_positive_int(raw.get("num_channels", 3),
                                       "encoder.num_channels"),
        )


@dataclass(frozen=True)
class PredictorConfig:
    num_frames: int
    input_dim: int
    hidden_dim: int
    output_dim: int
    depth: int
    heads: int
    mlp_dim: int
    dim_head: int
    dropout: float
    emb_dropout: float

    @classmethod
    def from_mapping(cls, value: object) -> "PredictorConfig":
        raw = _mapping(value, "predictor")
        fields = {"num_frames", "input_dim", "hidden_dim", "output_dim",
                  "depth", "heads", "mlp_dim", "dim_head", "dropout",
                  "emb_dropout"}
        _check_keys(raw, path="predictor", allowed=fields | {"_target_"},
                    required=fields)
        _target(raw, path="predictor", expected=_PREDICTOR_TARGET)
        return cls(
            num_frames=_positive_int(raw["num_frames"],
                                     "predictor.num_frames"),
            input_dim=_positive_int(raw["input_dim"], "predictor.input_dim"),
            hidden_dim=_positive_int(raw["hidden_dim"],
                                    "predictor.hidden_dim"),
            output_dim=_positive_int(raw["output_dim"],
                                    "predictor.output_dim"),
            depth=_positive_int(raw["depth"], "predictor.depth"),
            heads=_positive_int(raw["heads"], "predictor.heads"),
            mlp_dim=_positive_int(raw["mlp_dim"], "predictor.mlp_dim"),
            dim_head=_positive_int(raw["dim_head"], "predictor.dim_head"),
            dropout=_nonnegative_float(raw["dropout"], "predictor.dropout"),
            emb_dropout=_nonnegative_float(
                raw["emb_dropout"], "predictor.emb_dropout"),
        )


@dataclass(frozen=True)
class ActionEncoderConfig:
    input_dim: int
    emb_dim: int
    smoothed_dim: int
    mlp_scale: int

    @classmethod
    def from_mapping(cls, value: object) -> "ActionEncoderConfig":
        raw = _mapping(value, "action_encoder")
        _check_keys(
            raw, path="action_encoder",
            allowed={"_target_", "input_dim", "emb_dim", "smoothed_dim",
                     "mlp_scale"},
            required={"input_dim", "emb_dim"},
        )
        _target(raw, path="action_encoder", expected=_EMBEDDER_TARGET)
        input_dim = _positive_int(raw["input_dim"],
                                  "action_encoder.input_dim")
        return cls(
            input_dim=input_dim,
            emb_dim=_positive_int(raw["emb_dim"],
                                  "action_encoder.emb_dim"),
            smoothed_dim=_positive_int(raw.get("smoothed_dim", input_dim),
                                       "action_encoder.smoothed_dim"),
            mlp_scale=_positive_int(raw.get("mlp_scale", 4),
                                    "action_encoder.mlp_scale"),
        )


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    hidden_dim: int
    output_dim: int

    @classmethod
    def from_mapping(cls, value: object, path: str) -> "MLPConfig":
        raw = _mapping(value, path)
        _check_keys(
            raw, path=path,
            allowed={"_target_", "input_dim", "hidden_dim", "output_dim",
                     "norm_fn"},
            required={"input_dim", "hidden_dim", "output_dim"},
        )
        _target(raw, path=path, expected=_MLP_TARGET)
        if "norm_fn" in raw:
            norm = _mapping(raw["norm_fn"], f"{path}.norm_fn")
            _check_keys(norm, path=f"{path}.norm_fn",
                        allowed={"_target_", "_partial_"},
                        required={"_target_"})
            _target(norm, path=f"{path}.norm_fn",
                    expected=_BATCH_NORM_TARGET)
            if "_partial_" in norm and not _boolean(
                    norm["_partial_"], f"{path}.norm_fn._partial_"):
                raise OfficialLeWMConfigError(
                    f"{path}.norm_fn._partial_ must be true")
        return cls(
            input_dim=_positive_int(raw["input_dim"], f"{path}.input_dim"),
            hidden_dim=_positive_int(raw["hidden_dim"],
                                    f"{path}.hidden_dim"),
            output_dim=_positive_int(raw["output_dim"],
                                    f"{path}.output_dim"),
        )


@dataclass(frozen=True)
class OfficialLeWMConfig:
    encoder: EncoderConfig
    predictor: PredictorConfig
    action_encoder: ActionEncoderConfig
    projector: MLPConfig
    pred_proj: MLPConfig
    source: str | None = None

    @property
    def image_size(self) -> int:
        return self.encoder.image_size

    @property
    def history(self) -> int:
        return self.predictor.num_frames

    @property
    def action_dim(self) -> int:
        return self.action_encoder.input_dim

    @property
    def latent_dim(self) -> int:
        return self.projector.output_dim

    @classmethod
    def from_mapping(cls, value: object, *,
                     source: str | None = None) -> "OfficialLeWMConfig":
        raw = _mapping(value, "config")
        sections = {"encoder", "predictor", "action_encoder", "projector",
                    "pred_proj"}
        _check_keys(raw, path="config", allowed=sections | {"_target_"},
                    required=sections)
        _target(raw, path="config", expected=_ROOT_TARGET)
        config = cls(
            encoder=EncoderConfig.from_mapping(raw["encoder"]),
            predictor=PredictorConfig.from_mapping(raw["predictor"]),
            action_encoder=ActionEncoderConfig.from_mapping(
                raw["action_encoder"]),
            projector=MLPConfig.from_mapping(raw["projector"], "projector"),
            pred_proj=MLPConfig.from_mapping(raw["pred_proj"], "pred_proj"),
            source=source,
        )
        config.validate_dimensions()
        return config

    @classmethod
    def from_json(cls, path: str | Path) -> "OfficialLeWMConfig":
        config_path = Path(path)
        try:
            value = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise OfficialLeWMConfigError(
                f"cannot read official config {config_path}: {error}") from error
        return cls.from_mapping(value, source=str(config_path))

    def validate_dimensions(self) -> None:
        relations = (
            (self.encoder.hidden_size, self.projector.input_dim,
             "encoder hidden size", "projector input"),
            (self.projector.output_dim, self.predictor.input_dim,
             "projector output", "predictor input"),
            (self.action_encoder.emb_dim, self.predictor.input_dim,
             "action embedding", "predictor conditioning input"),
            (self.predictor.output_dim, self.pred_proj.input_dim,
             "predictor output", "prediction-projector input"),
            (self.pred_proj.output_dim, self.projector.output_dim,
             "prediction-projector output", "latent dimension"),
        )
        for left, right, left_name, right_name in relations:
            if left != right:
                raise OfficialLeWMConfigError(
                    f"dimension mismatch: {left_name}={left}, "
                    f"{right_name}={right}")


@dataclass(frozen=True)
class OfficialCheckpointIdentity:
    """Content identity for an environment-specific official bundle."""

    repo_id: str
    revision: str
    config_sha256: str
    weights_sha256: str
    weights_size: int


def build_official_lewm_model(config: OfficialLeWMConfig) -> OfficialLeWM:
    """Construct the exact module topology described by an official config."""
    encoder_config = config.encoder
    vit = ViTConfig(
        image_size=encoder_config.image_size,
        patch_size=encoder_config.patch_size,
        num_channels=encoder_config.num_channels,
        hidden_size=encoder_config.hidden_size,
        num_hidden_layers=encoder_config.num_hidden_layers,
        num_attention_heads=encoder_config.num_attention_heads,
        intermediate_size=encoder_config.intermediate_size,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    predictor_config = config.predictor
    action_config = config.action_encoder
    projector_config = config.projector
    pred_proj_config = config.pred_proj
    model = OfficialLeWM(
        encoder=ViTModel(vit, add_pooling_layer=False),
        predictor=Predictor(
            num_frames=predictor_config.num_frames,
            input_dim=predictor_config.input_dim,
            hidden_dim=predictor_config.hidden_dim,
            output_dim=predictor_config.output_dim,
            depth=predictor_config.depth,
            heads=predictor_config.heads,
            mlp_dim=predictor_config.mlp_dim,
            dim_head=predictor_config.dim_head,
            dropout=predictor_config.dropout,
            emb_dropout=predictor_config.emb_dropout,
        ),
        action_encoder=Embedder(
            input_dim=action_config.input_dim,
            smoothed_dim=action_config.smoothed_dim,
            emb_dim=action_config.emb_dim,
            mlp_scale=action_config.mlp_scale,
        ),
        projector=MLP(
            input_dim=projector_config.input_dim,
            hidden_dim=projector_config.hidden_dim,
            output_dim=projector_config.output_dim,
        ),
        pred_proj=MLP(
            input_dim=pred_proj_config.input_dim,
            hidden_dim=pred_proj_config.hidden_dim,
            output_dim=pred_proj_config.output_dim,
        ),
    )
    # Plain dataclasses are safe non-module attributes and let downstream
    # shape validation derive values from the loaded official configuration.
    model.official_config = config
    return model


def load_official_state_dict(model: OfficialLeWM,
                             state: object) -> None:
    """Preflight and strictly load a raw official state dict.

    Unlike ``nn.Module.load_state_dict(strict=True)`` alone, this also rejects
    dtype drift and wrapper checkpoints before any tensor is copied.
    """
    if not isinstance(state, Mapping) or not all(
            isinstance(key, str) for key in state):
        raise OfficialLeWMCheckpointError(
            "official weights must be a raw string-keyed state dict")
    expected = model.state_dict()
    actual_keys = set(state)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    invalid_types = sorted(
        key for key in actual_keys & expected_keys
        if not isinstance(state[key], torch.Tensor)
    )
    shape_mismatch = sorted(
        (key, tuple(state[key].shape), tuple(expected[key].shape))
        for key in actual_keys & expected_keys
        if isinstance(state[key], torch.Tensor)
        and tuple(state[key].shape) != tuple(expected[key].shape)
    )
    dtype_mismatch = sorted(
        (key, str(state[key].dtype), str(expected[key].dtype))
        for key in actual_keys & expected_keys
        if isinstance(state[key], torch.Tensor)
        and tuple(state[key].shape) == tuple(expected[key].shape)
        and state[key].dtype != expected[key].dtype
    )
    if missing or unexpected or invalid_types or shape_mismatch or dtype_mismatch:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:5]}")
        if invalid_types:
            details.append(f"non_tensor={invalid_types[:5]}")
        if shape_mismatch:
            details.append(f"shape_mismatch={shape_mismatch[:3]}")
        if dtype_mismatch:
            details.append(f"dtype_mismatch={dtype_mismatch[:3]}")
        raise OfficialLeWMCheckpointError(
            "official checkpoint does not exactly match config: "
            + "; ".join(details))
    try:
        model.load_state_dict(dict(state), strict=True)
    except RuntimeError as error:  # defensive: preflight should catch these
        raise OfficialLeWMCheckpointError(
            f"strict official checkpoint load failed: {error}") from error


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_official_bundle(config_path: str | Path,
                           weights_path: str | Path,
                           identity: OfficialCheckpointIdentity) -> None:
    config_file = Path(config_path)
    weights_file = Path(weights_path)
    if not config_file.is_file():
        raise OfficialLeWMCheckpointError(
            f"official config not found: {config_file}")
    if not weights_file.is_file():
        raise OfficialLeWMCheckpointError(
            f"official weights not found: {weights_file}")
    actual_config_hash = sha256_file(config_file)
    if actual_config_hash != identity.config_sha256:
        raise OfficialLeWMCheckpointError(
            f"{identity.repo_id} config SHA-256 mismatch: "
            f"expected {identity.config_sha256}, got {actual_config_hash}")
    actual_size = weights_file.stat().st_size
    if actual_size != identity.weights_size:
        raise OfficialLeWMCheckpointError(
            f"{identity.repo_id} weights size mismatch: "
            f"expected {identity.weights_size}, got {actual_size}")
    actual_weights_hash = sha256_file(weights_file)
    if actual_weights_hash != identity.weights_sha256:
        raise OfficialLeWMCheckpointError(
            f"{identity.repo_id} weights SHA-256 mismatch: "
            f"expected {identity.weights_sha256}, got {actual_weights_hash}")


def load_official_lewm_checkpoint(
        config_path: str | Path,
        weights_path: str | Path,
        device: torch.device | str = "cpu",
        *, identity: OfficialCheckpointIdentity | None = None,
        ) -> OfficialLeWM:
    """Load, freeze, and return one official config/checkpoint pair."""
    config_file = Path(config_path)
    weights_file = Path(weights_path)
    if identity is not None:
        verify_official_bundle(config_file, weights_file, identity)
    elif not weights_file.is_file():
        raise OfficialLeWMCheckpointError(
            f"official weights not found: {weights_file}")
    config = OfficialLeWMConfig.from_json(config_file)
    model = build_official_lewm_model(config)
    try:
        state = torch.load(weights_file, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch 2.0 compatibility
        state = torch.load(weights_file, map_location="cpu")
    except (OSError, RuntimeError) as error:
        raise OfficialLeWMCheckpointError(
            f"cannot read official weights {weights_file}: {error}") from error
    load_official_state_dict(model, state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.checkpoint_identity = identity
    return model.to(device)


def load_official_lewm_bundle(
        directory: str | Path,
        device: torch.device | str = "cpu",
        *, identity: OfficialCheckpointIdentity | None = None,
        ) -> OfficialLeWM:
    root = Path(directory)
    return load_official_lewm_checkpoint(
        root / "config.json", root / "weights.pt", device,
        identity=identity)


def preprocess_frames_for_config(
        frames: torch.Tensor,
        config: OfficialLeWMConfig,
        ) -> torch.Tensor:
    """Apply official normalization and the configured spatial resize."""
    return preprocess_frames(frames, image_size=config.image_size)
