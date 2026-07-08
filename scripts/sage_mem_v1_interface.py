#!/usr/bin/env python3
"""Runtime contract for the separately delivered SAGE-Mem model/host APIs."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Mapping


MODEL_API_VERSION = "sage_mem_v1_api_v1"
HOST_API_VERSION = "sage_mem_v1_host_adapter_v1"


class SageMemInterfaceError(RuntimeError):
    """A separately delivered integration does not implement the sealed API."""


@dataclass(frozen=True)
class ModelContract:
    module: ModuleType
    builder: Any
    required_output_keys: tuple[str, ...]


@dataclass(frozen=True)
class HostAdapterContract:
    module: ModuleType
    builder: Any


def _import(module_name: str, label: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except Exception as error:
        raise SageMemInterfaceError(
            f"{label} module {module_name!r} is unavailable; integration must "
            "be delivered and sealed before smoke/full execution"
        ) from error


def _reject_label_parameters(callable_: Any, label: str) -> None:
    parameters = inspect.signature(callable_).parameters
    forbidden = {"label", "labels", "target_label", "oracle_state"}
    found = forbidden.intersection(parameters)
    if found:
        raise SageMemInterfaceError(
            f"{label} exposes forbidden semantic inputs: {sorted(found)}")


def _require_keyword_parameters(callable_: Any, required: set[str],
                                label: str) -> None:
    parameters = inspect.signature(callable_).parameters
    accepts_kwargs = any(
        value.kind is inspect.Parameter.VAR_KEYWORD
        for value in parameters.values())
    missing = required.difference(parameters)
    if missing and not accepts_kwargs:
        raise SageMemInterfaceError(
            f"{label} does not accept required keywords: {sorted(missing)}")


def load_model_contract(config: Mapping[str, Any]) -> ModelContract:
    module_name = str(config["module"])
    module = _import(module_name, "SAGE-Mem model")
    if getattr(module, "SAGE_MEM_API_VERSION", None) != config["api_version"] \
            or config["api_version"] != MODEL_API_VERSION:
        raise SageMemInterfaceError("SAGE-Mem model API version mismatch")
    builder = getattr(module, str(config["builder"]), None)
    if not callable(builder):
        raise SageMemInterfaceError("SAGE-Mem builder is missing or not callable")
    _reject_label_parameters(builder, "model builder")
    _require_keyword_parameters(
        builder, {"embed_dim", "action_dim", "variant", "config"},
        "model builder")
    return ModelContract(
        module=module,
        builder=builder,
        required_output_keys=tuple(config["required_output_keys"]),
    )


def validate_model_instance(model: Any, contract: ModelContract) -> None:
    required_methods = (
        "forward_sequence", "describe", "parameter_count",
        "persistent_state_floats", "estimate_flops",
    )
    missing = [name for name in required_methods
               if not callable(getattr(model, name, None))]
    if missing:
        raise SageMemInterfaceError(f"model methods missing: {missing}")
    _reject_label_parameters(model.forward_sequence, "forward_sequence")
    description = model.describe()
    if not isinstance(description, Mapping) \
            or description.get("api_version") != MODEL_API_VERSION:
        raise SageMemInterfaceError(
            "model describe() lacks the sealed API version")
    for name, value in (
        ("parameter_count", model.parameter_count()),
        ("persistent_state_floats", model.persistent_state_floats()),
        ("estimate_flops", model.estimate_flops(
            batch_size=1, timesteps=20, tokens=1)),
    ):
        if not isinstance(value, int) or value < 0:
            raise SageMemInterfaceError(f"{name} must return a non-negative int")


def validate_forward_output(output: Any, contract: ModelContract) -> None:
    if not isinstance(output, Mapping):
        raise SageMemInterfaceError("forward_sequence must return a mapping")
    missing = set(contract.required_output_keys).difference(output)
    if missing:
        raise SageMemInterfaceError(
            f"forward_sequence output keys missing: {sorted(missing)}")


def load_host_adapter_contract(config: Mapping[str, Any]) -> HostAdapterContract:
    module_name = str(config["module"])
    module = _import(module_name, "SAGE-Mem host adapter")
    if getattr(module, "SAGE_MEM_HOST_ADAPTER_API_VERSION", None) != \
            config["api_version"] or config["api_version"] != HOST_API_VERSION:
        raise SageMemInterfaceError("host adapter API version mismatch")
    builder = getattr(module, str(config["builder"]), None)
    if not callable(builder):
        raise SageMemInterfaceError("host adapter builder is missing or not callable")
    _require_keyword_parameters(builder, {"cohort", "spec"},
                                "host adapter builder")
    return HostAdapterContract(module=module, builder=builder)


def validate_host_adapter_instance(
        adapter: Any, *, cohort: str, api_version: str) -> Mapping[str, Any]:
    requirements = {
        "smoke": {"model_contract"},
        "run_development_cell": {
            "arm", "seed", "output_directory", "model_contract",
            "development_manifest"},
        "prepare_fresh_banks": {
            "split_counts", "seed_registry", "forbidden_parent_artifacts",
            "model_contract"},
        "run_formal_cell": {
            "arm", "seed", "output_directory", "model_contract",
            "prepared"},
    }
    if not callable(getattr(adapter, "describe", None)):
        raise SageMemInterfaceError("host adapter describe() is missing")
    for name, keywords in requirements.items():
        method = getattr(adapter, name, None)
        if not callable(method):
            raise SageMemInterfaceError(f"host adapter method missing: {name}")
        _require_keyword_parameters(method, keywords, f"host adapter {name}")
    description = adapter.describe()
    required_description = {
        "api_version", "cohort", "family", "task", "embed_dim",
        "action_dim", "tokens", "classes", "development_source",
        "development_source_policy", "semantic_labels_for_training",
        "candidate_spatial_path", "formal_status",
    }
    if not isinstance(description, Mapping) \
            or required_description.difference(description):
        raise SageMemInterfaceError(
            "host adapter description is incomplete")
    if description["api_version"] != api_version \
            or description["cohort"] != cohort:
        raise SageMemInterfaceError(
            "host adapter description identity mismatch")
    if description["semantic_labels_for_training"] is not False \
            or description["formal_status"] != "pending_fresh_bank_builder" \
            or description["development_source_policy"] != \
            "manifest-selected parent TRAIN only":
        raise SageMemInterfaceError(
            "host adapter data/label/formal boundary mismatch")
    for name in ("embed_dim", "action_dim", "tokens", "classes"):
        value = description[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise SageMemInterfaceError(
                f"host adapter {name} must be a positive integer")
    return description


def integration_requirements() -> dict[str, Any]:
    """Machine-readable integration points for the pending implementation."""
    return {
        "model_module_constant": {"SAGE_MEM_API_VERSION": MODEL_API_VERSION},
        "model_builder_signature": (
            "build_sage_mem_v1(*, embed_dim, action_dim, variant, config)"),
        "model_methods": [
            "forward_sequence(features, actions, *, reset_mask=None)",
            "describe()", "parameter_count()", "persistent_state_floats()",
            "estimate_flops(*, batch_size, timesteps, tokens)",
        ],
        "forward_output_keys": [
            "fused", "prior", "posterior", "exposure", "diagnostics"],
        "host_module_constant": {
            "SAGE_MEM_HOST_ADAPTER_API_VERSION": HOST_API_VERSION},
        "host_builder_signature": "build_host_adapter(*, cohort, spec)",
        "host_resource_responsibility": (
            "adapter profiles forward FLOPs, persistent-state floats, peak "
            "CUDA bytes, and wall time for every arm"),
        "semantic_label_boundary": (
            "labels may enter only frozen post-hoc readouts and external "
            "consumers; never model construction, forward, or training loss"),
    }
