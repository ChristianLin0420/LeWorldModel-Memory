"""Deterministic, fail-closed artifact IO for official task studies."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    """Hash dtype, shape, and contiguous bytes of one array."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8).tobytes())
    return digest.hexdigest()


def sha256_arrays(values: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode("utf-8"))
        digest.update(sha256_array(np.asarray(value)).encode("ascii"))
    return digest.hexdigest()


def stable_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def atomic_text(path: str | Path, text: str, *, overwrite: bool = False) -> str:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def _zip_info(name: str, compression_level: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    info._compresslevel = compression_level
    return info


def write_npz(path: str | Path, arrays: Mapping[str, np.ndarray], *,
              compression_level: int = 1, overwrite: bool = False) -> str:
    """Atomically write a byte-stable allow-pickle-free NPZ archive."""

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be in [0,9]")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for name, value in arrays.items():
                if not name or "/" in name or name.endswith(".npy"):
                    raise ValueError(f"invalid NPZ member {name!r}")
                array = np.asanyarray(value)
                if array.dtype.hasobject:
                    raise ValueError(f"object arrays are forbidden: {name}")
                with archive.open(
                        _zip_info(name, compression_level), "w",
                        force_zip64=True) as member:
                    np.lib.format.write_array(member, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return sha256_file(path)


def array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
        }
        for name, value in arrays.items()
    }


def write_npz_with_sidecar(
        path: str | Path, arrays: Mapping[str, np.ndarray],
        metadata: Mapping[str, Any], *, compression_level: int = 1,
        overwrite: bool = False) -> dict[str, Any]:
    path = Path(path)
    payload = dict(arrays)
    payload["meta_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    artifact_hash = write_npz(
        path, payload, compression_level=compression_level,
        overwrite=overwrite)
    sidecar = {
        **dict(metadata),
        "arrays": array_manifest(payload),
        "artifact": {
            "path": path.name,
            "sha256": artifact_hash,
            "compression_level": compression_level,
        },
    }
    sidecar_path = path.with_suffix(path.suffix + ".json")
    sidecar_hash = atomic_text(
        sidecar_path, stable_json(sidecar), overwrite=overwrite)
    return {
        "path": str(path),
        "sha256": artifact_hash,
        "sidecar": str(sidecar_path),
        "sidecar_sha256": sidecar_hash,
    }


def load_verified_npz(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    path = Path(path)
    sidecar_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"missing artifact or sidecar for {path}")
    sidecar = json.loads(sidecar_path.read_text())
    expected = sidecar.get("artifact", {}).get("sha256")
    actual = sha256_file(path)
    if expected != actual:
        raise ValueError(f"artifact hash mismatch for {path}: {actual} != {expected}")
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    embedded = json.loads(str(arrays.pop("meta_json", np.asarray("{}"))))
    comparable = {
        key: value for key, value in sidecar.items()
        if key not in ("arrays", "artifact")
    }
    if embedded != comparable:
        raise ValueError(f"embedded metadata differs from sidecar for {path}")
    return arrays, sidecar
