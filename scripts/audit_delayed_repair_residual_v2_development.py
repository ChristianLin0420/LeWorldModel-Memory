#!/usr/bin/env python3
"""Write one locked, unlabeled development health receipt for repair V2."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.delayed_repair_residual_v2_spec import (  # noqa: E402
    DEFAULT_LOCK,
    DEFAULT_SPEC,
    TASKS,
    build_development_receipt,
    development_receipt_path,
    load_locked_spec,
    stable_json,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.execute:
        raise SystemExit(
            "refusing to write development receipt without explicit --execute")
    spec = load_locked_spec(args.spec, args.lock)
    output = development_receipt_path(spec, args.task)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    receipt = build_development_receipt(spec, args.task)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(stable_json(receipt))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[delayed-residual-v2-development] task={args.task} "
          f"passed={receipt['health']['passed']} output={output}", flush=True)


if __name__ == "__main__":
    main()
