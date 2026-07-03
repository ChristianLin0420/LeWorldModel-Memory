#!/usr/bin/env python3
"""Build a deterministic, curated, anonymous V18 code/result supplement.

The builder is allowlist-only.  It reads the private frozen protocol to select
and verify source bytes, but never publishes that protocol or any raw output,
log, checkpoint, rollout, W&B directory, Git metadata, or private receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v18_release_common as common


ARCHIVE_ROOT = "v18-anonymous-supplement"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
EXPECTED_FROZEN_SOURCES = {
    "docs/V18_LEWM_V8_CONFIRMATION.md",
    "lewm/__init__.py",
    "lewm/models/__init__.py",
    "lewm/models/cf_ebo.py",
    "lewm/models/cf_hiro.py",
    "lewm/models/cvpf.py",
    "lewm/models/encoder.py",
    "lewm/models/leworldmodel.py",
    "lewm/models/memory.py",
    "lewm/models/memory_model.py",
    "lewm/models/sigreg.py",
    "lewm/models/siro.py",
    "scripts/analyze_lewm_v8_v18.py",
    "scripts/hacssm_v10_data.py",
    "scripts/hacssm_v11_data.py",
    "scripts/hacssm_v18_data.py",
    "scripts/run_autovisreg_v17.py",
    "scripts/run_lewm_v8_v18.py",
    "scripts/train_hacssm_v10.py",
    "scripts/train_hacssm_v11.py",
    "scripts/train_lewm_v8_v18.py",
    "scripts/train_subjepa_v16.py",
}
V18_TESTS = {
    "scripts/test_analyze_lewm_v8_v18.py",
    "scripts/test_run_lewm_v8_v18.py",
    "scripts/test_train_lewm_v8_v18.py",
}
EXPECTED_REDACTED_FROZEN_SOURCES = {
    "scripts/run_autovisreg_v17.py",
    "scripts/train_hacssm_v10.py",
    "scripts/train_hacssm_v11.py",
    "scripts/train_subjepa_v16.py",
}
SUPPORTING_FILES: set[str] = set()
PUBLIC_RELEASE_TOOLS = {
    "scripts/build_v18_code_supplement.py",
    "scripts/build_v18_review_artifact.py",
    "scripts/plot_v18_paper.py",
    "scripts/render_v18_paper.py",
    "scripts/v18_release_common.py",
    "paper/README.md",
    "paper/build_paper.py",
    "paper/check_v18_paper.py",
    "paper/iclr2026_conference.sty",
    "paper/main.tex",
    "templates/ICLR.template.md",
}
PAPER_FILES = {
    "abstract.tex",
    "appendix.tex",
    "body.tex",
    "iclr2026_conference.sty",
    "main.pdf",
    "main.tex",
    "paper_build_manifest.json",
    "refs.tex",
}
TEXT_SUFFIXES = {
    ".csv", ".json", ".md", ".py", ".sty", ".tex", ".toml",
    ".txt", ".yaml", ".yml",
}
ALLOWED_SUFFIXES = TEXT_SUFFIXES | {".pdf", ".png"}
BANNED_COMPONENTS = {
    ".git", ".paper-draft", ".pytest_cache", ".venv", "__pycache__",
    "logs", "outputs", "wandb",
}
BANNED_SUFFIXES = {
    ".log", ".npz", ".pt", ".pth", ".wandb", ".ckpt", ".aux",
    ".fls", ".fdb_latexmk",
}
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
LOCAL_PATH_RE = re.compile(
    r"(?:/home/[A-Z0-9._-]+/|/Users/[A-Z0-9._-]+/|[A-Z]:\\Users\\[A-Z0-9._-]+\\)",
    re.I,
)
IDENTITY_REMOTE_RE = re.compile(
    r"(?:https?://(?:www\.)?(?:github|gitlab|bitbucket|wandb)\.[^\s)>\]}]+|"
    r"git@(?:github|gitlab|bitbucket)\.[^\s:]+:[^\s]+)",
    re.I,
)
SECRET_RE = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"\bgh[pousr]_[A-Za-z0-9]{20,}\b|"
    r"\bgithub_pat_[A-Za-z0-9_]{20,}\b|"
    r"\bAKIA[0-9A-Z]{16}\b|"
    r"\bsk-[A-Za-z0-9_-]{24,}\b)",
    re.I,
)


@dataclass(frozen=True)
class Replacement:
    kind: str
    source: str
    public: str

    def manifest_value(self) -> dict[str, Any]:
        return {
            "category": self.kind,
            "scan_passed": True,
        }


def safe_relative(value: str | Path) -> PurePosixPath:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise common.ReleaseValidationError(f"unsafe supplement path {value!r}")
    lowered = {part.casefold() for part in path.parts}
    if lowered & BANNED_COMPONENTS:
        raise common.ReleaseValidationError(f"banned supplement path component in {path}")
    if path.suffix.casefold() in BANNED_SUFFIXES:
        raise common.ReleaseValidationError(f"banned private artifact suffix in {path}")
    if path.name != ".gitignore" and path.suffix.casefold() not in ALLOWED_SUFFIXES:
        raise common.ReleaseValidationError(f"unapproved supplement file type {path}")
    return path


def read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = common.read_json(path)
    if not isinstance(value, dict):
        raise common.ReleaseValidationError(f"{label} is not a JSON object")
    return value


def git_remote(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def replacement_policy(
    repo: Path,
    result_root: Path,
    protocol: Mapping[str, Any],
) -> list[Replacement]:
    candidates = [
        Replacement("wandb_entity", str(protocol.get("wandb_entity", "")), "anonymous-review-entity"),
        Replacement("wandb_project", str(protocol.get("wandb_project", "")), "anonymous-review-project"),
        Replacement("result_root", str(result_root), "$RESULTS"),
        Replacement("protocol_output_root", str(protocol.get("output_root", "")), "$RESULTS"),
        Replacement("protocol_log_root", str(protocol.get("log_root", "")), "$LOGS"),
        Replacement("protocol_python", str(protocol.get("python", "")), "$PYTHON"),
        Replacement("repository_root", str(repo), "$REPO"),
        Replacement("git_commit", str(protocol.get("git_commit", "")), "WITHHELD_GIT_COMMIT"),
        Replacement(
            "git_upstream_commit",
            str(protocol.get("git_upstream_commit", "")),
            "WITHHELD_UPSTREAM_COMMIT",
        ),
        Replacement("git_remote", git_remote(repo) or "", "WITHHELD_GIT_REMOTE"),
    ]
    # Longest first prevents a home-path replacement from obscuring a more
    # specific repository/result path. Exact duplicate private values are kept once.
    chosen: dict[str, Replacement] = {}
    for item in candidates:
        if len(item.source) >= 4 and item.source not in chosen:
            chosen[item.source] = item
    return sorted(chosen.values(), key=lambda item: (-len(item.source), item.kind))


def redact_text(text: str, policy: Iterable[Replacement]) -> tuple[str, list[dict[str, Any]]]:
    receipts: list[dict[str, Any]] = []
    for item in policy:
        count = text.count(item.source)
        if count:
            text = text.replace(item.source, item.public)
            receipt = item.manifest_value()
            receipt["occurrences"] = count
            receipts.append(receipt)
    return text, receipts


def redact_source_text(
    relative: str,
    text: str,
    policy: Iterable[Replacement],
) -> tuple[str, list[dict[str, Any]]]:
    text, receipts = redact_text(text, policy)
    if relative == "scripts/run_autovisreg_v17.py":
        assignments = (
            (
                'WANDB_ENTITY = "anonymous-review-entity"',
                'WANDB_ENTITY = os.environ.get("V18_WANDB_ENTITY", "anonymous-review-entity")',
                "wandb_entity_environment_parameterization",
            ),
            (
                'WANDB_PROJECT = "anonymous-review-project"',
                'WANDB_PROJECT = os.environ.get("V18_WANDB_PROJECT", "anonymous-review-project")',
                "wandb_project_environment_parameterization",
            ),
        )
        for source, public, kind in assignments:
            if text.count(source) != 1:
                raise common.ReleaseValidationError(
                    f"expected exactly one neutral W&B assignment for {kind}"
                )
            text = text.replace(source, public, 1)
            receipts.append({
                "category": kind,
                "scan_passed": True,
                "occurrences": 1,
            })
    return text, receipts


def scan_name(name: str, tokens: Iterable[str]) -> None:
    safe_relative(name)
    folded = unicodedata.normalize("NFKC", name).casefold()
    leaks = [
        token for token in tokens
        if token and unicodedata.normalize("NFKC", token).casefold() in folded
    ]
    if leaks or EMAIL_RE.search(name) or IDENTITY_REMOTE_RE.search(name) \
            or LOCAL_PATH_RE.search(name) or SECRET_RE.search(name):
        raise common.ReleaseValidationError("identity-bearing archive member name detected")


def scan_output_name(name: str, tokens: Iterable[str]) -> None:
    folded = unicodedata.normalize("NFKC", name).casefold()
    leaks = [
        token for token in tokens
        if token and unicodedata.normalize("NFKC", token).casefold() in folded
    ]
    if leaks or EMAIL_RE.search(name) or IDENTITY_REMOTE_RE.search(name) \
            or LOCAL_PATH_RE.search(name) or SECRET_RE.search(name):
        raise common.ReleaseValidationError("identity-bearing archive output name detected")


def scan_bytes(
    path: Path,
    relative: str,
    tokens: Iterable[str],
    *,
    pdftotext: str,
) -> None:
    data = path.read_bytes()
    lowered = data.lower()
    byte_leaks = [
        token for token in tokens
        if token and token.encode("utf-8", "ignore").lower() in lowered
    ]
    if byte_leaks:
        raise common.ReleaseValidationError(
            f"{len(byte_leaks)} known identity token(s) remain in {relative}"
        )
    suffix = path.suffix.casefold()
    if suffix in TEXT_SUFFIXES:
        text = data.decode("utf-8", errors="strict")
    elif suffix == ".pdf":
        result = subprocess.run(
            [pdftotext, "-layout", str(path), "-"],
            capture_output=True,
            text=True,
            check=True,
        )
        text = result.stdout
    else:
        text = ""
    normalized = unicodedata.normalize("NFKC", text).casefold() if text else ""
    text_token_leaks = [
        token for token in tokens
        if token and unicodedata.normalize("NFKC", token).casefold() in normalized
    ]
    if text and (
        text_token_leaks
        or EMAIL_RE.search(text)
        or IDENTITY_REMOTE_RE.search(text)
        or LOCAL_PATH_RE.search(text)
        or SECRET_RE.search(text)
    ):
        raise common.ReleaseValidationError(
            f"email, identity remote, or local path remains in {relative}"
        )


def copy_with_receipt(
    source: Path,
    staging: Path,
    relative: str,
    *,
    category: str,
    source_class: str,
    policy: list[Replacement],
    permit_redaction: bool,
    expected_original_sha256: str | None = None,
) -> dict[str, Any]:
    destination_relative = safe_relative(relative)
    if not source.is_file() or source.is_symlink():
        raise common.ReleaseValidationError(
            f"missing regular allowlisted source file {destination_relative}"
        )
    original = common.sha256(source)
    if expected_original_sha256 is not None and original != expected_original_sha256:
        raise common.ReleaseValidationError(
            f"private source hash differs for {destination_relative}"
        )
    destination = staging.joinpath(*destination_relative.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    redactions: list[dict[str, Any]] = []
    if permit_redaction and source.suffix.casefold() in TEXT_SUFFIXES:
        text = source.read_text(encoding="utf-8", errors="strict")
        text, redactions = redact_source_text(relative, text, policy)
        common.atomic_write_text(destination, text)
    else:
        shutil.copyfile(source, destination)
    public_sha256 = common.sha256(destination)
    receipt = {
        "path": destination_relative.as_posix(),
        "category": category,
        "source_class": source_class,
        "public_sha256": public_sha256,
        "public_size_bytes": destination.stat().st_size,
        "archive_mode": "100644",
        "redactions": redactions,
    }
    if expected_original_sha256 is not None:
        if redactions:
            receipt["frozen_execution_sha256"] = original
        else:
            if public_sha256 != original:
                raise common.ReleaseValidationError(
                    f"unredacted frozen source bytes differ for {destination_relative}"
                )
            receipt["matches_frozen_execution"] = True
    return receipt


def add_generated(
    staging: Path,
    relative: str,
    text: str,
    *,
    category: str,
) -> dict[str, Any]:
    destination_relative = safe_relative(relative)
    destination = staging.joinpath(*destination_relative.parts)
    common.atomic_write_text(destination, text)
    return {
        "path": destination_relative.as_posix(),
        "category": category,
        "source_class": "generated_public",
        "public_sha256": common.sha256(destination),
        "public_size_bytes": destination.stat().st_size,
        "archive_mode": "100644",
        "redactions": [],
    }


def validate_review_artifact(review: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = review / "review_manifest.json"
    manifest = read_json_object(manifest_path, "review manifest")
    if manifest.get("schema_version") != 2 \
            or manifest.get("scientific_label") != bundle["report"]["scientific_label"] \
            or manifest.get("cells") != 200 \
            or manifest.get("contrasts") != 33 \
            or manifest.get("private_repository_review_safe") is not False:
        raise common.ReleaseValidationError("review artifact release receipt differs")
    if manifest.get("source_result_hashes") != bundle["hashes"]:
        raise common.ReleaseValidationError("review artifact/result source hashes differ")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise common.ReleaseValidationError("review artifact file manifest is absent")
    actual = {
        path.name for path in review.iterdir()
        if path.is_file() and path.name != "review_manifest.json"
    }
    if actual != set(files):
        raise common.ReleaseValidationError("review artifact file set differs")
    for name, digest in files.items():
        safe_relative(name)
        common.require_hash(digest, f"review artifact {name}")
        if common.sha256(review / name) != digest:
            raise common.ReleaseValidationError(f"review artifact hash differs for {name}")
    return manifest


def validate_manuscript(manuscript: Path, review_manifest: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = manuscript.with_suffix(".manifest.json")
    manifest = read_json_object(manifest_path, "manuscript manifest")
    if manifest.get("schema_version") != 2 \
            or manifest.get("scientific_label") != "CONFIRMATION_FAILED" \
            or manifest.get("manuscript_sha256") != common.sha256(manuscript) \
            or manifest.get("restart_interruptions") != 2 \
            or manifest.get("telemetry_disclosure_present") is not True \
            or manifest.get("llm_usage_statement_present") is not True:
        raise common.ReleaseValidationError("manuscript release receipt differs")
    source_hashes = review_manifest["source_result_hashes"]
    if manifest.get("canonical_commands_sha256") != review_manifest.get(
            "canonical_commands_sha256"):
        raise common.ReleaseValidationError(
            "manuscript/review canonical command binding differs"
        )
    for review_name, manuscript_name in (
        ("confirmation_analysis.json", "analysis_sha256"),
        ("confirmation_cells.csv", "cells_sha256"),
        ("confirmation_contrasts.csv", "contrasts_sha256"),
        ("confirmation_protocol.json", "protocol_sha256"),
        ("confirmation_runs.json", "runs_sha256"),
        ("confirmation_attempts.json", "attempts_sha256"),
        ("confirmation_summary.json", "summary_sha256"),
    ):
        if manifest.get(manuscript_name) != source_hashes.get(review_name):
            raise common.ReleaseValidationError(
                f"manuscript/review source binding differs for {review_name}"
            )
    figures = manuscript.parent / "figures"
    figure_manifest = read_json_object(figures / "fig_v18_manifest.json", "figure manifest")
    secondary = figure_manifest.get("descriptive_secondary")
    secondary_rows = secondary.get("public_prior_slices", []) \
        if isinstance(secondary, Mapping) else []
    if figure_manifest.get("schema_version") != 3 \
            or figure_manifest.get("artifact_kind") != "v18_provenance_bound_paper_figures" \
            or not isinstance(secondary, Mapping) \
            or secondary.get("official_decision_changed") is not False \
            or secondary.get("decision_gates_defined") is not False \
            or len(secondary_rows) != 200:
        raise common.ReleaseValidationError("descriptive figure provenance differs")
    if figure_manifest.get("analysis_sha256") != manifest["analysis_sha256"] \
            or figure_manifest.get("cells_sha256") != manifest["cells_sha256"] \
            or figure_manifest.get("contrasts_sha256") != manifest["contrasts_sha256"]:
        raise common.ReleaseValidationError("figure/manuscript provenance differs")
    figure_files = figure_manifest.get("figures")
    if not isinstance(figure_files, Mapping) or set(figure_files) != {
        "fig_v18_architecture.pdf", "fig_v18_architecture.png",
        "fig_v18_evidence.pdf", "fig_v18_evidence.png",
        "fig_v18_secondary.pdf", "fig_v18_secondary.png",
        "fig_v18_task_design.pdf", "fig_v18_task_design.png",
    }:
        raise common.ReleaseValidationError("figure release set differs")
    for name, digest in figure_files.items():
        if common.sha256(figures / name) != digest:
            raise common.ReleaseValidationError(f"figure hash differs for {name}")
    return manifest


def validate_paper(
    paper: Path,
    paper_check_path: Path,
    manuscript: Path,
    review: Path,
) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    check = read_json_object(paper_check_path, "paper check")
    if check.get("status") != "PASS" \
            or check.get("scientific_label") != "CONFIRMATION_FAILED" \
            or check.get("style") != "iclr2026_conference" \
            or int(check.get("main_pages", 99)) > int(check.get("max_main_pages", 9)) \
            or check.get("manuscript_sha256") != common.sha256(manuscript) \
            or check.get("review_manifest_sha256") != common.sha256(review / "review_manifest.json") \
            or check.get("pdf_sha256") != common.sha256(paper / "main.pdf") \
            or check.get("llm_usage_statement_present") is not True:
        raise common.ReleaseValidationError("final paper check is stale or failed")
    build = read_json_object(paper / "paper_build_manifest.json", "paper build manifest")
    if check.get("paper_build_manifest_sha256") != common.sha256(
        paper / "paper_build_manifest.json"
    ):
        raise common.ReleaseValidationError("paper check/build manifest binding differs")
    for name, digest in build.get("fragments", {}).items():
        if name not in {"abstract.tex", "appendix.tex", "body.tex", "refs.tex"} \
                or common.sha256(paper / name) != digest:
            raise common.ReleaseValidationError(f"paper fragment differs for {name}")
    figure_names: set[str] = set()
    for name, digest in build.get("figures", {}).items():
        safe_relative(name)
        if common.sha256(paper / "figures" / name) != digest:
            raise common.ReleaseValidationError(f"paper figure differs for {name}")
        figure_names.add(name)
    return check, build, figure_names


def supplement_readme() -> str:
    return """# Anonymous V18 code and result supplement

This curated archive contains only the source files hash-bound by the frozen
V18 protocol, the three V18 contract tests, minimal supporting configuration,
the analysis-rendered anonymous manuscript, checked paper build, and the
redacted review result artifact.

It intentionally excludes Git metadata and remotes, raw data caches, outputs,
logs, W&B directories, checkpoints, rollout arrays, histories, historical
identity-bearing research notes, `.paper-draft`, and private execution receipts.
The frozen source's private W&B defaults and any exact local paths are replaced
with inert anonymous placeholders. For the 22 frozen execution sources,
`MANIFEST.json` records a public hash plus an exact-match receipt when bytes are
unchanged; only the four redacted frozen files retain separate frozen-execution
and public hashes. Other review, release, test, and generated files publish only
their public hashes. Identity-scan receipts contain category and pass status,
never private token values, counts, or unsalted token digests.

The redacted source is suitable for review and code inspection. It cannot resume
the identity-bearing private run or satisfy that run's original source hash
guard. Scientific values are supplied only by `review_artifact/`; the complete
decision remains conjunctive and is `CONFIRMATION_FAILED`.

The V18 runner intentionally requires a clean committed worktree. Git history is
excluded for anonymity; initialize a new, unrelated deterministic anonymous
repository before launching any new experiment from this public copy:

```bash
python tools/bootstrap_anonymous_git.py
```

To check the implementation contracts:

```bash
python -m pytest -q scripts/test_train_lewm_v8_v18.py \
  scripts/test_run_lewm_v8_v18.py scripts/test_analyze_lewm_v8_v18.py
```

To rebuild the LaTeX fragments and PDF, use Pandoc 3.10 and TeX Live 2026:

```bash
python release_tools/paper/build_paper.py \
  --source docs/ICLR.md --paper-dir paper \
  --review-artifact review_artifact --pandoc /path/to/pandoc-3.10/bin/pandoc
PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH" \
python release_tools/paper/check_v18_paper.py --compile \
  --paper-dir paper --manuscript docs/ICLR.md \
  --review-artifact review_artifact --output paper/paper_check.rebuilt.json
```

The checked submission uses the pinned official ICLR 2026 style only as the
available format check. It must be rechecked against the official ICLR 2027
package and final author guide when those materials are available.

Verify all public file hashes before use:

```bash
python tools/verify_supplement.py
```
"""


def requirements_text() -> str:
    return """# Minimal V18 scientific/test dependencies; install PyTorch for your CUDA platform.
torch>=2.0
torchvision>=0.15
numpy
scipy
scikit-learn
tqdm
wandb
dm_control
pytest
"""


def gitignore_text() -> str:
    return """__pycache__/
.pytest_cache/
.venv/
*.py[cod]
*.aux
*.fdb_latexmk
*.fls
*.log
*.out
*.toc
*.pt
*.pth
*.npz
*.wandb
outputs/
logs/
wandb/
paper/paper_check.rebuilt.json
"""


def git_bootstrap_text() -> str:
    return '''#!/usr/bin/env python3
"""Create a deterministic, no-remote Git snapshot for the public supplement."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, env=env,
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def main() -> None:
    if (ROOT / ".git").exists():
        raise SystemExit("refusing: .git already exists")
    run("init", "--initial-branch=anonymous-v18")
    hooks = ROOT / ".git" / "no-hooks"
    hooks.mkdir()
    for key, value in (
        ("user.name", "anonymous"),
        ("user.email", "anonymous"),
        ("commit.gpgsign", "false"),
        ("tag.gpgsign", "false"),
        ("core.hooksPath", str(hooks)),
        ("core.autocrlf", "false"),
        ("core.filemode", "false"),
    ):
        run("config", "--local", key, value)
    run("add", "--all")
    environment = os.environ.copy()
    environment.update({
        "GIT_AUTHOR_NAME": "anonymous",
        "GIT_AUTHOR_EMAIL": "anonymous",
        "GIT_COMMITTER_NAME": "anonymous",
        "GIT_COMMITTER_EMAIL": "anonymous",
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
        "TZ": "UTC",
        "LC_ALL": "C",
    })
    run("commit", "--no-gpg-sign", "-m", "anonymous-v18-public-snapshot", env=environment)
    if run("remote"):
        raise SystemExit("anonymous bootstrap unexpectedly has a remote")
    if run("status", "--porcelain", "--untracked-files=all"):
        raise SystemExit("anonymous bootstrap worktree is not clean")
    print(run("rev-parse", "HEAD"))


if __name__ == "__main__":
    main()
'''


def supplement_verifier_text() -> str:
    return '''#!/usr/bin/env python3
"""Verify the extracted anonymous V18 supplement manifest and file bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest_path = ROOT / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload_digest = manifest.pop("manifest_payload_sha256")
    canonical = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != payload_digest:
        raise SystemExit("manifest payload hash differs")
    records = manifest["files"]
    expected = {row["path"] for row in records} | {"MANIFEST.json"}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(ROOT).parts
    }
    if expected != actual or len(expected) != len(records) + 1:
        raise SystemExit("supplement file set differs")
    folded: set[str] = set()
    for row in records:
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit("unsafe manifest path")
        key = relative.as_posix().casefold()
        if key in folded:
            raise SystemExit("casefold path collision")
        folded.add(key)
        path = ROOT.joinpath(*relative.parts)
        if path.stat().st_size != row["public_size_bytes"]:
            raise SystemExit(f"size differs for {relative}")
        if digest(path) != row["public_sha256"]:
            raise SystemExit(f"hash differs for {relative}")
        if row["archive_mode"] != "100644":
            raise SystemExit(f"mode receipt differs for {relative}")
    print(f"verified {len(records)} public files plus MANIFEST.json")


if __name__ == "__main__":
    main()
'''


def load_forbidden_files(paths: Iterable[Path]) -> tuple[list[str], list[dict[str, Any]]]:
    values: list[str] = []
    categories: set[str] = set()
    for path_value in paths:
        path = path_value.resolve()
        if not path.is_file() or path.is_symlink():
            raise common.ReleaseValidationError("identity-token file is not a regular file")
        if path.stat().st_mode & 0o077:
            raise common.ReleaseValidationError(
                "identity-token file must be mode 0600 (no group/other permissions)"
            )
        payload = common.read_json(path)
        if not isinstance(payload, Mapping):
            raise common.ReleaseValidationError("identity-token file must contain an object")
        for category, raw in sorted(payload.items()):
            if not isinstance(category, str) or not re.fullmatch(r"[a-z0-9_-]+", category):
                raise common.ReleaseValidationError("identity-token category is invalid")
            items = raw if isinstance(raw, list) else [raw]
            if not items or not all(isinstance(item, str) and len(item) >= 3 for item in items):
                raise common.ReleaseValidationError("identity-token value is invalid")
            categories.add(category)
            for item in items:
                values.append(item)
    receipts = [
        {
            "category": category,
            "scan_passed": True,
        }
        for category in sorted(categories)
    ]
    return values, receipts


def validate_public_identity_receipts(manifest: Mapping[str, Any]) -> None:
    """Reject public guess-and-confirm receipts for low-entropy private tokens."""

    banned_keys = {
        "source_value_sha256",
        "value_sha256",
        "private_token_sha256",
        "identity_token_sha256",
        "private_original_sha256",
        "public_redacted_sha256",
        "private_protocol_sha256",
        "private_analysis_sha256",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            overlap = banned_keys & set(value)
            if overlap:
                raise common.ReleaseValidationError(
                    f"public manifest contains forbidden private receipt field(s): {sorted(overlap)}"
                )
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(manifest)
    for field in ("redaction_policy", "additional_forbidden_token_receipts"):
        rows = manifest.get(field)
        if not isinstance(rows, list):
            raise common.ReleaseValidationError(f"public identity receipt list missing: {field}")
        for row in rows:
            required = {"category", "scan_passed"}
            if not isinstance(row, Mapping) \
                    or set(row) != required \
                    or not isinstance(row.get("category"), str) \
                    or row.get("scan_passed") is not True:
                raise common.ReleaseValidationError(
                    f"public identity receipt is not category/scan-passed-only: {field}"
                )


def validate_member_names(names: Iterable[str]) -> None:
    values = list(names)
    normalized = [unicodedata.normalize("NFC", value) for value in values]
    folded = [value.casefold() for value in normalized]
    if len(set(values)) != len(values) \
            or len(set(normalized)) != len(normalized) \
            or len(set(folded)) != len(folded):
        raise common.ReleaseValidationError(
            "duplicate, Unicode-normalization, or casefold ZIP member collision"
        )


def write_deterministic_zip(source: Path, destination: Path) -> None:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    names = [f"{ARCHIVE_ROOT}/{path.relative_to(source).as_posix()}" for path in files]
    validate_member_names(names)
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=False,
        strict_timestamps=True,
    ) as archive:
        for path, name in zip(files, names, strict=True):
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = (0o100644 << 16)
            info.flag_bits = 0x800
            info.extra = b""
            info.comment = b""
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_STORED)


def verify_zip(
    archive_path: Path,
    staging: Path,
    tokens: Iterable[str],
    *,
    pdftotext: str,
) -> None:
    expected = {
        f"{ARCHIVE_ROOT}/{path.relative_to(staging).as_posix()}": common.sha256(path)
        for path in staging.rglob("*") if path.is_file()
    }
    with zipfile.ZipFile(archive_path, "r", allowZip64=False) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        validate_member_names(names)
        if names != sorted(expected) or set(names) != set(expected):
            raise common.ReleaseValidationError("deterministic ZIP member set/order differs")
        if archive.comment:
            raise common.ReleaseValidationError("ZIP archive comment must be empty")
        if archive.testzip() is not None:
            raise common.ReleaseValidationError("ZIP CRC validation failed")
        total_size = 0
        extracted = Path(tempfile.mkdtemp(prefix=".v18-zip-verify-"))
        try:
            for info in infos:
                total_size += info.file_size
                if info.file_size > 32 * 1024 * 1024:
                    raise common.ReleaseValidationError("ZIP member exceeds 32 MiB limit")
                if info.flag_bits & 0x1 or info.extra or info.comment or info.is_dir():
                    raise common.ReleaseValidationError("ZIP member has unsafe metadata")
                if "\\" in info.filename or not info.filename.startswith(f"{ARCHIVE_ROOT}/"):
                    raise common.ReleaseValidationError("ZIP member path is unsafe")
                inner = info.filename.removeprefix(f"{ARCHIVE_ROOT}/")
                scan_name(inner, tokens)
                if info.date_time != ZIP_TIMESTAMP \
                        or info.compress_type != zipfile.ZIP_STORED \
                        or (info.external_attr >> 16) != 0o100644:
                    raise common.ReleaseValidationError(f"ZIP metadata differs for {info.filename}")
                data = archive.read(info)
                digest = hashlib.sha256(data).hexdigest()
                if digest != expected[info.filename]:
                    raise common.ReleaseValidationError(f"ZIP member bytes differ for {info.filename}")
                destination = extracted.joinpath(*PurePosixPath(inner).parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                scan_bytes(destination, inner, tokens, pdftotext=pdftotext)
            if total_size > 256 * 1024 * 1024:
                raise common.ReleaseValidationError("ZIP uncompressed payload exceeds 256 MiB")
        finally:
            shutil.rmtree(extracted, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--review-artifact", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--paper-check", type=Path, required=True)
    parser.add_argument("--release-tool-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--pdftotext", default=shutil.which("pdftotext"))
    parser.add_argument(
        "--forbid", action="append", default=[], metavar="TOKEN",
        help="additional identity token that must not occur; it is not auto-redacted",
    )
    parser.add_argument(
        "--forbid-file", action="append", default=[], type=Path, metavar="MODE_0600_JSON",
        help="mode-0600 JSON object of category -> token(s); values are never published",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pdftotext:
        raise common.ReleaseValidationError("pdftotext is required for PDF identity scanning")
    repo = args.repo_root.resolve()
    result_root = args.result_root.resolve()
    review = args.review_artifact.resolve()
    manuscript = args.manuscript.resolve()
    paper = args.paper_dir.resolve()
    paper_check_path = args.paper_check.resolve()
    tools = args.release_tool_root.resolve()
    output = args.output_dir.resolve()
    output_zip = args.output_zip.resolve()
    sidecar = output_zip.with_suffix(output_zip.suffix + ".sha256")
    for path in (output, output_zip, sidecar):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite supplement output {path}")

    bundle = common.load_complete_bundle(result_root, require_failure=True)
    protocol = bundle["protocol"]
    source_hashes = protocol.get("source_sha256")
    if not isinstance(source_hashes, Mapping) or set(source_hashes) != EXPECTED_FROZEN_SOURCES:
        raise common.ReleaseValidationError("protocol source allowlist differs from frozen V18")
    review_manifest = validate_review_artifact(review, bundle)
    manuscript_manifest = validate_manuscript(manuscript, review_manifest)
    paper_check, paper_build, paper_figure_names = validate_paper(
        paper, paper_check_path, manuscript, review
    )
    policy = replacement_policy(repo, result_root, protocol)
    secret_tokens, secret_receipts = load_forbidden_files(args.forbid_file)
    forbidden = {
        *(item.source for item in policy),
        *args.forbid,
        *secret_tokens,
    } - {""}

    output.parent.mkdir(parents=True, exist_ok=True)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".v18-supplement-", dir=output.parent))
    zip_staging = Path(tempfile.mkdtemp(prefix=".v18-zip-", dir=output_zip.parent))
    records: list[dict[str, Any]] = []
    try:
        for relative in sorted(EXPECTED_FROZEN_SOURCES):
            records.append(copy_with_receipt(
                repo / relative,
                staging,
                relative,
                category="frozen_v18_source",
                source_class="frozen_execution_source",
                policy=policy,
                permit_redaction=True,
                expected_original_sha256=str(source_hashes[relative]),
            ))
        redacted_frozen = {
            row["path"]
            for row in records
            if row["source_class"] == "frozen_execution_source"
            and "frozen_execution_sha256" in row
        }
        if redacted_frozen != EXPECTED_REDACTED_FROZEN_SOURCES:
            raise common.ReleaseValidationError(
                "redacted frozen-source set differs from the four expected files"
            )
        if sum(
            row.get("matches_frozen_execution") is True
            for row in records
            if row["source_class"] == "frozen_execution_source"
        ) != len(EXPECTED_FROZEN_SOURCES) - len(EXPECTED_REDACTED_FROZEN_SOURCES):
            raise common.ReleaseValidationError("unchanged frozen-source match count differs")
        for relative in sorted(V18_TESTS):
            records.append(copy_with_receipt(
                repo / relative, staging, relative,
                category="v18_contract_test",
                source_class="repository_contract_test_source",
                policy=policy,
                permit_redaction=True,
            ))
        for relative in sorted(SUPPORTING_FILES):
            records.append(copy_with_receipt(
                repo / relative, staging, relative,
                category="supporting_configuration",
                source_class="repository_supporting_source",
                policy=policy,
                permit_redaction=True,
            ))
        for relative in sorted(PUBLIC_RELEASE_TOOLS):
            records.append(copy_with_receipt(
                tools / relative,
                staging,
                "release_tools/" + relative,
                category="paper_release_tool",
                source_class="review_safe_release_source",
                policy=policy,
                permit_redaction=False,
            ))

        records.append(copy_with_receipt(
            manuscript, staging, "docs/ICLR.md",
            category="anonymous_manuscript",
            source_class="review_safe_release_artifact",
            policy=policy,
            permit_redaction=False,
        ))
        records.append(copy_with_receipt(
            manuscript.with_suffix(".manifest.json"),
            staging,
            "docs/ICLR.manifest.json",
            category="anonymous_manuscript",
            source_class="review_safe_release_artifact",
            policy=policy,
            permit_redaction=False,
        ))
        source_figures = manuscript.parent / "figures"
        for name in sorted({"fig_v18_manifest.json", *read_json_object(
                source_figures / "fig_v18_manifest.json", "figure manifest"
        )["figures"]}):
            records.append(copy_with_receipt(
                source_figures / name,
                staging,
                f"docs/figures/{name}",
                category="anonymous_figure",
                source_class="review_safe_release_artifact",
                policy=policy,
                permit_redaction=False,
            ))

        for name in sorted(PAPER_FILES):
            records.append(copy_with_receipt(
                paper / name,
                staging,
                f"paper/{name}",
                category="checked_paper",
                source_class="review_safe_release_artifact",
                policy=policy,
                permit_redaction=False,
            ))
        for name in sorted(paper_figure_names):
            records.append(copy_with_receipt(
                paper / "figures" / name,
                staging,
                f"paper/figures/{name}",
                category="checked_paper_figure",
                source_class="review_safe_release_artifact",
                policy=policy,
                permit_redaction=False,
            ))
        records.append(copy_with_receipt(
            paper_check_path,
            staging,
            "paper/paper_check.json",
            category="checked_paper",
            source_class="review_safe_release_artifact",
            policy=policy,
            permit_redaction=False,
        ))

        for name in sorted({"review_manifest.json", *review_manifest["files"]}):
            records.append(copy_with_receipt(
                review / name,
                staging,
                f"review_artifact/{name}",
                category="anonymous_result_artifact",
                source_class="review_safe_release_artifact",
                policy=policy,
                permit_redaction=False,
            ))

        records.append(add_generated(
            staging, "README.md", supplement_readme(), category="supplement_documentation"
        ))
        records.append(add_generated(
            staging,
            "requirements-v18.txt",
            requirements_text(),
            category="supporting_configuration",
        ))
        records.append(add_generated(
            staging, ".gitignore", gitignore_text(), category="supporting_configuration"
        ))
        records.append(add_generated(
            staging,
            "tools/bootstrap_anonymous_git.py",
            git_bootstrap_text(),
            category="supplement_tool",
        ))
        records.append(add_generated(
            staging,
            "tools/verify_supplement.py",
            supplement_verifier_text(),
            category="supplement_tool",
        ))
        records.sort(key=lambda row: row["path"])
        if len({row["path"] for row in records}) != len(records):
            raise common.ReleaseValidationError("duplicate curated supplement destination")

        manifest = {
            "schema_version": 1,
            "scope": "anonymous_v18_code_and_result_supplement",
            "scientific_label": bundle["report"]["scientific_label"],
            "cells": 200,
            "contrasts": 33,
            "archive_contract": {
                "root": ARCHIVE_ROOT,
                "timestamp": list(ZIP_TIMESTAMP),
                "compression": "ZIP_STORED",
                "file_mode": "100644",
                "member_order": "lexicographic",
            },
            "input_bindings": {
                "frozen_protocol_sha256": bundle["hashes"]["confirmation_protocol.json"],
                "analysis_sha256": bundle["hashes"]["confirmation_analysis.json"],
                "review_manifest_sha256": common.sha256(review / "review_manifest.json"),
                "manuscript_manifest_sha256": common.sha256(
                    manuscript.with_suffix(".manifest.json")
                ),
                "paper_build_manifest_sha256": common.sha256(
                    paper / "paper_build_manifest.json"
                ),
                "paper_check_sha256": common.sha256(paper_check_path),
                "paper_pdf_sha256": paper_check["pdf_sha256"],
            },
            "redaction_policy": [item.manifest_value() for item in policy],
            "additional_forbidden_token_receipts": secret_receipts + ([{
                "category": "command_line_forbid",
                "scan_passed": True,
            }] if args.forbid else []),
            "identity_receipt_policy": {
                "private_token_values_published": False,
                "private_token_digests_published": False,
                "receipt_granularity": "category_and_scan_passed_only",
            },
            "excluded_classes": [
                ".git and remotes",
                "raw data caches and outputs",
                "logs and W&B directories",
                "checkpoints, histories, and rollout arrays",
                "historical identity-bearing documents and .paper-draft",
                "private protocol, runs, attempts, summary, and restart receipts",
            ],
            "files": records,
            "file_count_excluding_manifest": len(records),
            "manifest_excludes_its_own_hash": True,
            "toolchain": {
                "python": sys.version.split()[0],
                "zipfile": "stdlib",
            },
            "builder_sha256": common.sha256(Path(__file__).resolve()),
        }
        validate_public_identity_receipts(manifest)
        manifest["manifest_payload_sha256"] = common.json_sha256(manifest)
        common.atomic_write_json(staging / "MANIFEST.json", manifest)

        for path in sorted(item for item in staging.rglob("*") if item.is_file()):
            path.chmod(0o644)
            relative = path.relative_to(staging).as_posix()
            scan_name(relative, forbidden)
            scan_bytes(path, relative, forbidden, pdftotext=args.pdftotext)
            if path.suffix.casefold() == ".py":
                compile(path.read_text(encoding="utf-8"), relative, "exec")

        first_zip = zip_staging / "first.zip"
        second_zip = zip_staging / "second.zip"
        write_deterministic_zip(staging, first_zip)
        write_deterministic_zip(staging, second_zip)
        if common.sha256(first_zip) != common.sha256(second_zip):
            raise common.ReleaseValidationError("two deterministic ZIP builds differ")
        verify_zip(first_zip, staging, forbidden, pdftotext=args.pdftotext)
        scan_output_name(output_zip.name, forbidden)

        os.replace(staging, output)
        os.replace(first_zip, output_zip)
        output_zip.chmod(0o644)
        zip_digest = common.sha256(output_zip)
        common.atomic_write_text(sidecar, f"{zip_digest}  {output_zip.name}\n")
        sidecar.chmod(0o644)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(zip_staging, ignore_errors=True)

    print(json.dumps({
        "output_dir": str(output),
        "output_zip": str(output_zip),
        "zip_sha256": common.sha256(output_zip),
        "sha256_sidecar": str(sidecar),
        "files_excluding_manifest": len(records),
        "source_redactions": sum(
            sum(item["occurrences"] for item in row["redactions"])
            for row in records
        ),
        "scientific_label": bundle["report"]["scientific_label"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
