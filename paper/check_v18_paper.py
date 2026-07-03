#!/usr/bin/env python3
"""Compile and fail-closed check the anonymous V18 ICLR paper."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v18_release_common as common

OFFICIAL_ICLR2026_SHA256 = (
    "a4852f68e080d6c5245057ca2039100b409e31727898aa93c03d78ddb84374a3"
)
LLM_USAGE_STATEMENT = (
    "OpenAI Codex assisted with code review, experiment monitoring, artifact "
    "auditing, deterministic result-to-manuscript tooling, and manuscript "
    "drafting/editing. The authors verified the executed code, artifacts, "
    "statistics, citations, and final claims and retain responsibility for the work."
)
GENERATED_TEX = (
    "main.tex",
    "abstract.tex",
    "body.tex",
    "refs.tex",
    "appendix.tex",
)


def command_version(command: str, *arguments: str) -> str:
    result = subprocess.run(
        [command, *arguments], capture_output=True, text=True, check=True
    )
    lines = (result.stdout or result.stderr).splitlines()
    return lines[0] if lines else "unknown"


def run_compile(paper_dir: Path, latexmk: str) -> str:
    environment = os.environ.copy()
    environment.setdefault("SOURCE_DATE_EPOCH", "1783036800")  # 2026-07-03 UTC
    command = [
        latexmk,
        "-pdf",
        "-halt-on-error",
        "-interaction=nonstopmode",
        "-file-line-error",
        "main.tex",
    ]
    result = subprocess.run(
        command,
        cwd=paper_dir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    transcript = result.stdout + "\n" + result.stderr
    common.atomic_write_text(paper_dir / "latexmk.stdout.log", transcript)
    if result.returncode:
        raise common.ReleaseValidationError(
            f"latexmk failed with exit {result.returncode}; see latexmk.stdout.log"
        )
    return transcript


def parse_pdfinfo(pdfinfo: str, pdf: Path) -> dict[str, str]:
    result = subprocess.run(
        [pdfinfo, str(pdf)], capture_output=True, text=True, check=True
    )
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def pdf_text(pdftotext: str, pdf: Path) -> str:
    result = subprocess.run(
        [pdftotext, "-layout", str(pdf), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def pdf_text_flow(pdftotext: str, pdf: Path) -> str:
    result = subprocess.run(
        [pdftotext, str(pdf), "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def validate_build_manifest(paper_dir: Path, manuscript: Path, review: Path) -> dict[str, Any]:
    path = paper_dir / "paper_build_manifest.json"
    if not path.is_file():
        raise common.ReleaseValidationError("paper fragment build manifest is absent")
    manifest = common.read_json(path)
    if not isinstance(manifest, dict) or manifest.get("scope") != "v18_paper_fragments":
        raise common.ReleaseValidationError("paper fragment manifest is malformed")
    manuscript_manifest = common.read_json(manuscript.with_suffix(".manifest.json"))
    if not isinstance(manuscript_manifest, dict) \
            or manuscript_manifest.get("schema_version") != 2 \
            or manuscript_manifest.get("scientific_label") != "CONFIRMATION_FAILED" \
            or manuscript_manifest.get("restart_interruptions") != 2 \
            or manuscript_manifest.get("telemetry_disclosure_present") is not True \
            or manuscript_manifest.get("llm_usage_statement_present") is not True:
        raise common.ReleaseValidationError(
            "manuscript manifest lacks falsification/restart/telemetry/LLM receipts"
        )
    for key, expected in {
        "renderer_sha256": common.sha256(ROOT / "scripts" / "render_v18_paper.py"),
        "common_validator_sha256": common.sha256(ROOT / "scripts" / "v18_release_common.py"),
    }.items():
        if manuscript_manifest.get(key) != expected:
            raise common.ReleaseValidationError(f"manuscript renderer provenance differs for {key}")
    checks = {
        "manuscript_sha256": common.sha256(manuscript),
        "manuscript_manifest_sha256": common.sha256(
            manuscript.with_suffix(".manifest.json")
        ),
        "review_manifest_sha256": common.sha256(review / "review_manifest.json"),
        "main_tex_sha256": common.sha256(paper_dir / "main.tex"),
        "style_sha256": common.sha256(paper_dir / "iclr2026_conference.sty"),
        "builder_sha256": common.sha256(Path(__file__).with_name("build_paper.py")),
        "llm_usage_statement_present": True,
        "source_figure_manifest_sha256": common.sha256(
            manuscript.parent / "figures" / "fig_v18_manifest.json"
        ),
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise common.ReleaseValidationError(f"paper build manifest differs for {key}")
    if manuscript_manifest.get("figure_manifest_sha256") != checks[
            "source_figure_manifest_sha256"]:
        raise common.ReleaseValidationError("manuscript/figure manifest binding differs")
    for name, expected in manifest.get("fragments", {}).items():
        path = paper_dir / name
        if not path.is_file() or common.sha256(path) != expected:
            raise common.ReleaseValidationError(f"paper fragment hash differs for {name}")
    for name, expected in manifest.get("figures", {}).items():
        path = paper_dir / "figures" / name
        if not path.is_file() or common.sha256(path) != expected:
            raise common.ReleaseValidationError(f"paper figure hash differs for {name}")
    return manifest


def validate_review_manifest(review: Path) -> dict[str, Any]:
    manifest_path = review / "review_manifest.json"
    manifest = common.read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
        raise common.ReleaseValidationError("review manifest is not schema v2")
    if manifest.get("scientific_label") != "CONFIRMATION_FAILED" \
            or manifest.get("cells") != 200 or manifest.get("contrasts") != 33:
        raise common.ReleaseValidationError("review manifest is not the complete falsification")
    if manifest.get("restart_schema_version") != 2 \
            or manifest.get("restart_interruptions") != 2:
        raise common.ReleaseValidationError("review bundle lacks v2 two-restart provenance")
    expected_tool_hashes = {
        "builder_sha256": common.sha256(ROOT / "scripts" / "build_v18_review_artifact.py"),
        "common_validator_sha256": common.sha256(ROOT / "scripts" / "v18_release_common.py"),
    }
    for key, expected in expected_tool_hashes.items():
        if manifest.get(key) != expected:
            raise common.ReleaseValidationError(f"review builder provenance differs for {key}")
    expected_names = set(manifest.get("files", {}))
    actual_names = {
        path.name for path in review.iterdir()
        if path.is_file() and path.name != "review_manifest.json"
    }
    if expected_names != actual_names:
        raise common.ReleaseValidationError("review manifest file set differs")
    for name, digest in manifest["files"].items():
        common.require_hash(digest, f"review artifact {name}")
        if common.sha256(review / name) != digest:
            raise common.ReleaseValidationError(f"review artifact hash differs for {name}")
    return manifest


def parse_main_page(aux: str) -> int:
    matches = re.findall(
        r"\\newlabel\{v18-main-end\}\{\{[^}]*\}\{([0-9]+)\}", aux
    )
    if len(matches) != 1:
        raise common.ReleaseValidationError(
            f"expected one machine-readable main-page marker, found {matches}"
        )
    return int(matches[0])


def validate_main_figure_pages(aux: str, main_page: int) -> dict[str, int]:
    expected = {
        "fig:fig-v18-architecture",
        "fig:fig-v18-evidence",
        "fig:fig-v18-secondary",
    }
    observed: dict[str, int] = {}
    for label in expected:
        matches = re.findall(
            r"\\newlabel\{" + re.escape(label)
            + r"\}\{\{[^}]*\}\{([0-9]+)\}", aux
        )
        if len(matches) != 1:
            raise common.ReleaseValidationError(
                f"expected one page receipt for main figure {label}, found {matches}"
            )
        observed[label] = int(matches[0])
        if observed[label] > main_page:
            raise common.ReleaseValidationError(
                f"main-text figure {label} floated after the main-page marker"
            )
    return observed


def warning_checks(log: str, max_overfull: float) -> tuple[float, int]:
    fatal_patterns = (
        r"LaTeX Warning: Citation .* undefined",
        r"Package natbib Warning: Citation .* undefined",
        r"There were undefined references",
        r"Reference .* undefined",
        r"Label\(s\) may have changed",
        r"Rerun to get cross-references right",
        r"multiply defined",
        r"Emergency stop",
        r"Fatal error occurred",
    )
    problems = [pattern for pattern in fatal_patterns if re.search(pattern, log, re.I)]
    if problems:
        raise common.ReleaseValidationError(f"LaTeX log has unresolved warnings: {problems}")
    values = [
        float(value)
        for value in re.findall(
            r"Overfull \\[hv]box \(([0-9]+(?:\.[0-9]+)?)pt too (?:wide|high)\)", log
        )
    ]
    maximum = max(values, default=0.0)
    if maximum > max_overfull:
        raise common.ReleaseValidationError(
            f"serious overfull box: {maximum:.3f}pt exceeds {max_overfull:.3f}pt"
        )
    return maximum, len(values)


def citation_checks(aux: str, log: str, pdf: str) -> tuple[int, int]:
    citation_groups = re.findall(r"\\citation\{([^}]*)\}", aux)
    cited = {
        key.strip() for group in citation_groups for key in group.split(",") if key.strip()
    }
    bibitems = set(re.findall(r"\\bibcite\{([^}]*)\}", aux))
    if not cited or cited != bibitems or len(cited) != 18:
        raise common.ReleaseValidationError(
            "citation/bibliography mismatch: "
            f"cited={len(cited)}, bibliography={len(bibitems)}, "
            f"missing={sorted(cited - bibitems)}, uncited={sorted(bibitems - cited)}"
        )
    if "Package natbib" not in log and "natbib.sty" not in log:
        raise common.ReleaseValidationError("LaTeX log does not show natbib citation handling")
    if re.search(r"\[[0-9]+(?:\s*[-,;]\s*[0-9]+)*\]", pdf):
        raise common.ReleaseValidationError("PDF appears to contain numeric bracket citations")
    if not re.search(r"\([A-Z][A-Za-zÀ-ž.-]+(?: et al\.)?,?\s+(?:19|20)\d{2}", pdf):
        raise common.ReleaseValidationError("PDF lacks an author-year citation receipt")
    return len(cited), len(bibitems)


def identity_checks(
    manuscript: Path,
    review: Path,
    tex_paths: list[Path],
    pdf: str,
    info: dict[str, str],
    extra: list[str],
) -> None:
    tokens = {
        "/home/",
        "wandb.ai/",
        *extra,
    }
    textual = [manuscript, *tex_paths]
    textual.extend(path for path in review.iterdir() if path.is_file())
    common.scan_forbidden(textual, tokens)
    email_pattern = re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I
    )
    for path in textual:
        if email_pattern.search(path.read_text(encoding="utf-8", errors="strict")):
            raise common.ReleaseValidationError(f"email address appears in {path}")
    combined = pdf + "\n" + "\n".join(f"{key}: {value}" for key, value in info.items())
    leaks = [token for token in tokens if token and token.casefold() in combined.casefold()]
    if leaks:
        raise common.ReleaseValidationError(f"identity leak in PDF or metadata: {leaks}")
    if email_pattern.search(combined):
        raise common.ReleaseValidationError("email address appears in PDF or metadata")
    if info.get("Author", "").strip() not in {"", "Anonymous authors"}:
        raise common.ReleaseValidationError(f"identity-bearing PDF Author: {info['Author']!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--review-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-main-pages", type=int, default=9)
    parser.add_argument("--max-overfull-pt", type=float, default=2.0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--latexmk", default=shutil.which("latexmk"))
    parser.add_argument("--pdfinfo", default=shutil.which("pdfinfo"))
    parser.add_argument("--pdftotext", default=shutil.which("pdftotext"))
    parser.add_argument("--forbid", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paper_dir = args.paper_dir.resolve()
    manuscript = args.manuscript.resolve()
    review = args.review_artifact.resolve()
    output = args.output.resolve()
    for command, label in (
        (args.pdfinfo, "pdfinfo"),
        (args.pdftotext, "pdftotext"),
    ):
        if not command:
            raise common.ReleaseValidationError(f"{label} is unavailable")
    review_manifest = validate_review_manifest(review)
    build_manifest = validate_build_manifest(paper_dir, manuscript, review)
    style = paper_dir / "iclr2026_conference.sty"
    if common.sha256(style) != OFFICIAL_ICLR2026_SHA256:
        raise common.ReleaseValidationError(
            "ICLR style bytes are not the pinned official 2026 style"
        )
    if args.compile:
        if not args.latexmk:
            raise common.ReleaseValidationError("latexmk is unavailable")
        run_compile(paper_dir, args.latexmk)

    required = [paper_dir / name for name in ("main.pdf", "main.log", "main.aux", "main.fls")]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise common.ReleaseValidationError(f"paper outputs are absent or empty: {missing}")
    pdf_path, log_path, aux_path, fls_path = required
    log = log_path.read_text(encoding="utf-8", errors="replace")
    aux = aux_path.read_text(encoding="utf-8", errors="replace")
    fls = fls_path.read_text(encoding="utf-8", errors="replace")
    if "iclr2026_conference.sty" not in fls or "iclr2027_conference.sty" in fls:
        raise common.ReleaseValidationError("recorder does not prove use of ICLR 2026 style")
    local_style_lines = [
        line.removeprefix("INPUT ") for line in fls.splitlines()
        if line.startswith("INPUT ") and line.endswith("iclr2026_conference.sty")
    ]
    def recorder_path(value: str) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else paper_dir / path).resolve()
    if not local_style_lines or not any(recorder_path(line) == style for line in local_style_lines):
        raise common.ReleaseValidationError("TeX loaded a nonlocal ICLR style file")

    maximum_overfull, overfull_count = warning_checks(log, args.max_overfull_pt)
    info = parse_pdfinfo(args.pdfinfo, pdf_path)
    try:
        total_pages = int(info["Pages"])
    except (KeyError, TypeError, ValueError) as exc:
        raise common.ReleaseValidationError("pdfinfo did not report a valid page count") from exc
    main_pages = parse_main_page(aux)
    if not 1 <= main_pages <= args.max_main_pages:
        raise common.ReleaseValidationError(
            f"main text is {main_pages} pages; limit is {args.max_main_pages}"
        )
    if total_pages < main_pages:
        raise common.ReleaseValidationError("total PDF pages are fewer than main pages")
    main_figure_pages = validate_main_figure_pages(aux, main_pages)
    extracted = pdf_text(args.pdftotext, pdf_path)
    extracted_flow = pdf_text_flow(args.pdftotext, pdf_path)
    cited, bibitems = citation_checks(aux, log, extracted)

    normalize = lambda value: re.sub(r"\s+", " ", value).strip()
    manuscript_text = manuscript.read_text(encoding="utf-8", errors="strict")
    appendix_text = (paper_dir / "appendix.tex").read_text(
        encoding="utf-8", errors="strict"
    )
    required_llm = normalize(LLM_USAGE_STATEMENT)
    if manuscript_text.count(LLM_USAGE_STATEMENT) != 1 \
            or required_llm not in normalize(appendix_text) \
            or required_llm not in normalize(extracted_flow):
        raise common.ReleaseValidationError(
            "required LLM Usage Statement is absent from manuscript, appendix, or PDF"
        )

    tex_paths = [paper_dir / name for name in GENERATED_TEX]
    for path in [manuscript, *tex_paths]:
        text = path.read_text(encoding="utf-8", errors="strict")
        leftovers = common.PLACEHOLDER.findall(text)
        if leftovers:
            raise common.ReleaseValidationError(f"unrendered placeholders in {path}: {leftovers}")
    if common.PLACEHOLDER.search(extracted):
        raise common.ReleaseValidationError("unrendered placeholder appears in PDF")
    if any(character in manuscript.read_text(encoding="utf-8") for character in "“”‘’"):
        raise common.ReleaseValidationError("manuscript contains unnormalized smart quotes")
    identity_checks(manuscript, review, tex_paths, extracted, info, args.forbid)

    manifest = {
        "schema_version": 1,
        "scope": "v18_final_paper_check",
        "status": "PASS",
        "style": "iclr2026_conference",
        "style_sha256": common.sha256(style),
        "official_style_sha256": OFFICIAL_ICLR2026_SHA256,
        "main_pages": main_pages,
        "max_main_pages": args.max_main_pages,
        "total_pages": total_pages,
        "main_figure_pages": main_figure_pages,
        "overfull_boxes": overfull_count,
        "maximum_overfull_pt": maximum_overfull,
        "maximum_allowed_overfull_pt": args.max_overfull_pt,
        "cited_keys": cited,
        "bibliography_entries": bibitems,
        "scientific_label": review_manifest["scientific_label"],
        "restart_schema_version": review_manifest["restart_schema_version"],
        "restart_interruptions": review_manifest["restart_interruptions"],
        "llm_usage_statement_present": True,
        "manuscript_sha256": common.sha256(manuscript),
        "manuscript_manifest_sha256": common.sha256(manuscript.with_suffix(".manifest.json")),
        "review_manifest_sha256": common.sha256(review / "review_manifest.json"),
        "paper_build_manifest_sha256": common.sha256(paper_dir / "paper_build_manifest.json"),
        "pdf_sha256": common.sha256(pdf_path),
        "log_sha256": common.sha256(log_path),
        "aux_sha256": common.sha256(aux_path),
        "checker_sha256": common.sha256(Path(__file__).resolve()),
        "tools": {
            "latexmk": command_version(args.latexmk, "-v") if args.latexmk else None,
            "pdfinfo": command_version(args.pdfinfo, "-v"),
            "pdftotext": command_version(args.pdftotext, "-v"),
        },
        "pdf_metadata": {
            key: info.get(key, "")
            for key in ("Title", "Subject", "Author", "Creator", "Producer")
        },
    }
    common.atomic_write_json(output, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
