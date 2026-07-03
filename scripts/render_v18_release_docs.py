#!/usr/bin/env python3
"""Deterministically render the V18 README and evidence-record patch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import v18_release_common as common


def pct(value: Any) -> str:
    return f"{100.0 * float(value):+.2f}%"


def ci(contrast: Mapping[str, Any]) -> str:
    return (
        f"[{pct(contrast['bootstrap']['ci95_low'])}, "
        f"{pct(contrast['bootstrap']['ci95_high'])}]"
    )


def gate(value: bool) -> str:
    return "PASS" if value else "FAIL"


def scalar(value: Any) -> str:
    number = float(value)
    if number == 0.0:
        return "0"
    if abs(number) < 1e-3 or abs(number) >= 1e3:
        return f"{number:.3e}"
    return f"{number:.4f}"


def values(bundle: Mapping[str, Any], pdf_check: Mapping[str, Any] | None) -> dict[str, str]:
    report = bundle["report"]
    contrasts = {
        name: report["contrasts"][key]
        for name, key in common.CONTRAST_KEYS.items()
    }
    gates = {
        name: bool(report["gates"][key])
        for name, key in common.GATE_KEYS.items()
    }
    rendered: dict[str, str] = {}
    for name in common.CONTRAST_KEYS:
        contrast = contrasts[name]
        rendered.update({
            f"{name}_MEAN": pct(contrast["mean_paired_relative_reduction"]),
            f"{name}_CI": ci(contrast),
            f"{name}_WINS": str(int(contrast["paired_wins"])),
            f"{name}_TASKS": str(int(contrast["task_mean_wins"])),
            f"{name}_GATE": gate(gates[name]),
        })
    representation = report["representation"]["observed"]
    convergence = report["convergence"]["observed"]
    passed = bool(report["official_confirmation_result"])
    if passed:
        interpretation = (
            "STABILIZED_LEWM_V8_CONFIRMATION_PASS: all registered integrity, "
            "superiority, intervention, noninferiority, clean-state, representation, "
            "and convergence clauses pass."
        )
        decision = (
            "The complete registered conjunction passes and confirms SAS-PC/V8 under "
            "the frozen host, cohort, endpoints, and claim boundary."
        )
        framing = (
            "The registered conjunction confirms SAS-PC/V8 only within the frozen "
            "VICReg-trained LeWM-derived host and evaluation boundary."
        )
    else:
        interpretation = (
            "CONFIRMATION_FAILED: at least one registered conjunctive clause fails, so "
            "SAS-PC/V8 is not confirmed as a generally superior persistent carrier; "
            "favorable individual comparisons remain descriptive and cannot rescue it."
        )
        decision = (
            "The registered conjunction fails, so SAS-PC/V8 is not confirmed as a "
            "generally superior persistent carrier; favorable individual contrasts "
            "cannot rescue the frozen decision."
        )
        framing = (
            "The submission is a complete frozen falsification: SAS-PC/V8 is not "
            "confirmed as a generally superior persistent carrier, and favorable "
            "subsets do not override the failed conjunction."
        )
    finished = str(bundle["summary"].get("finished_at", ""))
    if len(finished) < 10:
        raise common.ReleaseValidationError("summary lacks a final ISO date")
    rendered.update({
        "V18_STATUS": str(report["status"]),
        "V18_SCIENTIFIC_LABEL": str(report["scientific_label"]),
        "V18_OFFICIAL_CONFIRMATION_RESULT": str(passed).lower(),
        "V18_COMPLETED_VALID_CELLS": str(report["completed_valid_cells"]),
        "V18_ARTIFACT_INTEGRITY": gate(bool(report["artifact_integrity_passed"])),
        "V18_MIN_VARIANCE": scalar(representation["minimum_channel_variance"]),
        "V18_MIN_RANK": f"{float(representation['minimum_effective_rank']):.2f}",
        "V18_VARIANCE_PASSING_CELLS": str(representation["variance_passing_cells"]),
        "V18_RANK_PASSING_CELLS": str(representation["rank_passing_cells"]),
        "V18_REPRESENTATION_GATE": gate(report["gates"]["healthy_representation"]),
        "V18_MAX_LATE_CHANGE": pct(convergence["maximum_absolute_relative_change"]),
        "V18_CONVERGED_CELLS": str(convergence["passing_cells"]),
        "V18_CONVERGENCE_GATE": gate(report["gates"]["convergence"]),
        "V18_ANALYSIS_SHA256": bundle["hashes"]["confirmation_analysis.json"],
        "V18_CELLS_SHA256": bundle["hashes"]["confirmation_cells.csv"],
        "V18_CONTRASTS_SHA256": bundle["hashes"]["confirmation_contrasts.csv"],
        "V18_RESULT_INTERPRETATION": interpretation,
        "V18_DECISION_SENTENCE": decision,
        "V18_SUBMISSION_FRAMING": framing,
        "V18_FINAL_DATE": finished[:10],
    })
    if pdf_check is not None:
        if pdf_check.get("status") != "PASS" \
                or pdf_check.get("style") != "iclr2026_conference" \
                or pdf_check.get("llm_usage_statement_present") is not True:
            raise common.ReleaseValidationError("PDF check did not pass official ICLR 2026 style")
        rendered.update({
            "V18_PDF_TOTAL_PAGES": str(pdf_check["total_pages"]),
            "V18_PDF_MAIN_PAGES": str(pdf_check["main_pages"]),
            "V18_RELEASE_STATUS": (
                "FORMAT_CHECK_COMPLETE_UNDER_OFFICIAL_ICLR_2026_STYLE; "
                "SUBMISSION_BLOCKED_PENDING_OFFICIAL_ICLR_2027_TEMPLATE_AND_FINAL_AUTHOR_GUIDE"
            ),
        })
    return rendered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--readme-template", type=Path)
    parser.add_argument("--readme-output", type=Path)
    parser.add_argument("--evidence-template", type=Path)
    parser.add_argument("--evidence-output", type=Path)
    parser.add_argument("--pdf-check", type=Path)
    parser.add_argument("--manifest-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = (
        (args.readme_template, args.readme_output, "README"),
        (args.evidence_template, args.evidence_output, "evidence patch"),
    )
    if not any(template and output for template, output, _ in pairs):
        raise SystemExit("at least one matching template/output pair is required")
    if any(bool(template) != bool(output) for template, output, _ in pairs):
        raise SystemExit("each supplied template requires its matching output")
    bundle = common.load_complete_bundle(args.root.resolve())
    pdf_check = common.read_json(args.pdf_check.resolve()) if args.pdf_check else None
    if args.evidence_template and pdf_check is None:
        raise SystemExit("evidence rendering requires --pdf-check for page-bound placeholders")
    substitutions = values(bundle, pdf_check)
    outputs: dict[str, str] = {}
    templates: dict[str, str] = {}
    for template_arg, output_arg, label in pairs:
        if not template_arg:
            continue
        template = template_arg.resolve()
        output = output_arg.resolve()
        text = common.render_template(
            template.read_text(encoding="utf-8"), substitutions, label=label
        )
        common.atomic_write_text(output, text)
        outputs[output.name] = common.sha256(output)
        templates[template.name] = common.sha256(template)
    manifest = {
        "schema_version": 1,
        "scope": "v18_deterministic_release_docs",
        "scientific_label": bundle["report"]["scientific_label"],
        "source_result_hashes": bundle["hashes"],
        "templates": templates,
        "outputs": outputs,
        "pdf_check_sha256": common.sha256(args.pdf_check.resolve()) if args.pdf_check else None,
        "renderer_sha256": common.sha256(Path(__file__).resolve()),
        "common_validator_sha256": common.sha256(Path(common.__file__).resolve()),
    }
    common.atomic_write_json(args.manifest_output.resolve(), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
