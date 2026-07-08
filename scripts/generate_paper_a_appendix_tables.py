#!/usr/bin/env python3
"""Generate receipt-bound appendix tables and a main-text claim ledger.

The two independent post-lock audit receipts are the trust boundary.  No
matched-color, DINO-WM PushT, or PointMaze summary is hashed or parsed until
both receipts exist, are complete, and agree on every overlapping summary
identity.  Without ``--execute`` this script is read-only and prints a plan.
With ``--execute`` it atomically creates one caller-selected output directory
outside all experiment roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_SCRIPT = Path(__file__).resolve().relative_to(ROOT)
DEFAULT_COMPLETION_RECEIPT = Path(
    "outputs/paper_a_cross_wave_completion/receipt.json")
DEFAULT_STATISTICS_RECEIPT = Path(
    "outputs/paper_a_statistics_independent/receipt.json")

MATCHED_SUMMARY = Path("outputs/paper_a_matched_color_v1_1/summary.json")
PUSHT_SUMMARY = Path(
    "outputs/dinowm_wave2_spatial_carrier_v1_1/formal/summary.json")
POINTMAZE_SUMMARY = Path(
    "outputs/dinowm_pointmaze_wave3/formal/summary.json")
POINTMAZE_CARRIER = Path(
    "outputs/dinowm_pointmaze_wave3/formal/carrier_summary.json")
POINTMAZE_USE = Path(
    "outputs/dinowm_pointmaze_wave3/formal/external_use_summary.json")

EXPERIMENT_ROOTS = (
    Path("outputs/paper_a_matched_color_v1_1"),
    Path("outputs/dinowm_wave2_spatial_carrier_v1_1"),
    Path("outputs/dinowm_pointmaze_wave3"),
)

ARMS = ("none", "gru", "lstm", "ssm", "fixed_trust")
AGES = (4, 8, 15)
PARAMETERS = {
    "none": 0,
    "gru": 298_368,
    "lstm": 299_632,
    "ssm": 299_520,
    "fixed_trust": 299_520,
}
ARM_LABELS = {
    "none": "No state",
    "gru": "GRU",
    "lstm": "LSTM",
    "ssm": "State-space",
    "fixed_trust": "Fixed-trust",
}
TASK_LABELS = {
    "transient-visual-token-recall": "Token recall",
    "multi-item-visual-binding-recall": "Binding recall",
}
APPENDIX_TABLE_FILENAMES = (
    "matched_color_results.tex",
    "dinowm_pusht_results.tex",
    "pointmaze_carrier_results.tex",
    "pointmaze_gate_checklist.tex",
    "pointmaze_external_use_results.tex",
)
APPENDIX_TABLE_COMMANDS = {
    "matched_color_results.tex": "PaperMatchedColorResults",
    "dinowm_pusht_results.tex": "PaperDinoPushTResults",
    "pointmaze_carrier_results.tex": "PaperPointMazeCarrierResults",
    "pointmaze_gate_checklist.tex": "PaperPointMazeGateChecklist",
    "pointmaze_external_use_results.tex": "PaperPointMazeExternalUseResults",
}
MAIN_CLAIM_LEDGER_FILENAME = "main_claim_ledger.tex"
APPENDIX_BUNDLE_FILENAME = "all_tables.tex"


class GenerationFailure(RuntimeError):
    """A completed receipt, source, or table contract was invalid."""


class NotReady(GenerationFailure):
    """Both independent audit receipts do not yet exist."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GenerationFailure(message)


def stable_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def repository_path(root: Path, value: str | Path) -> Path:
    base = root.resolve()
    candidate = Path(value)
    result = candidate.resolve() if candidate.is_absolute() \
        else (base / candidate).resolve()
    try:
        result.relative_to(base)
    except ValueError as error:
        raise GenerationFailure(f"path leaves repository: {value}") from error
    return result


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def sha256_file(path: Path) -> str:
    require(path.is_file(), f"missing file: {path}")
    before = path.stat()
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            f"file changed while hashing: {path}")
    return value.hexdigest()


def generator_identity(root: Path) -> dict[str, Any]:
    """Bind generation to the repository's current table-generator source."""

    root = root.resolve()
    path = repository_path(root, GENERATOR_SCRIPT)
    require(path.is_file(), f"missing table-generator script: {path}")
    return {
        "path": str(path.relative_to(root)),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise GenerationFailure(f"invalid {label}: {path}: {error}") from error
    require(isinstance(value, dict), f"{label} is not a mapping: {path}")
    return value


def _digest(value: Any, label: str) -> str:
    require(isinstance(value, str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", value)),
            f"{label} is not a SHA-256 digest")
    return value


def _number(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)), f"{label} is not finite numeric")
    return float(value)


def _stat(record: Any, label: str,
          *, interval: str = "ci95") -> Mapping[str, Any]:
    require(isinstance(record, Mapping), f"{label} is not a statistic")
    mean = _number(record.get("mean"), f"{label}.mean")
    ci = record.get(interval)
    require(isinstance(ci, list) and len(ci) == 2,
            f"{label}.{interval} is not a two-sided interval")
    low = _number(ci[0], f"{label}.{interval}[0]")
    high = _number(ci[1], f"{label}.{interval}[1]")
    require(low <= high, f"{label}.{interval} is reversed")
    # Bootstrap point estimates may lie a few ulps outside percentile bounds,
    # so validate the record but do not impose mean-within-CI.
    del mean
    return record


def _protected_roots(root: Path) -> tuple[Path, ...]:
    return tuple(repository_path(root, value) for value in EXPERIMENT_ROOTS)


def _require_external_path(root: Path, path: Path, label: str) -> Path:
    resolved = repository_path(root, path)
    require(not any(_is_within(resolved, protected)
                    for protected in _protected_roots(root)),
            f"{label} must be outside experiment roots: {resolved}")
    return resolved


def _receipt_hashes(completion: Mapping[str, Any],
                    statistics: Mapping[str, Any]) -> dict[str, str]:
    require(completion.get("schema") ==
            "paper_a_cross_wave_completion_receipt_v1"
            and completion.get("status") == "complete"
            and completion.get("read_only_verification") is True
            and completion.get("scientific_cross_wave_aggregation") is False
            and completion.get("sealed_locks_modified") is False
            and completion.get("paper_files_modified") is False,
            "cross-study completion receipt is not complete/read-only")
    require(statistics.get("schema") ==
            "paper_a_statistics_independent_receipt_v1"
            and statistics.get("status") == "verified"
            and statistics.get("read_only") is True
            and statistics.get("statistics_computed") is True
            and statistics.get("scientific_cross_family_pooling") is False
            and statistics.get("imports_producer_statistics") is False
            and statistics.get("experiment_roots_modified") is False,
            "independent statistics receipt is not verified/read-only")

    completion_waves = completion.get("waves")
    statistics_waves = statistics.get("waves")
    require(isinstance(completion_waves, Mapping)
            and set(completion_waves) == {"wave1_1", "wave2_v1_1", "wave3"}
            and isinstance(statistics_waves, Mapping)
            and set(statistics_waves) == {"wave2", "wave3"},
            "independent receipt study set differs")
    require(all(isinstance(value, Mapping)
                and value.get("status") == "verified"
                for value in completion_waves.values())
            and all(isinstance(value, Mapping)
                    and value.get("status") == "verified"
                    for value in statistics_waves.values()),
            "an independently audited study is not verified")

    hashes = {
        "matched": _digest(
            completion_waves["wave1_1"].get("summary_sha256"),
            "matched-color summary receipt"),
        "pusht": _digest(
            completion_waves["wave2_v1_1"].get("summary_sha256"),
            "DINO-WM PushT summary receipt"),
        "pointmaze": _digest(
            completion_waves["wave3"].get("summary_sha256"),
            "PointMaze summary receipt"),
        "pointmaze_carrier": _digest(
            completion_waves["wave3"].get("carrier_summary_sha256"),
            "PointMaze carrier receipt"),
        "pointmaze_use": _digest(
            completion_waves["wave3"].get("external_use_summary_sha256"),
            "PointMaze external-use receipt"),
    }
    overlaps = {
        "pusht": statistics_waves["wave2"].get("summary_sha256"),
        "pointmaze": statistics_waves["wave3"].get(
            "combined_summary_sha256"),
        "pointmaze_carrier": statistics_waves["wave3"].get(
            "carrier_summary_sha256"),
        "pointmaze_use": statistics_waves["wave3"].get(
            "external_use_summary_sha256"),
    }
    for name, value in overlaps.items():
        require(_digest(value, f"statistics receipt {name}") == hashes[name],
                f"independent receipts disagree on {name} summary")
    return hashes


def load_verified_summaries(
        root: Path, completion_receipt: Path,
        statistics_receipt: Path) \
        -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Read receipts first; only then hash and parse completed summaries."""

    root = root.resolve()
    completion_path = _require_external_path(
        root, completion_receipt, "completion receipt")
    statistics_path = _require_external_path(
        root, statistics_receipt, "statistics receipt")
    missing = [str(path.relative_to(root)) for path in
               (completion_path, statistics_path) if not path.is_file()]
    if missing:
        raise NotReady("missing independent audit receipts: "
                       + ", ".join(sorted(missing)))

    completion = read_json(completion_path, "completion receipt")
    statistics = read_json(statistics_path, "statistics receipt")
    expected_hashes = _receipt_hashes(completion, statistics)
    paths = {
        "matched": repository_path(root, MATCHED_SUMMARY),
        "pusht": repository_path(root, PUSHT_SUMMARY),
        "pointmaze": repository_path(root, POINTMAZE_SUMMARY),
        "pointmaze_carrier": repository_path(root, POINTMAZE_CARRIER),
        "pointmaze_use": repository_path(root, POINTMAZE_USE),
    }
    actual_hashes: dict[str, str] = {}
    for name, path in paths.items():
        actual_hashes[name] = sha256_file(path)
        require(actual_hashes[name] == expected_hashes[name],
                f"{name} summary hash differs from both audit receipts")

    summaries = {
        name: read_json(path, f"verified {name} summary")
        for name, path in paths.items()
    }
    identities = {
        "completion_receipt": {
            "path": str(completion_path.relative_to(root)),
            "sha256": sha256_file(completion_path),
        },
        "statistics_receipt": {
            "path": str(statistics_path.relative_to(root)),
            "sha256": sha256_file(statistics_path),
        },
        "summaries": {
            name: {"path": str(paths[name].relative_to(root)),
                   "sha256": actual_hashes[name]}
            for name in paths
        },
    }
    return summaries, identities


def _validate_matched(summary: Mapping[str, Any]) -> None:
    require(summary.get("schema_version") == 1
            and summary.get("status") == "complete"
            and summary.get("study") == "paper-a-matched-color-v1-1"
            and summary.get("branch") ==
            "admission-informed-matched-color-v1-1"
            and summary.get("ages") == list(AGES)
            and summary.get("arms") == list(ARMS)
            and summary.get("seeds") == list(range(5)),
            "matched-color summary contract differs")
    hosts = summary.get("hosts")
    require(isinstance(hosts, Mapping)
            and set(hosts) == {"reacher", "pusht"},
            "matched-color host set differs")
    for host in ("reacher", "pusht"):
        arms = hosts[host].get("arms")
        require(isinstance(arms, Mapping) and set(arms) == set(ARMS),
                f"matched-color arm set differs for {host}")
        for arm in ARMS:
            require(set(arms[arm]) == {
                "age-4", "age-8", "age-15", "shortest_minus_longest"},
                f"matched-color age set differs for {host}/{arm}")
            for age in AGES:
                _stat(arms[arm][f"age-{age}"],
                      f"matched-color {host}/{arm}/age-{age}")
            _stat(arms[arm]["shortest_minus_longest"],
                  f"matched-color {host}/{arm}/age4-age15")
    interaction = _stat(summary.get("primary_ranking_interaction"),
                        "matched-color registered interaction")
    _stat(interaction, "matched-color registered interaction", interval="ci90")
    require(_number(interaction.get("equivalence_margin"),
                    "matched-color equivalence margin") == 0.05
            and isinstance(interaction.get("equivalent_within_margin"), bool)
            and isinstance(interaction.get("resolved_nonzero"), bool),
            "matched-color interaction metadata differs")


def _validate_pusht(summary: Mapping[str, Any]) -> None:
    require(summary.get("schema") ==
            "dinowm_wave2_spatial_carrier_summary_v1"
            and summary.get("status") == "complete"
            and summary.get("grid") == {
                "tasks": 2, "arms": 5, "seeds": 5, "cells": 50},
            "DINO-WM PushT summary contract differs")
    results = summary.get("results")
    require(isinstance(results, Mapping)
            and set(results) == set(TASK_LABELS),
            "DINO-WM PushT task set differs")
    for task in TASK_LABELS:
        task_record = results[task]
        require(set(task_record.get("ages", {})) == {str(age) for age in AGES},
                f"DINO-WM PushT age set differs for {task}")
        for age in AGES:
            record = task_record["ages"][str(age)]
            require(set(record.get("arms", {})) == set(ARMS)
                    and set(record.get("paired_vs_none", {}))
                    == set(ARMS) - {"none"}
                    and set(record.get("full_vs_context_reset", {}))
                    == set(ARMS),
                    f"DINO-WM PushT arm/contrast set differs for {task}/{age}")
            for arm in ARMS:
                arm_record = record["arms"][arm]
                _stat(arm_record.get("balanced_accuracy"),
                      f"DINO-WM PushT {task}/{age}/{arm} absolute")
                require(arm_record.get("parameters") == PARAMETERS[arm],
                        f"DINO-WM PushT parameter count differs for {arm}")
                _stat(record["full_vs_context_reset"][arm],
                      f"DINO-WM PushT {task}/{age}/{arm} reset contrast")
                if arm != "none":
                    _stat(record["paired_vs_none"][arm],
                          f"DINO-WM PushT {task}/{age}/{arm} no-state contrast")


def _validate_pointmaze(summaries: Mapping[str, Mapping[str, Any]]) -> None:
    combined = summaries["pointmaze"]
    carrier = summaries["pointmaze_carrier"]
    use = summaries["pointmaze_use"]
    require(combined.get("schema") == "dinowm_pointmaze_wave3_summary_v1"
            and combined.get("status") == "complete"
            and carrier.get("schema") ==
            "dinowm_pointmaze_wave3_carrier_summary_v1"
            and carrier.get("status") == "complete"
            and use.get("schema") == "dinowm_pointmaze_wave3_external_use_v1"
            and use.get("status") == "complete",
            "PointMaze summary contract differs")
    protocol = combined.get("protocol_sha256")
    require(isinstance(protocol, str)
            and carrier.get("protocol_sha256") == protocol
            and use.get("protocol_sha256") == protocol
            and combined.get("carrier_summary_path") == "carrier_summary.json"
            and combined.get("external_use_summary_path")
            == "external_use_summary.json",
            "PointMaze summary linkage differs")
    require(combined.get("admission") is not None
            and combined.get("controller_gate") is not None
            and combined["controller_gate"] == use.get("controller_gate"),
            "PointMaze gate linkage differs")
    results = carrier.get("results")
    require(isinstance(results, Mapping)
            and set(results) == {str(age) for age in AGES}
            and carrier.get("grid") == {
                "tasks": 1, "arms": 5, "seeds": 5, "cells": 25},
            "PointMaze carrier grid differs")
    for age in AGES:
        record = results[str(age)]
        require(set(record.get("arms", {})) == set(ARMS)
                and set(record.get("paired_vs_none", {}))
                == set(ARMS) - {"none"}
                and set(record.get("full_vs_context_reset", {})) == set(ARMS),
                f"PointMaze arm/contrast set differs for age {age}")
        for arm in ARMS:
            arm_record = record["arms"][arm]
            _stat(arm_record.get("balanced_accuracy"),
                  f"PointMaze {age}/{arm} absolute")
            require(arm_record.get("parameters") == PARAMETERS[arm],
                    f"PointMaze parameter count differs for {arm}")
            _stat(record["full_vs_context_reset"][arm],
                  f"PointMaze {age}/{arm} reset contrast")
            if arm != "none":
                _stat(record["paired_vs_none"][arm],
                      f"PointMaze {age}/{arm} no-state contrast")
    arms = use.get("arms")
    require(isinstance(arms, Mapping) and set(arms) == set(ARMS),
            "PointMaze external-use arm set differs")
    for arm in ARMS:
        record = arms[arm]
        require(set(record) == {
            "goal_accuracy", "executed_success", "contrast_vs_none",
            "contrast_vs_random", "resolved_execution_gain"},
            f"PointMaze external-use fields differ for {arm}")
        for key in ("goal_accuracy", "executed_success", "contrast_vs_none",
                    "contrast_vs_random"):
            _stat(record[key], f"PointMaze external use {arm}/{key}")
        require(isinstance(record["resolved_execution_gain"], bool),
                f"PointMaze resolved-use flag differs for {arm}")
    _stat(use.get("realized_random_goal"), "PointMaze random-goal success")
    oracle = _number(use.get("oracle_executed_success"),
                     "PointMaze oracle execution")
    require(0.0 <= oracle <= 1.0,
            "PointMaze oracle execution is outside [0,1]")


def validate_summaries(summaries: Mapping[str, Mapping[str, Any]]) -> None:
    require(set(summaries) == {
        "matched", "pusht", "pointmaze", "pointmaze_carrier",
        "pointmaze_use"}, "verified summary set differs")
    _validate_matched(summaries["matched"])
    _validate_pusht(summaries["pusht"])
    _validate_pointmaze(summaries)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character)
                   for character in value)


def _format_number(value: float, *, signed: bool) -> str:
    number = 100.0 * value
    if abs(number) < 0.05:
        number = 0.0
    return f"{number:+.1f}" if signed else f"{number:.1f}"


def format_stat(record: Mapping[str, Any], *, signed: bool = False,
                interval: str = "ci95") -> str:
    checked = _stat(record, "table statistic", interval=interval)
    low, high = checked[interval]
    return (f"${_format_number(float(checked['mean']), signed=signed)}$ "
            f"$[{_format_number(float(low), signed=signed)},"
            f"{_format_number(float(high), signed=signed)}]$")


def format_parameters(value: int) -> str:
    return f"{value:,}".replace(",", r"{,}")


def _table_prefix(caption: str, label: str, columns: str,
                  header: str, *, font_size: str = "footnotesize") \
        -> list[str]:
    require("wave" not in caption.lower(),
            "displayed table captions may not contain 'Wave'")
    require(font_size in {"footnotesize", "scriptsize"},
            "unsupported appendix table font size")
    return [
        r"\begin{table}[H]",
        r"\centering",
        f"\\{font_size}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]


def _table_suffix(note: str) -> list[str]:
    return [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        (r"\parbox{0.98\textwidth}{\scriptsize " + note + "}"),
        r"\end{table}",
        "",
    ]


def render_matched_table(summary: Mapping[str, Any]) -> str:
    lines = _table_prefix(
        "Matched-color retention across hosts and evidence ages.",
        "tab:appendix-matched-color", "llrrrr",
        (r"Host & Carrier & Age 4 & Age 8 & Age 15 & "
         r"$\Delta_{4-15}$ \\"), font_size="scriptsize")
    for host_index, host in enumerate(("reacher", "pusht")):
        for arm_index, arm in enumerate(ARMS):
            prefix = ("Reacher" if host == "reacher" else "PushT") \
                if arm_index == 0 else ""
            record = summary["hosts"][host]["arms"][arm]
            lines.append(
                f"{prefix} & {ARM_LABELS[arm]} & "
                + " & ".join(format_stat(record[f"age-{age}"])
                              for age in AGES)
                + " & "
                + format_stat(record["shortest_minus_longest"], signed=True)
                + r" \\")
        if host_index == 0:
            lines.append(r"\addlinespace[2pt]")
    interaction = summary["primary_ranking_interaction"]
    low90, high90 = interaction["ci90"]
    ci90_text = (
        f"$[{_format_number(float(low90), signed=True)},"
        f"{_format_number(float(high90), signed=True)}]$")
    status = (r"equivalent within $\pm5$ points"
              if interaction["equivalent_within_margin"]
              else r"not equivalent within $\pm5$ points")
    lines.extend([
        r"\midrule",
        (r"\multicolumn{2}{l}{Registered host ranking interaction} & "
         + r"\multicolumn{4}{l}{"
         + format_stat(interaction, signed=True)
         + "; 90\\% CI "
         + ci90_text
         + f"; {status}" + r"} \\"),
    ])
    lines.extend(_table_suffix(
        "Entries are percentage-point balanced accuracy with 95\\% intervals; "
        "$\\Delta_{4-15}$ is the paired shortest-minus-longest contrast. "
        "The registered interaction is the PushT-minus-Reacher difference in "
        "fixed-trust-minus-state-space accuracy at age 15."))
    return "\n".join(lines)


def render_pusht_table(summary: Mapping[str, Any]) -> str:
    lines = _table_prefix(
        "DINO-WM PushT carrier results on two registered visual-memory tasks.",
        "tab:appendix-dinowm-pusht", "lllrrrr",
        (r"Task & Age & Carrier & BAcc & $\Delta$ no state & "
         r"$\Delta$ reset & Params \\"), font_size="scriptsize")
    for task_index, task in enumerate(TASK_LABELS):
        for age_index, age in enumerate(AGES):
            record = summary["results"][task]["ages"][str(age)]
            for arm_index, arm in enumerate(ARMS):
                task_label = TASK_LABELS[task] \
                    if age_index == arm_index == 0 else ""
                age_label = str(age) if arm_index == 0 else ""
                delta_none = (r"\textit{reference}"
                              if arm == "none" else format_stat(
                                  record["paired_vs_none"][arm], signed=True))
                lines.append(
                    f"{task_label} & {age_label} & {ARM_LABELS[arm]} & "
                    f"{format_stat(record['arms'][arm]['balanced_accuracy'])} & "
                    f"{delta_none} & "
                    f"{format_stat(record['full_vs_context_reset'][arm], signed=True)} & "
                    f"{format_parameters(PARAMETERS[arm])} " + r"\\")
            if age_index < len(AGES) - 1:
                lines.append(r"\addlinespace[1pt]")
        if task_index == 0:
            lines.extend([r"\midrule", r"\addlinespace[1pt]"])
    lines.extend(_table_suffix(
        "All performance entries are percentage points with 95\\% intervals. "
        "Absolute BAcc is class balanced; contrasts preserve matched carrier "
        "seeds and class-stratified held-out episodes. Params counts trainable "
        "carrier parameters only."))
    return "\n".join(lines)


def render_pointmaze_carrier_table(summary: Mapping[str, Any]) -> str:
    lines = _table_prefix(
        "DINO-WM PointMaze persistent-carrier results.",
        "tab:appendix-pointmaze-carriers", "llrrrr",
        (r"Age & Carrier & BAcc & $\Delta$ no state & "
         r"$\Delta$ reset & Params \\"))
    for age_index, age in enumerate(AGES):
        record = summary["results"][str(age)]
        for arm_index, arm in enumerate(ARMS):
            delta_none = (r"\textit{reference}"
                          if arm == "none" else format_stat(
                              record["paired_vs_none"][arm], signed=True))
            lines.append(
                f"{age if arm_index == 0 else ''} & {ARM_LABELS[arm]} & "
                f"{format_stat(record['arms'][arm]['balanced_accuracy'])} & "
                f"{delta_none} & "
                f"{format_stat(record['full_vs_context_reset'][arm], signed=True)} & "
                f"{format_parameters(PARAMETERS[arm])} " + r"\\")
        if age_index < len(AGES) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines.extend(_table_suffix(
        "Entries are percentage points with 95\\% intervals from matched "
        "carrier-seed by equal-native-episode cluster resampling; all four "
        "counterfactual labels remain within their native episode cluster."))
    return "\n".join(lines)


def _gate_rows(combined: Mapping[str, Any]) \
        -> list[tuple[str, str, str, bool]]:
    admission = combined["admission"]
    controller = combined["controller_gate"]
    require(admission.get("admitted") is True
            and admission.get("all_gates_required") is True
            and controller.get("admitted") is True,
            "PointMaze prerequisite gate is not admitted")
    cue = admission["cue_encoding"]
    shortcuts = admission["shortcuts"]
    shortcut_names = (
        ("no_cue_visual", "No-cue visual shortcut"),
        ("action_only", "Action-only shortcut"),
        ("proprio_only", "Proprio-only shortcut"),
    )
    rows: list[tuple[str, str, str, bool]] = [
        ("Old-observation requirement", "four labels; no post-cue leak",
         "cue-only counterfactual", admission["requirement"]["pass"] is True),
        ("Visible-cue encoding",
         f"{100.0 * float(cue['balanced_accuracy']):.1f} BAcc; "
         f"{100.0 * min(cue['per_class_recall']):.1f} min recall",
         (f"$\\geq${100.0 * cue['thresholds']['balanced_accuracy_minimum']:.1f}; "
          f"$\\geq${100.0 * cue['thresholds']['per_class_recall_minimum']:.1f}"),
         cue["pass"] is True),
    ]
    for key, label in shortcut_names:
        values = [shortcuts[str(age)][key] for age in AGES]
        rows.append((
            label,
            f"max {100.0 * max(float(value['balanced_accuracy']) for value in values):.1f}",
            f"$\\leq${100.0 * min(float(value['maximum']) for value in values):.1f}",
            all(value["pass"] is True for value in values),
        ))
    counterfactual = admission["cue_only_counterfactual"]
    rows.extend([
        ("Cue-only invariance",
         (f"outside={counterfactual['outside_declared_mask_changed_pixels']}; "
          f"post-cue={counterfactual['post_cue_differing_pixels']}"),
         "both 0", counterfactual["pass"] is True),
        ("Frozen host", "digest unchanged", "exact equality",
         admission["frozen_host"]["pass"] is True),
        ("Controller oracle",
         f"{100.0 * controller['oracle_executed_success']:.1f}",
         f"$\\geq${100.0 * controller['thresholds']['oracle_success_minimum']:.1f}",
         controller["oracle_executed_success"]
         >= controller["thresholds"]["oracle_success_minimum"]),
        ("Worst-goal oracle",
         f"{100.0 * min(controller['oracle_per_class_executed_success']):.1f}",
         (f"$\\geq${100.0 * controller['thresholds']['oracle_per_class_success_minimum']:.1f}"),
         min(controller["oracle_per_class_executed_success"])
         >= controller["thresholds"]["oracle_per_class_success_minimum"]),
        ("Off-diagonal false success",
         f"{100.0 * controller['off_diagonal_false_success']:.1f}",
         (f"$\\leq${100.0 * controller['thresholds']['off_diagonal_false_success_maximum']:.1f}"),
         controller["off_diagonal_false_success"]
         <= controller["thresholds"]["off_diagonal_false_success_maximum"]),
        ("Deterministic replay",
         f"{100.0 * controller['deterministic_replay_fidelity']:.1f}",
         (f"$\\geq${100.0 * controller['thresholds']['deterministic_reset_replay_minimum']:.1f}"),
         controller["deterministic_replay_fidelity"]
         >= controller["thresholds"]["deterministic_reset_replay_minimum"]),
    ])
    require(all(row[3] for row in rows),
            "PointMaze gate checklist contains a failed row")
    return rows


def render_pointmaze_gate_table(combined: Mapping[str, Any]) -> str:
    lines = _table_prefix(
        "PointMaze prerequisite admission and controller checklist.",
        "tab:appendix-pointmaze-gates", r"lp{0.34\textwidth}ll",
        "Gate & Observed & Criterion & Status \\\\")
    for label, observed, criterion, passed in _gate_rows(combined):
        lines.append(
            f"{_latex_escape(label)} & {observed} & {criterion} & "
            + (r"\textsc{pass}" if passed else r"\textsc{fail}")
            + r" \\")
    lines.extend(_table_suffix(
        "Every registered prerequisite must pass before carrier training or "
        "external-use evaluation. Shortcut values are the maximum across "
        "evidence ages; controller values are percentages."))
    return "\n".join(lines)


def render_pointmaze_use_table(summary: Mapping[str, Any]) -> str:
    lines = _table_prefix(
        "PointMaze external-use results under the fixed waypoint controller.",
        "tab:appendix-pointmaze-use", "lrrrrc",
        (r"Carrier & Goal accuracy & Executed success & $\Delta$ no state & "
         r"$\Delta$ random & Resolved \\"), font_size="scriptsize")
    for arm in ARMS:
        record = summary["arms"][arm]
        lines.append(
            f"{ARM_LABELS[arm]} & {format_stat(record['goal_accuracy'])} & "
            f"{format_stat(record['executed_success'])} & "
            f"{format_stat(record['contrast_vs_none'], signed=True)} & "
            f"{format_stat(record['contrast_vs_random'], signed=True)} & "
            + (r"\textsc{yes}" if record["resolved_execution_gain"]
               else r"\textsc{no}") + r" \\")
    lines.extend([
        r"\midrule",
        (r"\multicolumn{6}{l}{Realized random-goal success: "
         + format_stat(summary["realized_random_goal"])
         + f"; oracle executed success: "
         f"{100.0 * float(summary['oracle_executed_success']):.1f}." + r"} \\"),
    ])
    lines.extend(_table_suffix(
        "Entries are percentage points with 95\\% native-episode cluster "
        "intervals. Consumers are shared and arm blind; execution uses the "
        "fixed released waypoint controller rather than native model planning."))
    return "\n".join(lines)


def _resolved_stat_arms(records: Mapping[str, Mapping[str, Any]]) -> str:
    """List learned arms with a strictly positive paired CI95 lower bound."""
    resolved = []
    for arm in ARMS:
        if arm == "none":
            continue
        record = _stat(records[arm], f"claim-ledger {arm} contrast")
        if float(record["ci95"][0]) > 0.0:
            resolved.append(ARM_LABELS[arm])
    return ", ".join(resolved) if resolved else "None resolved"


def _resolved_execution_arms(summary: Mapping[str, Any]) -> str:
    resolved = [
        ARM_LABELS[arm] for arm in ARMS
        if summary["arms"][arm]["resolved_execution_gain"] is True
    ]
    return ", ".join(resolved) if resolved else "None resolved"


def _compact_all_learned(value: str) -> str:
    all_learned = ", ".join(
        ARM_LABELS[arm] for arm in ARMS if arm != "none")
    return "All learned" if value == all_learned else value


def render_main_claim_ledger(
        summaries: Mapping[str, Mapping[str, Any]]) -> str:
    """Render the neutral, task-level claim ledger used in the main paper."""
    rows = [
        (
            "LeWM matched color (Reacher / PushT)", r"\Pass\ 3/3 each",
            "Age curves (both hosts)", r"\NA", r"\NA",
        ),
    ]
    pusht = summaries["pusht"]["results"]
    for task, display in (
            ("transient-visual-token-recall", "DINO-WM PushT token"),
            ("multi-item-visual-binding-recall", "DINO-WM PushT binding")):
        registered = pusht[task]["ages"]["15"]
        rows.append((
            display, r"\Pass\ 3/3",
            _resolved_stat_arms(registered["paired_vs_none"]),
            _resolved_stat_arms(registered["full_vs_context_reset"]),
            r"\NA",
        ))
    pointmaze = summaries["pointmaze_carrier"]["results"]["15"]
    rows.append((
        "DINO-WM PointMaze goal", r"\Pass\ 3/3 + controller",
        _compact_all_learned(_resolved_stat_arms(pointmaze["paired_vs_none"])),
        _compact_all_learned(
            _resolved_stat_arms(pointmaze["full_vs_context_reset"])),
        _compact_all_learned(
            _resolved_execution_arms(summaries["pointmaze_use"])),
    ))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        (r"\caption{Task-level claim ledger. A check denotes passed requirement, "
         r"cue-encoding, and shortcut gates; carrier cells list arms with a "
         r"positive registered age-15 paired lower bound. ``None resolved'' "
         r"means tested but unresolved; \NA\ means not tested. No pooled "
         r"score or cross-family ranking is computed.}"),
        r"\label{tab:claim-ledger-v2}",
        (r"\begin{tabularx}{\textwidth}{@{}"
         r">{\raggedright\arraybackslash}p{0.26\textwidth}"
         r">{\centering\arraybackslash}p{0.16\textwidth}"
         r"*{3}{>{\centering\arraybackslash}X}@{}}"),
        r"\toprule",
        r"Study & Gates & vs. no state & vs. reset & Executed use \\",
        r"\midrule",
    ]
    for index, row in enumerate(rows):
        lines.append(" & ".join(row) + r" \\")
        if index == 0:
            lines.append(r"\addlinespace[1pt]")
    lines.extend([
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table*}",
        "",
    ])
    return "\n".join(lines)


def render_tables(summaries: Mapping[str, Mapping[str, Any]]) \
        -> dict[str, str]:
    validate_summaries(summaries)
    appendix_files = {
        "matched_color_results.tex": render_matched_table(
            summaries["matched"]),
        "dinowm_pusht_results.tex": render_pusht_table(
            summaries["pusht"]),
        "pointmaze_carrier_results.tex": render_pointmaze_carrier_table(
            summaries["pointmaze_carrier"]),
        "pointmaze_gate_checklist.tex": render_pointmaze_gate_table(
            summaries["pointmaze"]),
        "pointmaze_external_use_results.tex": render_pointmaze_use_table(
            summaries["pointmaze_use"]),
    }
    require(tuple(appendix_files) == APPENDIX_TABLE_FILENAMES,
            "appendix table set or order differs")
    files = dict(appendix_files)
    files[MAIN_CLAIM_LEDGER_FILENAME] = render_main_claim_ledger(summaries)
    # Define long commands rather than emitting relative \input paths.  The
    # bundle is compiled once, while the appendix invokes each table beside its
    # owning study; this preserves the exact source set and prevents deferred
    # result floats from spilling into unrelated legacy sections.
    bundle_parts: list[str] = []
    require(set(APPENDIX_TABLE_COMMANDS) == set(appendix_files),
            "appendix table command map differs")
    for name, content in appendix_files.items():
        command = APPENDIX_TABLE_COMMANDS[name]
        bundle_parts.extend([
            f"\\long\\def\\{command}{{",
            content,
            "}",
            "",
        ])
    files[APPENDIX_BUNDLE_FILENAME] = "\n".join(bundle_parts)
    require("tab:claim-ledger-v2" not in files[APPENDIX_BUNDLE_FILENAME],
            "main claim ledger leaked into appendix table bundle")
    return files


def _content_digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def build_plan(files: Mapping[str, str], identities: Mapping[str, Any],
               output_dir: Path, generator: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "paper_a_appendix_result_tables_v1",
        "status": "ready",
        "read_only_sources": True,
        "output_directory": str(output_dir),
        "generator": dict(generator),
        "source_identities": identities,
        "tables": {
            name: {"bytes": len(content.encode("utf-8")),
                   "sha256": _content_digest(content)}
            for name, content in sorted(files.items())
        },
        "display_contract": {
            "font_size": "footnotesize; scriptsize for wide tables",
            "main_claim_ledger_font_size": "scriptsize",
            "captions_contain_wave_labels": False,
            "scientific_cross_family_pooling": False,
            "main_claim_ledger_label": "tab:claim-ledger-v2",
            "all_tables_excludes": [MAIN_CLAIM_LEDGER_FILENAME],
        },
    }


def emit_output(root: Path, output_dir: Path, files: Mapping[str, str],
                plan: Mapping[str, Any], *, execute: bool) -> bool:
    if not execute:
        return False
    root = root.resolve()
    require(plan.get("generator") == generator_identity(root),
            "table-generator identity changed before publication")
    target = _require_external_path(root, output_dir, "output directory")
    require(target != root and not target.exists(),
            f"output directory already exists or is invalid: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent))
    try:
        for name, content in files.items():
            require(Path(name).name == name and name.endswith(".tex"),
                    f"invalid generated table filename: {name}")
            path = staging / name
            with path.open("x", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            require(sha256_file(path) == plan["tables"][name]["sha256"],
                    f"staged table digest differs: {name}")
        completed = dict(plan)
        completed["status"] = "complete"
        manifest_content = stable_json(completed)
        with (staging / "manifest.json").open("x", encoding="utf-8") as stream:
            stream.write(manifest_content)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        require(not target.exists(), f"output directory appeared: {target}")
        os.rename(staging, target)
        parent_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return True


def generate(root: Path, completion_receipt: Path,
             statistics_receipt: Path, output_dir: Path, *,
             execute: bool) -> dict[str, Any]:
    summaries, identities = load_verified_summaries(
        root, completion_receipt, statistics_receipt)
    files = render_tables(summaries)
    target = _require_external_path(root, output_dir, "output directory")
    require(target != root.resolve() and not target.exists(),
            f"output directory already exists or is invalid: {target}")
    plan = build_plan(
        files, identities, target.relative_to(root.resolve()),
        generator_identity(root))
    wrote = emit_output(root, output_dir, files, plan, execute=execute)
    result = dict(plan)
    result["status"] = "complete" if wrote else "ready"
    result["wrote_output"] = wrote
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--completion-receipt", type=Path,
                        default=DEFAULT_COMPLETION_RECEIPT)
    parser.add_argument("--statistics-receipt", type=Path,
                        default=DEFAULT_STATISTICS_RECEIPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = generate(
            args.root, args.completion_receipt, args.statistics_receipt,
            args.output_dir, execute=bool(args.execute))
        print(stable_json(result), end="")
        return 0
    except NotReady as error:
        print(stable_json({
            "schema": "paper_a_appendix_result_tables_v1",
            "status": "incomplete", "wrote_output": False,
            "reason": str(error),
        }), end="")
        return 2
    except GenerationFailure as error:
        print(stable_json({
            "schema": "paper_a_appendix_result_tables_v1",
            "status": "failed", "wrote_output": False,
            "reason": str(error),
        }), end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
