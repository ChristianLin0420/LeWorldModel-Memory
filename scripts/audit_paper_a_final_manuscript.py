#!/usr/bin/env python3
"""Fail-closed, read-only audit of the final Paper-A manuscript build.

The auditor consumes only the compiled PDF/build sidecars, final manuscript
sources, optional generated-asset manifests, and optional completed audit
receipts.  It never parses experiment outcomes.  By default it prints a
receipt; ``--execute`` may atomically create one receipt outside experiment
roots, but only after every check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = Path("paper_a/main.pdf")
DEFAULT_AUX = Path("paper_a/main.aux")
DEFAULT_LOG = Path("paper_a/main.log")
DEFAULT_MAIN_SOURCE = Path("paper_a/main.tex")
DEFAULT_SOURCES = (
    Path("paper_a/main.tex"),
    Path("paper_a/abstract.tex"),
    Path("paper_a/body.tex"),
    Path("paper_a/refs.tex"),
    Path("paper_a/appendix.tex"),
    Path("paper_a/generated_results/all_tables.tex"),
    Path("paper_a/generated_results/main_claim_ledger.tex"),
)
DEFAULT_MAIN_TEXT_SOURCES = (
    Path("paper_a/main.tex"),
    Path("paper_a/abstract.tex"),
    Path("paper_a/body.tex"),
    Path("paper_a/generated_results/main_claim_ledger.tex"),
)
DEFAULT_TITLE = (
    "When Do Latent World Models Remember? A Controlled Audit of LeWM and DINO-WM")
DEFAULT_RECEIPT = Path("outputs/paper_a_final_manuscript_audit/receipt.json")
DEFAULT_COMPLETION_RECEIPT = Path(
    "outputs/paper_a_cross_wave_completion/receipt.json")
DEFAULT_STATISTICS_RECEIPT = Path(
    "outputs/paper_a_statistics_independent/receipt.json")
DEFAULT_TABLE_MANIFEST = Path("paper_a/generated_results/manifest.json")
TABLE_GENERATOR = Path("scripts/generate_paper_a_appendix_tables.py")
DEFAULT_FIGURE_MANIFESTS = (
    Path("paper_a/figures/manifest.json"),
    Path("outputs/paper_a_figures/manifest.json"),
    Path("outputs/paper_a_final_figures/manifest.json"),
)

EXPERIMENT_ROOTS = (
    Path("outputs/paper_a_matched_color_v1_1"),
    Path("outputs/dinowm_wave2_spatial_carrier_v1_1"),
    Path("outputs/dinowm_pointmaze_wave3"),
)
SUMMARY_PATHS = {
    "matched": Path("outputs/paper_a_matched_color_v1_1/summary.json"),
    "pusht": Path(
        "outputs/dinowm_wave2_spatial_carrier_v1_1/formal/summary.json"),
    "pointmaze": Path(
        "outputs/dinowm_pointmaze_wave3/formal/summary.json"),
    "pointmaze_carrier": Path(
        "outputs/dinowm_pointmaze_wave3/formal/carrier_summary.json"),
    "pointmaze_use": Path(
        "outputs/dinowm_pointmaze_wave3/formal/external_use_summary.json"),
}

EXPECTED_MAIN_FIGURES = {
    "fig:architecture-v2", "fig:tasks-v2", "fig:matched-v2",
    "fig:dinowm-carrier", "fig:pointmaze",
}
EXPECTED_MAIN_TABLES = {"tab:design-v2", "tab:claim-ledger-v2"}
MIN_MAIN_PAGE_ALNUM = 500

PLACEHOLDER_PATTERNS = (
    re.compile(r"\bTODO\b", re.IGNORECASE),
    re.compile(r"\bTBD\b", re.IGNORECASE),
    re.compile(r"\bFIXME\b", re.IGNORECASE),
    re.compile(r"\bXXX\b", re.IGNORECASE),
    re.compile(r"\bPLACEHOLDER\b", re.IGNORECASE),
    re.compile(r"\bLOREM\s+IPSUM\b", re.IGNORECASE),
    re.compile(r"\bCITATION\s+NEEDED\b", re.IGNORECASE),
    re.compile(r"\[\s*INSERT(?:\s+[^]]*)?\]", re.IGNORECASE),
)
MAIN_SHORTHAND_PATTERNS = (
    re.compile(r"\bWave\b"),
    re.compile(r"\bT1\b"),
    re.compile(r"\bT3\b"),
    re.compile(r"\(\s*ours\s*\)", re.IGNORECASE),
)


class ManuscriptAuditFailure(RuntimeError):
    """A final manuscript or provenance invariant failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ManuscriptAuditFailure(message)


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
        raise ManuscriptAuditFailure(f"path leaves repository: {value}") \
            from error
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


def read_text(path: Path, label: str) -> str:
    require(path.is_file(), f"missing {label}: {path}")
    try:
        return path.read_text(errors="strict")
    except (OSError, UnicodeError) as error:
        raise ManuscriptAuditFailure(f"invalid {label}: {path}: {error}") \
            from error


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(read_text(path, label))
    except json.JSONDecodeError as error:
        raise ManuscriptAuditFailure(f"invalid {label}: {path}: {error}") \
            from error
    require(isinstance(value, dict), f"{label} is not a mapping: {path}")
    return value


def _digest(value: Any, label: str) -> str:
    require(isinstance(value, str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", value)),
            f"{label} is not a SHA-256 digest")
    return value


def run_tool(arguments: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            list(arguments), check=False, capture_output=True, text=True)
    except OSError as error:
        raise ManuscriptAuditFailure(
            f"failed to run {arguments[0]}: {error}") from error
    require(result.returncode == 0,
            f"{arguments[0]} failed ({result.returncode}): "
            f"{result.stderr.strip()}")
    return result.stdout


def parse_pdfinfo(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip()
    require("Title" in result and "Pages" in result,
            "pdfinfo omitted title or page count")
    require(result["Pages"].isdigit(), "PDF page count is not an integer")
    return result


def parse_destinations(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    pattern = re.compile(r'^\s*(\d+)\s+\[.*\]\s+"([^"]+)"\s*$')
    for line in text.splitlines():
        match = pattern.match(line)
        if match:
            page, name = int(match.group(1)), match.group(2)
            require(name not in result, f"duplicate PDF destination: {name}")
            result[name] = page
    require(result, "PDF contains no named destinations")
    return result


def parse_pdffonts(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    pattern = re.compile(
        r"^(\S+)\s+(.+?)\s+(\S+)\s+(yes|no)\s+(yes|no)\s+"
        r"(yes|no)\s+\d+\s+\d+\s*$", re.IGNORECASE)
    for line in text.splitlines():
        match = pattern.match(line)
        if not match or line.lower().startswith("name "):
            continue
        rows.append({
            "name": match.group(1), "type": match.group(2).strip(),
            "encoding": match.group(3), "embedded": match.group(4).lower(),
            "subset": match.group(5).lower(), "unicode": match.group(6).lower(),
        })
    require(rows, "pdffonts returned no font records")
    return rows


def _braced_groups(text: str) -> list[str]:
    groups: list[str] = []
    cursor = 0
    while cursor < len(text):
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        require(text[cursor] == "{", "malformed TeX braced group")
        depth, start = 0, cursor + 1
        cursor += 1
        while cursor < len(text):
            character = text[cursor]
            if character == "{" and (cursor == 0 or text[cursor - 1] != "\\"):
                depth += 1
            elif character == "}" and (cursor == 0 or text[cursor - 1] != "\\"):
                if depth == 0:
                    groups.append(text[start:cursor])
                    cursor += 1
                    break
                depth -= 1
            cursor += 1
        else:
            raise ManuscriptAuditFailure("unterminated TeX braced group")
    return groups


def parse_aux_labels(text: str) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    prefix = re.compile(r"^\\newlabel\{([^}]+)\}(.*)$")
    for line in text.splitlines():
        match = prefix.match(line.strip())
        if not match:
            continue
        name, remainder = match.group(1), match.group(2)
        outer = _braced_groups(remainder)
        require(len(outer) == 1,
                f"malformed aux label wrapper: {name}")
        fields = _braced_groups(outer[0])
        require(len(fields) >= 4, f"malformed aux label fields: {name}")
        require(name not in labels, f"duplicate aux label: {name}")
        try:
            page = int(fields[1])
        except ValueError as error:
            raise ManuscriptAuditFailure(
                f"aux label page is not numeric: {name}") from error
        labels[name] = {
            "number": fields[0], "page": page, "text": fields[2],
            "destination": fields[3],
        }
    require(labels, "aux file contains no labels")
    return labels


def strip_tex_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        result, index = [], 0
        while index < len(line):
            if line[index] == "%":
                slashes = 0
                back = index - 1
                while back >= 0 and line[back] == "\\":
                    slashes += 1
                    back -= 1
                if slashes % 2 == 0:
                    break
            result.append(line[index])
            index += 1
        lines.append("".join(result))
    return "\n".join(lines)


def normalize_title(text: str) -> str:
    value = text.replace(r"\\", " ")
    value = re.sub(r"\\[A-Za-z]+\*?(?:\[[^]]*\])?", " ", value)
    value = value.replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", value).strip()


def _first_braced_group(text: str) -> tuple[str, int]:
    require(text.startswith("{"), "expected a leading TeX braced group")
    depth = 0
    for cursor in range(1, len(text)):
        character = text[cursor]
        escaped = cursor > 0 and text[cursor - 1] == "\\"
        if character == "{" and not escaped:
            depth += 1
        elif character == "}" and not escaped:
            if depth == 0:
                return text[1:cursor], cursor + 1
            depth -= 1
    raise ManuscriptAuditFailure("unterminated TeX braced group")


def extract_titles(source: str) -> list[str]:
    titles: list[str] = []
    cursor = 0
    token = r"\title"
    while True:
        index = source.find(token, cursor)
        if index < 0:
            break
        brace = source.find("{", index + len(token))
        require(brace >= 0, "title declaration lacks a braced value")
        value, consumed = _first_braced_group(source[brace:])
        titles.append(normalize_title(value))
        cursor = brace + consumed
    require(titles, "main source contains no title declaration")
    return titles


def _compact_letters(text: str) -> str:
    return re.sub(r"[^A-Za-z]", "", text).upper()


def has_references_heading(text: str) -> bool:
    for line in text.splitlines()[:40]:
        letters = re.sub(r"[^A-Za-z]", "", line).upper()
        if letters == "REFERENCES":
            return True
    return False


def find_forbidden(text: str, patterns: Sequence[re.Pattern[str]],
                   label: str) -> None:
    for pattern in patterns:
        match = pattern.search(text)
        require(match is None,
                f"{label} contains forbidden marker {pattern.pattern!r}")


def validate_log(log: str, *, pages: int, pdf_size: int,
                 pdf_name: str) -> None:
    forbidden = (
        re.compile(r"Overfull \\[hv]box", re.IGNORECASE),
        re.compile(r"(?:LaTeX|Package .*?) Warning:.*undefined",
                   re.IGNORECASE),
        re.compile(r"(?:Citation|Reference).*undefined", re.IGNORECASE),
        re.compile(r"There were undefined references", re.IGNORECASE),
        re.compile(r"multiply defined", re.IGNORECASE),
        re.compile(r"Label\(s\) may have changed", re.IGNORECASE),
        re.compile(r"Rerun to get cross-references right", re.IGNORECASE),
        re.compile(r"Please \(re\)run", re.IGNORECASE),
        re.compile(r"rerunfilecheck Warning", re.IGNORECASE),
    )
    for pattern in forbidden:
        require(pattern.search(log) is None,
                f"build log contains forbidden diagnostic: {pattern.pattern}")
    output = re.search(
        r"Output written on\s+([^\s]+)\s+\((\d+) pages?,\s+(\d+) bytes\)\.",
        log)
    require(output is not None, "build log lacks final PDF output record")
    require(Path(output.group(1)).name == pdf_name
            and int(output.group(2)) == pages
            and int(output.group(3)) == pdf_size,
            "build log PDF record differs from current PDF")


def compiled_local_sources(log: str, paper_directory: Path) -> set[Path]:
    names = set(re.findall(r"\((\.\.?/[^\s()]+\.tex)", log))
    return {(paper_directory / name).resolve() for name in names}


def included_graphics(source_text: str, source_directory: Path) -> set[Path]:
    search_directories = [source_directory]
    for declaration in re.finditer(
            r"\\graphicspath\{((?:\{[^}]+\})+)\}", source_text):
        for value in re.findall(r"\{([^}]+)\}", declaration.group(1)):
            candidate = (source_directory / value).resolve()
            if candidate not in search_directories:
                search_directories.append(candidate)
    graphics: set[Path] = set()
    for match in re.finditer(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",
                             source_text):
        value = Path(match.group(1))
        candidates = []
        for directory in search_directories:
            if value.suffix:
                candidates.append((directory / value).resolve())
            else:
                candidates.extend((directory / value).with_suffix(suffix)
                                  for suffix in (".pdf", ".png", ".jpg", ".jpeg"))
        existing = [candidate for candidate in candidates if candidate.is_file()]
        require(len(existing) == 1,
                f"graphic path is missing or ambiguous: {value}")
        graphics.add(existing[0])
    return graphics


def _label_physical_page(label: Mapping[str, Any],
                         destinations: Mapping[str, int], name: str) -> int:
    destination = label["destination"]
    require(destination in destinations,
            f"label destination is absent from PDF: {name}/{destination}")
    return destinations[destination]


def validate_labels_and_pagination(
        aux: str, destinations: Mapping[str, int], page_text: Mapping[int, str],
        total_pages: int, main_source: str,
        main_text_source: str) -> dict[str, Any]:
    labels = parse_aux_labels(aux)
    marker = labels.get("paper-a-main-end")
    require(marker is not None, "paper-a-main-end label is missing")
    marker_physical = _label_physical_page(
        marker, destinations, "paper-a-main-end")
    require(marker["page"] == 9 and marker_physical == 9,
            "paper-a-main-end is not on physical page 9")

    require(has_references_heading(page_text[10]),
            "References does not start on physical page 10")
    require(not has_references_heading(page_text[9]),
            "References appears on physical page 9")
    require(total_pages >= 11, "PDF does not include appendix pages")
    appendix_labels = [
        (name, value) for name, value in labels.items()
        if name.startswith("app:")]
    require(appendix_labels
            and min(_label_physical_page(value, destinations, name)
                    for name, value in appendix_labels) >= 11
            and destinations.get("section.A") == 11,
            "appendix does not start after References on physical page 11")

    clean_main = strip_tex_comments(main_source)
    marker_index = clean_main.find(r"\label{paper-a-main-end}")
    refs_index = clean_main.find(r"\input{refs.tex}")
    appendix_index = clean_main.find(r"\appendix")
    require(0 <= marker_index < refs_index < appendix_index,
            "main source marker/References/appendix order differs")
    between = clean_main[marker_index:refs_index]
    require(r"\clearpage" in between,
            "main source lacks a page break before References")
    require(r"\clearpage" in clean_main[refs_index:appendix_index],
            "main source lacks a page break before appendix")

    source_labels = set(re.findall(
        r"\\label\{((?:fig|tab):[^}]+)\}",
        strip_tex_comments(main_text_source)))
    source_main_figures = {
        name for name in source_labels if name.startswith("fig:")}
    source_main_tables = {
        name for name in source_labels if name.startswith("tab:")}
    aux_main_figures = {
        name for name, value in labels.items()
        if name.startswith("fig:")
        and _label_physical_page(value, destinations, name) <= 9
    }
    aux_main_tables = {
        name for name, value in labels.items()
        if name.startswith("tab:")
        and _label_physical_page(value, destinations, name) <= 9
    }
    require(aux_main_figures == EXPECTED_MAIN_FIGURES
            and source_main_figures == EXPECTED_MAIN_FIGURES,
            f"main figure labels differ: {sorted(aux_main_figures)}")
    require(aux_main_tables == EXPECTED_MAIN_TABLES
            and source_main_tables == EXPECTED_MAIN_TABLES,
            f"main table labels differ: {sorted(aux_main_tables)}")
    return {
        "main_end_aux_page": marker["page"],
        "main_end_physical_page": marker_physical,
        "references_physical_page": 10,
        "appendix_first_physical_page": 11,
        "main_figure_labels": sorted(aux_main_figures),
        "main_table_labels": sorted(aux_main_tables),
    }


def _artifact_record_path(root: Path, manifest_path: Path, key: str,
                          record: Mapping[str, Any]) -> Path | None:
    if "path" in record:
        return repository_path(root, record["path"])
    candidate = (manifest_path.parent / key).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def collect_manifest_records(
        root: Path, manifest_path: Path, value: Any,
        prefix: tuple[str, ...] = ()) -> list[tuple[str, Path, Mapping[str, Any]]]:
    result: list[tuple[str, Path, Mapping[str, Any]]] = []
    if isinstance(value, Mapping):
        if "sha256" in value:
            key = prefix[-1] if prefix else "artifact"
            path = _artifact_record_path(root, manifest_path, key, value)
            if path is not None:
                result.append((".".join(prefix), path, value))
        for key, child in value.items():
            if isinstance(child, (Mapping, list)):
                result.extend(collect_manifest_records(
                    root, manifest_path, child, prefix + (str(key),)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(collect_manifest_records(
                root, manifest_path, child, prefix + (str(index),)))
    return result


def validate_manifest_records(
        root: Path, manifest_path: Path, manifest: Mapping[str, Any],
        label: str) -> dict[str, Any]:
    if "status" in manifest:
        require(manifest["status"] in {"complete", "verified"},
                f"{label} is not complete")
    records = collect_manifest_records(root, manifest_path, manifest)
    require(records, f"{label} contains no hash-bound artifacts")
    seen: set[Path] = set()
    for name, path, record in records:
        if path in seen:
            continue
        seen.add(path)
        expected = _digest(record.get("sha256"), f"{label}/{name}")
        require(path.is_file(), f"{label} artifact is missing: {path}")
        if "size" in record:
            require(isinstance(record["size"], int)
                    and not isinstance(record["size"], bool)
                    and path.stat().st_size == record["size"],
                    f"{label} artifact size differs: {path}")
        if "bytes" in record:
            require(isinstance(record["bytes"], int)
                    and not isinstance(record["bytes"], bool)
                    and path.stat().st_size == record["bytes"],
                    f"{label} artifact byte count differs: {path}")
        require(sha256_file(path) == expected,
                f"{label} artifact hash differs: {path}")
    return {
        "path": str(manifest_path.relative_to(root.resolve())),
        "sha256": sha256_file(manifest_path),
        "artifacts_verified": len(seen),
        "artifact_paths": sorted(str(path.relative_to(root.resolve()))
                                 for path in seen),
    }


def _receipt_summary_hashes(completion: Mapping[str, Any],
                            statistics: Mapping[str, Any]) -> dict[str, str]:
    require(completion.get("schema") ==
            "paper_a_cross_wave_completion_receipt_v1"
            and completion.get("status") == "complete"
            and completion.get("read_only_verification") is True
            and completion.get("scientific_cross_wave_aggregation") is False
            and completion.get("sealed_locks_modified") is False
            and completion.get("paper_files_modified") is False,
            "completion audit receipt is not final/read-only")
    require(statistics.get("schema") ==
            "paper_a_statistics_independent_receipt_v1"
            and statistics.get("status") == "verified"
            and statistics.get("read_only") is True
            and statistics.get("statistics_computed") is True
            and statistics.get("scientific_cross_family_pooling") is False
            and statistics.get("imports_producer_statistics") is False
            and statistics.get("experiment_roots_modified") is False,
            "statistics audit receipt is not final/read-only")
    cw, sw = completion.get("waves"), statistics.get("waves")
    require(isinstance(cw, Mapping) and isinstance(sw, Mapping)
            and set(cw) == {"wave1_1", "wave2_v1_1", "wave3"}
            and set(sw) == {"wave2", "wave3"}
            and all(value.get("status") == "verified" for value in cw.values())
            and all(value.get("status") == "verified" for value in sw.values()),
            "audit receipt study set/status differs")
    hashes = {
        "matched": _digest(cw["wave1_1"].get("summary_sha256"),
                           "matched summary receipt"),
        "pusht": _digest(cw["wave2_v1_1"].get("summary_sha256"),
                         "PushT summary receipt"),
        "pointmaze": _digest(cw["wave3"].get("summary_sha256"),
                             "PointMaze summary receipt"),
        "pointmaze_carrier": _digest(
            cw["wave3"].get("carrier_summary_sha256"),
            "PointMaze carrier receipt"),
        "pointmaze_use": _digest(
            cw["wave3"].get("external_use_summary_sha256"),
            "PointMaze use receipt"),
    }
    overlaps = {
        "pusht": sw["wave2"].get("summary_sha256"),
        "pointmaze": sw["wave3"].get("combined_summary_sha256"),
        "pointmaze_carrier": sw["wave3"].get("carrier_summary_sha256"),
        "pointmaze_use": sw["wave3"].get("external_use_summary_sha256"),
    }
    for name, value in overlaps.items():
        require(_digest(value, f"statistics {name} receipt") == hashes[name],
                f"audit receipts disagree on {name} summary")
    return hashes


def validate_optional_bindings(
        root: Path, *, completion_receipt: Path,
        statistics_receipt: Path, table_manifest: Path,
        figure_manifests: Sequence[Path],
        main_graphics: set[Path]) -> dict[str, Any]:
    root = root.resolve()
    completion_path = repository_path(root, completion_receipt)
    statistics_path = repository_path(root, statistics_receipt)
    completion_exists = completion_path.is_file()
    statistics_exists = statistics_path.is_file()
    require(completion_exists == statistics_exists,
            "exactly one independent audit receipt is present")

    result: dict[str, Any] = {
        "audit_receipts_present": completion_exists,
        "table_manifest_present": False,
        "figure_manifests": [],
    }
    receipt_hashes: dict[str, str] | None = None
    if completion_exists:
        completion = read_json(completion_path, "completion audit receipt")
        statistics = read_json(statistics_path, "statistics audit receipt")
        receipt_hashes = _receipt_summary_hashes(completion, statistics)
        # Only after both receipts are valid and mutually consistent may any
        # experiment summary be touched, and even then it is hashed, not parsed.
        for name, relative in SUMMARY_PATHS.items():
            require(sha256_file(repository_path(root, relative))
                    == receipt_hashes[name],
                    f"current {name} summary differs from audit receipts")
        result["audit_receipts"] = {
            "completion": {
                "path": str(completion_path.relative_to(root)),
                "sha256": sha256_file(completion_path),
            },
            "statistics": {
                "path": str(statistics_path.relative_to(root)),
                "sha256": sha256_file(statistics_path),
            },
            "summary_sha256": receipt_hashes,
        }

    table_path = repository_path(root, table_manifest)
    if table_path.is_file():
        require(completion_exists,
                "generated table manifest exists before both audit receipts")
        manifest = read_json(table_path, "generated table manifest")
        require(manifest.get("schema") == "paper_a_appendix_result_tables_v1"
                and manifest.get("status") == "complete",
                "generated table manifest is not complete")
        generator = manifest.get("generator")
        generator_path = repository_path(root, TABLE_GENERATOR)
        require(isinstance(generator, Mapping)
                and repository_path(root, generator.get("path", ""))
                == generator_path
                and generator_path.is_file()
                and isinstance(generator.get("size"), int)
                and not isinstance(generator.get("size"), bool)
                and generator["size"] == generator_path.stat().st_size
                and _digest(generator.get("sha256"),
                            "table manifest generator")
                == sha256_file(generator_path),
                "generated table manifest uses a stale generator")
        identities = manifest.get("source_identities")
        require(isinstance(identities, Mapping),
                "generated table manifest lacks source identities")
        for key, expected_path in (
                ("completion_receipt", completion_path),
                ("statistics_receipt", statistics_path)):
            record = identities.get(key)
            require(isinstance(record, Mapping)
                    and repository_path(root, record.get("path", ""))
                    == expected_path
                    and _digest(record.get("sha256"),
                                f"table manifest {key}")
                    == sha256_file(expected_path),
                    f"generated table manifest differs from {key}")
        summaries = identities.get("summaries")
        require(isinstance(summaries, Mapping)
                and set(summaries) == set(SUMMARY_PATHS),
                "generated table manifest summary set differs")
        assert receipt_hashes is not None
        for name, relative in SUMMARY_PATHS.items():
            record = summaries[name]
            require(isinstance(record, Mapping)
                    and repository_path(root, record.get("path", ""))
                    == repository_path(root, relative)
                    and _digest(record.get("sha256"),
                                f"table manifest {name} summary")
                    == receipt_hashes[name],
                    f"generated table manifest differs from {name} receipt")
        result["table_manifest"] = validate_manifest_records(
            root, table_path, manifest, "generated table manifest")
        result["table_manifest_present"] = True

    candidates = [repository_path(root, path) for path in figure_manifests]
    for manifest_path in candidates:
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path, "generated figure manifest")
        require(manifest.get("schema") == "paper_a_figure_manifest_v1"
                and manifest.get("status") == "complete",
                "generated figure manifest is not complete")
        generator = manifest.get("generator")
        generator_path = repository_path(
            root, "scripts/plot_paper_a_strengthened.py")
        require(isinstance(generator, Mapping)
                and repository_path(root, generator.get("path", ""))
                == generator_path
                and _digest(generator.get("sha256"),
                            "figure manifest generator")
                == sha256_file(generator_path),
                "generated figure manifest uses a stale generator")
        require(completion_exists and receipt_hashes is not None,
                "generated figure manifest exists before both audit receipts")
        manifest_receipts = manifest.get("audit_receipts")
        require(isinstance(manifest_receipts, Mapping),
                "generated figure manifest lacks audit receipts")
        for key, expected_path in (
                ("completion", completion_path),
                ("statistics", statistics_path)):
            record = manifest_receipts.get(key)
            require(isinstance(record, Mapping)
                    and repository_path(root, record.get("path", ""))
                    == expected_path
                    and _digest(record.get("sha256"),
                                f"figure manifest {key} receipt")
                    == sha256_file(expected_path),
                    f"generated figure manifest differs from {key} receipt")
        manifest_summaries = manifest.get("summaries")
        require(isinstance(manifest_summaries, Mapping)
                and set(manifest_summaries) == set(SUMMARY_PATHS),
                "generated figure manifest summary set differs")
        for name, relative in SUMMARY_PATHS.items():
            record = manifest_summaries[name]
            require(isinstance(record, Mapping)
                    and repository_path(root, record.get("path", ""))
                    == repository_path(root, relative)
                    and _digest(record.get("sha256"),
                                f"figure manifest {name} summary")
                    == receipt_hashes[name],
                    f"generated figure manifest differs from {name} receipt")
        records = collect_manifest_records(root, manifest_path, manifest)
        protected = tuple(repository_path(root, value)
                          for value in EXPERIMENT_ROOTS)
        references_experiment = any(
            any(_is_within(path, experiment) for experiment in protected)
            for _name, path, _record in records)
        require(not references_experiment or completion_exists,
                "figure manifest references outcomes before audit receipts")
        verified = validate_manifest_records(
            root, manifest_path, manifest, "generated figure manifest")
        # ``artifact_paths`` are root-relative strings; resolve against root.
        covered = {repository_path(root, value)
                   for value in verified["artifact_paths"]}
        require(main_graphics.issubset(covered),
                "figure manifest does not bind every main-text graphic")
        result["figure_manifests"].append(verified)
    return result


def audit_manuscript(
        root: Path, *, pdf: Path, aux: Path, log: Path,
        main_source: Path, sources: Sequence[Path],
        main_text_sources: Sequence[Path], expected_title: str,
        completion_receipt: Path, statistics_receipt: Path,
        table_manifest: Path, figure_manifests: Sequence[Path]) \
        -> dict[str, Any]:
    root = root.resolve()
    pdf_path = repository_path(root, pdf)
    aux_path = repository_path(root, aux)
    log_path = repository_path(root, log)
    main_path = repository_path(root, main_source)
    source_paths = tuple(repository_path(root, value) for value in sources)
    main_text_paths = tuple(repository_path(root, value)
                            for value in main_text_sources)
    require(pdf_path.is_file() and aux_path.is_file() and log_path.is_file(),
            "compiled PDF, aux, or log is missing")
    require(main_path in source_paths
            and set(main_text_paths).issubset(set(source_paths)),
            "main/final source sets are inconsistent")

    aux_text = read_text(aux_path, "aux file")
    log_text = read_text(log_path, "build log")
    source_text = {path: read_text(path, "final source")
                   for path in source_paths}
    main_text = "\n".join(source_text[path] for path in main_text_paths)
    all_source_text = "\n".join(source_text.values())
    main_source_text = source_text[main_path]

    pdf_info = parse_pdfinfo(run_tool(("pdfinfo", str(pdf_path))))
    pages = int(pdf_info["Pages"])
    require(pages >= 11, "PDF does not contain References plus appendix")
    require(pdf_info["Title"] == expected_title,
            "PDF metadata title differs from final title")
    titles = extract_titles(strip_tex_comments(main_source_text))
    require(expected_title in titles,
            "final title is absent from main source")
    destinations = parse_destinations(
        run_tool(("pdfinfo", "-dests", str(pdf_path))))
    fonts = parse_pdffonts(run_tool(("pdffonts", str(pdf_path))))
    require(all(record["embedded"] == "yes" for record in fonts),
            "PDF contains a non-embedded font")
    require(all("type 3" not in record["type"].lower() for record in fonts),
            "PDF contains a Type 3 font")

    page_text = {
        page: run_tool(("pdftotext", "-f", str(page), "-l", str(page),
                        "-layout", str(pdf_path), "-"))
        for page in range(1, 12)
    }
    for page in range(1, 10):
        count = len(re.sub(r"[^A-Za-z0-9]", "", page_text[page]))
        require(count >= MIN_MAIN_PAGE_ALNUM,
                f"main physical page {page} has too little extracted text: "
                f"{count} < {MIN_MAIN_PAGE_ALNUM}")
    require(_compact_letters(expected_title)
            in _compact_letters(page_text[1]),
            "first-page title text differs from final title")

    pagination = validate_labels_and_pagination(
        aux_text, destinations, page_text, pages, main_source_text, main_text)
    validate_log(log_text, pages=pages, pdf_size=pdf_path.stat().st_size,
                 pdf_name=pdf_path.name)
    compiled = compiled_local_sources(log_text, main_path.parent)
    require(compiled == set(source_paths),
            "declared final sources differ from TeX inputs recorded in log")
    require(pdf_path.stat().st_mtime_ns
            >= max(path.stat().st_mtime_ns for path in source_paths),
            "PDF is older than a final source; rebuild is required")

    clean_all_sources = strip_tex_comments(all_source_text)
    clean_main_text = strip_tex_comments(main_text)
    find_forbidden(clean_all_sources, PLACEHOLDER_PATTERNS, "final sources")
    find_forbidden(clean_main_text, MAIN_SHORTHAND_PATTERNS, "main-text sources")
    full_pdf_text = run_tool(("pdftotext", "-layout", str(pdf_path), "-"))
    find_forbidden(full_pdf_text, PLACEHOLDER_PATTERNS, "PDF text")
    find_forbidden("\n".join(page_text[page] for page in range(1, 10)),
                   MAIN_SHORTHAND_PATTERNS, "main PDF text")

    graphics = included_graphics(clean_main_text, main_path.parent)
    require(len(graphics) == len(EXPECTED_MAIN_FIGURES)
            and all(path.is_file() for path in graphics),
            "main-text graphic set differs or contains a missing file")
    bindings = validate_optional_bindings(
        root, completion_receipt=completion_receipt,
        statistics_receipt=statistics_receipt,
        table_manifest=table_manifest,
        figure_manifests=figure_manifests, main_graphics=graphics)

    identities = {
        "pdf": {"path": str(pdf_path.relative_to(root)),
                "size": pdf_path.stat().st_size,
                "sha256": sha256_file(pdf_path)},
        "aux": {"path": str(aux_path.relative_to(root)),
                "size": aux_path.stat().st_size,
                "sha256": sha256_file(aux_path)},
        "log": {"path": str(log_path.relative_to(root)),
                "size": log_path.stat().st_size,
                "sha256": sha256_file(log_path)},
        "sources": {
            str(path.relative_to(root)): {
                "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in source_paths
        },
        "main_graphics": {
            str(path.relative_to(root)): {
                "size": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(graphics)
        },
    }
    return {
        "schema": "paper_a_final_manuscript_audit_v1",
        "status": "verified",
        "read_only": True,
        "paper_modified": False,
        "experiment_outcomes_parsed": False,
        "title": expected_title,
        "physical_pages": pages,
        "main_pages": 9,
        "references_page": 10,
        "appendix_pages": pages - 10,
        "pagination": pagination,
        "fonts": {
            "count": len(fonts), "all_embedded": True,
            "type3_count": 0,
        },
        "main_page_alphanumeric_counts": {
            str(page): len(re.sub(r"[^A-Za-z0-9]", "", page_text[page]))
            for page in range(1, 10)
        },
        "diagnostics": {
            "overfull_boxes": 0,
            "undefined_references_or_citations": 0,
            "rerun_warnings": 0,
            "placeholder_markers": 0,
            "forbidden_main_shorthand_labels": 0,
        },
        "bindings": bindings,
        "identities": identities,
    }


def emit_receipt(root: Path, destination: Path,
                 payload: Mapping[str, Any], *, execute: bool) -> bool:
    if not execute:
        return False
    require(payload.get("status") == "verified",
            "refusing to write a failed manuscript audit")
    root = root.resolve()
    target = repository_path(root, destination)
    protected = tuple(repository_path(root, value)
                      for value in EXPERIMENT_ROOTS)
    require(not any(_is_within(target, value) for value in protected),
            "manuscript receipt must be outside experiment roots")
    require(not target.exists(), f"receipt already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(stable_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
        require(not target.exists(), f"receipt appeared concurrently: {target}")
        os.link(temporary_path, target)
        parent_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)
    return True


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--aux", type=Path, default=DEFAULT_AUX)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--main-source", type=Path, default=DEFAULT_MAIN_SOURCE)
    parser.add_argument("--source", type=Path, action="append")
    parser.add_argument("--main-text-source", type=Path, action="append")
    parser.add_argument("--expected-title", default=DEFAULT_TITLE)
    parser.add_argument("--completion-receipt", type=Path,
                        default=DEFAULT_COMPLETION_RECEIPT)
    parser.add_argument("--statistics-receipt", type=Path,
                        default=DEFAULT_STATISTICS_RECEIPT)
    parser.add_argument("--table-manifest", type=Path,
                        default=DEFAULT_TABLE_MANIFEST)
    parser.add_argument("--figure-manifest", type=Path, action="append")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = audit_manuscript(
            args.root, pdf=args.pdf, aux=args.aux, log=args.log,
            main_source=args.main_source,
            sources=tuple(args.source or DEFAULT_SOURCES),
            main_text_sources=tuple(
                args.main_text_source or DEFAULT_MAIN_TEXT_SOURCES),
            expected_title=args.expected_title,
            completion_receipt=args.completion_receipt,
            statistics_receipt=args.statistics_receipt,
            table_manifest=args.table_manifest,
            figure_manifests=tuple(
                args.figure_manifest or DEFAULT_FIGURE_MANIFESTS))
        bindings = payload["bindings"]
        require(bindings.get("audit_receipts_present") is True
                and bindings.get("table_manifest_present") is True
                and len(bindings.get("figure_manifests", [])) >= 1,
                "final audit receipts/table/figure manifests are incomplete")
        payload["auditor"] = {
            "path": str(Path(__file__).resolve().relative_to(
                Path(args.root).resolve())),
            "sha256": sha256_file(Path(__file__).resolve()),
        }
        wrote = emit_receipt(
            args.root, args.receipt, payload, execute=bool(args.execute))
        payload["receipt_written"] = wrote
        print(stable_json(payload), end="")
        return 0
    except ManuscriptAuditFailure as error:
        print(stable_json({
            "schema": "paper_a_final_manuscript_audit_v1",
            "status": "failed", "read_only": True,
            "receipt_written": False, "reason": str(error),
        }), end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
