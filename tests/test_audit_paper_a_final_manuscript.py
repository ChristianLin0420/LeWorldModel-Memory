from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from scripts import audit_paper_a_final_manuscript as audit


def _newlabel(name: str, number: str, page: int, destination: str) -> str:
    return (f"\\newlabel{{{name}}}{{{{{number}}}{{{page}}}{{caption}}"
            f"{{{destination}}}{{}}}}")


def _fixture_build(root: Path) -> dict[str, object]:
    paper = root / "paper_a"
    figures = paper / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    generated = paper / "generated_results"
    generated.mkdir(parents=True, exist_ok=True)
    main = paper / "main.tex"
    abstract = paper / "abstract.tex"
    body = paper / "body.tex"
    refs = paper / "refs.tex"
    appendix = paper / "appendix.tex"
    all_tables = generated / "all_tables.tex"
    claim_ledger = generated / "main_claim_ledger.tex"
    main.write_text(
        "\\title{" + audit.DEFAULT_TITLE + "}\n"
        "\\begin{document}\n\\input{abstract.tex}\n\\input{body.tex}\n"
        "\\phantomsection\\label{paper-a-main-end}\n\\clearpage\n"
        "\\input{refs.tex}\n\\clearpage\n\\appendix\n"
        "\\input{appendix.tex}\n\\end{document}\n")
    abstract.write_text("A complete abstract with no pending markers.\n")
    body_lines = []
    figure_pages = {}
    for index, label in enumerate(sorted(audit.EXPECTED_MAIN_FIGURES), 1):
        filename = f"figure_{index}.pdf"
        (figures / filename).write_bytes(b"figure" + bytes([index]))
        body_lines.append(
            f"\\begin{{figure}}\\includegraphics{{figures/{filename}}}"
            f"\\caption{{Figure {index}}}\\label{{{label}}}\\end{{figure}}")
        figure_pages[label] = min(index + 1, 6)
    body_lines.append(
        "\\begin{table}\\caption{Design table}\\label{tab:design-v2}"
        "x\\end{table}")
    body_lines.append(r"\input{generated_results/main_claim_ledger.tex}")
    body_lines.append("Complete main prose without experimental shorthand.")
    body.write_text("\n".join(body_lines) + "\n")
    claim_ledger.write_text(
        "\\begin{table}\\caption{Claim ledger}"
        "\\label{tab:claim-ledger-v2}x\\end{table}\n")
    all_tables.write_text("Generated appendix result tables.\n")
    refs.write_text("\\begin{thebibliography}{1}\\bibitem{x} X.\\end{thebibliography}\n")
    appendix.write_text(
        "\\input{generated_results/all_tables.tex}\n"
        "\\section{Appendix contract}\\label{app:protocol} Complete.\n")

    aux_lines = []
    destinations = {}
    for index, label in enumerate(sorted(audit.EXPECTED_MAIN_FIGURES), 1):
        destination = f"figure.caption.{index}"
        page = figure_pages[label]
        aux_lines.append(_newlabel(label, str(index), page, destination))
        destinations[destination] = page
    for index, label in enumerate(sorted(audit.EXPECTED_MAIN_TABLES), 1):
        aux_lines.append(_newlabel(
            label, str(index), 6 + index, f"table.caption.{index}"))
    aux_lines.extend([
        _newlabel("paper-a-main-end", "9", 9, "section*.15"),
        _newlabel("app:protocol", "A", 11, "section.A"),
    ])
    destinations.update({
        "table.caption.1": 7, "table.caption.2": 8,
        "section*.15": 9, "section.A": 11,
    })
    aux_path = paper / "main.aux"
    aux_path.write_text("\n".join(aux_lines) + "\n")
    pdf_path = paper / "main.pdf"
    pdf_path.write_bytes(b"%PDF synthetic final manuscript\n")
    log_path = paper / "main.log"
    log_path.write_text(
        "(./main.tex) (./abstract.tex) (./body.tex) "
        "(./generated_results/main_claim_ledger.tex) (./refs.tex) "
        "(./appendix.tex) (./generated_results/all_tables.tex)\n"
        f"Output written on main.pdf (12 pages, {pdf_path.stat().st_size} bytes).\n")
    # Sources were created before the PDF. Some filesystems have coarse mtimes,
    # so explicitly make the PDF one nanosecond newer than every source.
    sources = (main, abstract, body, refs, appendix, all_tables, claim_ledger)
    newest = max(path.stat().st_mtime_ns for path in sources)
    os.utime(pdf_path, ns=(newest + 1, newest + 1))

    page_text = {
        page: (audit.DEFAULT_TITLE + "\n" if page == 1 else "")
        + ("Conclusion\n" if page == 9 else "")
        + ("R E F E R E N C E S\n" if page == 10 else "")
        + ("A APPENDIX CONTRACT\n" if page == 11 else "")
        + ("substantive manuscript text " * 40)
        for page in range(1, 13)
    }
    return {
        "paper": paper,
        "pdf": pdf_path,
        "aux": aux_path,
        "log": log_path,
        "sources": sources,
        "main_text_sources": (main, abstract, body, claim_ledger),
        "destinations": destinations,
        "page_text": page_text,
    }


def _mock_tools(monkeypatch: pytest.MonkeyPatch,
                fixture: dict[str, object], *, title: str | None = None,
                fonts: str | None = None) -> None:
    destinations = fixture["destinations"]
    page_text = fixture["page_text"]

    def run(arguments: tuple[str, ...] | list[str]) -> str:
        if arguments[0] == "pdfinfo" and "-dests" in arguments:
            return "Page Destination Name\n" + "\n".join(
                f'{page:4d} [ XYZ 0 0 null ] "{name}"'
                for name, page in destinations.items()) + "\n"
        if arguments[0] == "pdfinfo":
            return f"Title: {title or audit.DEFAULT_TITLE}\nPages: 12\n"
        if arguments[0] == "pdffonts":
            return fonts or (
                "name type encoding emb sub uni object ID\n"
                "------------------------------------------\n"
                "AAAAAA+Times Type 1 Custom yes yes yes 1 0\n")
        if arguments[0] == "pdftotext":
            if "-f" in arguments:
                page = int(arguments[arguments.index("-f") + 1])
                return page_text[page]
            return "\f".join(page_text.values())
        raise AssertionError(arguments)

    monkeypatch.setattr(audit, "run_tool", run)


def _audit_fixture(root: Path, fixture: dict[str, object]) -> dict[str, object]:
    return audit.audit_manuscript(
        root,
        pdf=Path("paper_a/main.pdf"), aux=Path("paper_a/main.aux"),
        log=Path("paper_a/main.log"),
        main_source=Path("paper_a/main.tex"),
        sources=tuple(path.relative_to(root) for path in fixture["sources"]),
        main_text_sources=tuple(
            path.relative_to(root) for path in fixture["main_text_sources"]),
        expected_title=audit.DEFAULT_TITLE,
        completion_receipt=audit.DEFAULT_COMPLETION_RECEIPT,
        statistics_receipt=audit.DEFAULT_STATISTICS_RECEIPT,
        table_manifest=audit.DEFAULT_TABLE_MANIFEST,
        figure_manifests=audit.DEFAULT_FIGURE_MANIFESTS)


def test_graphicspath_resolution_matches_latex_search(tmp_path: Path) -> None:
    paper = tmp_path / "paper_a"
    figures = paper / "figures"
    figures.mkdir(parents=True)
    expected = figures / "result.pdf"
    expected.write_bytes(b"figure")
    source = (r"\graphicspath{{figures/}}" + "\n"
              + r"\includegraphics[width=\linewidth]{result.pdf}")
    assert audit.included_graphics(source, paper) == {expected.resolve()}


def test_default_sources_include_separate_main_claim_ledger() -> None:
    ledger = Path("paper_a/generated_results/main_claim_ledger.tex")
    appendix_bundle = Path("paper_a/generated_results/all_tables.tex")
    assert ledger in audit.DEFAULT_SOURCES
    assert ledger in audit.DEFAULT_MAIN_TEXT_SOURCES
    assert appendix_bundle in audit.DEFAULT_SOURCES
    assert appendix_bundle not in audit.DEFAULT_MAIN_TEXT_SOURCES


def test_full_synthetic_manuscript_passes_and_receipt_is_opt_in(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture_build(tmp_path)
    _mock_tools(monkeypatch, fixture)
    result = _audit_fixture(tmp_path, fixture)
    assert result["status"] == "verified"
    assert result["pagination"]["main_end_physical_page"] == 9
    assert result["references_page"] == 10
    assert result["appendix_pages"] == 2
    assert result["fonts"] == {
        "count": 1, "all_embedded": True, "type3_count": 0}
    receipt = Path("outputs/final_audit/receipt.json")
    assert audit.emit_receipt(
        tmp_path, receipt, result, execute=False) is False
    assert not (tmp_path / receipt).exists()
    assert audit.emit_receipt(
        tmp_path, receipt, result, execute=True) is True
    assert json.loads((tmp_path / receipt).read_text())["status"] == "verified"


def test_physical_marker_and_reference_boundaries_are_fail_closed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture_build(tmp_path)
    _mock_tools(monkeypatch, fixture)
    fixture["destinations"]["section*.15"] = 8
    with pytest.raises(audit.ManuscriptAuditFailure, match="physical page 9"):
        _audit_fixture(tmp_path, fixture)

    fixture = _fixture_build(tmp_path)
    fixture["page_text"][9] = "R E F E R E N C E S\n" + "text " * 200
    _mock_tools(monkeypatch, fixture)
    with pytest.raises(audit.ManuscriptAuditFailure,
                       match="References appears"):
        _audit_fixture(tmp_path, fixture)


@pytest.mark.parametrize("diagnostic", [
    "Overfull \\hbox (1.0pt too wide)",
    "LaTeX Warning: Reference `x' on page 1 undefined",
    "LaTeX Warning: Label(s) may have changed. Rerun",
    "Package rerunfilecheck Warning: Rerun required",
])
def test_build_log_rejects_layout_reference_and_rerun_warnings(
        diagnostic: str) -> None:
    log = diagnostic + "\nOutput written on main.pdf (12 pages, 10 bytes).\n"
    with pytest.raises(audit.ManuscriptAuditFailure,
                       match="forbidden diagnostic"):
        audit.validate_log(log, pages=12, pdf_size=10, pdf_name="main.pdf")


@pytest.mark.parametrize("font_line, message", [
    ("AAAA Type 1 Custom no no yes 1 0", "non-embedded"),
    ("AAAA Type 3 Custom yes no yes 1 0", "Type 3"),
])
def test_font_embedding_contract(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        font_line: str, message: str) -> None:
    fixture = _fixture_build(tmp_path)
    _mock_tools(monkeypatch, fixture, fonts=(
        "name type encoding emb sub uni object ID\n" + font_line + "\n"))
    with pytest.raises(audit.ManuscriptAuditFailure, match=message):
        _audit_fixture(tmp_path, fixture)


def test_title_labels_text_density_and_forbidden_markers(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture_build(tmp_path)
    _mock_tools(monkeypatch, fixture, title="Wrong title")
    with pytest.raises(audit.ManuscriptAuditFailure, match="metadata title"):
        _audit_fixture(tmp_path, fixture)

    fixture = _fixture_build(tmp_path)
    fixture["page_text"][4] = "tiny"
    _mock_tools(monkeypatch, fixture)
    with pytest.raises(audit.ManuscriptAuditFailure, match="too little"):
        _audit_fixture(tmp_path, fixture)

    fixture = _fixture_build(tmp_path)
    body = tmp_path / "paper_a/body.tex"
    body.write_text(body.read_text() + "The Wave 3 carrier (ours) is TODO.\n")
    pdf = tmp_path / "paper_a/main.pdf"
    os.utime(pdf, ns=(body.stat().st_mtime_ns + 1,
                      body.stat().st_mtime_ns + 1))
    _mock_tools(monkeypatch, fixture)
    with pytest.raises(audit.ManuscriptAuditFailure,
                       match="forbidden marker|forbidden marker"):
        _audit_fixture(tmp_path, fixture)

    fixture = _fixture_build(tmp_path)
    aux = tmp_path / "paper_a/main.aux"
    missing_table = sorted(audit.EXPECTED_MAIN_TABLES)[1]
    aux.write_text(aux.read_text().replace(
        _newlabel(missing_table, "2", 8, "table.caption.2") + "\n", ""))
    _mock_tools(monkeypatch, fixture)
    with pytest.raises(audit.ManuscriptAuditFailure, match="table labels"):
        _audit_fixture(tmp_path, fixture)


def _write_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt_pair(root: Path) -> tuple[Path, Path]:
    hashes = {}
    for name, relative in audit.SUMMARY_PATHS.items():
        hashes[name] = _write_json(root / relative, {"complete": name})
    completion = {
        "schema": "paper_a_cross_wave_completion_receipt_v1",
        "status": "complete", "read_only_verification": True,
        "scientific_cross_wave_aggregation": False,
        "sealed_locks_modified": False, "paper_files_modified": False,
        "waves": {
            "wave1_1": {"status": "verified",
                        "summary_sha256": hashes["matched"]},
            "wave2_v1_1": {"status": "verified",
                            "summary_sha256": hashes["pusht"]},
            "wave3": {"status": "verified",
                      "summary_sha256": hashes["pointmaze"],
                      "carrier_summary_sha256": hashes["pointmaze_carrier"],
                      "external_use_summary_sha256": hashes["pointmaze_use"]},
        },
    }
    statistics = {
        "schema": "paper_a_statistics_independent_receipt_v1",
        "status": "verified", "read_only": True,
        "statistics_computed": True, "imports_producer_statistics": False,
        "scientific_cross_family_pooling": False,
        "experiment_roots_modified": False,
        "waves": {
            "wave2": {"status": "verified",
                      "summary_sha256": hashes["pusht"]},
            "wave3": {"status": "verified",
                      "combined_summary_sha256": hashes["pointmaze"],
                      "carrier_summary_sha256": hashes["pointmaze_carrier"],
                      "external_use_summary_sha256": hashes["pointmaze_use"]},
        },
    }
    completion_path = root / audit.DEFAULT_COMPLETION_RECEIPT
    statistics_path = root / audit.DEFAULT_STATISTICS_RECEIPT
    _write_json(completion_path, completion)
    _write_json(statistics_path, statistics)
    return completion_path, statistics_path


def test_optional_receipts_bind_hashes_without_parsing_outcomes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _receipt_pair(tmp_path)
    original_read = audit.read_json

    def reject_summary_parse(path: Path, label: str) -> dict[str, object]:
        assert path not in {
            (tmp_path / relative).resolve()
            for relative in audit.SUMMARY_PATHS.values()}
        return original_read(path, label)

    monkeypatch.setattr(audit, "read_json", reject_summary_parse)
    result = audit.validate_optional_bindings(
        tmp_path, completion_receipt=audit.DEFAULT_COMPLETION_RECEIPT,
        statistics_receipt=audit.DEFAULT_STATISTICS_RECEIPT,
        table_manifest=audit.DEFAULT_TABLE_MANIFEST,
        figure_manifests=(), main_graphics=set())
    assert result["audit_receipts_present"] is True
    assert result["audit_receipts"]["summary_sha256"]["pusht"]


def test_single_or_invalid_receipt_never_hashes_partial_outcomes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    completion, statistics = _receipt_pair(tmp_path)
    statistics.unlink()
    touched: list[Path] = []
    original_hash = audit.sha256_file

    def track(path: Path) -> str:
        if path.resolve() in {
                (tmp_path / relative).resolve()
                for relative in audit.SUMMARY_PATHS.values()}:
            touched.append(path)
        return original_hash(path)

    monkeypatch.setattr(audit, "sha256_file", track)
    with pytest.raises(audit.ManuscriptAuditFailure, match="exactly one"):
        audit.validate_optional_bindings(
            tmp_path, completion_receipt=completion.relative_to(tmp_path),
            statistics_receipt=audit.DEFAULT_STATISTICS_RECEIPT,
            table_manifest=audit.DEFAULT_TABLE_MANIFEST,
            figure_manifests=(), main_graphics=set())
    assert touched == []


def test_generated_table_manifest_binds_receipts_summaries_and_artifacts(
        tmp_path: Path) -> None:
    completion, statistics = _receipt_pair(tmp_path)
    generator = tmp_path / audit.TABLE_GENERATOR
    generator.parent.mkdir(parents=True, exist_ok=True)
    generator_bytes = (audit.ROOT / audit.TABLE_GENERATOR).read_bytes()
    generator.write_bytes(generator_bytes)
    output = tmp_path / audit.DEFAULT_TABLE_MANIFEST.parent
    output.mkdir(parents=True, exist_ok=True)
    table = output / "result_table.tex"
    table.write_text("\\begin{tabular}{c}verified\\end{tabular}\n")
    summary_records = {
        name: {
            "path": str(relative),
            "sha256": hashlib.sha256(
                (tmp_path / relative).read_bytes()).hexdigest(),
        }
        for name, relative in audit.SUMMARY_PATHS.items()
    }
    manifest = {
        "schema": "paper_a_appendix_result_tables_v1",
        "status": "complete",
        "generator": {
            "path": str(audit.TABLE_GENERATOR),
            "size": generator.stat().st_size,
            "sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        },
        "source_identities": {
            "completion_receipt": {
                "path": str(completion.relative_to(tmp_path)),
                "sha256": hashlib.sha256(
                    completion.read_bytes()).hexdigest(),
            },
            "statistics_receipt": {
                "path": str(statistics.relative_to(tmp_path)),
                "sha256": hashlib.sha256(
                    statistics.read_bytes()).hexdigest(),
            },
            "summaries": summary_records,
        },
        "tables": {
            table.name: {
                "bytes": table.stat().st_size,
                "sha256": hashlib.sha256(table.read_bytes()).hexdigest(),
            },
        },
    }
    _write_json(tmp_path / audit.DEFAULT_TABLE_MANIFEST, manifest)
    result = audit.validate_optional_bindings(
        tmp_path, completion_receipt=audit.DEFAULT_COMPLETION_RECEIPT,
        statistics_receipt=audit.DEFAULT_STATISTICS_RECEIPT,
        table_manifest=audit.DEFAULT_TABLE_MANIFEST,
        figure_manifests=(), main_graphics=set())
    assert result["table_manifest_present"] is True
    assert result["table_manifest"]["artifacts_verified"] == 9

    manifest_without_generator = dict(manifest)
    manifest_without_generator.pop("generator")
    _write_json(tmp_path / audit.DEFAULT_TABLE_MANIFEST,
                manifest_without_generator)
    with pytest.raises(audit.ManuscriptAuditFailure, match="stale generator"):
        audit.validate_optional_bindings(
            tmp_path, completion_receipt=audit.DEFAULT_COMPLETION_RECEIPT,
            statistics_receipt=audit.DEFAULT_STATISTICS_RECEIPT,
            table_manifest=audit.DEFAULT_TABLE_MANIFEST,
            figure_manifests=(), main_graphics=set())

    _write_json(tmp_path / audit.DEFAULT_TABLE_MANIFEST, manifest)
    generator.write_text("tampered generator\n")
    with pytest.raises(audit.ManuscriptAuditFailure, match="stale generator"):
        audit.validate_optional_bindings(
            tmp_path, completion_receipt=audit.DEFAULT_COMPLETION_RECEIPT,
            statistics_receipt=audit.DEFAULT_STATISTICS_RECEIPT,
            table_manifest=audit.DEFAULT_TABLE_MANIFEST,
            figure_manifests=(), main_graphics=set())

    generator.write_bytes(generator_bytes)
    table.write_text("tampered\n")
    with pytest.raises(audit.ManuscriptAuditFailure,
                       match="byte count differs|hash differs"):
        audit.validate_optional_bindings(
            tmp_path, completion_receipt=audit.DEFAULT_COMPLETION_RECEIPT,
            statistics_receipt=audit.DEFAULT_STATISTICS_RECEIPT,
            table_manifest=audit.DEFAULT_TABLE_MANIFEST,
            figure_manifests=(), main_graphics=set())


def test_generated_figure_manifest_covers_and_binds_all_main_graphics(
        tmp_path: Path) -> None:
    completion, statistics = _receipt_pair(tmp_path)
    generator = tmp_path / "scripts/plot_paper_a_strengthened.py"
    generator.parent.mkdir(parents=True)
    generator.write_bytes(
        (audit.ROOT / "scripts/plot_paper_a_strengthened.py").read_bytes())
    directory = tmp_path / "paper_a/figures"
    directory.mkdir(parents=True)
    graphics: set[Path] = set()
    artifacts: dict[str, dict[str, object]] = {}
    for index in range(5):
        path = directory / f"main_{index}.pdf"
        path.write_bytes(f"figure-{index}".encode())
        graphics.add(path.resolve())
        artifacts[path.name] = {
            "path": str(path.relative_to(tmp_path)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest_path = directory / "manifest.json"
    manifest = {
        "schema": "paper_a_figure_manifest_v1",
        "status": "complete",
        "generator": {
            "path": str(generator.relative_to(tmp_path)),
            "sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
        },
        "audit_receipts": {
            "completion": {
                "path": str(completion.relative_to(tmp_path)),
                "sha256": hashlib.sha256(
                    completion.read_bytes()).hexdigest(),
            },
            "statistics": {
                "path": str(statistics.relative_to(tmp_path)),
                "sha256": hashlib.sha256(
                    statistics.read_bytes()).hexdigest(),
            },
        },
        "summaries": {
            name: {
                "path": str(relative),
                "sha256": hashlib.sha256(
                    (tmp_path / relative).read_bytes()).hexdigest(),
            }
            for name, relative in audit.SUMMARY_PATHS.items()
        },
        "artifacts": artifacts,
    }
    _write_json(manifest_path, manifest)
    result = audit.validate_optional_bindings(
        tmp_path, completion_receipt=audit.DEFAULT_COMPLETION_RECEIPT,
        statistics_receipt=audit.DEFAULT_STATISTICS_RECEIPT,
        table_manifest=audit.DEFAULT_TABLE_MANIFEST,
        figure_manifests=(manifest_path.relative_to(tmp_path),),
        main_graphics=graphics)
    assert result["figure_manifests"][0]["artifacts_verified"] == 13

    artifacts.pop("main_4.pdf")
    _write_json(manifest_path, manifest)
    with pytest.raises(audit.ManuscriptAuditFailure,
                       match="does not bind every main-text graphic"):
        audit.validate_optional_bindings(
            tmp_path, completion_receipt=audit.DEFAULT_COMPLETION_RECEIPT,
            statistics_receipt=audit.DEFAULT_STATISTICS_RECEIPT,
            table_manifest=audit.DEFAULT_TABLE_MANIFEST,
            figure_manifests=(manifest_path.relative_to(tmp_path),),
            main_graphics=graphics)


def test_receipt_cannot_be_written_inside_experiment_roots(
        tmp_path: Path) -> None:
    payload = {"status": "verified"}
    destination = audit.EXPERIMENT_ROOTS[0] / "final-audit.json"
    with pytest.raises(audit.ManuscriptAuditFailure,
                       match="outside experiment roots"):
        audit.emit_receipt(tmp_path, destination, payload, execute=True)
    assert not (tmp_path / destination).exists()
