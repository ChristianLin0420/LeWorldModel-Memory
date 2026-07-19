#!/usr/bin/env python3
"""Overnight auto-resume for non-manual OGBench memory validation.

The controller is deliberately conservative:
1. wait for the active breadth wave to finish;
2. generate figures and update the HTML status block;
3. rerun any failed env-age rows on a larger 768-episode confirmation bank;
4. launch extra locomotion validation envs one by one so one bad env does not
   block the entire overnight run.

It never promotes a result to a paper claim; it only records evidence and
starts the next validation stage.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path("python")

STATUS_ROOT = ROOT / "outputs" / "random_patchset_view_color_jepa_autorun_v1"
STATUS_PATH = STATUS_ROOT / "status.json"
HTML_PATH = ROOT / "docs" / "mesm_nvidia_plan.html"
PAPER_B = ROOT / "paper_b"
PAPER_AUTO_TEX = PAPER_B / "generated_results" / "nonmanual_breadth_auto.tex"
PAPER_FIGURES = PAPER_B / "figures"
START = "<!-- NONMANUAL_AUTORUN_STATUS_START -->"
END = "<!-- NONMANUAL_AUTORUN_STATUS_END -->"

BREADTH = ROOT / "outputs" / "random_patchset_view_color_jepa_breadth_v1"
BREADTH_PID = BREADTH / "orchestrator_breadth.pid"
CONFIRM = ROOT / "outputs" / "random_patchset_view_color_jepa_breadth_confirm_v1"
EXTRA_ROOT = ROOT / "outputs" / "random_patchset_view_color_jepa_extra_v1"

RUNNER = ROOT / "scripts" / "launch_random_patchset_view_color_jepa_ogbench.py"
PLOTTER = ROOT / "scripts" / "plot_masked_evidence_jepa_ogbench.py"
COMMON = [
    "--gpus", "0", "1", "2",
    "--img-size", "64",
    "--epochs", "72",
    "--batch-size", "64",
    "--dim", "224",
    "--slots", "12",
    "--heads", "4",
    "--resume",
]


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_status(payload: dict[str, Any]) -> None:
    STATUS_ROOT.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = now()
    STATUS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    update_html(payload)


def update_html(payload: dict[str, Any]) -> None:
    if not HTML_PATH.exists():
        return
    text = HTML_PATH.read_text()
    if START not in text or END not in text:
        return
    stage = payload.get("stage", "unknown")
    detail = payload.get("detail", "")
    rows = payload.get("rows", [])
    row_html = ""
    if rows:
        items = "".join(
            f"<li><code>{row.get('env_name')}</code> age {row.get('age')}: "
            f"{row.get('pass_count')}/{row.get('seed_count')} seeds, "
            f"full={row.get('full_bacc_mean', 0):.3f}, "
            f"reset={row.get('reset_bacc_mean', 0):.3f}, "
            f"recent={row.get('no_state_bacc_mean', 0):.3f}</li>"
            for row in rows[:10]
        )
        row_html = f"<ul>{items}</ul>"
    figure_html = ""
    figures = payload.get("figures", [])
    if figures:
        cards = []
        for figure in figures:
            src = figure.get("src", "")
            title = figure.get("title", "Auto-generated figure")
            caption = figure.get("caption", "")
            if not src:
                continue
            cards.append(
                f"<div class=\"render-card\" style=\"margin-top:12px\">"
                f"<img src=\"{src}\" alt=\"{title}\">"
                f"<div class=\"render-label\"><span>{title}</span><span>{caption}</span></div>"
                f"</div>"
            )
        figure_html = "".join(cards)
    block = (
        f"{START}\n"
        f"        <div class=\"callout\" style=\"margin-top:16px\">\n"
        f"          <strong>Overnight auto-resume status</strong>\n"
        f"          <p><b>{stage}</b> — {detail}</p>\n"
        f"          {row_html}\n"
        f"          {figure_html}\n"
        f"          <p><small>Last update: {payload.get('updated_at', now())}. "
        f"Machine-readable status: <code>outputs/random_patchset_view_color_jepa_autorun_v1/status.json</code>.</small></p>\n"
        f"        </div>\n"
        f"        {END}"
    )
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    HTML_PATH.write_text(before + block + after)


def pid_alive(pid_file: Path) -> bool:
    if not pid_file.exists():
        return False
    pid = pid_file.read_text().strip()
    if not pid:
        return False
    return subprocess.run(["ps", "-p", pid], stdout=subprocess.DEVNULL).returncode == 0


def wait_pid(pid_file: Path, label: str) -> None:
    while pid_alive(pid_file):
        result_count = len(list(pid_file.parent.glob("**/result.json")))
        write_status({
            "stage": f"waiting: {label}",
            "detail": f"{result_count} cell result files are present; process is still active.",
        })
        time.sleep(180)


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            write_status({
                "stage": "running subprocess",
                "detail": f"{' '.join(command[:4])} ... pid={proc.pid}",
            })
            time.sleep(180)
        return int(proc.returncode)


def plot_summary(summary: Path, output: Path) -> None:
    if summary.exists():
        subprocess.run([str(PYTHON), str(PLOTTER), "--summary", str(summary), "--output", str(output)], cwd=ROOT, check=False)


def tex_escape(value: object) -> str:
    text = str(value)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def paper_figure_name(prefix: str, source: Path) -> Path:
    safe = source.parent.name.replace("-", "_").replace("/", "_")
    return PAPER_FIGURES / f"{prefix}_{safe}.pdf"


def render_summary_table(summary: Path, title: str, source_label: str, figure: Path | None = None) -> str:
    if not summary.exists():
        return (
            f"\\paragraph{{{tex_escape(title)}.}} "
            f"No completed summary was available at generation time for "
            f"\\path{{{tex_escape(source_label)}}}.\n\n"
        )
    data = read_json(summary)
    rows = data.get("rows", [])
    if not rows:
        return (
            f"\\paragraph{{{tex_escape(title)}.}} "
            f"The summary at \\path{{{tex_escape(source_label)}}} contains no completed rows.\n\n"
        )
    pass_rows = sum(1 for row in rows if row.get("all_pass"))
    tex = [
        f"\\paragraph{{{tex_escape(title)}.}}",
        f"This generated block reports {pass_rows}/{len(rows)} environment-age rows passing the",
        "registered non-manual gate. Mixed rows are treated as stress-test limits or",
        "confirmation targets, not as headline wins.",
        "",
    ]
    if figure is not None and figure.exists():
        tex.extend([
            "\\begin{figure}[!tbp]",
            "\\centering",
            f"\\includegraphics[width=\\linewidth]{{figures/{figure.name}}}",
            f"\\caption{{Generated non-manual breadth plot for {tex_escape(title)}.}}",
            "\\end{figure}",
            "",
        ])
    tex.extend([
        "\\begin{table}[!tbp]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\renewcommand{\\arraystretch}{1.05}",
        f"\\caption{{Generated non-manual breadth table for {tex_escape(title)}.}}",
        "\\begin{tabular}{lccccccl}",
        "\\toprule",
        "Deck & Age & Seeds & Pass & Full & Reset & Recent & Status\\\\",
        "\\midrule",
    ])
    for row in rows:
        status = r"\textsc{pass}" if row.get("all_pass") else r"\textsc{mixed}"
        tex.append(
            f"{tex_escape(row.get('env_name', 'unknown'))} & "
            f"{int(row.get('age', -1))} & "
            f"{int(row.get('seed_count', 0))} & "
            f"{int(row.get('pass_count', 0))}/{int(row.get('seed_count', 0))} & "
            f"{float(row.get('full_bacc_mean', 0.0)):.3f} & "
            f"{float(row.get('reset_bacc_mean', 0.0)):.3f} & "
            f"{float(row.get('no_state_bacc_mean', 0.0)):.3f} & "
            f"{status}\\\\"
        )
    tex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(tex)


def compile_paper() -> int:
    tectonic = ROOT / ".tools" / "tectonic"
    if not tectonic.exists():
        return 127
    log_path = STATUS_ROOT / "logs" / "paper_b_compile.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as handle:
        handle.write(f"\n===== compile {now()} =====\n".encode())
        proc = subprocess.run([str(tectonic), "-X", "compile", "main.tex"], cwd=PAPER_B, stdout=handle, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def write_paper_appendix(stage_note: str, specs: list[tuple[str, Path, Path | None]]) -> int:
    PAPER_AUTO_TEX.parent.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    blocks = [
        "\\section{Generated non-manual breadth monitor}",
        "\\label{app:nonmanual-breadth-auto}",
        "",
        "This section is generated automatically by the overnight controller.  It is",
        "included to remove ambiguity between clean paper-level claims and broader",
        "stress tests.  A row marked mixed is not promoted to a main-text claim; it",
        "identifies an environment-age setting that requires confirmation or a design",
        "change.",
        "",
        f"\\paragraph{{Generation status.}} {tex_escape(stage_note)} Last update: {tex_escape(now())}.",
        "",
    ]
    for title, summary, figure in specs:
        blocks.append(render_summary_table(summary, title, str(summary.relative_to(ROOT)), figure))
    PAPER_AUTO_TEX.write_text("\n".join(blocks) + "\n")
    return compile_paper()


def ensure_summary(output: Path) -> Path:
    """Build a robust env-age summary directly from per-cell result files.

    The launchers normally call the runner's aggregate mode at the end, but the
    overnight controller should still work when the launcher exits after all
    cells yet before writing summary.json.
    """
    summary_path = output / "summary.json"
    cells: list[dict[str, Any]] = []
    for result_path in sorted(output.glob("**/result.json")):
        try:
            cell = read_json(result_path)
        except Exception:
            continue
        if "readout" not in cell or "gate" not in cell:
            continue
        cells.append(cell)
    if not cells:
        if not summary_path.exists():
            summary_path.write_text(stable_json({
                "schema": "random_patchset_jepa_ogbench_summary_v1",
                "output": str(output),
                "rows": [],
                "cell_count": 0,
                "warning": "no completed per-cell result files found",
            }))
        return summary_path

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[(str(cell["env_name"]), int(cell["age"]))].append(cell)

    rows: list[dict[str, Any]] = []
    for (env_name, age), values in sorted(grouped.items()):
        def metric(arm: str, key: str) -> float:
            xs = [float(v["readout"][arm][key]) for v in values if arm in v.get("readout", {})]
            return float(sum(xs) / len(xs)) if xs else 0.0

        def retrieval(key: str) -> float:
            xs = [float(v.get("retrieval", {}).get("full", {}).get(key, 0.0)) for v in values]
            return float(sum(xs) / len(xs)) if xs else 0.0

        pass_count = sum(1 for v in values if bool(v.get("gate", {}).get("pass", False)))
        rows.append({
            "env_name": env_name,
            "age": int(age),
            "seed_count": int(len(values)),
            "seeds": sorted(int(v.get("seed", -1)) for v in values),
            "pass_count": int(pass_count),
            "all_pass": bool(pass_count == len(values)),
            "full_bacc_mean": metric("full", "balanced_accuracy"),
            "reset_bacc_mean": metric("reset", "balanced_accuracy"),
            "no_state_bacc_mean": metric("no_state", "balanced_accuracy"),
            "retrieval_top1_mean": retrieval("top1"),
            "failed_seeds": [
                int(v.get("seed", -1))
                for v in values
                if not bool(v.get("gate", {}).get("pass", False))
            ],
        })

    summary_path.write_text(stable_json({
        "schema": "random_patchset_jepa_ogbench_summary_v1",
        "output": str(output),
        "cell_count": int(len(cells)),
        "env_age_count": int(len(rows)),
        "rows": rows,
    }))
    return summary_path


def failed_rows(summary: Path) -> list[dict[str, Any]]:
    if not summary.exists():
        return []
    data = read_json(summary)
    return [row for row in data.get("rows", []) if not row.get("all_pass", False)]


def launch_confirmation(rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    envs = sorted({str(row["env_name"]) for row in rows})
    ages = sorted({str(int(row["age"])) for row in rows})
    command = [
        str(PYTHON), "-u", str(RUNNER),
        "--output", str(CONFIRM),
        "--envs", *envs,
        "--ages", *ages,
        "--seeds", "0", "1", "2",
        "--episodes", "768",
        *COMMON,
    ]
    write_status({
        "stage": "launching confirmation",
        "detail": f"Rerunning failed breadth rows on 768 episodes: envs={envs}, ages={ages}.",
        "rows": rows,
    })
    return run_command(command, CONFIRM / "logs" / "orchestrator_confirm.log")


def launch_extra_env(env_name: str) -> int:
    output = EXTRA_ROOT / env_name.replace("/", "_")
    command = [
        str(PYTHON), "-u", str(RUNNER),
        "--output", str(output),
        "--envs", env_name,
        "--ages", "4", "8", "15",
        "--seeds", "0", "1", "2",
        "--episodes", "384",
        *COMMON,
    ]
    write_status({
        "stage": "launching extra env",
        "detail": f"Running extra validation env {env_name}.",
    })
    return run_command(command, output / "logs" / "orchestrator_extra.log")


def main() -> None:
    write_status({"stage": "started", "detail": "Auto-resume controller is active."})
    wait_pid(BREADTH_PID, "breadth wave")
    breadth_summary = ensure_summary(BREADTH)
    breadth_figure = ROOT / "docs" / "assets" / "random_patchset_view_color_jepa_breadth_summary.svg"
    breadth_pdf = PAPER_FIGURES / "fig_b_nonmanual_breadth_auto.pdf"
    plot_summary(breadth_summary, breadth_figure)
    plot_summary(breadth_summary, breadth_pdf)
    bad = failed_rows(breadth_summary)
    compile_code = write_paper_appendix(
        f"Breadth wave completed with {len(bad)} mixed environment-age rows.",
        [("Random patch-set breadth wave", breadth_summary, breadth_pdf)],
    )
    write_status({
        "stage": "breadth completed",
        "detail": f"{len(bad)} failed env-age rows found in breadth summary. Paper rebuild exit code {compile_code}.",
        "rows": bad,
        "figures": [
            {
                "src": "assets/random_patchset_view_color_jepa_breadth_summary.svg",
                "title": "Random patch-set JEPA breadth summary",
                "caption": "active breadth wave",
            }
        ],
    })

    if bad:
        code = launch_confirmation(bad)
        confirm_summary = ensure_summary(CONFIRM)
        confirm_figure = ROOT / "docs" / "assets" / "random_patchset_view_color_jepa_breadth_confirm.svg"
        confirm_pdf = PAPER_FIGURES / "fig_b_nonmanual_breadth_confirm_auto.pdf"
        plot_summary(confirm_summary, confirm_figure)
        plot_summary(confirm_summary, confirm_pdf)
        confirm_bad = failed_rows(confirm_summary)
        compile_code = write_paper_appendix(
            f"Large-bank confirmation completed with {len(confirm_bad)} mixed environment-age rows.",
            [
                ("Random patch-set breadth wave", breadth_summary, breadth_pdf),
                ("Large-bank confirmation", confirm_summary, confirm_pdf),
            ],
        )
        write_status({
            "stage": "confirmation completed",
            "detail": f"Confirmation exit code {code}; failed rows: {len(confirm_bad)}. Paper rebuild exit code {compile_code}.",
            "rows": confirm_bad,
            "figures": [
                {
                    "src": "assets/random_patchset_view_color_jepa_breadth_confirm.svg",
                    "title": "Large-bank confirmation summary",
                    "caption": "failed rows rerun with 768 episodes",
                }
            ],
        })

    extra_envs = [
        "antmaze-large-navigate-v0",
        "antmaze-giant-navigate-v0",
        "humanoidmaze-large-navigate-v0",
    ]
    extra_results = []
    for env_name in extra_envs:
        code = launch_extra_env(env_name)
        output = EXTRA_ROOT / env_name.replace("/", "_")
        summary = ensure_summary(output)
        figure_name = f"random_patchset_view_color_jepa_extra_{env_name.replace('-', '_')}.svg"
        pdf_name = f"fig_b_nonmanual_extra_{env_name.replace('-', '_')}.pdf"
        plot_summary(summary, ROOT / "docs" / "assets" / figure_name)
        plot_summary(summary, PAPER_FIGURES / pdf_name)
        extra_results.append({
            "env_name": env_name,
            "exit_code": code,
            "failed_rows": failed_rows(summary),
            "figure": f"assets/{figure_name}",
            "paper_figure": f"figures/{pdf_name}",
            "summary": str(summary),
        })
        specs = [("Random patch-set breadth wave", breadth_summary, breadth_pdf)]
        if bad:
            specs.append(("Large-bank confirmation", CONFIRM / "summary.json", PAPER_FIGURES / "fig_b_nonmanual_breadth_confirm_auto.pdf"))
        for item in extra_results:
            specs.append((
                item["env_name"],
                Path(item["summary"]),
                PAPER_B / item["paper_figure"],
            ))
        write_paper_appendix(f"Extra environment {env_name} completed.", specs)
    figures = [
        {
            "src": item["figure"],
            "title": item["env_name"],
            "caption": f"exit={item['exit_code']}, failed rows={len(item['failed_rows'])}",
        }
        for item in extra_results
    ]
    final_specs = [("Random patch-set breadth wave", breadth_summary, breadth_pdf)]
    if bad:
        final_specs.append(("Large-bank confirmation", CONFIRM / "summary.json", PAPER_FIGURES / "fig_b_nonmanual_breadth_confirm_auto.pdf"))
    for item in extra_results:
        final_specs.append((item["env_name"], Path(item["summary"]), PAPER_B / item["paper_figure"]))
    compile_code = write_paper_appendix("Overnight sequence completed.", final_specs)
    write_status({
        "stage": "overnight sequence completed",
        "detail": f"Breadth, confirmation, and extra-env validation sequence finished. Paper rebuild exit code {compile_code}.",
        "extra_results": extra_results,
        "figures": figures,
    })


if __name__ == "__main__":
    main()
