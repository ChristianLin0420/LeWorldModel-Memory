#!/usr/bin/env python3
"""Monitor temporal-coverage JEPA weak-env sweep and update the HTML report."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path("python")

OUTPUT = ROOT / "outputs" / "multiview_patchset_color_jepa_weak_v1"
PID_FILE = OUTPUT / "orchestrator_multiview.pid"
STATUS = OUTPUT / "monitor_status.json"
HTML = ROOT / "docs" / "mesm_nvidia_plan.html"
PLOTTER = ROOT / "scripts" / "plot_masked_evidence_jepa_ogbench.py"
RUNNER = ROOT / "scripts" / "run_multiview_patchset_color_jepa_ogbench.py"
FIGURE = ROOT / "docs" / "assets" / "multiview_patchset_color_jepa_weak_summary.svg"
DONE_NOTICE = OUTPUT / "DONE.txt"
START = "<!-- MULTIVIEW_WEAK_STATUS_START -->"
END = "<!-- MULTIVIEW_WEAK_STATUS_END -->"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def pid_alive() -> bool:
    if not PID_FILE.exists():
        return False
    pid = PID_FILE.read_text().strip()
    return bool(pid) and subprocess.run(["ps", "-p", pid], stdout=subprocess.DEVNULL).returncode == 0


def summarize_from_cells() -> Path:
    summary = OUTPUT / "summary.json"
    cells = []
    for path in sorted(OUTPUT.glob("**/result.json")):
        try:
            cell = read_json(path)
        except Exception:
            continue
        if "readout" in cell and "gate" in cell:
            cells.append(cell)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[(str(cell["env_name"]), int(cell["age"]))].append(cell)
    rows = []
    for (env_name, age), values in sorted(grouped.items()):
        def mean(arm: str) -> float:
            xs = [float(v["readout"][arm]["balanced_accuracy"]) for v in values]
            return float(sum(xs) / len(xs)) if xs else 0.0

        def retrieval_top1() -> float:
            xs = [float(v.get("retrieval", {}).get("full", {}).get("top1", 0.0)) for v in values]
            return float(sum(xs) / len(xs)) if xs else 0.0

        pass_count = sum(1 for v in values if bool(v.get("gate", {}).get("pass", False)))
        rows.append({
            "env_name": env_name,
            "age": int(age),
            "seed_count": int(len(values)),
            "seeds": sorted(int(v.get("seed", -1)) for v in values),
            "pass_count": int(pass_count),
            "all_pass": bool(pass_count == len(values)),
            "full_bacc_mean": mean("full"),
            "reset_bacc_mean": mean("reset"),
            "no_state_bacc_mean": mean("no_state"),
            "retrieval_top1_mean": retrieval_top1(),
        })
    summary.write_text(stable_json({
        "schema": "multiview_patchset_color_jepa_weak_summary_v1",
        "cell_count": int(len(cells)),
        "env_age_count": int(len(rows)),
        "rows": rows,
    }))
    return summary


def update_html(payload: dict[str, Any]) -> None:
    if not HTML.exists():
        return
    figure_html = ""
    if FIGURE.exists():
        figure_html = (
            "<div class=\"render-card\" style=\"margin-top:14px\">"
            "<img src=\"assets/multiview_patchset_color_jepa_weak_summary.svg\" "
            "alt=\"Temporal-coverage JEPA weak-env summary\">"
            "<div class=\"render-label\"><span>Temporal-coverage weak-env sweep</span>"
            "<span>automatic monitor</span></div></div>"
        )
    rows = payload.get("rows", [])
    row_html = ""
    if rows:
        passed = sum(1 for row in rows if row.get("all_pass", False))
        min_age15 = min(
            (float(row["full_bacc_mean"]) for row in rows if int(row["age"]) == 15),
            default=0.0,
        )
        worst = min(rows, key=lambda row: float(row["full_bacc_mean"]))
        body = "".join(
            "<tr>"
            f"<td><code>{row['env_name']}</code></td>"
            f"<td class=\"num\">{row['age']}</td>"
            f"<td class=\"num\">{row['pass_count']} / {row['seed_count']}</td>"
            f"<td class=\"num\">{row['full_bacc_mean']:.3f}</td>"
            f"<td class=\"num\">{row['reset_bacc_mean']:.3f}</td>"
            f"<td class=\"num\">{row['no_state_bacc_mean']:.3f}</td>"
            f"<td><span class=\"status-pill {'pass' if row.get('all_pass', False) else 'fail'}\">"
            f"{'Pass' if row.get('all_pass', False) else 'Mixed'}</span></td>"
            "</tr>"
            for row in rows
        )
        row_html = (
            "<div class=\"matrix\" style=\"margin-top:14px\">"
            f"<div class=\"cell\"><b>{passed}/{len(rows)}</b><p>env-age rows pass the memory gate.</p></div>"
            f"<div class=\"cell\"><b>{min_age15:.3f}</b><p>worst age-15 full-memory readout across environments.</p></div>"
            f"<div class=\"cell\"><b>{worst['env_name']}</b><p>hardest row: age {worst['age']} with "
            f"{worst['pass_count']}/{worst['seed_count']} seeds passing.</p></div>"
            "<div class=\"cell\"><b>Success-rate pending</b><p>This sweep measures retention/readout. "
            "Native success requires a fixed controller and is launched separately for supported OGBench envs.</p></div>"
            "</div>"
            "<div class=\"table-wrap\" style=\"margin-top:14px\">"
            "<table><caption>Compact retention gate summary; not native environment success.</caption>"
            "<thead><tr><th>Environment</th><th>Age</th><th>Seeds</th><th>Full</th>"
            "<th>Reset</th><th>Recent</th><th>Status</th></tr></thead>"
            f"<tbody>{body}</tbody></table></div>"
            "<div class=\"table-wrap\" style=\"margin-top:14px\">"
            "<table><caption>Native environment success-rate evidence currently available. "
            "These are fixed-controller use checks, not native planner claims. "
            "The new temporal-coverage current-method all-env supervisor is active at "
            "<code>outputs/multiview_patchset_color_jepa_native_use_all_v1</code>.</caption>"
            "<thead><tr><th>Setting</th><th>Env / host</th><th>Age</th><th>Full-memory success</th>"
            "<th>Main baseline</th><th>Random / no-state</th><th>Oracle</th><th>Status</th></tr></thead>"
            "<tbody>"
            "<tr><td>Checkpointed PushT</td><td>DINO-WM</td><td class=\"num\">15</td>"
            "<td class=\"num\">0.972</td><td class=\"num\">reset 0.167</td>"
            "<td class=\"num\">no-state 0.168</td><td class=\"num\">—</td>"
            "<td><span class=\"status-pill pass\">Available</span></td></tr>"
            "<tr><td>Checkpointed PushT</td><td>LeWM</td><td class=\"num\">15</td>"
            "<td class=\"num\">0.926</td><td class=\"num\">reset 0.156</td>"
            "<td class=\"num\">no-state 0.151</td><td class=\"num\">—</td>"
            "<td><span class=\"status-pill pass\">Available</span></td></tr>"
            "<tr><td>Fixed-panel ME-JEPA</td><td>PointMaze-large</td><td class=\"num\">4 / 8 / 15</td>"
            "<td class=\"num\">1.000 / 1.000 / 1.000</td>"
            "<td class=\"num\">recent 0.255 / 0.281 / 0.264</td>"
            "<td class=\"num\">random 0.238</td><td class=\"num\">1.000</td>"
            "<td><span class=\"status-pill pass\">Available</span></td></tr>"
            "<tr><td>Fixed-panel ME-JEPA</td><td>Cube-single</td><td class=\"num\">4 / 8 / 15</td>"
            "<td class=\"num\">1.000 / 1.000 / 0.996</td>"
            "<td class=\"num\">recent 0.281 / 0.255 / 0.264</td>"
            "<td class=\"num\">random 0.238</td><td class=\"num\">1.000</td>"
            "<td><span class=\"status-pill pass\">Available</span></td></tr>"
            "<tr><td>Feature-host bridge</td><td>PointMaze-large + Cube-single</td>"
            "<td class=\"num\">4 / 8 / 15</td><td class=\"num\">1.000 all rows</td>"
            "<td class=\"num\">reset 0.250</td><td class=\"num\">random/no-state 0.250</td>"
            "<td class=\"num\">1.000</td><td><span class=\"status-pill pass\">Available</span></td></tr>"
            "<tr><td>Temporal-coverage current method</td><td>All tested OGBench envs</td>"
            "<td class=\"num\">4 / 8 / 15</td><td class=\"num\">pending</td>"
            "<td class=\"num\">reset / recent</td><td class=\"num\">random</td><td class=\"num\">oracle</td>"
            "<td><span class=\"status-pill partial\">Running all-env supervisor</span></td></tr>"
            "</tbody></table></div>"
            "<p>Insight: the retention sweep proves old evidence survives the context boundary; "
            "the fixed-controller success rows show when that memory readout changes the final "
            "environment outcome. The missing comparison is the exact temporal-coverage current-method "
            "success rate, so the all-env supervisor is now scheduled at "
            "<code>outputs/multiview_patchset_color_jepa_native_use_all_v1</code>. It covers PointMaze "
            "medium/large/giant/teleport, Cube single/double/triple, Scene, Puzzle, AntMaze large/giant, "
            "and HumanoidMaze large. AntMaze and HumanoidMaze will be reported as controller-unavailable "
            "unless a valid low-level policy is added; Scene will only be reported if its fixed-controller "
            "oracle gate passes.</p>"
        )
    block = f"""
        {START}
        <section class=\"section-block\" id=\"multiview-weak-status\">
          <div class=\"section-kicker\">Active improvement run</div>
          <h2>Temporal-coverage salient JEPA weak-env sweep</h2>
          <p><b>{payload.get('stage', 'unknown')}</b> — {payload.get('detail', '')}</p>
          {row_html}
          {figure_html}
          <p><small>Last update: {payload.get('updated_at', now())}. Machine-readable status: <code>outputs/multiview_patchset_color_jepa_weak_v1/monitor_status.json</code>.</small></p>
        </section>
        {END}
"""
    text = HTML.read_text()
    if START in text and END in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        HTML.write_text(before + block + after)
    else:
        HTML.write_text(text.replace("</main>", block + "\n</main>"))


def write_status(payload: dict[str, Any]) -> None:
    payload["updated_at"] = now()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(stable_json(payload))
    update_html(payload)


def emit_completion_notice(payload: dict[str, Any]) -> None:
    message = (
        f"Temporal-coverage JEPA sweep completed at {payload.get('updated_at', now())}.\n"
        f"{payload.get('detail', '')}\n"
        f"Status: {STATUS.relative_to(ROOT)}\n"
        f"HTML: docs/mesm_nvidia_plan.html\n"
    )
    DONE_NOTICE.write_text(message)
    subprocess.run(
        ["bash", "-lc", f"command -v notify-send >/dev/null && notify-send 'LeWorldModel-Memory run complete' {json.dumps(payload.get('detail', 'sweep completed'))} || true"],
        cwd=ROOT,
        check=False,
    )


def main() -> None:
    while pid_alive():
        count = len(list(OUTPUT.glob("**/result.json")))
        write_status({
            "stage": "running",
            "detail": f"{count}/27 cells completed.",
        })
        time.sleep(180)
    summary = summarize_from_cells()
    subprocess.run([str(PYTHON), str(PLOTTER), "--summary", str(summary), "--output", str(FIGURE)], cwd=ROOT, check=False)
    data = read_json(summary)
    rows = data.get("rows", [])
    mixed = [row for row in rows if not row.get("all_pass", False)]
    payload = {
        "stage": "completed",
        "detail": f"{len(rows) - len(mixed)}/{len(rows)} env-age rows passed; {len(mixed)} mixed rows remain.",
        "rows": rows,
        "summary": str(summary),
        "figure": str(FIGURE),
    }
    write_status(payload)
    payload = read_json(STATUS)
    emit_completion_notice(payload)


if __name__ == "__main__":
    main()
