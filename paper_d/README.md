# Causal-Effect Memory for Raw Visual Events — ICLR 2027 paper

This directory is self-contained for LaTeX compilation after figures and tables
have been generated. It does not modify or depend on `paper_c/` at build time.

## Generate figures and tables

Run from the repository root:

```bash
.venv/bin/python scripts/plot_paper_d.py
```

Required Python packages: `numpy`, `matplotlib`, `Pillow`, and `cairosvg`.

The raw generator reads:

- `outputs/cem_raw_ogbench/report.json`
- `outputs/cem_raw_ogbench/cells/<env>/s<seed>/{result.json,decision_log.json}`
- original render caches under
  `outputs/multiview_patchset_color_jepa_native_v1/cache/`

`plot_paper_d.py` deterministically selects visibly changing unmodified frames,
embeds them in the self-contained architecture SVG, writes
`generated_results/architecture_example_receipt.json`, and generates all
white-background paper figures.

The controlled-protocol generator reads these historical artifact families:

- `outputs/cem_auto_discovery_v1/report.json`
- `outputs/cem_auto_discovery_v2/report.json`
- `outputs/cem_event_versioning_v1/report.json`
- `outputs/cem_event_versioning_dinowm_official_v1/report.json`
- `outputs/cem_v3_report.json`
- `outputs/cem_lewm_semantic_adapter_v1/report.json`
- `outputs/cem_lewm_memory_tokens_v1/report.json`
- `outputs/cem_lewm_memory_experts_v1/report.json`
- `outputs/cem_lewm_adaln_memory_v1/report.json`
- `outputs/cem_lewm_hybrid_interface_v1/report.json`
- `outputs/cem_lewm_lora_memory_v1/report.json`

The script writes publication artifacts to `paper_d/figures/` and
`paper_d/generated_results/`. Those generated files remove the LaTeX build's
dependency on the gitignored `outputs/` directory. The breadth host is a
DINO-feature action-conditioned world model, not official DINO-WM; the
official Wall result is retained as a separate controlled protocol.

## Build the PDF

```bash
cd paper_d
/home/chrislin/.TinyTeX/bin/x86_64-linux/pdflatex -interaction=nonstopmode -halt-on-error main.tex
/home/chrislin/.TinyTeX/bin/x86_64-linux/pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Validation commands:

```bash
pdfinfo main.pdf
rg 'Undefined|LaTeX Warning: (Reference|Citation)|^!|Overfull' main.log
mkdir -p /tmp/paper_d_pages
pdftoppm -png -r 110 main.pdf /tmp/paper_d_pages/page
```

The main paper plus references must occupy pages 1--9. The appendix begins on
page 10 after the explicit `\clearpage` in `main.tex`. Inspect page 1, the raw
architecture page, the raw results page, page 9, and page 10 after every layout
change.

## Raw protocol reproduction

```bash
.venv/bin/python -m pytest -q scripts/test_run_cem_raw_ogbench.py
.venv/bin/python scripts/launch_cem_raw_ogbench.py --smoke --gpus 1 2
.venv/bin/python scripts/launch_cem_raw_ogbench.py --campaign --gpus 0 1 2
```

The launcher rejects GPU3. The no-manual-event contract and ignored cache keys
are recorded in every receipt and result.

The copied `iclr2027_conference.sty` and `natbib.sty` are unchanged from the
proven `paper_c` build contract.
