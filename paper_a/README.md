# Paper A build

`main.pdf` is the anonymous ICLR-format manuscript generated from validated
experiment summaries and vector figures.

From the repository root, rebuild it with:

```bash
.venv/bin/python scripts/audit_paper_a_cross_wave_completion.py --execute
.venv/bin/python scripts/audit_paper_a_statistics_independent.py --execute
.venv/bin/python scripts/generate_paper_a_appendix_tables.py \
  --output-dir paper_a/generated_results --execute
.venv/bin/python scripts/plot_paper_a_strengthened.py
cd paper_a
PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH" \
  latexmk -g -pdf -interaction=nonstopmode -halt-on-error main.tex
cd ..
.venv/bin/python scripts/audit_paper_a_final_manuscript.py --execute
```

The plotter and table generator refuse to read pending DINO-WM outcomes until
the official verifiers and both independent post-lock audit receipts bind the
completed summary hashes. The plotter then reads the authenticated Reacher,
PushT, matched-host, context/rollout, decision-use, DINO-WM, repair, and
failed-admission summaries.
It writes the `fig_mem_*` PDF/PNG assets used by `body.tex` and `appendix.tex`.
The table generator writes a complete appendix bundle plus a separate
task-level main-paper claim ledger; both are bound in its manifest.
Figure generation is staged under `paper_a/`; validated assets are published
to `paper_a/figures` with the source-bound manifest committed last.
The manuscript then compiles directly with the bundled ICLR 2026 style.

The manuscript has nine main-text pages; references begin on page 10 after the
`paper-a-main-end` marker. An extended appendix follows the references and
contains complete memory-module grids, cue-offset checks, context/rollout
controls, decision-use results, the DINO-WM portability audit, repair
diagnostics, failed task admissions, and the reproducibility ledger.
The final command writes
`outputs/paper_a_final_manuscript_audit/receipt.json` only after checking the
compiled-source set, exact pagination, figure/table bindings, embedded fonts,
and the two independent experiment-audit receipts.
