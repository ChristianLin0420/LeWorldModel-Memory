# ICLR paper build

This directory contains the anonymous paper shell and provenance-checking build
tools for an analysis-rendered V18 manuscript.

## Files

- `main.tex` — anonymous document shell.
- `iclr2026_conference.sty` and `natbib.sty` — unmodified files from the latest
  publicly available official ICLR template at build time.
- `build_paper.py` — validates rendered Markdown and provenance-bound figures,
  then generates `abstract.tex`, `body.tex`, `refs.tex`, and `appendix.tex`.
- the rendered manuscript's sibling `figures/` directory — source figures and
  their provenance manifest.
- `figures/` — vector-PDF build copies of the architecture, registered-effect,
  task/corruption, and rank-distribution figures, written only after their
  source manifest hashes validate.
- the `--review-artifact` directory — redacted result/receipt bundle for
  double-blind review, including attempts, summary, and restart-audit v2.
- `main.pdf` — compiled anonymous paper.

The ICLR 2027 author guide and official style were not public when this kit was
prepared. The
current PDF therefore uses the official ICLR 2026 style as the closest public
format check. Replace it with the official 2027 package and recheck the page
limit before submission. Under the ICLR 2026 rule, the main body is at most nine
pages; references and appendices follow it.

## Regenerate and rebuild

In a completed release, the Markdown manuscript and adjacent manifest bind to
the anonymized JSON/CSV bundle. The provenance-bound figures are adjacent to
the manuscript. From the repository root, regenerate private-result-bound
public inputs with explicit checked-in paths:

```bash
.venv/bin/python scripts/plot_v18_paper.py \
  --root outputs/lewm_v8_v18_confirmation \
  --output-dir docs/figures

.venv/bin/python scripts/render_v18_paper.py \
  --root outputs/lewm_v8_v18_confirmation \
  --log-root logs/lewm_v8_v18_confirmation \
  --restart-audit docs/V18_RESTART_AUDIT.json \
  --template templates/ICLR.template.md \
  --output docs/ICLR.md \
  --manifest-output docs/ICLR.manifest.json

REVIEW_STAGE="$(mktemp -d)"
.venv/bin/python scripts/build_v18_review_artifact.py \
  --root outputs/lewm_v8_v18_confirmation \
  --log-root logs/lewm_v8_v18_confirmation \
  --protocol-document docs/V18_LEWM_V8_CONFIRMATION.md \
  --restart-audit docs/V18_RESTART_AUDIT.json \
  --output "$REVIEW_STAGE/review_artifact"

# Existing checked-in bundles are write-protected. Compare the staged bundle;
# replace paper/review_artifact only after all validation and review checks pass.
diff -ru paper/review_artifact "$REVIEW_STAGE/review_artifact"
```

The clean-clone paper rebuild needs no private output tree once `docs/ICLR.md`,
`docs/figures`, and `paper/review_artifact` are present:

```bash
.venv/bin/python paper/build_paper.py \
  --source docs/ICLR.md --paper-dir paper \
  --review-artifact paper/review_artifact \
  --pandoc /tmp/pandoc-3.10/bin/pandoc
PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH" \
  .venv/bin/python paper/check_v18_paper.py --compile \
    --paper-dir paper --manuscript docs/ICLR.md \
    --review-artifact paper/review_artifact \
    --output paper/paper_check.json
```

Authors with the private raw-artifact tree can regenerate the figures first
with the explicit `scripts/plot_v18_paper.py --root ... --output-dir ...`
command above. The plotter validates
the complete write-once analysis bundle and emits the vector architecture,
registered-effect forest, task/corruption, and rank-distribution figures plus
`fig_v18_manifest.json`; raw
checkpoints, epoch histories, rollout arrays, and identity-bearing remote
receipts are intentionally excluded from the double-blind repository.

The Markdown build fails on unresolved result placeholders, nonfinite tokens,
absent figures, known legacy figure hashes, or manifest mismatches. The final
checker fails on undefined/non-author-year citations, overfull boxes above 2 pt,
identity leaks or metadata, unofficial style bytes, and a main body over nine
pages.

Toolchain used for the checked build: TinyTeX (TeX Live 2026), `latexmk`, and
Pandoc 3.10.
