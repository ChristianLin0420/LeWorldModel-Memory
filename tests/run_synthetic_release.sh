#!/usr/bin/env bash
set -euo pipefail

KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_FULL="${PYTHON_FULL:-python}"
PANDOC="${PANDOC:-/tmp/pandoc-3.10/bin/pandoc}"
BASE="${BASE:-$KIT/_synthetic}"

cd "$KIT"
"$PYTHON_FULL" tests/make_synthetic_v18_bundle.py --output "$BASE"
"$PYTHON_FULL" scripts/bind_v18_restart_audit.py \
  --root "$BASE/result" \
  --record "$BASE/provenance/restart_interruptions.record.json" \
  --log-root "$BASE/logs" \
  --output "$BASE/provenance/v18_restart_audit.portable.v2.json"
"$PYTHON_FULL" tests/verify_synthetic_release.py \
  --base "$BASE" --prepare-canonical
"$PYTHON_FULL" scripts/plot_v18_paper.py \
  --root "$BASE/result" --output-dir "$BASE/generated/figures"
"$PYTHON_FULL" scripts/render_v18_paper.py \
  --root "$BASE/result" \
  --template "$KIT/templates/ICLR.template.md" \
  --restart-audit "$BASE/provenance/v18_restart_audit.v2.json" \
  --log-root "$BASE/logs" \
  --output "$BASE/generated/ICLR.md"
"$PYTHON_FULL" scripts/build_v18_review_artifact.py \
  --root "$BASE/result" \
  --protocol-document "$BASE/provenance/V18_LEWM_V8_CONFIRMATION.md" \
  --restart-audit "$BASE/provenance/v18_restart_audit.v2.json" \
  --log-root "$BASE/logs" \
  --output "$BASE/generated/review_artifact" \
  --forbid synthetic-private
cp paper/main.tex paper/iclr2026_conference.sty paper/natbib.sty "$BASE/paper/"
"$PYTHON_FULL" paper/build_paper.py \
  --source "$BASE/generated/ICLR.md" \
  --paper-dir "$BASE/paper" \
  --review-artifact "$BASE/generated/review_artifact" \
  --pandoc "$PANDOC"
"$PYTHON_FULL" paper/check_v18_paper.py --compile \
  --paper-dir "$BASE/paper" \
  --manuscript "$BASE/generated/ICLR.md" \
  --review-artifact "$BASE/generated/review_artifact" \
  --output "$BASE/generated/paper_check.json"
"$PYTHON_FULL" scripts/render_v18_release_docs.py \
  --root "$BASE/result" \
  --readme-template templates/README.template.md \
  --readme-output "$BASE/generated/README.final.md" \
  --evidence-template templates/LEARNABLE_MEMORY.v18.patch.template.md \
  --evidence-output "$BASE/generated/LEARNABLE_MEMORY.v18.patch.md" \
  --pdf-check "$BASE/generated/paper_check.json" \
  --manifest-output "$BASE/generated/release_docs.manifest.json"
"$PYTHON_FULL" scripts/build_v18_code_supplement.py \
  --repo-root "$BASE/private_repo" \
  --result-root "$BASE/result" \
  --review-artifact "$BASE/generated/review_artifact" \
  --manuscript "$BASE/generated/ICLR.md" \
  --paper-dir "$BASE/paper" \
  --paper-check "$BASE/generated/paper_check.json" \
  --release-tool-root "$KIT" \
  --output-dir "$BASE/v18-anonymous-supplement" \
  --output-zip "$BASE/v18-anonymous-supplement.zip" \
  --forbid-file "$BASE/private_identity_tokens.json"
(
  cd /tmp
  umask 077
  HOME="$BASE/fake-home" USER=fake-user HOSTNAME=fake-host \
  TZ=Pacific/Honolulu LC_ALL=C PYTHONHASHSEED=123 \
  "$PYTHON_FULL" "$KIT/scripts/build_v18_code_supplement.py" \
    --repo-root "$BASE/private_repo" \
    --result-root "$BASE/result" \
    --review-artifact "$BASE/generated/review_artifact" \
    --manuscript "$BASE/generated/ICLR.md" \
    --paper-dir "$BASE/paper" \
    --paper-check "$BASE/generated/paper_check.json" \
    --release-tool-root "$KIT" \
    --output-dir "$BASE/v18-anonymous-supplement-repeat" \
    --output-zip "$BASE/v18-anonymous-supplement-repeat.zip" \
    --forbid-file "$BASE/private_identity_tokens.json"
)
"$PYTHON_FULL" tests/verify_synthetic_release.py --base "$BASE"
