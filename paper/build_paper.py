r"""Convert docs/ICLR.md -> ICLR LaTeX fragments (abstract.tex, body.tex, refs.tex).

Strategy: pandoc mangles raw LaTeX math in markdown ($...$ breaks before digits; \(..\) is
stripped to literal parens). So we *tokenize* every math span and unicode char to an inert
ASCII placeholder before pandoc, then restore the real LaTeX in pandoc's output. Figures and
equations become raw LaTeX blocks (with tokenized content) that pandoc passes through verbatim.
"""
import argparse
import re
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import v18_release_common as common

SRC = ROOT / 'generated' / 'ICLR.md'
OUT = ROOT / 'paper'
RESULTS = ROOT / 'generated' / 'review_artifact'
PANDOC = shutil.which('pandoc') or '/tmp/pandoc-3.10/bin/pandoc'
LLM_USAGE_STATEMENT = (
    'OpenAI Codex assisted with code review, experiment monitoring, artifact '
    'auditing, deterministic result-to-manuscript tooling, and manuscript '
    'drafting/editing. The authors verified the executed code, artifacts, '
    'statistics, citations, and final claims and retain responsibility for the work.'
)
STALE_FIGURE_HASHES = {
    '3aab1670',  # legacy T-Maze/Distractor/Recall/Occlusion bar chart
    '2aa8f910',  # legacy POPGym Arcade summary
}

UNI = {
    '±': r'\(\pm\)', '—': '---', '§': r'\S{}', '×': r'\(\times\)', '–': '--',
    '−': r'\(-\)', '→': r'\(\to\)', '·': r'\(\cdot\)', '≈': r'\(\approx\)',
    'Δ': r'\(\Delta\)', 'τ': r'\(\tau\)', '≥': r'\(\ge\)', '≠': r'\(\ne\)', '≪': r'\(\ll\)',
    '“': '``', '”': "''", '‘': '`', '’': "'",
}

MAP = {}
_ctr = [0]


def tok(latex):
    _ctr[0] += 1
    t = 'XXMATH%05dXX' % _ctr[0]
    MAP[t] = latex
    return t


def tokenize(text):
    text = re.sub(r'\$\$(.+?)\$\$',
                  lambda m: tok(r'\begin{equation}' + m.group(1).strip() + r'\end{equation}'),
                  text, flags=re.S)
    text = re.sub(r'\$([^$\n]+?)\$', lambda m: tok(r'\(' + m.group(1) + r'\)'), text)
    for ch, lx in UNI.items():
        if ch in text:
            text = text.replace(ch, tok(lx))
    return text


def restore(text):
    for t, lx in MAP.items():
        text = text.replace(t, lx)
    return text


def md_inline_to_tex(s):
    # escape LaTeX specials in manually-built (non-pandoc) text: refs + figure captions.
    # (math/unicode are still inert tokens at this point, so this can't corrupt them.)
    for ch in ('&', '_', '#', '%'):
        s = re.sub(r'(?<!\\)' + re.escape(ch), '\\' + ch, s)
    s = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', s)
    s = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\\emph{\1}', s)
    return s


def strip_header_num(line):
    return re.sub(r'^(#{2,3})\s+\d+(?:\.\d+)*[a-z]?\.?\s+', r'\1 ', line)


def convert_figures(lines):
    out, i = [], 0
    img = re.compile(r'^!\[[^\]]*\]\(([^)]+)\)\s*$')
    while i < len(lines):
        if img.match(lines[i]):
            imgs = []
            while i < len(lines) and img.match(lines[i]):
                imgs.append(img.match(lines[i]).group(1)); i += 1
            j = i
            while j < len(lines) and lines[j].strip() == '':
                j += 1
            cap = ''
            if j < len(lines) and lines[j].strip().startswith('*') and lines[j].strip().endswith('*'):
                cap = lines[j].strip().strip('*').strip(); i = j + 1
            cap = md_inline_to_tex(cap)
            cap = re.sub(r'^Figures?\s+\S+?\.\s*', '', cap)
            stem = Path(imgs[0]).stem if len(imgs) == 1 else ''
            placement = {
                'fig_v18_architecture': '!tbp',
                'fig_v18_evidence': '!t',
                'fig_v18_secondary': '!t',
                'fig_v18_task_design': '!tbp',
            }.get(stem, '!tbp')
            out += ['', rf'\begin{{figure}}[{placement}]', r'\centering']
            if len(imgs) == 1:
                paper_path = str(Path(imgs[0]).with_suffix('.pdf'))
                out.append(r'\includegraphics[width=\linewidth]{%s}' % paper_path)
            else:
                w = 0.95 / len(imgs)
                for k, p in enumerate(imgs):
                    p = str(Path(p).with_suffix('.pdf'))
                    out.append(r'\includegraphics[width=%.2f\linewidth]{%s}%s'
                               % (w, p, '' if k == len(imgs) - 1 else r'\hfill'))
            if cap:
                out.append(r'\caption{%s}' % cap)
            if len(imgs) == 1:
                label = Path(imgs[0]).stem.replace('_', '-')
                out.append(r'\label{fig:%s}' % label)
            out += [r'\end{figure}', '']
        else:
            out.append(lines[i]); i += 1
    return out


def style_longtables(text):
    """Hook for table-level postprocessing; captions carry the shared theme."""

    return text


def run_pandoc(md, *, shift=False):
    command = [PANDOC, '-f', 'markdown+raw_tex', '-t', 'latex']
    if shift:
        command.append('--shift-heading-level-by=-1')
    result = subprocess.run(command, input=md, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"pandoc failed with exit {result.returncode}:\n{result.stderr.strip()}")
    return result.stdout


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manuscript_bundle():
    manuscript_manifest_path = SRC.with_suffix('.manifest.json')
    analysis_path = RESULTS / 'confirmation_analysis.json'
    cells_path = RESULTS / 'confirmation_cells.csv'
    contrasts_path = RESULTS / 'confirmation_contrasts.csv'
    review_manifest_path = RESULTS / 'review_manifest.json'
    required = (
        manuscript_manifest_path, analysis_path, cells_path, contrasts_path,
        review_manifest_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f'missing manuscript/result provenance files: {missing}')
    manuscript = json.loads(manuscript_manifest_path.read_text(encoding='utf-8'))
    report = json.loads(analysis_path.read_text(encoding='utf-8'))
    review = json.loads(review_manifest_path.read_text(encoding='utf-8'))
    checks = {
        'manuscript SHA-256': (manuscript.get('manuscript_sha256'), sha256(SRC)),
        'analysis SHA-256': (manuscript.get('analysis_sha256'), sha256(analysis_path)),
        'cell CSV SHA-256': (manuscript.get('cells_sha256'), sha256(cells_path)),
        'contrast CSV SHA-256': (
            manuscript.get('contrasts_sha256'), sha256(contrasts_path)),
        'analysis-bound cell CSV': (report.get('cells_csv_sha256'), sha256(cells_path)),
        'analysis-bound contrast CSV': (
            report.get('contrasts_csv_sha256'), sha256(contrasts_path)),
        'scientific label': (
            manuscript.get('scientific_label'), report.get('scientific_label')),
    }
    mismatches = [
        f'{name}: expected={expected!r}, actual={actual!r}'
        for name, (expected, actual) in checks.items()
        if expected != actual
    ]
    if report.get('status') != 'COMPLETE' \
            or report.get('artifact_integrity_passed') is not True \
            or report.get('completed_valid_cells') != 200:
        mismatches.append('analysis report is not a complete 200-cell valid result')
    if report.get('scientific_label') != 'CONFIRMATION_FAILED' \
            or report.get('official_confirmation_result') is not False:
        mismatches.append(
            'falsification manuscript requires CONFIRMATION_FAILED and '
            'official_confirmation_result=false')
    if manuscript.get('schema_version') != 2:
        mismatches.append('manuscript manifest is not provenance schema v2')
    if manuscript.get('llm_usage_statement_present') is not True \
            or SRC.read_text(encoding='utf-8').count(LLM_USAGE_STATEMENT) != 1:
        mismatches.append('manuscript lacks the required LLM Usage Statement receipt')
    review_files = review.get('files', {})
    for path in sorted(RESULTS.iterdir()):
        if path.is_file() and path.name != 'review_manifest.json':
            if review_files.get(path.name) != sha256(path):
                mismatches.append(f'review artifact hash differs for {path.name}')
    source_hashes = review.get('source_result_hashes', {})
    for key, manifest_key in (
            ('confirmation_analysis.json', 'analysis_sha256'),
            ('confirmation_cells.csv', 'cells_sha256'),
            ('confirmation_contrasts.csv', 'contrasts_sha256')):
        if source_hashes.get(key) != manuscript.get(manifest_key):
            mismatches.append(f'review/manuscript source binding differs for {key}')
    if review.get('source_restart_audit_sha256') != manuscript.get('restart_audit_sha256'):
        mismatches.append('review/manuscript restart-audit source binding differs')
    if review.get('canonical_commands_sha256') != manuscript.get(
            'canonical_commands_sha256'):
        mismatches.append('review/manuscript canonical command binding differs')
    if review.get('restart_schema_version') != 2 \
            or review.get('restart_interruptions') != 2:
        mismatches.append('review artifact lacks the two-interruption v2 audit')
    if mismatches:
        raise RuntimeError('manuscript/result provenance mismatch: ' + '; '.join(mismatches))
    return manuscript, review


def validate_and_copy_figures(source, manuscript_manifest):
    paths = re.findall(r'^!\[[^\]]*\]\(([^)]+)\)\s*$', source, flags=re.M)
    expected_paths = {
        'figures/fig_v18_architecture.png',
        'figures/fig_v18_evidence.png',
        'figures/fig_v18_secondary.png',
        'figures/fig_v18_task_design.png',
    }
    if set(paths) != expected_paths:
        raise RuntimeError(f'unexpected paper figure set: {paths!r}')
    manifest_path = SRC.parent / 'figures' / 'fig_v18_manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError('missing V18 figure provenance manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    secondary = manifest.get('descriptive_secondary')
    secondary_rows = secondary.get('public_prior_slices', []) \
        if isinstance(secondary, dict) else []
    secondary_grid = {
        (row.get('task'), row.get('seed'), row.get('design'))
        for row in secondary_rows if isinstance(row, dict)
    }
    if manifest.get('schema_version') != 3 \
            or manifest.get('artifact_kind') != 'v18_provenance_bound_paper_figures' \
            or not isinstance(secondary, dict) \
            or secondary.get('official_decision_changed') is not False \
            or secondary.get('decision_gates_defined') is not False \
            or secondary.get('claim_scope') != 'descriptive decomposition only' \
            or len(secondary_rows) != 200 or len(secondary_grid) != 200 \
            or not re.fullmatch(r'[0-9a-f]{64}', str(
                secondary.get('metrics_json_manifest_sha256', ''))):
        raise RuntimeError('descriptive figure provenance payload differs')
    if manuscript_manifest.get('figure_manifest_sha256') != sha256(manifest_path):
        raise RuntimeError('manuscript was rendered from a different figure manifest')
    figure_checks = {
        'analysis_sha256': manuscript_manifest['analysis_sha256'],
        'cells_sha256': manuscript_manifest['cells_sha256'],
        'contrasts_sha256': manuscript_manifest['contrasts_sha256'],
        'protocol_sha256': manuscript_manifest['protocol_sha256'],
        'summary_sha256': manuscript_manifest['summary_sha256'],
        'scientific_label': manuscript_manifest['scientific_label'],
    }
    mismatches = [
        f'{key}: figure={manifest.get(key)!r}, manuscript={expected!r}'
        for key, expected in figure_checks.items()
        if manifest.get(key) != expected
    ]
    if mismatches:
        raise RuntimeError(
            'figure/manuscript provenance mismatch: ' + '; '.join(mismatches))
    expected_files = {
        Path(relative).with_suffix(suffix).name
        for relative in expected_paths
        for suffix in ('.png', '.pdf')
    }
    if set(manifest.get('figures', {})) != expected_files:
        raise RuntimeError('figure provenance file set differs')
    destination = OUT / 'figures'
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob('fig_v18_*'):
        if stale.is_file():
            stale.unlink()
    for relative in paths:
        path = SRC.parent / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f'missing paper figure {path}')
        digest = sha256(path)
        if digest[:8] in STALE_FIGURE_HASHES:
            raise RuntimeError(f'refusing known stale legacy figure {path}')
        expected = manifest.get('figures', {}).get(path.name)
        if expected != digest:
            raise RuntimeError(f'figure provenance hash differs for {path.name}')
        pdf_path = path.with_suffix('.pdf')
        pdf_digest = sha256(pdf_path) if pdf_path.is_file() else None
        if manifest.get('figures', {}).get(pdf_path.name) != pdf_digest:
            raise RuntimeError(f'figure provenance hash differs for {pdf_path.name}')
        shutil.copy2(pdf_path, destination / pdf_path.name)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SRC)
    parser.add_argument('--paper-dir', type=Path, default=OUT)
    parser.add_argument('--review-artifact', type=Path, default=RESULTS)
    parser.add_argument('--pandoc', type=Path, default=Path(PANDOC))
    return parser.parse_args()


def main():
    global MAP, SRC, OUT, RESULTS, PANDOC
    args = parse_args()
    SRC = args.source.resolve()
    OUT = args.paper_dir.resolve()
    RESULTS = args.review_artifact.resolve()
    PANDOC = str(args.pandoc.resolve())
    MAP = {}
    _ctr[0] = 0
    if not Path(PANDOC).is_file():
        raise RuntimeError('pandoc is unavailable; install it or set it on PATH')
    version = subprocess.run(
        [PANDOC, '--version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    if not version.startswith('pandoc 3.10'):
        raise RuntimeError(f'paper build requires the pinned pandoc 3.10.x, found {version}')
    OUT.mkdir(parents=True, exist_ok=True)
    for required in ('main.tex', 'iclr2026_conference.sty'):
        if not (OUT / required).is_file():
            raise RuntimeError(f'missing paper build input {OUT / required}')
    source = SRC.read_text(encoding='utf-8')
    leftovers = sorted(set(re.findall(r'\{\{[A-Z0-9_]+\}\}', source)))
    if leftovers:
        raise RuntimeError(f'unrendered manuscript placeholders: {leftovers}')
    if re.search(r'(?i)(?:^|\W)(?:nan|[+-]?inf)(?:$|\W)', source):
        raise RuntimeError('manuscript contains a nonfinite numeric token')
    if r'\cite{' in source or r'\citep{' not in source:
        raise RuntimeError('manuscript must use parenthetical natbib \\citep citations')
    manuscript_manifest, review_manifest = validate_manuscript_bundle()
    validate_and_copy_figures(source, manuscript_manifest)
    lines = source.split('\n')
    def idx(p): return next(k for k, l in enumerate(lines) if l.strip().startswith(p))
    i_abs, i_intro, i_refs = idx('## Abstract'), idx('## 1. Introduction'), idx('## References')
    i_appendix = idx('## Appendix')
    abstract = '\n'.join(lines[i_abs + 1:i_intro])
    body = '\n'.join(lines[i_intro:i_refs])
    refs_raw = '\n'.join(lines[i_refs + 1:i_appendix])
    appendix = '\n'.join(lines[i_appendix:])

    # abstract
    atex = restore(run_pandoc(tokenize(abstract)))

    # body
    btok = tokenize(body)
    bl = [strip_header_num(l) for l in btok.split('\n')]
    bl = [r'\section*{Reproducibility Statement}' if l.strip() == '## Reproducibility Statement' else l
          for l in bl]
    bl = convert_figures(bl)
    btex = style_longtables(run_pandoc('\n'.join(bl), shift=True))

    # appendix: strip literal Appendix A/B labels because LaTeX numbers them.
    atok = tokenize(appendix)
    al = [re.sub(r'^(##)\s+Appendix\s+[A-Z]\.\s+', r'\1 ', line)
          for line in atok.split('\n')]
    al = [re.sub(r'^(###)\s+[A-Z]\.\d+\s+', r'\1 ', line) for line in al]
    al = [strip_header_num(line) for line in al]
    al = convert_figures(al)
    aptex = style_longtables(run_pandoc('\n'.join(al), shift=True))

    # references (manual; not pandoc'd)
    entries = [e.strip().replace('\n', ' ') for e in refs_raw.split('·') if e.strip()]
    out = [r'\begin{thebibliography}{99}\small']
    indexed_entries = sorted(
        enumerate(entries, 1),
        key=lambda item: item[1].split(',', 1)[0].casefold(),
    )
    for k, e in indexed_entries:
        years = re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)', e)
        if not years:
            raise RuntimeError(f'reference {k} has no publication year: {e}')
        author_block = e.split('. ', 1)[0].strip().rstrip('.')
        if ' et al' in author_block:
            citation_author = author_block.rstrip('.') + '.'
        else:
            authors = [author.strip() for author in author_block.split(',')]
            if len(authors) == 1:
                citation_author = authors[0]
            elif len(authors) == 2:
                citation_author = authors[0] + r' \& ' + authors[1]
            else:
                citation_author = authors[0] + ' et al.'
        for ch, lx in UNI.items():
            e = e.replace(ch, lx)
        citation_author = md_inline_to_tex(citation_author)
        out.append(r'\bibitem[%s(%s)]{ref%d} %s' % (
            citation_author, years[-1], k, md_inline_to_tex(e)))
    out.append(r'\end{thebibliography}')
    fragments = {
        'abstract.tex': atex,
        'body.tex': restore(btex),
        'appendix.tex': '\\appendix\n' + restore(aptex),
        'refs.tex': '\n'.join(out) + '\n',
    }
    staging = Path(tempfile.mkdtemp(prefix='.paper-fragments-', dir=OUT))
    try:
        for name, value in fragments.items():
            (staging / name).write_text(value, encoding='utf-8')
        for name in fragments:
            os.replace(staging / name, OUT / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    build_manifest = {
        'schema_version': 1,
        'scope': 'v18_paper_fragments',
        'pandoc_version': version,
        'manuscript_sha256': sha256(SRC),
        'manuscript_manifest_sha256': sha256(SRC.with_suffix('.manifest.json')),
        'review_manifest_sha256': sha256(RESULTS / 'review_manifest.json'),
        'main_tex_sha256': sha256(OUT / 'main.tex'),
        'style_sha256': sha256(OUT / 'iclr2026_conference.sty'),
        'llm_usage_statement_present': True,
        'source_figure_manifest_sha256': sha256(
            SRC.parent / 'figures' / 'fig_v18_manifest.json'),
        'fragments': {name: sha256(OUT / name) for name in fragments},
        'figures': {
            path.name: sha256(path)
            for path in sorted((OUT / 'figures').glob('*')) if path.is_file()
        },
        'builder_sha256': sha256(Path(__file__).resolve()),
    }
    common.atomic_write_json(OUT / 'paper_build_manifest.json', build_manifest)
    print('tokens: %d | refs: %d | abstract %d chars' % (len(MAP), len(entries), len(atex)))


if __name__ == '__main__':
    main()
