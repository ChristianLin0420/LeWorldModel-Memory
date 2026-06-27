"""Convert docs/ICLR.md -> ICLR LaTeX fragments (abstract.tex, body.tex, refs.tex).

Strategy: pandoc mangles raw LaTeX math in markdown ($...$ breaks before digits; \(..\) is
stripped to literal parens). So we *tokenize* every math span and unicode char to an inert
ASCII placeholder before pandoc, then restore the real LaTeX in pandoc's output. Figures and
equations become raw LaTeX blocks (with tokenized content) that pandoc passes through verbatim.
"""
import re
import subprocess
from pathlib import Path

SRC = Path('docs/ICLR.md')
OUT = Path('paper')
PANDOC = '/tmp/pandoc-3.10/bin/pandoc'

UNI = {
    '±': r'\(\pm\)', '—': '---', '§': r'\S{}', '×': r'\(\times\)', '–': '--',
    '−': r'\(-\)', '→': r'\(\to\)', '·': r'\(\cdot\)', '≈': r'\(\approx\)',
    'Δ': r'\(\Delta\)', 'τ': r'\(\tau\)', '≥': r'\(\ge\)', '≠': r'\(\ne\)', '≪': r'\(\ll\)',
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
            # keep the markdown's literal "Figure N." (body text references it by number),
            # bold the label, and use \caption* so LaTeX does not re-number (which would
            # collapse the double-image blocks and desync from the in-text references).
            cap = md_inline_to_tex(cap)
            cap = re.sub(r'^(Figures?\s+\S+?\.)', r'\\textbf{\1}', cap)
            out += ['', r'\begin{figure}[t]', r'\centering']
            if len(imgs) == 1:
                out.append(r'\includegraphics[width=0.82\linewidth]{%s}' % imgs[0])
            else:
                w = 0.95 / len(imgs)
                for k, p in enumerate(imgs):
                    out.append(r'\includegraphics[width=%.2f\linewidth]{%s}%s'
                               % (w, p, '' if k == len(imgs) - 1 else r'\hfill'))
            if cap:
                out.append(r'\caption*{%s}' % cap)
            out += [r'\end{figure}', '']
        else:
            out.append(lines[i]); i += 1
    return out


def pandoc(md, extra=''):
    fmt = 'markdown+raw_tex' + extra
    return subprocess.run([PANDOC, '-f', fmt, '-t', 'latex'] +
                          (['--shift-heading-level-by=-1'] if extra == '_body' else []),
                          input=md, capture_output=True, text=True, check=True).stdout


def main():
    lines = SRC.read_text(encoding='utf-8').split('\n')
    def idx(p): return next(k for k, l in enumerate(lines) if l.strip().startswith(p))
    i_abs, i_intro, i_refs = idx('## Abstract'), idx('## 1. Introduction'), idx('## References')
    abstract = '\n'.join(lines[i_abs + 1:i_intro])
    body = '\n'.join(lines[i_intro:i_refs])
    refs_raw = '\n'.join(lines[i_refs + 1:])

    # abstract
    atex = restore(subprocess.run([PANDOC, '-f', 'markdown+raw_tex', '-t', 'latex'],
                                  input=tokenize(abstract), capture_output=True, text=True,
                                  check=True).stdout)
    (OUT / 'abstract.tex').write_text(atex, encoding='utf-8')

    # body
    btok = tokenize(body)
    bl = [strip_header_num(l) for l in btok.split('\n')]
    bl = [r'\section*{Reproducibility Statement}' if l.strip() == '## Reproducibility Statement' else l
          for l in bl]
    bl = convert_figures(bl)
    btex = subprocess.run([PANDOC, '-f', 'markdown+raw_tex', '-t', 'latex',
                           '--shift-heading-level-by=-1'],
                          input='\n'.join(bl), capture_output=True, text=True, check=True).stdout
    (OUT / 'body.tex').write_text(restore(btex), encoding='utf-8')

    # references (manual; not pandoc'd)
    entries = [e.strip().replace('\n', ' ') for e in refs_raw.split('·') if e.strip()]
    out = [r'\begin{thebibliography}{99}\small']
    for k, e in enumerate(entries, 1):
        for ch, lx in UNI.items():
            e = e.replace(ch, lx)
        out.append(r'\bibitem{ref%d} %s' % (k, md_inline_to_tex(e)))
    out.append(r'\end{thebibliography}')
    (OUT / 'refs.tex').write_text('\n'.join(out) + '\n', encoding='utf-8')
    print('tokens: %d | refs: %d | abstract %d chars' % (len(MAP), len(entries), len(atex)))


if __name__ == '__main__':
    main()
