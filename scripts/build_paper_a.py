#!/usr/bin/env python3
"""Build the Paper A PDF: docs/PAPER_A.md -> paper_a/main.pdf.

Lean sibling of paper/build_paper.py (which is V18-locked): tokenize math
spans so pandoc cannot mangle them, convert body to LaTeX with the pinned
pandoc, wrap in the ICLR 2026 style (copied from paper/), compile with
TinyTeX latexmk.  Draft mode: textual citations, \\iclrfinalcopy off.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "PAPER_A.md"
OUT = ROOT / "paper_a"
PANDOC = "/tmp/pandoc-3.10/bin/pandoc"
TEXBIN = Path.home() / ".TinyTeX" / "bin" / "x86_64-linux"

UNI = {
    "±": r"\(\pm\)", "—": "---", "§": r"\S{}", "×": r"\(\times\)",
    "–": "--", "−": r"\(-\)", "→": r"\(\to\)", "·": r"\(\cdot\)",
    "≈": r"\(\approx\)", "Δ": r"\(\Delta\)", "τ": r"\(\tau\)",
    "≥": r"\(\ge\)", "≤": r"\(\le\)", "≠": r"\(\ne\)", "≫": r"\(\gg\)",
    "≪": r"\(\ll\)", "ξ": r"\(\xi\)", "σ": r"\(\sigma\)",
    "“": "``", "”": "''", "‘": "`", "’": "'", "✓": r"\checkmark{}",
    "🤖": "", "★": r"\(\star\)",
}

MAP: dict[str, str] = {}
_counter = [0]


def tok(latex: str) -> str:
    _counter[0] += 1
    token = "XXMATHTOK%05dXX" % _counter[0]
    MAP[token] = latex
    return token


def tokenize(text: str) -> str:
    text = re.sub(r"\$([^$\n]+?)\$",
                  lambda m: tok(r"\(" + m.group(1) + r"\)"), text)
    for char, latex in UNI.items():
        text = text.replace(char, tok(latex) if latex.startswith("\\") or
                            latex.startswith("\\(") else latex)
    return text


def restore(text: str) -> str:
    for token, latex in MAP.items():
        text = text.replace(token, latex)
    return text


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    for name in ("iclr2026_conference.sty", "natbib.sty"):
        shutil.copy(ROOT / "paper" / name, OUT / name)
    shutil.copy(ROOT / "templates" / "PAPER_A.refs.tex", OUT / "refs.tex")
    for fig in (ROOT / "docs" / "figures").glob("fig_a_*.pdf"):
        shutil.copy(fig, OUT / "figures" / fig.name)

    text = SRC.read_text()
    title_match = re.match(r"# (.+)\n", text)
    title = title_match.group(1)
    text = text[title_match.end():]
    abstract_match = re.search(r"## Abstract\n\n(.+?)\n\n## ", text, re.S)
    abstract = abstract_match.group(1)
    body = text[abstract_match.end() - len("## "):]
    # png figure refs -> pdf (vector in the paper)
    body = re.sub(r"figures/(fig_a_\w+)\.png", r"figures/\1.pdf", body)

    # LaTeX numbers sections itself: strip the manual "N." / "N.M" heading
    # prefixes and promote H2 -> \section
    body = re.sub(r"^## \d+\. ", "## ", body, flags=re.M)
    body = re.sub(r"^### \d+\.\d+ ", "### ", body, flags=re.M)
    body_tex = subprocess.run(
        [PANDOC, "-f", "markdown+tex_math_dollars", "-t", "latex",
         "--wrap=preserve", "--shift-heading-level-by=-1"],
        input=tokenize(body), text=True, capture_output=True,
        check=True).stdout
    abstract_tex = subprocess.run(
        [PANDOC, "-f", "markdown", "-t", "latex", "--wrap=preserve"],
        input=tokenize(abstract), text=True, capture_output=True,
        check=True).stdout
    body_tex = restore(body_tex)
    abstract_tex = restore(abstract_tex)
    # normalize pandoc's image options (keepaspectratio,alt={...}) to a
    # plain columnwidth constraint
    body_tex = re.sub(r"\\includegraphics(\[[^{]*?\])?\{",
                      r"\\includegraphics[width=\\columnwidth]{", body_tex)
    # pandoc's calc-based column widths break under the ICLR style; use
    # fixed fractions and plain tabulars instead of longtables
    body_tex = re.sub(
        r"\(\\(?:linewidth|columnwidth) - \d+\\tabcolsep\) \* "
        r"\\real\{([\d.]+)\}", r"\1\\linewidth", body_tex)

    def to_table(match: re.Match) -> str:
        spec, inner = match.group(1).replace("@{}", ""), match.group(2)
        caption = ""
        cap = re.search(r"\\caption\{((?:[^{}]|\{[^{}]*\})*)\}"
                        r"\\tabularnewline\n?", inner)
        if cap:
            caption = f"\\caption{{{cap.group(1)}}}\n"
            inner = inner[:cap.start()] + inner[cap.end():]
        # captioned longtables carry a duplicate header block
        inner = re.sub(r"\\endfirsthead.*?\\endhead", "", inner, flags=re.S)
        inner = inner.replace("\\endhead", "")
        inner = re.sub(r"\\bottomrule(\\noalign\{\})?\s*\\endlastfoot", "",
                       inner)
        inner = inner.replace("\\noalign{}", "")
        inner = inner.replace("\\toprule\n",
                              "\\toprule\\rowcolor{NVIDIAHeader}\n", 1)
        return ("\\begin{table}[h]\\small\\centering\n" + caption
                + f"\\begin{{tabular}}{{{spec}}}" + inner.rstrip()
                + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")

    body_tex = re.sub(
        r"\\begin\{longtable\}\[\]\{(.*?)\}\n(.*?)\\end\{longtable\}",
        to_table, body_tex, flags=re.S)
    # references (natbib thebibliography) precede the lettered appendix
    body_tex = re.sub(r"(\\phantomsection\\label\{[^}]*\}\n)?APPENDIXMARKER",
                      "", body_tex)
    body_tex = body_tex.replace("APPENDIXMARKER", "")
    marker = r"\section{What the certificates caught"
    if marker in body_tex:
        body_tex = body_tex.replace(
            marker, "\\input{refs}\n\\appendix\n" + marker, 1)

    (OUT / "body.tex").write_text(body_tex)
    (OUT / "abstract.tex").write_text(abstract_tex)
    (OUT / "main.tex").write_text(r"""\documentclass{article}
\usepackage{iclr2026_conference,times}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx,float,booktabs,array,longtable,multirow,calc}
\usepackage{xcolor,colortbl,microtype,caption}
\usepackage[hidelinks]{hyperref}
\usepackage{url}
\graphicspath{{figures/}}
\emergencystretch=2em

% Restrained NVIDIA-inspired academic visual system (house style).
\definecolor{NVIDIAGreen}{HTML}{76B900}
\definecolor{NVIDIADark}{HTML}{4B780A}
\definecolor{NVIDIACharcoal}{HTML}{252A2E}
\definecolor{NVIDIAPale}{HTML}{F1F7E8}
\definecolor{NVIDIAHeader}{HTML}{E6F0D6}
\renewcommand{\floatpagefraction}{0.9}
\setlength{\textfloatsep}{7pt plus 2pt minus 2pt}
\setlength{\floatsep}{7pt plus 2pt minus 2pt}
\setlength{\intextsep}{7pt plus 2pt minus 2pt}
\captionsetup{
  font=footnotesize,
  labelfont={bf,color=NVIDIADark},
  textfont={color=NVIDIACharcoal},
  skip=3pt
}

\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\title{""" + title + r"""}
\author{Anonymous authors\\Paper under double-blind review}
\begin{document}
\maketitle
\begin{abstract}
\input{abstract}
\end{abstract}
\input{body}
\end{document}
""")
    env = {"PATH": f"{TEXBIN}:/usr/bin:/bin"}
    result = subprocess.run(
        [str(TEXBIN / "latexmk"), "-pdf", "-interaction=nonstopmode",
         "-halt-on-error", "main.tex"],
        cwd=OUT, env=env, capture_output=True, text=True)
    log = OUT / "latexmk.log"
    log.write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-30:])
        raise SystemExit(f"latexmk failed:\n{tail}")
    pages = re.search(r"Output written on main\.pdf \((\d+) page",
                      result.stdout)
    print(f"[build-a] paper_a/main.pdf built"
          f" ({pages.group(1) if pages else '?'} pages)")


if __name__ == "__main__":
    main()
