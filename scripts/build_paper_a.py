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
PANDOC = shutil.which("pandoc") or "/tmp/pandoc-3.10/bin/pandoc"
TEXBIN = Path.home() / ".TinyTeX" / "bin" / "x86_64-linux"
SOURCE_DATE_EPOCH = "1783296000"  # 2026-07-06 00:00:00 UTC

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
    text = re.sub(
        r"\$\$(.+?)\$\$",
        lambda m: tok(r"\begin{equation}" + m.group(1).strip()
                      + r"\end{equation}"),
        text, flags=re.S)
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


def normalize_generated_source(text: str) -> str:
    """Correct release terminology without modifying the source template."""
    text = re.sub(
        r"(\\begin\{figure\})\[!h\]", r"\1[!ht]", text)
    text = text.replace(
        "fixed log-spaced half-lives from 2 to 96 steps",
        "fixed log-spaced e-folding time constants from 2 to 96 steps")
    text = text.replace(
        "20,000 crossed seed-bootstrap draws",
        "20,000 task-stratified paired optimizer-seed bootstrap draws")
    text = text.replace(
        "crossed-bootstrap 95% interval",
        "task-stratified paired optimizer-seed bootstrap 95% interval")
    text = text.replace(
        "registered crossed-bootstrap interval",
        "registered task-stratified paired optimizer-seed bootstrap interval")
    text = text.replace(
        "task-stratified paired-seed bootstrap",
        "task-stratified paired optimizer-seed bootstrap")
    text = text.replace(
        "The crossed bootstrap independently resamples the task and "
        "paired-seed axes on each of 20,000 draws.",
        "The task-stratified paired optimizer-seed bootstrap independently "
        "resamples paired optimizer-seed cells within each task on each of "
        "20,000 draws.")
    text = text.replace(
        "Whiskers show the recorded across-bank variation.",
        "In panel (b), whiskers show checkpoint-seed variation on one fixed "
        "240-episode bank (three seeds for the filter and GRU; ten for the "
        "delta cell).")
    text = text.replace(
        "Whiskers show checkpoint-seed variation on one fixed 240-episode "
        "bank (three filter/GRU seeds and ten delta seeds).",
        "In panel (b), whiskers show checkpoint-seed variation on one fixed "
        "240-episode bank (three seeds for the filter and GRU; ten for the "
        "delta cell).")
    return text


def main() -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    for name in ("iclr2026_conference.sty", "natbib.sty"):
        shutil.copy(ROOT / "paper" / name, OUT / name)
    shutil.copy(ROOT / "templates" / "PAPER_A.refs.tex", OUT / "refs.tex")
    for fig in (ROOT / "docs" / "figures").glob("fig_a_*.pdf"):
        shutil.copy(fig, OUT / "figures" / fig.name)

    text = normalize_generated_source(SRC.read_text())
    title_match = re.match(r"# (.+)\n", text)
    title = title_match.group(1)
    text = text[title_match.end():]
    abstract_match = re.search(r"## Abstract\n\n(.+?)\n\n## ", text, re.S)
    abstract = abstract_match.group(1)
    body = text[abstract_match.end() - len("## "):]
    if "APPENDIXMARKER" in body:
        body, appendix = body.split("APPENDIXMARKER", 1)
    else:
        appendix = ""
    # png figure refs -> pdf (vector in the paper)
    body = re.sub(r"figures/(fig_a_\w+)\.png", r"figures/\1.pdf", body)
    appendix = re.sub(r"figures/(fig_a_\w+)\.png", r"figures/\1.pdf",
                      appendix)

    # LaTeX numbers sections itself: strip the manual "N." / "N.M" heading
    # prefixes and promote H2 -> \section
    def strip_heading_numbers(fragment: str) -> str:
        fragment = re.sub(r"^## \d+\. ", "## ", fragment, flags=re.M)
        return re.sub(r"^### \d+\.\d+ ", "### ", fragment, flags=re.M)

    def pandoc_latex(fragment: str, *, shift: bool) -> str:
        command = [PANDOC, "-f", "markdown+tex_math_dollars+raw_tex",
                   "-t", "latex", "--wrap=preserve"]
        if shift:
            command.append("--shift-heading-level-by=-1")
        return subprocess.run(
            command, input=tokenize(fragment), text=True,
            capture_output=True, check=True).stdout

    body_tex = pandoc_latex(strip_heading_numbers(body), shift=True)
    appendix_tex = pandoc_latex(strip_heading_numbers(appendix), shift=True)
    abstract_tex = subprocess.run(
        [PANDOC, "-f", "markdown+tex_math_dollars+raw_tex", "-t", "latex",
         "--wrap=preserve"],
        input=tokenize(abstract), text=True, capture_output=True,
        check=True).stdout
    body_tex = restore(body_tex)
    appendix_tex = restore(appendix_tex)
    abstract_tex = restore(abstract_tex)
    # Normalize any remaining Pandoc image wrappers.  The non-greedy match is
    # anchored on our known figure filename, so braces inside alt text cannot
    # defeat the width constraint.
    def normalize_images(fragment: str) -> str:
        fragment = re.sub(
            r"\\pandocbounded\{\\includegraphics\[.*?\]"
            r"\{(figures/fig_a_\w+\.pdf)\}\}",
            r"\\includegraphics[width=\\linewidth]{\1}", fragment,
            flags=re.S)
        return re.sub(
            r"\\includegraphics(?:\[[^\]]*\])?"
            r"\{(figures/fig_a_\w+\.pdf)\}",
            r"\\includegraphics[width=\\linewidth]{\1}", fragment)

    body_tex = normalize_images(body_tex)
    appendix_tex = normalize_images(appendix_tex)
    # pandoc's calc-based column widths break under the ICLR style; use
    # fixed fractions and plain tabulars instead of longtables
    def shrink_column(match: re.Match) -> str:
        # Keep generated fallback tables inside the text block after TeX adds
        # inter-column padding.  Hand-authored tables use tabularx instead.
        return f"{0.88 * float(match.group(1)):.5f}\\linewidth"

    for name, fragment in (("body", body_tex), ("appendix", appendix_tex)):
        fragment = re.sub(
            r"\(\\(?:linewidth|columnwidth) - \d+\\tabcolsep\) \* "
            r"\\real\{([\d.]+)\}", shrink_column, fragment)
        if name == "body":
            body_tex = fragment
        else:
            appendix_tex = fragment

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
    appendix_tex = re.sub(
        r"\\begin\{longtable\}\[\]\{(.*?)\}\n(.*?)\\end\{longtable\}",
        to_table, appendix_tex, flags=re.S)

    (OUT / "body.tex").write_text(body_tex)
    (OUT / "appendix.tex").write_text(appendix_tex)
    (OUT / "abstract.tex").write_text(abstract_tex)
    display_title = title.replace(": ", ":\\\\\n", 1)
    metadata_title = title.replace("\\", " ")
    (OUT / "main.tex").write_text(r"""\documentclass{article}
\usepackage{iclr2026_conference,times}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx,float,booktabs,array,longtable,multirow,calc,tabularx}
\usepackage{xcolor,colortbl,microtype,caption}
\usepackage[hidelinks]{hyperref}
\usepackage{url}
\hypersetup{
  pdfauthor={},
  pdftitle={""" + metadata_title + r"""},
  pdfsubject={Anonymous ICLR submission}
}
\graphicspath{{figures/}}
\emergencystretch=2em

% Restrained NVIDIA-inspired academic visual system (house style).
\definecolor{NVIDIAGreen}{HTML}{76B900}
\definecolor{NVIDIADark}{HTML}{4B780A}
\definecolor{NVIDIACharcoal}{HTML}{252A2E}
\definecolor{NVIDIAPale}{HTML}{F1F7E8}
\definecolor{NVIDIAHeader}{HTML}{E6F0D6}
\definecolor{TableGray}{HTML}{F1F3F4}
\definecolor{FailAmber}{HTML}{A45A1C}
\newcommand{\TblPass}{\textcolor{NVIDIADark}{\ensuremath{\checkmark}}}
\newcommand{\TblFail}{\textcolor{FailAmber}{\ensuremath{\times}}}
\newcommand{\TblPart}{\textcolor{FailAmber}{\ensuremath{\triangle}}}
\newcommand{\TblUnknown}{\textcolor{NVIDIACharcoal}{\textbf{?}}}
\newcommand{\TblNA}{\textcolor{NVIDIACharcoal!45}{\textemdash}}
\renewcommand{\floatpagefraction}{0.9}
\setlength{\textfloatsep}{7pt plus 2pt minus 2pt}
\setlength{\floatsep}{7pt plus 2pt minus 2pt}
\setlength{\intextsep}{7pt plus 2pt minus 2pt}
\setlength{\abovedisplayskip}{5pt plus 2pt minus 2pt}
\setlength{\belowdisplayskip}{5pt plus 2pt minus 2pt}
\captionsetup{
  font=footnotesize,
  labelfont={bf,color=NVIDIAGreen},
  textfont={color=NVIDIACharcoal},
  skip=3pt
}

\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\pandocbounded}[1]{#1}
\providecommand{\real}[1]{#1}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\renewcommand{\arraystretch}{1.06}
\title{""" + display_title + r"""}
\author{Anonymous authors\\Paper under double-blind review}
\begin{document}
\maketitle
\begin{abstract}
\input{abstract.tex}
\end{abstract}
\input{body.tex}

% Machine-readable main-text page marker. References and appendices are excluded.
\phantomsection\label{paper-a-main-end}
\clearpage

\input{refs.tex}

\clearpage
\appendix
\raggedbottom
\input{appendix.tex}
\end{document}
""")
    env = {
        "PATH": f"{TEXBIN}:/usr/bin:/bin",
        "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH,
        "FORCE_SOURCE_DATE": "1",
        "TZ": "UTC",
        "LC_ALL": "C",
    }
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
    if pages is None and (OUT / "main.log").is_file():
        pages = re.search(r"Output written on main\.pdf \((\d+) page",
                          (OUT / "main.log").read_text())
    print(f"[build-a] paper_a/main.pdf built"
          f" ({pages.group(1) if pages else '?'} pages)")


if __name__ == "__main__":
    main()
