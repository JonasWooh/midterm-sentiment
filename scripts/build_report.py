"""Typeset reports/midterm_sentiment_report.md as a PDF via XeLaTeX.

    python scripts/build_report.py

Layout follows a Word-style academic report: US Letter, Times New Roman,
10.5 pt justified body, bold 13/11 pt run-in-free headings, three-rule tables
with a bold "Table n." caption above, full-width figures with a bold
"Figure n." caption below.

* The Markdown is parsed with markdown-it (CommonMark + pipe tables).
* A pandoc-style ``Table: caption`` paragraph directly after a table becomes
  that table's caption; an image alone in a paragraph becomes a figure whose
  caption is the alt text.
* Table column widths use the CSS automatic table-layout algorithm on text
  widths measured from the Times New Roman font files. Tables and figures are
  floats (every table fits on one page); longtable is not used because it
  ignores floats already placed at the top of a page and can overfill it.
* The build fails if TeX reports an overfull page, an oversized float or a
  missing glyph.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
MD = REP / "midterm_sentiment_report.md"
PDF = REP / "midterm_sentiment_report.pdf"

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
FONT_REGULAR = FONT_DIR / "Times New Roman.ttf"
FONT_BOLD = FONT_DIR / "Times New Roman Bold.ttf"

# page geometry in TeX points (1in = 72.27pt); must match the preamble
TEXT_W = (8.5 - 2 * 0.85) * 72.27
TABLE_PT = 9.0
TABCOLSEP = 5.0

PREAMBLE = r"""
\documentclass[10pt]{article}
\usepackage[letterpaper,margin=0.85in,top=0.8in,bottom=0.85in]{geometry}
\usepackage{fontspec}
\setmainfont{Times New Roman}
\PassOptionsToPackage{hyphens}{url}
\usepackage{graphicx,array,booktabs,caption,url}
\usepackage[hidelinks,bookmarksnumbered=false]{hyperref}
\tracinglostchars=3
\pagestyle{empty}
\urlstyle{same}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt plus 1pt}
\renewcommand{\normalsize}{\fontsize{10.5}{13.2}\selectfont}
\renewcommand{\small}{\fontsize{9.5}{11.6}\selectfont}
\normalsize
\emergencystretch=1.5em
\makeatletter
\renewcommand{\section}{\@startsection{section}{1}{0pt}{12pt plus 3pt minus 1pt}{5pt}{\fontsize{13}{15.5}\selectfont\bfseries\raggedright}}
\renewcommand{\subsection}{\@startsection{subsection}{2}{0pt}{9pt plus 2pt minus 1pt}{3pt}{\fontsize{11}{13.2}\selectfont\bfseries\raggedright}}
\newcommand{\centeredsection}[1]{\@startsection{section}{1}{0pt}{12pt plus 3pt}{5pt}{\fontsize{13}{15.5}\selectfont\bfseries\centering}*{#1}}
\setlength{\leftmargini}{1.6em}
\setlength{\leftmarginii}{1.5em}
\def\@listi{\leftmargin\leftmargini \topsep 2pt plus 1pt \partopsep 0pt \itemsep 1pt \parsep 2pt}
\def\@listii{\leftmargin\leftmarginii \labelwidth\leftmarginii \advance\labelwidth-\labelsep \topsep 1pt \partopsep 0pt \itemsep 1pt \parsep 2pt}
\let\@listI\@listi
\renewcommand{\labelitemi}{\textbullet}
\renewcommand{\labelitemii}{\textendash}
\makeatother
\newcounter{refn}
\newenvironment{references}{\list{[\arabic{refn}]}{\usecounter{refn}\setlength{\labelwidth}{2.2em}\setlength{\labelsep}{0.5em}\setlength{\leftmargin}{2.7em}\setlength{\topsep}{2pt}\setlength{\itemsep}{3pt}\setlength{\parsep}{0pt}}\raggedright}{\endlist}
\DeclareCaptionFont{ten}{\fontsize{10}{12}\selectfont}
\captionsetup{font=ten,labelfont=bf,textfont=bf,labelsep=period,justification=raggedright,singlelinecheck=false}
\captionsetup[table]{skip=5pt,position=top}
\captionsetup[figure]{skip=6pt}
\setlength{\heavyrulewidth}{0.7pt}
\setlength{\lightrulewidth}{0.45pt}
\setlength{\tabcolsep}{%(tabcolsep)spt}
\setlength{\textfloatsep}{12pt plus 3pt minus 2pt}
\setlength{\floatsep}{12pt plus 3pt minus 2pt}
\setlength{\intextsep}{10pt plus 3pt minus 2pt}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.8}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.75}
\graphicspath{{%(graphics)s/}}
"""

LATEX_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def esc(text: str) -> str:
    return "".join(LATEX_SPECIAL.get(c, c) for c in text)


def esc_code(text: str) -> str:
    """Inline code is set in the body font, with line breaks allowed after separators."""
    return "".join(LATEX_SPECIAL.get(c, c) + (r"\allowbreak{}" if c in "/_-.,=" else "") for c in text)


def esc_url(url: str) -> str:
    return url.replace("\\", "/").replace("%", r"\%").replace("#", r"\#")


# ---------------------------------------------------------------- inline

def inline(node: SyntaxTreeNode) -> str:
    out = []
    for ch in node.children:
        t = ch.type
        if t == "text":
            out.append(esc(ch.content))
        elif t == "softbreak":
            out.append(" ")
        elif t == "hardbreak":
            out.append(r"\\")
        elif t == "strong":
            out.append(r"\textbf{" + inline(ch) + "}")
        elif t == "em":
            out.append(r"\emph{" + inline(ch) + "}")
        elif t == "code_inline":
            out.append(esc_code(ch.content))
        elif t == "link":
            if plain(ch) == ch.attrs["href"]:  # autolink: print the URL itself, breakable
                out.append(r"\url{" + ch.attrs["href"] + "}")
            else:
                out.append(r"\href{" + esc_url(ch.attrs["href"]) + "}{" + inline(ch) + "}")
        elif t == "image":
            raise ValueError("images are only supported as stand-alone figure paragraphs")
        else:
            raise ValueError(f"unsupported inline token: {t}")
    return "".join(out)


def plain(node: SyntaxTreeNode, mark_breaks: bool = False) -> str:
    """Rendered text of an inline node, for width measurement. With mark_breaks,
    BREAK is inserted at the non-space line-break points TeX may use: after
    explicit hyphens/dashes, and after the separators esc_code() makes breakable."""
    if node.type == "text":
        return re.sub(r"([-–—])", r"\1" + BREAK, node.content) if mark_breaks else node.content
    if node.type == "code_inline":
        return re.sub(r"([/_\-.,=])", r"\1" + BREAK, node.content) if mark_breaks else node.content
    if node.type in ("softbreak", "hardbreak"):
        return " "
    return "".join(plain(c, mark_breaks) for c in node.children)


BREAK = "\x00"


def is_bold(node: SyntaxTreeNode) -> bool:
    kids = [c for c in node.children if not (c.type == "text" and not c.content.strip())]
    return len(kids) > 0 and all(c.type == "strong" for c in kids)


# ---------------------------------------------------------------- tables

class Measure:
    def __init__(self, size_pt: float):
        scale = 20  # measure at a larger size for sub-point precision
        self.k = 1.0 / scale
        self.fonts = {False: ImageFont.truetype(str(FONT_REGULAR), int(size_pt * scale)),
                      True: ImageFont.truetype(str(FONT_BOLD), int(size_pt * scale))}

    def width(self, text: str, bold: bool) -> float:
        return self.fonts[bold].getlength(text) * self.k

    def max_word(self, marked: str, bold: bool) -> float:
        """Widest unbreakable chunk of text marked by plain(mark_breaks=True)."""
        words = re.split(r"\s+|" + BREAK, marked)
        return max((self.width(w, bold) for w in words if w), default=0.0)


def column_widths(cells: list[list[tuple[str, bool]]]) -> tuple[list[float], list[bool]]:
    """CSS automatic table layout: min = longest unbreakable word, max = unwrapped
    content; widths fill the text block. Returns content widths and whether each
    column wraps."""
    m = Measure(TABLE_PT)
    ncol = len(cells[0])
    pad = 2 * TABCOLSEP
    mins = [max(m.max_word(t, b) for t, b in (row[j] for row in cells)) + pad for j in range(ncol)]
    maxs = [max(m.width(t.replace(BREAK, ""), b) for t, b in (row[j] for row in cells)) + pad
            for j in range(ncol)]
    if sum(mins) > TEXT_W:
        raise ValueError(f"table cannot fit: minimum width {sum(mins):.0f}pt > {TEXT_W:.0f}pt")
    if sum(maxs) <= TEXT_W:
        widths = [w * TEXT_W / sum(maxs) for w in maxs]
    else:
        f = (TEXT_W - sum(mins)) / (sum(maxs) - sum(mins))
        widths = [lo + (hi - lo) * f for lo, hi in zip(mins, maxs)]
    wraps = [w < hi - 0.01 for w, hi in zip(widths, maxs)]
    return [w - pad for w in widths], wraps


def table(node: SyntaxTreeNode, caption: str | None) -> str:
    rows = [tr for sec in node.children for tr in sec.children]
    cell_nodes = [[td.children[0] if td.children else None for td in tr.children] for tr in rows]
    cells = [[(plain(c, mark_breaks=True) if c else "", (i == 0) or (c is not None and is_bold(c)))
              for c in row]
             for i, row in enumerate(cell_nodes)]
    widths, wraps = column_widths(cells)
    spec = []
    for j, (w, wrap) in enumerate(zip(widths, wraps)):
        align = r"\raggedright" if (j == 0 or wrap) else r"\centering"
        frac = (w + 2 * TABCOLSEP) / TEXT_W
        spec.append(r">{" + align + r"\arraybackslash}p{\dimexpr" + f"{frac:.5f}" + r"\linewidth-2\tabcolsep\relax}")

    def render(row, header=False):
        out = []
        for c in row:
            s = inline(c) if c else ""
            out.append(r"\textbf{" + s + "}" if header and s else s)
        return " & ".join(out) + r" \\"

    head = render(cell_nodes[0], header=True)
    lines = [r"\begin{table}[htbp]"]
    if caption is not None:
        lines.append(r"\caption{" + caption + "}")
    lines += [r"\fontsize{" + f"{TABLE_PT}" + r"}{11}\selectfont\renewcommand{\arraystretch}{1.42}",
              r"\begin{tabular}{" + "".join(spec) + "}",
              r"\toprule", head, r"\midrule"]
    lines += [render(r) for r in cell_nodes[1:]]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------- blocks

def figure(img: SyntaxTreeNode) -> str:
    src = img.attrs["src"]
    alt = plain(img)
    return "\n".join([
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\includegraphics[width=\linewidth,height=0.85\textheight,keepaspectratio]{" + src + "}",
        r"\caption{" + esc(alt) + "}",
        r"\end{figure}", ""])


def is_figure_paragraph(node: SyntaxTreeNode) -> bool:
    if node.type != "paragraph":
        return False
    kids = node.children[0].children
    return len(kids) == 1 and kids[0].type == "image"


TABLE_CAPTION = re.compile(r"^Table:\s+")


def table_caption(node: SyntaxTreeNode | None) -> str | None:
    if node is None or node.type != "paragraph":
        return None
    first = node.children[0].children[0]
    if first.type != "text" or not TABLE_CAPTION.match(first.content):
        return None
    first.token.content = TABLE_CAPTION.sub("", first.content)
    return inline(node.children[0])


class Writer:
    def __init__(self):
        self.seen_numbered = False
        self.back_matter = False

    def heading(self, node: SyntaxTreeNode) -> str:
        level = int(node.tag[1])
        title = inline(node.children[0])
        text = plain(node.children[0])
        numbered = bool(re.match(r"^\d+(\.\d+)*\.?\s", text))
        self.seen_numbered |= numbered
        bm = r"\pdfbookmark[" + str(level - 1) + "]{" + esc(text) + "}{" + re.sub(r"\W+", "-", text) + "}"
        if level == 1 and not numbered and not self.seen_numbered:
            return r"\centeredsection{" + title + "}" + bm + "\n"  # abstract-style front matter
        if level == 1 and not numbered:  # back matter (references): own page, smaller type
            self.back_matter = True
            return r"\clearpage\section*{" + title + "}" + bm + "\n\\small\n"
        cmd = {1: "section", 2: "subsection"}[level]
        return "\\" + cmd + "*{" + title + "}" + bm + "\n"

    def blocks(self, nodes: list[SyntaxTreeNode]) -> str:
        out, i = [], 0
        while i < len(nodes):
            n = nodes[i]
            if n.type == "heading":
                out.append(self.heading(n))
            elif n.type == "table":
                nxt = nodes[i + 1] if i + 1 < len(nodes) else None
                cap = table_caption(nxt)
                out.append(table(n, cap))
                if cap is not None:
                    i += 1
            elif is_figure_paragraph(n):
                out.append(figure(n.children[0].children[0]))
            elif n.type == "paragraph":
                out.append(inline(n.children[0]) + "\n")
            elif n.type in ("bullet_list", "ordered_list"):
                out.append(self.lst(n))
            elif n.type == "hr":
                out.append(r"\par\noindent\rule{\linewidth}{0.4pt}" + "\n")
            else:
                raise ValueError(f"unsupported block: {n.type}")
            i += 1
        return "\n".join(out)

    def lst(self, node: SyntaxTreeNode) -> str:
        env = "itemize" if node.type == "bullet_list" else "enumerate"
        start = node.attrs.get("start") if env == "enumerate" else None
        if env == "enumerate" and self.back_matter:  # numbered reference list, labels [n]
            assert start in (None, 1), "reference list must start at 1"
            env, start = "references", None
        lines = [r"\begin{" + env + "}"]
        if start and int(start) != 1:
            lines.append(r"\setcounter{enumi}{" + str(int(start) - 1) + "}")
        for item in node.children:
            lines.append(r"\item " + self.blocks(item.children).strip())
        lines.append(r"\end{" + env + "}")
        return "\n".join(lines) + "\n"


def title_block(meta: dict) -> str:
    parts = [r"\begin{center}",
             r"{\fontsize{16}{19}\selectfont\bfseries " + esc(meta["title"]) + r"\par}"]
    if meta.get("subtitle"):
        parts.append(r"\vspace{5pt}{\fontsize{11}{13.5}\selectfont\itshape " + esc(meta["subtitle"]) + r"\par}")
    for key in ("author", "date"):
        if meta.get(key):
            parts.append(r"\vspace{3pt}{\fontsize{11}{13.5}\selectfont " + esc(meta[key]) + r"\par}")
    if "github" in meta:  # repository line; left blank (reserved) until a URL is given
        url = meta["github"] or ""
        body = r"\textit{\url{" + url + "}}" if url else ""
        parts.append(r"\vspace{3pt}{\fontsize{11}{13.5}\selectfont\makebox[24em][l]{GitHub: " + body + r"}\par}")
    parts.append(r"\end{center}")
    parts.append(r"\vspace{-4pt}")
    return "\n".join(parts) + "\n"


def to_latex(md_text: str) -> str:
    m = re.match(r"^---\n(.*?)\n---\n", md_text, re.S)
    meta = yaml.safe_load(m.group(1)) if m else {}
    body = md_text[m.end():] if m else md_text
    mdi = MarkdownIt("commonmark", {"typographer": True}).enable(["table", "smartquotes"])
    tree = SyntaxTreeNode(mdi.parse(body))
    pdf_meta = (r"\hypersetup{pdftitle={" + esc(meta.get("title", "")) + "},pdfauthor={"
                + esc(meta.get("author", "")) + "}}")
    return "\n".join([
        PREAMBLE % {"tabcolsep": TABCOLSEP, "graphics": REP.as_posix()},
        pdf_meta,
        r"\begin{document}",
        title_block(meta) if meta else "",
        Writer().blocks(tree.children),
        r"\end{document}", ""])


def main() -> None:
    tex = to_latex(MD.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "report.tex"
        src.write_text(tex, encoding="utf-8")
        for _ in range(2):  # second pass settles float placement and bookmarks
            r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", src.name],
                               cwd=tmp, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stdout[-4000:])
        log = (Path(tmp) / "report.log").read_text(encoding="utf-8", errors="replace")
        problems = re.findall(r"^(Overfull \\[hv]box.*|LaTeX Warning: Float too large.*|Missing character.*)$", log, re.M)
        if problems:
            raise RuntimeError("layout problems:\n" + "\n".join(problems))
        shutil.copy2(Path(tmp) / "report.pdf", PDF)
    print("built:", PDF.relative_to(ROOT))


if __name__ == "__main__":
    main()
