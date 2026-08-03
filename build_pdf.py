#!/usr/bin/env python3
"""
Build a timestamped PDF of the SeeSense project book from the Markdown chapters.

The .md files in this folder are the SOURCE OF TRUTH. This script never writes
to them — it only reads them, in filename order, and renders build artefacts
into build/.

Usage:  ./build_pdf.sh          (wrapper that picks the right Python)
        python build_pdf.py     (if `markdown` is importable)

Output (build/):
    SeeSense_Project_Book_YYYY-MM-DD_HHMM.md    merged Markdown
    SeeSense_Project_Book_YYYY-MM-DD_HHMM.html  styled, self-contained
    SeeSense_Project_Book_YYYY-MM-DD_HHMM.pdf   printed via headless Chrome

PDF rendering uses Google Chrome in headless mode, which is present on macOS
by default. If Chrome is missing, the HTML is still produced and can be opened
in any browser and printed to PDF manually (Cmd-P → Save as PDF).
"""

import datetime
import glob
import html as html_mod
import os
import re
import shutil
import subprocess
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(HERE, "build")

# Chapter files, in order. Deliberately explicit rather than a bare glob, so a
# stray .md dropped in this folder does not silently end up in the book.
CHAPTER_GLOBS = [
    "01_front_matter.md",
    "02_ch1_introduction.md",
    "03_ch2_literature_review.md",
    "04_ch3_system_architecture.md",
    "05_ch3_data.md",
    "06_ch3_model_training.md",
    "07_ch3_server.md",
    "08_ch3_client.md",
    "09_ch3_deployment_eval.md",
    "10_ch4_results.md",
    "11_ch5_challenges.md",
    "12_ch6_conclusion.md",
    "13_ch7_references.md",
    "14_appendices.md",
]

CSS = """
@page { size: A4; margin: 17mm 16mm 15mm 16mm; }
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  font-family: "Charter", "Georgia", "Times New Roman", serif;
  font-size: 10pt; line-height: 1.42; color: #16181d; background: #fff;
  margin: 0; padding: 0; max-width: 100%;
}
.buildstamp {
  font-family: "SF Mono", Menlo, monospace; font-size: 8pt; color: #6b7280;
  border-bottom: 1px solid #e5e7eb; padding-bottom: 6px; margin-bottom: 26px;
}
h1 {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 17pt; font-weight: 700; color: #0b1220;
  margin: 0 0 11px; padding-top: 4px; line-height: 1.2;
  border-bottom: 2px solid #0b1220; padding-bottom: 8px;
  page-break-before: always; page-break-after: avoid;
}
h1:first-of-type { page-break-before: avoid; }
h2 {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 12.5pt; font-weight: 700; color: #0b1220;
  margin: 18px 0 7px; page-break-after: avoid; line-height: 1.25;
}
h3 {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 10.8pt; font-weight: 700; color: #1f2937;
  margin: 14px 0 6px; page-break-after: avoid;
}
h4 { font-size: 10.5pt; font-weight: 700; margin: 16px 0 6px; page-break-after: avoid; }
p { margin: 0 0 7px; text-align: justify; hyphens: auto; orphans: 2; widows: 2; }
ul, ol { margin: 0 0 8px; padding-inline-start: 22px; }
li { margin-bottom: 2px; }
strong { font-weight: 700; color: #0b1220; }
code {
  font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 8.8pt;
  background: #f3f4f6; padding: 1px 4px; border-radius: 3px; color: #a3115f;
}
pre {
  background: #f7f8fa; border: 1px solid #e5e7eb; border-left: 3px solid #64748b;
  border-radius: 4px; padding: 8px 10px; overflow-x: auto;
  font-size: 8pt; line-height: 1.35; margin: 8px 0 10px;
  page-break-inside: avoid; white-space: pre-wrap; word-wrap: break-word;
}
pre code { background: none; padding: 0; color: #1f2937; font-size: inherit; }
table {
  border-collapse: collapse; width: 100%; margin: 8px 0 12px;
  font-size: 8.3pt; page-break-inside: avoid;
}
th, td { border: 1px solid #d5d9e0; padding: 3px 6px; text-align: left; vertical-align: top; }
th { background: #eef1f5; font-weight: 700; font-family: "Helvetica Neue", Helvetica, sans-serif; }
tr:nth-child(even) td { background: #fafbfc; }
blockquote {
  border-left: 3px solid #94a3b8; background: #f8fafc;
  margin: 9px 0; padding: 6px 10px; color: #334155; font-style: italic;
  page-break-inside: avoid;
}
blockquote p { margin: 0 0 4px; text-align: left; }
blockquote code { font-style: normal; background: #fff4d6; color: #92400e; }
hr { border: none; border-top: 1px solid #d5d9e0; margin: 14px 0; }
a { color: #1d4ed8; text-decoration: none; word-break: break-word; }
.todo {
  background: #fff3cd; border: 1px solid #f0c36d; border-radius: 3px;
  padding: 1px 5px; font-family: "SF Mono", Menlo, monospace; font-size: 8.3pt;
  color: #7a4f01; font-style: normal;
}
.figslot {
  background: #eef6ff; border: 1px dashed #7aa7d9; border-radius: 3px;
  padding: 1px 5px; font-family: "SF Mono", Menlo, monospace; font-size: 8.3pt;
  color: #1a4a7a; font-style: normal;
}
"""


def read_chapters():
    parts, missing = [], []
    for name in CHAPTER_GLOBS:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            missing.append(name)
            continue
        with open(path, encoding="utf-8") as fh:
            parts.append(fh.read().rstrip() + "\n")
    if missing:
        print("WARNING: missing chapter files: " + ", ".join(missing), file=sys.stderr)
    return "\n\n".join(parts)


def highlight_placeholders(html):
    """Make [[TODO: ...]] and [[FIGURE: ...]] visually obvious in the PDF."""
    def todo(m):
        detail = (m.group(1) or "").strip()
        return '<span class="todo">[[TODO%s]]</span>' % (": " + detail if detail else "")

    def fig(m):
        return '<span class="figslot">[[FIGURE: %s]]</span>' % (m.group(1) or "").strip()

    # Matches both "[[TODO]]" (common inside result tables) and "[[TODO: detail]]".
    html = re.sub(r"\[\[TODO(?::\s*(.*?))?\]\]", todo, html, flags=re.S)
    html = re.sub(r"\[\[FIGURE(?::\s*(.*?))?\]\]", fig, html, flags=re.S)
    return html


def main():
    os.makedirs(BUILD, exist_ok=True)
    now = datetime.datetime.now()
    stamp = now.strftime("%Y-%m-%d_%H%M")
    human = now.strftime("%d %B %Y, %H:%M")
    base = os.path.join(BUILD, "SeeSense_Project_Book_%s" % stamp)

    md_text = read_chapters()
    words = len(md_text.split())
    n_todo = len(re.findall(r"\[\[TODO", md_text))
    n_fig = len(re.findall(r"\[\[FIGURE", md_text))

    with open(base + ".md", "w", encoding="utf-8") as fh:
        fh.write(md_text)

    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )
    body = highlight_placeholders(body)

    stampline = (
        "SeeSense — Final Project Book &nbsp;·&nbsp; DRAFT built %s "
        "&nbsp;·&nbsp; %s words &nbsp;·&nbsp; %d unresolved TODOs, %d figure slots "
        "&nbsp;·&nbsp; generated from the Markdown chapters — do not edit this PDF"
        % (html_mod.escape(human), format(words, ","), n_todo, n_fig)
    )

    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>SeeSense — Final Project Book (%s)</title>"
        "<style>%s</style></head><body>"
        "<div class='buildstamp'>%s</div>%s</body></html>"
        % (html_mod.escape(stamp), CSS, stampline, body)
    )

    with open(base + ".html", "w", encoding="utf-8") as fh:
        fh.write(page)

    chrome = None
    for cand in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        shutil.which("google-chrome"),
        shutil.which("chromium"),
    ):
        if cand and os.path.exists(cand):
            chrome = cand
            break

    print("merged   : %s.md   (%s words)" % (os.path.basename(base), format(words, ",")))
    print("html     : %s.html" % os.path.basename(base))

    if not chrome:
        print("\nChrome not found — PDF not generated.")
        print("Open the HTML in any browser and print to PDF (Cmd-P → Save as PDF).")
        return 0

    cmd = [
        chrome, "--headless", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
        "--virtual-time-budget=20000",
        "--print-to-pdf=" + base + ".pdf",
        "file://" + base + ".html",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if os.path.exists(base + ".pdf"):
        size = os.path.getsize(base + ".pdf") / 1024
        print("pdf      : %s.pdf   (%.0f KB)" % (os.path.basename(base), size))
        print("\nBuild directory: %s" % BUILD)
        print("Remember: edit the .md chapters, never the generated files.")
        return 0

    print("\nChrome failed to produce a PDF:", file=sys.stderr)
    print((res.stderr or "")[-1500:], file=sys.stderr)
    print("The HTML is still available; print it to PDF from a browser.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
