# SeeSense — Final Project Book

Source of the SeeSense final project book. Project #503, Deep Learning specialization,
College of Management Rishon LeZion. Supervisor: Dr. Moshe Butman.

Team: Oren Levy · Liad Lati · Omer Helfer · Shir Yahav
Code repository: <https://github.com/oren1levy/SeeSense>

---

## ⚠️ The Markdown files in this repo are the source of truth

The book lives here as **one Markdown file per chapter**. The PDF is *generated* from them.

That means:

- **Edit the `.md` files. Never edit a PDF, a Word file, or the merged Markdown in `build/`.**
  Anything in `build/` is overwritten on the next build and is not committed to git.
- If someone sends you a PDF of the book, it is a **snapshot**, not the document. Do not make
  changes in it — they will be lost. Make them here.
- Do not keep a private copy of a chapter on your laptop or in Drive and edit that. That is how
  two versions of Chapter 4 come to exist, and merging them by hand at 2 a.m. before submission is
  not something anyone should have to do.

Everyone works from this repo, so everyone always has everyone else's latest text.

## Repository contents

| File | Chapter |
|---|---|
| `01_front_matter.md` | Title page, acknowledgments, executive summary, contents, abbreviations, figures |
| `02_ch1_introduction.md` | 1 — Introduction |
| `03_ch2_literature_review.md` | 2 — Literature Review |
| `04_ch3_system_architecture.md` | 3.1 — System architecture |
| `05_ch3_data.md` | 3.2 — Data collection, preparation, verification |
| `06_ch3_model_training.md` | 3.3 — Model development and training |
| `07_ch3_server.md` | 3.4 — Server implementation |
| `08_ch3_client.md` | 3.5 — Client implementation |
| `09_ch3_deployment_eval.md` | 3.6 — Deployment and evaluation methodology |
| `10_ch4_results.md` | 4 — Results and Analysis |
| `11_ch5_challenges.md` | 5 — Engineering Challenges and Lessons Learned |
| `12_ch6_conclusion.md` | 6 — Conclusion and Future Work |
| `13_ch7_references.md` | 7 — References |
| `14_appendices.md` | Appendices A–D |
| `TODO_NUMBERS.md` | **Every number, figure and fact still missing.** Read before submitting |
| `figures/` | Images referenced by the chapters |
| `build_pdf.py` / `build_pdf.sh` | The PDF builder |

The number prefixes control the order chapters are merged in — don't rename files without also
updating `CHAPTER_GLOBS` in `build_pdf.py`.

## Building the PDF

```bash
./build_pdf.sh
```

First run creates a local `.venv/` and installs one package (~15 seconds); afterwards it is
instant. Rendering uses headless Google Chrome, already present on macOS.

Output lands in `build/`, stamped with the build date and time so earlier builds are kept:

```
build/SeeSense_Project_Book_YYYY-MM-DD_HHMM.pdf     ~51 A4 pages
build/SeeSense_Project_Book_YYYY-MM-DD_HHMM.html    same thing, in a browser
build/SeeSense_Project_Book_YYYY-MM-DD_HHMM.md      all chapters merged
```

`build/` is gitignored — the PDF is regenerable, and committing a 1.8 MB binary on every rebuild
would bloat the repo. **Anyone can rebuild it in seconds, so send people the repo, not the PDF.**

Every page carries a header showing the build time, word count, and how many `[[TODO]]` markers
and figure slots are still unresolved — so a printed draft always states how finished it is.
TODO markers render highlighted in yellow and figure slots in blue, so they are impossible to
miss when reviewing. Before the final submission build, delete the `.buildstamp` div from
`build_pdf.py` so the stamp does not appear on the submitted PDF.

## Working together without stepping on each other

Each of us can own chapters and edit them independently. Because the book is split by file, two
people editing different chapters will **never** conflict.

### The routine, every time you sit down to write

```bash
git pull                      # get everyone else's latest work FIRST
# ... edit your chapter(s) ...
./build_pdf.sh                # optional: check how it looks
git add 10_ch4_results.md     # add just what you changed
git commit -m "ch4: fill in final YOLO11 test metrics"
git push
```

**`git pull` before you start** is the important habit. If you edit for three hours on a stale
copy, you may have rewritten a paragraph somebody else already fixed.

**Commit and push in small pieces**, as you finish sections — not one enormous commit the night
before. Small commits are what make it possible to see who changed what, and to undo one change
without losing others.

**Write a useful commit message.** `ch5: add the ONNX thread-oversubscription story` tells the
team something; `update` does not.

### If you both edit the same file anyway

Git will ask you to resolve it on `git pull`. Open the file, look for the `<<<<<<<` / `=======` /
`>>>>>>>` markers, keep the text that should survive (usually both halves, merged sensibly),
delete the markers, then:

```bash
git add <file>
git commit
git push
```

Nothing is lost in a conflict — both versions are in the file, and you choose. If it looks
frightening, stop and ask before deleting anything.

### Suggested chapter ownership

Fill this in and keep it accurate; it is the cheapest way to avoid two people rewriting the same
section.

| Chapter | Owner |
|---|---|
| 1 Introduction, 2 Literature Review | |
| 3.2 Data | |
| 3.3 Model training | |
| 3.4 Server | |
| 3.5 Client | |
| 4 Results | |
| 5 Challenges, 6 Conclusion | |

Anyone may fix a typo anywhere. For anything larger than a typo in someone else's chapter, tell
them rather than silently rewriting it.

## Conventions inside the chapters

- **`[[TODO: ...]]`** — a value or claim that must be filled in or verified before submission.
  Every one is listed in `TODO_NUMBERS.md`. **None may survive into the final PDF.**
- **`[[FIGURE: ...]]`** — a figure slot. Put the image in `figures/`, then replace the placeholder
  with `![caption](figures/your-file.png)`.
- Tables are plain Markdown so they convert cleanly to PDF and Word.
- Keep lines wrapped at roughly 100 characters. This is not cosmetic: git compares line by line,
  so short lines mean a one-word change shows as a one-line diff instead of a whole paragraph, and
  conflicts become far easier to resolve.
- Cross-references are written as `§3.2.7` / `Table 4.1` / `Figure 3.3`. If you renumber a
  section, grep for its old number across all chapters before you commit.

## Current state

**~26,700 words · 51 A4 pages before figures are inserted.** Adding the twelve figures will take
it to roughly 57–61 pages. The example book we were given (RoadXpert) is 43 pages.

**What is finished:** the full narrative across all chapters — introduction, literature review,
architecture, the complete data pipeline, all five model-development stages with their real
measured metrics, server and client implementation, deployment, evaluation methodology, latency
and real-time analysis, the challenges chapter, conclusions, references and appendices.

**What is missing** — all of it tracked in `TODO_NUMBERS.md`:

1. **The final 17-class YOLO11 results.** Per-seed validation mAP, final test precision/recall/mAP,
   and per-class AP. This is the single biggest gap, and it is in the chapter the project is most
   graded on.
2. **Server and client per-stage latency numbers** from the admin dashboard.
3. **Twelve figures** — four copied straight out of the Ultralytics run directory, three small
   plots, three diagrams, and the real street photographs required by the final review meeting.
4. **A handful of facts to verify**, including the deployed `HIGH_RISK_CLASSES` list and the team
   role split in Appendix D.

`TODO_NUMBERS.md` also carries an **honesty checklist** — statements that are true today and would
become false if the work progresses (no user study yet, no offline mode, `manhole` never
evaluated). Re-read it before submitting: if the work got done, update the text; if it did not,
leave the limitation in. Stated limitations are a strength in a project book, and every one of
them was raised by the supervisor at some point.

## If you would rather work in Word

```bash
brew install pandoc
pandoc build/SeeSense_Project_Book_*.md -o SeeSense_Project_Book.docx
```

Be aware that the moment you start editing the `.docx`, it stops being the source of truth and
this repo goes stale. Either keep editing the Markdown and re-export, or abandon the Markdown
entirely — but do not edit both.
