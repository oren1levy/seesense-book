# TODO — everything that must be filled in or verified before submission

Every item corresponds to a `[[TODO: ...]]` or `[[FIGURE: ...]]` marker in the chapters, or to a
🔴 mark. None may survive into the final PDF. Anything that doesn't make sense (rather than being
merely missing) is in `AMBIGUOUS.md` — read both.

## A. Missing numbers — the model (highest priority)

| # | What | Where | Source |
|---|---|---|---|
| 1 | **Test-split P / R / mAP@0.50 / mAP@0.50:0.95 for the delivered model.** Every Stage IV number is validation; the frozen 5,957-image test split has never been evaluated | §4.3.1, Table 4.1 row IV | `model.val(split='test')` on `best.pt` |
| 2 | `manhole` has no val/test instances, so its printed AP is Ultralytics' placeholder | Table 3.10, §4.3.1 | re-split so the class has instances, re-run |

## B. Missing numbers — runtime (GPU deployment)

The book's headline GPU-era figures (server ~16.4 ms/frame, R₀ ~120 ms, ~50 FPS at ~120–130 ms,
GPU 20–30 ms vs CPU ~200 ms) come from the measurement note in `streamConfig.js` (2026-08-06) and
`DEPLOYMENT.md` — code-committed prose, not a saved artefact. To close the 🔴s in §4.4.2:

| # | What | Source |
|---|---|---|
| 3 | Reset persisted stats (admin page reset button — current data is contaminated by debug restarts and a ~13 s CUDA-warmup outlier), run one clean session, screenshot `GET /get_system_status` | admin dashboard on the GCP deployment |
| 4 | Server per-stage averages: `decode_quality`, `inference`, `tracking`, `danger_logic`, `db_write`, `response` | same |
| 5 | Client per-stage averages: `capture`, `encode`, `render`, `feedback` | same, "פירוט זמן עיבוד בלקוח" |
| 6 | Battery drain over a 30-minute continuous scanning session | one measured test on a real device |

## C. Figures to produce or drop in

**From the Ultralytics run directory (copy):** 4.1 `results.png` · 4.2
`confusion_matrix_normalized.png` · 4.3 curve plots — note Ultralytics 8.4.115 names them
`BoxPR_curve.png` / `BoxF1_curve.png` (the notebook's display cells looked for the old names and
printed "not found"; the files exist in the run directory) · 3.5 `train_batch*.jpg`.

**To plot (small matplotlib scripts):** 3.2 dataset composition + per-class counts (log scale) ·
3.4 custom-detector train/val loss (table in §3.3.2) · 4.4 latency breakdown + throughput vs
depth (Table 4.3; needs items 4–5).

**To draw:** 3.3 polygon → box conversion · 3.6 alert decision tree — draw the CURRENT rules
(§3.4.4): unchecked class → none; not approaching → none; approaching → fast=high any distance /
Close+Medium=high / Far=low.

**Done:** 3.1 architecture and frame lifecycle — `figures/fig-3-1-architecture.svg`, placed in
§3.1. It is a plain SVG, so any text in it can be edited in a text editor; `build_pdf.py` now
inlines figures automatically, so new ones only need `![alt](figures/name.svg)` in a chapter.

**To capture:** 3.7 annotated dashboard screenshot mid-detection · 4.5 three real street photos
with detections + three failure cases (blur, low light, distant object) — **required by meeting 6**.

## D. Facts to verify

| # | Item | Where |
|---|---|---|
| 7 | Team-role split for the later phases (planning docs cover early phases only) | Appendix D |
| 8 | Submission month on the title page (currently August 2026) | Front matter |
| 9 | Whether the `cross` ID-0 → crosswalk mapping was ever visually verified | §3.2.5, Appendix A |
| 10 | Whether the three real street photos were captured | §4.6 |
| 11 | "708 items transferred" — the docx states it directly, so this is now sourced; no action | §3.3.5 |

Closed since the last revision: the deployed `HIGH_RISK_CLASSES` default, all Appendix B.1
constants, the tracker constants and the health thresholds are **verified against the code**
(re-verified 2026-08-07 against commit `6d22728` — every motion constant unchanged); the
`MAX_INFLIGHT` 4-vs-5 discrepancy is gone; Table 3.9 is now the complete 20-epoch window and
Table 3.6's two wrong image counts are corrected, both from
`Model_Development_and_Training.docx`, which the model chapter now matches throughout.

## E. Optional strengthening, if time allows

- `model.val()` of the final model on the **10-class base test set**, so one Table 4.1 row is
  directly comparable to Stage II ("same test set, before and after").
- Move a stratified sample of `manhole` into val/test and re-run.
- One controlled `yolo26n` / `yolo26m` comparison at identical settings.
- Migrate the GPU VM to `me-west1` (Tel Aviv) if capacity appears — removes ~90–100 ms — and
  re-measure §4.4 (see `DEPLOYMENT.md`, "Migrating to Tel Aviv").

## F. Honesty checklist before submission

Statements that are true *now* and would become false if the work progresses. If the work got
done, update the text; if not, leave the limitation in — every one was raised by the supervisor.

- §3.6.2, §6.2: "no user study has been conducted."
- §2.7, §3.5.7, §6.2: "the monitoring half is implemented, the on-device model is not, and the
  automatic stop on a red connection is not wired."
- §4.3.1, §6.2: "`manhole` has never been evaluated."
- Table 4.1: the "not comparable across rows" note must stay.
- §3.3.4, Table 5.1: the oversampling evaluation was contaminated — the admission stays in.
- §3.4.4, Appendix B.1: the server's class list and high-risk default are still the legacy
  14-class configuration; only 13 of the model's 17 classes reach the user. If the code is
  reconciled before submission, update §1.4, the exec summary, §3.4.4, B.1 and §6.3 together.
