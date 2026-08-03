# TODO — everything that must be filled in or verified before submission

Every item below corresponds to a `[[TODO: ...]]` or `[[FIGURE: ...]]` marker in the chapters.
None of them should survive into the final PDF.

## A. Missing numbers — the final model (highest priority)

The book contains complete, real numbers for the four legacy training stages (they came from the
stored notebook outputs). **It does not contain the final 17-class YOLO11 results**, because
those artefacts are not in the material we have. Everything below comes from the winning run's
directory (`runs/detect/train*/`).

| # | What | Where in the book | Source |
|---|---|---|---|
| 1 | Per-seed validation mAP@0.50 and mAP@0.50:0.95 for seeds 0, 1, 2, and which was selected | §4.3.1 table | the three run directories |
| 2 | Final test Precision, Recall, mAP@0.50, mAP@0.50:0.95 | §4.3.1 and Table 4.1 row V | `model.val(split='test')` |
| 3 | Per-class AP@0.50 and AP@0.50:0.95 for all 17 classes | §4.3.1 per-class table | validation/test output |
| 4 | GPU used for the final runs (A100 / T4 / L4) | §4.1 table | Colab session |
| 5 | Inference time per image on GPU | §4.3.1 | `model.val()` speed line |

## B. Missing numbers — runtime

| # | What | Where | Source |
|---|---|---|---|
| 6 | Server per-stage latency shares: `decode_quality`, `inference`, `tracking`, `danger_logic`, `db_write` | §4.4.2 | `GET /get_system_status?range=live` on the admin dashboard |
| 7 | Client per-stage averages: `capture`, `encode`, `render`, `feedback` | §4.4.2 | same dashboard, "פירוט זמן עיבוד בלקוח" |
| 8 | Battery drain over a 30-minute continuous scanning session | Table 5.1, §6.2 | one measured test on a real device |

## C. Figures to produce or drop in

All figure slots are marked `[[FIGURE: ...]]` in the text. Put the images in `Final Book/figures/`.

**From the Ultralytics run directory (just copy them):**
- 4.1 `results.png` — training curves
- 4.2 `confusion_matrix_normalized.png`
- 4.3 `PR_curve.png` + `F1_curve.png`
- 3.5 `train_batch0.jpg`, `train_batch1.jpg` — augmented batches

**To produce from data (small matplotlib scripts):**
- 3.2 dataset composition by source + per-class counts on a log scale (two panels)
- 3.4 custom-detector train vs val loss (from the table in §3.3.2)
- 4.4 latency breakdown + throughput vs pipeline depth (two panels; needs items 6 and 7 above)

**To draw (diagrams):**
- 3.1 architecture and frame lifecycle — redraw the ASCII block diagram cleanly
- 3.3 polygon → bounding box conversion
- 3.6 alert classification decision tree

**To capture:**
- 3.7 annotated dashboard screenshot mid-detection
- 4.5 three real street photographs with detections, plus three failure cases (motion blur, low
  light, distant small object) — **required by meeting 6**

## D. Facts to verify

| # | Item | Where |
|---|---|---|
| 9 | The team-role split — the original planning docs only cover the early phases | Appendix D |
| 10 | Submission month on the title page (currently August 2026) | §Front matter |
| 11 | **Deployed `HIGH_RISK_CLASSES` default** — the book lists an 11-class injury-on-contact subset, inferred from the documented 14-class default. Copy the real list from the deployed `core/config.py` | §3.4.4, Appendix B.1 |
| 12 | Whether the `cross` ID-0 → crosswalk mapping was ever visually verified | §3.2.5, Appendix A |
| 13 | Whether the three real street photos were captured | §4.6 |
| 14 | **All other server constants in Appendix B.1** (thresholds, sensitivity profiles, TARGET_FPS, input-size clamp) came from `SERVER.md`, which documents a 14-class build. Spot-check them against the deployed config | Appendix B.1 |

### A note on which sources were trusted

Three descriptions of the server exist in this project and they disagree, because they were
written at different times:

| Source | Date | Classes it describes |
|---|---|---|
| `SeeSense/Server/core/config.py` (local checkout) | 31 Mar 2026 | 10 — **oldest, do not trust** |
| `Material for final book/SERVER.md` | 3 Aug 2026 | 14 |
| The actually deployed system | current | **17 — this is what the book states** |

The book was written from `SERVER.md` and then corrected to 17 classes. Anything in it that came
from `SERVER.md` and is *class-count dependent* is flagged above; everything else (the pipeline
structure, the optimisation log, the tracker and danger-logic constants) is class-independent and
should still be accurate.

## E. Optional strengthening, if time allows

- Run `model.val()` on the **10-class base test set** with the final model, so there is one row in
  Table 4.1 that is directly comparable to the Stage II baseline. This is the single most valuable
  extra experiment available: it would let the book say "the same test set, before and after"
  instead of only "different test sets, not comparable".
- Move a stratified sample of `manhole` images into val/test and re-run, so the class stops being
  unmeasurable.
- One controlled comparison against `yolo11s` at the same settings, to justify the nano choice by
  measurement.

## F. Honesty checklist before submission

The book currently makes several statements that are true *now* and would become false if the
work progresses. Re-read these before submitting:

- §3.6.2 and §6.2: "no user study has been conducted."
- §2.7 and §6.2: "the hybrid failover is designed, the monitoring half is implemented, the
  on-device model is not."
- §4.3.1, §6.2: "`manhole` has never been evaluated."
- Table 4.1: the "not comparable across rows" note must stay.
- §3.3.5 and Table 5.1: the oversampling evaluation was contaminated — this admission should stay in.

If any of these becomes false because the work was finished, update the text. If any stays true,
leave it in — a stated limitation is a strength in a project book, and every one of these was
raised by the supervisor at some point.
