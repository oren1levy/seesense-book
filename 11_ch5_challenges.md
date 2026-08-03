# Chapter 5 — Engineering Challenges and Lessons Learned

This chapter exists because the problems were the education. A book that reports only the final
architecture implies that the architecture was designed and then built; in reality almost every
design decision in Chapter 3 is the scar tissue of something that went wrong. The four narratives
below are the ones that most shaped the system; Table 5.1 summarises the rest.

## 5.1 The polygon bug — thousands of images vanishing in silence

**Symptom.** The merged dataset contained 84,940 images, but the `stairs` gap-fill set of ~1,950
images had contributed exactly **one**. No error, no warning, no failed assertion — and one is a
particularly deceptive number, since zero would have suggested a path problem while one suggests a
partially-working integration.

**Diagnosis.** We opened a raw label file instead of trusting the source's `data.yaml`, and found
polygon coordinates rather than five-number boxes: the dataset was annotated in YOLO
**segmentation** format. The merge parser contained `if len(p) != 5: continue`, so every polygon
line was skipped, and images left with zero valid boxes were never copied.

**Fix and payoff.** Detect the format by counting coordinates and, for polygons, compute the tight
enclosing box (§3.2.7). Re-running the resumable builder recovered 1,949 stairs images and,
through the same fix, 3,185 construction and 846 bench images vanishing for the same reason —
**6,199 recovered images, 6.8 % of the final dataset**.

**Lesson.** "Image count equals label count" is necessary but wholly insufficient, because a
silent format mismatch drops images and their labels *together* and every consistency check still
passes. The only reliable defence is to **open a raw label file from every new source by hand**
before integrating it. Two minutes per source; it would have saved days here.

## 5.2 The phantom combined dataset — training on data that was not there

**Symptom.** The combined-dataset run completed 50 epochs and reported plausible metrics
(P 0.671 / R 0.404 / mAP@0.50 0.4372) — slightly *worse* than the baseline it was meant to improve.

**Diagnosis.** The merge script printed `Skipping new train - missing folder` three times and
continued. The new-class dataset lived in `/content`, Colab's ephemeral storage, which had been
reclaimed between sessions. The "combined" dataset contained only the old data, under a YAML
declaring 14 classes of which four had no examples at all.

**Fix.** Persist all intermediate artefacts to Drive, never `/content`. **Fail loudly** — a missing
required input must raise, not print and continue. Verify the *output artefact* — image counts,
label counts, per-class distribution — before training, not the script that produced it.

**Lesson, and the most expensive one.** A long-running job that silently proceeds on the wrong data
is far worse than one that crashes immediately. It consumes GPU hours and, much worse, produces a
number that looks like a result. We nearly reported it as one. The verification protocol of
§3.2.10 exists entirely because of this run.

## 5.3 ONNX — 1.5× faster locally, 75× slower in production

**Symptom.** ONNX Runtime benchmarked roughly 1.5× faster than PyTorch on a development machine, so
inference was ported to it. On Railway it ran at **2,323 ms per frame** — about 75× slower than the
path it replaced.

**Diagnosis.** The container's cgroup limited it to 8 vCPU while `os.cpu_count()` reported 48. Both
runtimes sized their thread pools from the visible core count, spawning 48 threads for 8 vCPU of
work — six-fold oversubscription, with most time spent in context switches and cache thrashing
rather than arithmetic. ONNX Runtime's threading model amplified the effect far more than
PyTorch's. A controlled sweep on a 16-core machine measured 16 threads running **four times
slower** than 8 on the same workload.

**Fix.** Cap `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `torch.set_num_threads()` at 8, **before
importing** torch, NumPy or OpenCV, because those libraries read the values once at import.
Capping ONNX's threads helped but never beat PyTorch on the real server, so the ONNX path was
**removed entirely** rather than left as dead configuration.

**Lesson — the most valuable systems lesson of the project.** *Measure on the deployment target,
not the development machine.* A local benchmark was not merely optimistic here; it pointed in the
opposite direction from reality. Containers routinely lie about the machine they run on, and any
library auto-sizing a thread pool from the core count will get it wrong.

## 5.4 Alert flooding — the challenge that shaped the product

**Symptom.** The first working end-to-end version was intolerable within seconds. At ~20 FPS a
parked car in view produced twenty spoken announcements and twenty vibration bursts per second.

**Diagnosis.** Two independent causes. The system alerted on **presence** rather than **change**: an
object that was there last frame and is still there generated a fresh alert. And YOLO boxes jitter
by a few pixels every frame, so naive frame-to-frame comparison made even the "approaching" flag
flicker several times a second.

**Fix, in three parts.** A **four-frame motion window** instead of frame-to-frame comparison, so
jitter reads as `static` and only sustained growth reads as `approaching`. **Motion-first alert
classification**, so objects that are not approaching never raise a red alert. And **per-track
alert deduplication**, retaining the last announced level per track ID and firing only on a genuine
transition.

**Lesson.** For a non-visual interface, the hard problem is not producing information — it is
suppressing it. A visual overlay can update 22 times a second because the eye samples what it
wants; an audio channel is serial and blocking, and it competes with the environmental hearing the
user navigates by. §4.8 argues this is the project's central design contribution.

## 5.5 Summary of challenges, and process lessons

**Table 5.1 — Engineering challenges, root causes and resolutions**

| Challenge | Root cause | Resolution | Status |
|---|---|---|---|
| Ten incompatible class taxonomies (`bike`/`bicycle`, `Trashcan`/`waste_container`, `warning_column` = bollard) | No shared ontology across contributed datasets | Explicit per-source mapping dictionary; any unmapped class reported and dropped, never guessed | Resolved |
| Sources with meaningless class names (`'19'`, `'object'`) | Contributed datasets with numeric labels | ID-based mapping after counting the actual ID distribution | Resolved; the `cross` ID-0 assumption flagged |
| 6,199 images silently dropped | Parser accepted only 5-field lines; sources used polygons | Polygon → tight bounding-box conversion (§5.1) | Resolved |
| `manhole` absent from val and test | Random split over a 0.79 % class | Documented; redistribution pending | **Open** |
| 628:1 class imbalance | Real-world object frequency | Gap-fill collection, multi-source merging, augmentation; loss reweighting available | Partially resolved |
| Disk exhaustion (35 GB of archives) | Duplicated base set | Hardlinks, selective copying, resumable builder | Resolved |
| Custom detector did not generalise | Frozen backbone, simplified grid head, class imbalance | Pivot to YOLOv8 | Resolved |
| Contaminated oversampling evaluation | Duplication applied to all three splits | Train-split-only rule; test-set freeze | Resolved as policy |
| Training on data that was not there | Ephemeral `/content`; silent skip | Persist to Drive; fail loudly; verify artefacts (§5.2) | Resolved |
| Non-comparable metrics across stages | Different test sets per stage | Name the evaluation set on every row | Resolved |
| ONNX 75× slower in production | Container reported 48 cores, limited to 8 vCPU | Thread caps before import; ONNX removed (§5.3) | Resolved |
| False "connection lost" | Inference blocking the async event loop, stalling `/health` | `asyncio.to_thread` + global lock | Resolved |
| SOS froze the entire server | Synchronous SMTP on the event loop; `/health` timed out and every request stalled | All e-mail dispatched on daemon threads | Resolved |
| 71 ms per frame of database work | Settings read on every frame | In-memory cache; writes off the hot path | Resolved |
| Diagnostics never appeared in logs | Module imported before `logging.basicConfig()`, root logger still at WARNING | Moved the call into `load_model()` | Resolved |
| Timestamps three hours wrong | Naive datetime with no timezone marker crossing a process boundary | Normalise to UTC and pin formatting to `Asia/Jerusalem` | Resolved (3 pages outstanding) |
| Sensors unusable in development | `getUserMedia`, gyroscope and geolocation require a secure context | ngrok HTTPS tunnel + allowed-hosts entry | Resolved (process cost) |
| Alert flooding | Alerting on presence; box jitter | 4-frame window, motion-first logic, per-track dedup (§5.4) | Resolved |
| Over-sensitive health watchdog | Single-reading thresholds, tight values for mobile RTT | Consecutive streaks; announce-once flags | Partially resolved |
| iOS platform restrictions | Gesture-gated sensor permission; no Vibration API; no background execution | Gesture-gated request; capability detection with honest reporting | Mitigated |
| Motion blur and low light | Domain shift from clean academic imagery | Quality gates, augmentation, alignment gate | **Open** |
| Battery and thermal load | Continuous camera, encoding and radio | Start/stop control; capture early-out | Mitigated, unmeasured |
| Audio masked in noisy streets | Physical | Parallel haptic channel with rhythmically distinct patterns | Mitigated |
| Device and browser variability | Web platform capability surface is not constant | Layered fallbacks; honest capability reporting to the user | Resolved |

**Process lessons.**

*Breadth before depth was the right instruction.* The supervisor's direction to present one working
end-to-end pipeline before optimising accuracy was uncomfortable at the time, since it meant
demonstrating a system driven by a mediocre model. It was correct: almost every serious problem we
hit was an integration problem, and every one was found earlier because the whole pipeline existed
earlier.

*Train it yourself.* Stage I cost weeks and produced a model we discarded. It also produced the
only genuine understanding any of us has of what a detection head, a grid encoder, an assignment
strategy and a multi-component loss actually do — and the ability to read a YOLO training curve and
know what each loss term means.

*Freeze the test set.* Adopted after the Stage IV contamination and applied strictly thereafter. It
costs flexibility late in the project, which is precisely the point.

*Verify the artefact, not the script.* Every data disaster here — the polygon drop, the missing
manhole examples, the phantom combined dataset — would have been caught immediately by inspecting
the output: image counts, per-class per-split distribution, and one raw label file. That inspection
now takes ten minutes and is §3.2.10.

*Fail loudly.* Every silent skip in this project cost days. `print("Skipping…")` and continue is
not error handling.

*Measure on the target.* Stated three times in this chapter because it cost us three times.
