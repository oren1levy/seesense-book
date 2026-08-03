# Chapter 4 — Results and Analysis

## 4.1 Experimental Setup

| Training environment | | Deployment environment | |
|---|---|---|---|
| Platform | Google Colab | Host | Railway container, **CPU-only** |
| GPU (stages I–IV) | NVIDIA A100-SXM4-40GB | Cores reported vs. limit | 48 vs. 8 vCPU |
| GPU (stage V) | `[[TODO: confirm]]` | Thread cap applied | 8 |
| Frameworks | PyTorch 2.x, Ultralytics | Database | MongoDB Atlas |
| Storage | Google Drive (persistent) | Model | `seesense_model.pt`, ~6 MB, **17 classes** |
| Seeds | 42 (split); 0/1/2 (training) | Input size | 512×512 (clamp 160–640) |
| | | Confidence / NMS IoU | 0.4 / 0.45, overridden per profile |

The most important line is **CPU-only**. All latency figures in §4.4 are for CPU inference in a
resource-limited container, not for a GPU. This is a deliberate choice — it is what a project of
this scale can afford to run continuously — and it is why input size turned out to be the dominant
performance lever.

## 4.2 Dataset Results

| Property | Value |
|---|---|
| Classes / images / boxes | 17 · **91,139** · **964,837** |
| Splits | train 74,096 / val 11,086 / test 5,957 (≈ 81/12/7) |
| Sources merged | 10 (3 academic + 9 Roboflow) |
| Orphans, invalid labels, out-of-range IDs, corrupt images | **0** in every category |

**Growth over the project:** ~37,000 images at the POC presentation (Mar), ~57,000 at the April
review, >70,000 at 17 classes in June, 84,940 at the first 17-class merge, and **91,139** final.
The last step is the 6,199 images recovered by the polygon-to-box conversion of §3.2.7 — 6.8 % of
the final dataset that had been disappearing without any error.

**Balance achieved and not achieved.** The gap-fill campaign succeeded for the classes it targeted:
`crosswalk` went from 33 images to 3,511, a factor of 106, and twelve of seventeen classes now sit
above the 3,000-image working floor. Five do not: `dog` (2,981), `bollard` (2,699), `curb`
(2,258), `trash_can` (2,126) and `manhole` (722). The structural imbalance is unchanged and cannot
be removed by collection: `pole` contributes 453,239 of 964,837 boxes — 47 % of every annotation in
the dataset is a pole — against `manhole`'s 722, a ratio of 628:1.

## 4.3 Model Accuracy Across Development Stages

**Table 4.1 — Comparative evaluation across all five stages**

| Stage | Model | Evaluated on | P | R | mAP@0.50 | mAP@0.50:0.95 |
|---|---|---|---:|---:|---:|---:|
| I | Custom ResNet18 + hand-written head | Original validation | n/c | n/c | n/c | n/c |
| II | YOLOv8n, 10 classes | Original test (3,699 img / 86,381 inst.) | 0.669 | 0.409 | 0.4528 | 0.2847 |
| III | YOLOv8n fine-tuned, 14 classes | New-class test (784 img / 1,548 inst.) | 0.862 | 0.818 | 0.853 | 0.618 |
| IV | Stage III + weak-class oversampling | Oversampled test (1,082 img / 2,448 inst.) | 0.857 | 0.843 | 0.866 | 0.630 |
| — | Attempted combined dataset | Old-only test (merge failed) | 0.671 | 0.404 | 0.4372 | 0.2750 |
| **V** | **YOLO11n, 17 classes, best of 3 seeds** | **Final test (5,957 img)** | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` | `[[TODO]]` |

`n/c` = not computed; the custom detector was evaluated qualitatively and by loss, and inventing
metrics retrospectively would be dishonest. Stage I's conclusion: loss improved but validation
stalled at epoch 4, and detections were unreliable and pole-biased. Stage II: fast and usable, but
recall far too low for a safety system. Stage III: strong — on a smaller, easier, differently
distributed set. Stage IV: recall and mAP improved, evaluation methodology compromised. The
combined run: **invalid**, trained on 10-class data under a 14-class YAML.

**The rows are not directly comparable, and that is the most important thing to understand about
this table.** Stage II and the combined run used the same large, dense, heavily-imbalanced
ten-class test set. Stages III and IV used much smaller sets built from single-class sources, where
a typical image contains one clearly-photographed object against an uncluttered background. A
0.853 mAP@0.50 on 784 such images is genuinely good *on that data*; it is not evidence the model
became twice as good at the original task.

### 4.3.1 Final model results

`[[TODO: fill this subsection from the Stage V run.]]`

| Seed | Val mAP@0.50 | Val mAP@0.50:0.95 | Selected |
|---:|---:|---:|---|
| 0 | `[[TODO]]` | `[[TODO]]` | |
| 1 | `[[TODO]]` | `[[TODO]]` | |
| 2 | `[[TODO]]` | `[[TODO]]` | |

The spread across seeds is itself a result: it is the empirical variance of the training
procedure, and the correct context in which to read the headline number. Final test results on the
frozen 5,957-image split, plus per-class AP for all 17 classes, go here — noting that **`manhole`
has zero test instances and therefore no AP at all**.

> **Figure 4.1** — Training curves for the selected run: box, classification and DFL losses
> falling while precision, recall and both mAP measures rise. `[[FIGURE: results.png]]`

> **Figure 4.2** — Normalized confusion matrix on the 17-class test set.
> `[[FIGURE: confusion_matrix_normalized.png]]`

> **Figure 4.3** — Per-class precision–recall and F1-versus-confidence curves.
> `[[FIGURE: PR_curve.png and F1_curve.png]]`

**Expected confusion patterns**, stated before reading the matrix so it confirms or refutes a
prediction rather than merely being described: **`bollard` ↔ `pole`**, both narrow vertical street
furniture distinguished mainly by height, which a single-frame box conveys least well — Stage III
already showed bollards over-predicted by 20 %; **`curb` ↔ background**, a long low-contrast
horizontal edge whose boundaries are ambiguous even for a human annotator, meaning the ground
truth itself is noisy; **`manhole` ↔ background**, low-contrast, viewed at an extreme oblique angle
and trained on only 722 examples; **`crosswalk` ↔ background**, similar geometry plus the ~7 %
suspected label noise from the ID-0 mapping assumption; **`scooter` ↔ `motorcycle` ↔ `bicycle`**,
similar silhouettes and frequently occluded when parked in a row; and **`construction` internally
heterogeneous**, since it deliberately collapses cones, barriers, scaffolding and machinery.

### 4.3.2 What improved performance, and what depressed it

**Improvements:** more data for starved classes — `crosswalk` at 33 images could not be learned by
any method, at 3,511 it can, and this is the single largest factor; transfer learning from strong
pretrained weights plus the newer YOLO11 architecture; built-in augmentation; recovery of 6,199
polygon-labelled images adding real, non-synthetic signal; and multi-seed selection removing the
risk of shipping an unlucky initialisation.

**Factors that depress performance, and would be expected to:** **augmentation inflation**, since
the gap-fill sources are ~3× Roboflow-augmented and augmented copies are far less informative than
independent scenes; **domain shift** between clean academic imagery and messy real street
photographs; **nano-model capacity**, a 3 M-parameter network separating 17 classes with heavy
classes dominating the gradient and suppressing rare-class learning; **class heterogeneity by
design** in `construction`; and **label noise** from merging ten independently-annotated sources
with differing conventions — what counts as a "kerb" in one dataset may not in another.

### 4.3.3 From trained model to deployed model

The model evaluated here is the model the product runs: the Stage V 17-class YOLO11 weights
selected by validation mAP@50-95, shipped inside the Docker image rather than downloaded at
startup. Deploying a retrained model touches four places that must move together or the system
will report classes it cannot detect and fail to report classes it can: the weights, the server's
class list and default high-risk set, the client's Hebrew class-name map, and the settings class
grid. The client-side halves were, at the time of the code review in §3.5.10, still partially out
of sync — a cosmetic-label defect rather than a detection defect, but exactly the drift a single
shared class map would prevent (§6.3.1).

## 4.4 Runtime and Latency Results

### 4.4.1 The optimisation log

**Table 4.2 — Performance optimisations with measured effect**

| # | Change | Measured effect |
|---|---|---|
| 1 | Per-frame HTTP POST → **WebSocket streaming** | Removed per-frame handshake and header overhead; latency to **~33 ms** |
| 2 | Per-frame settings read → **in-memory cache** | **71 ms → 0 ms** per frame |
| 3 | **DB write off the hot path** (pre-generated ObjectId, daemon thread) | `record_id` returned before the insert; no I/O in the request path |
| 4 | **Inference to a worker thread** (`asyncio.to_thread`) | Event loop free; `/health` keeps answering, so the watchdog stops falsely reporting instability |
| 5 | **Global inference lock** | Prevents concurrent forward passes |
| 6 | **Input size 640 → 512** | **The single biggest win** — smaller uploads *and* faster inference → **~41 ms/frame** |
| 7 | Client `toDataURL` → **`toBlob`** | Async encode, no base64 round trip; quality checks moved onto the resized image |
| 8 | **Thread-pool capping at 8** | Container reported 48 cores while limited to 8 vCPU; 16 threads measured **4× slower** than 8 on a 16-core box |
| 9 | **ONNX Runtime port** | ~1.5× faster locally, **75× slower on the deployment target (2,323 ms/frame)** — reverted and removed |
| 10 | **Windowed motion + per-track alert dedup** | Not raw speed, but removed a flood of redundant alerts and work |

Items 8 and 9 share a root cause (§5.3) and are the most valuable entries precisely because they
are negative results: the local benchmark was not merely optimistic, it pointed in the opposite
direction from the truth.

### 4.4.2 Latency and throughput

Per-frame server latency at `INPUT_SIZE = 512` on the CPU-only container is approximately
**41 ms**, dominated by the `inference` stage, with `decode_quality` second and `tracking`,
`danger_logic` and `db_write` small to negligible. `[[TODO: fill the five server-stage and four
client-stage averages from GET /get_system_status.]]`

The network round trip from a mobile connection was approximately **131 ms** — the largest single
term in the end-to-end budget, three times the entire server-side processing cost. This is the
fundamental argument for the on-device model in future work: no amount of server optimisation can
remove a term not spent on the server.

**Table 4.3 — Throughput versus in-flight depth** (server ~41 ms, RTT ~131 ms)

| Depth | Throughput | Per-frame latency | Efficiency | Note |
|---:|---|---|---|---|
| 1 | ~7 FPS | ~131 ms | 30 % | Server idle for a full RTT between frames |
| 2 | ~13 FPS | ~172 ms | 55 % | |
| 4 | ~22 FPS | ~216 ms | ~96 % | Approaches the ~23.5 FPS ceiling |
| 5 (deployed) | ~22 FPS | ~220 ms+ | ~96 % | Marginal gain, additional queueing delay |

> **Figure 4.4** — Latency breakdown per pipeline stage (client, network, server), and throughput
> versus pipeline depth showing the knee at depth 4.
>
> `[[FIGURE: two-panel — stacked latency bar, and dual-axis depth curve from Table 4.3]]`

This is the classic bandwidth–delay shape: throughput rises with depth until it saturates at the
server's service rate, after which additional depth buys only queueing delay. Depth 4 sits at the
knee; the deployed value of 5 is one step past it, a discrepancy between the documented reasoning
and the shipped constant which should be resolved to 4.

**Four FPS numbers** are reported because they answer different questions: `server_capacity`
(`1/avg_latency`, the theoretical ceiling), `server_actual` (frames actually arriving),
`client_actual` (what the phone reports sending, revealing which side is the bottleneck) and
`overall` (total frames / uptime). Throughput is reported separately over a rolling ten-second
window and decays to zero when idle, so it answers "how many detections per second are flowing
right now".

## 4.5 Real-Time Analysis

The question posed at the fifth review meeting was direct: **are 7 FPS enough for pedestrian
walking pace, and what happens with fast objects?** It deserves an arithmetic answer.

The relevant quantity is not frame rate but **end-to-end latency** — the interval between a hazard
becoming visible and the user hearing the warning — which at the deployed operating point is
approximately **216 ms**.

| Moving party | Speed | Distance in 216 ms | in 150 ms |
|---|---|---:|---:|
| Walking pedestrian (the user) | 1.4 m/s | **0.30 m** | 0.21 m |
| Running person | 3.0 m/s | 0.65 m | 0.45 m |
| Cyclist / e-scooter | 6.0 m/s | 1.30 m | 0.90 m |
| Urban car | 13.9 m/s (50 km/h) | **3.00 m** | 2.08 m |
| Fast urban car | 19.4 m/s (70 km/h) | 4.19 m | 2.92 m |

A user walking at normal pace advances 30 cm between a hazard appearing and being warned about it
— roughly half a stride, a comfortable margin against a static obstacle detected at the "Close"
threshold. A car at 50 km/h closes 3 m during the same delay, which is why the alert logic
escalates a fast-approaching high-risk object to `high` at **any** distance rather than waiting for
the box to grow (§3.4.4): with a vehicle, the distance at the moment of detection is the entire
safety margin.

Frame rate determines how often the situation is *re-assessed*; latency how stale each assessment
is. At **7 FPS** the gap between assessments is 143 ms, during which a pedestrian moves 20 cm and a
car 2 m — adequate for static obstacles, marginal for vehicles. At the deployed **~22 FPS** the gap
is 45 ms — 6 cm of pedestrian motion, 63 cm of car motion — and every object is re-assessed about
22 times a second, which is also what makes the four-frame motion window meaningful: four frames is
~180 ms of history, long enough to distinguish genuine approach from box jitter and short enough
to stay responsive.

**Conclusion: 7 FPS is sufficient for walking-pace hazards but not comfortable for vehicles;
~22 FPS at ~216 ms end-to-end is a defensible operating point for urban pedestrian use.** Against
the original POC criterion — an alert within 200–300 ms under stable network conditions — 216 ms
is inside the target band.

Of that budget, ~131 ms is network and ~41 ms server processing. **The network is the largest term
and the one we cannot optimise from the server.** Three routes would reduce it: **on-device
inference**, removing the network term entirely — our own characterisation estimated 80–150 ms
inference on an Apple Neural Engine, giving 100–180 ms total; **edge deployment** geographically
closer to users; and **lower input size**, reducing both upload and inference at a measurable cost
in small-object recall — a knob the user could in principle be given as "prioritise
responsiveness" versus "prioritise detail".

## 4.6 Qualitative Results

`[[TODO: complete after running the final model on the captured street photographs.]]`

Three real street scenes were photographed by the team in the deployment environment and run
through the model, following the supervisor's instruction to validate against self-captured field
images rather than only benchmark data.

> **Figure 4.5** — Qualitative detections on real street photographs (a pavement with parked
> scooters, a crossing with a kerb transition, a construction area), and failure cases: motion
> blur while walking, low light, and distant small objects.
>
> `[[FIGURE: three annotated model outputs plus three failure examples]]`

Expected behaviour, to be confirmed or corrected against the actual outputs: **reliable** on large,
common, well-represented classes — `person`, `car`, `traffic_light`, `pole` — in daylight at
moderate distance; **reduced** on small or low-contrast classes — `manhole`, `curb`, distant
`bollard` — consistent with both their box statistics and their training representation;
**degraded under motion blur**, which the training data barely contains although a phone held by a
walking person produces it on a large fraction of frames — a domain-shift failure rather than a
capacity failure, whose fix is data rather than a bigger model; and **degraded in low light**,
where the quality gate correctly rejects the darkest frames but the band just above the threshold
produces noisy, low-confidence detections.

## 4.7 Comparison with Existing Approaches

**Table 4.4 — SeeSense compared with existing assistive approaches**

| Dimension | White cane | Guide dog | Ultrasonic aid | Scene-description app | **SeeSense** |
|---|---|---|---|---|---|
| Detection range | ~1 m | Situational | 1–5 m | Photo-dependent | Frame-dependent; vehicles at distance |
| Object identity | None | Implicit | None | Rich | **17 obstacle classes** |
| Continuous | Yes | Yes | Yes | No — on demand | **Yes, ~22 FPS** |
| Latency | Instant | Instant | Instant | Seconds | **~216 ms** |
| Motion awareness | No | Yes | No | No | **Yes — tracked approach and speed** |
| Cost / extra hardware | Very low / cane | Very high / dog | Low / device | Low / none | **Low / none** |
| Training required | Weeks | Months, both parties | Minimal | Minimal | **Minimal** |
| Works offline | Yes | Yes | Yes | Partly | **No** — current limitation |

Against the reference project **RoadXpert**, the comparison is instructive because the stacks
overlap while the design targets differ. RoadXpert serves a sighted two-wheel rider with
on-device TensorFlow.js inference at 5–10 FPS on mid-range Android and 15–20 FPS on desktop, no
network dependency, a visual overlay as the primary channel, region-of-interest alert triggering,
and 10 traffic-oriented classes (reported P 0.732 / R 0.568 / mAP@0.5 0.641 / mAP@0.5:0.95 0.422).
SeeSense serves a blind pedestrian with server-side PyTorch inference at ~22 FPS end-to-end, a
hard network dependency, speech and haptics as the primary channel, tracked-motion alert
triggering, and 17 pedestrian-hazard classes over a 91,139-image dataset.

The two projects made opposite calls on the same trade-off, and both are defensible. RoadXpert
bought independence from the network at the cost of model capacity and frame rate; SeeSense bought
capacity and frame rate at the cost of a hard network dependency. For a rider moving at 25 km/h
through variable coverage, offline operation is arguably worth more; for a pedestrian whose
hazards need finer discrimination, the larger model and higher frame rate arguably are. The
correct end state, as both projects note, is the hybrid.

## 4.8 Discussion of Findings

**What the results establish.** A self-trained nano detector on a self-assembled dataset is
sufficient to drive a usable assistive system, and the trajectory — from a hand-written detector
that could not generalise, through a baseline with 0.409 recall, to a 17-class model on 91,139
verified images — demonstrates that the limiting factor at every stage was **data, not
architecture**. The single largest accuracy improvement came from taking `crosswalk` from 33 images
to 3,511, not from changing any model. **The systems engineering matters as much as the model**:
the difference between 33 ms and 41 ms of server latency is small, but the difference between an
alert on every frame and an alert on every *change* is the difference between a product and an
unusable demo — and the difference between 2,323 ms and 41 ms per frame came from a thread-count
environment variable. And **measurement on the deployment target is not optional**: the ONNX
experiment benchmarked 1.5× faster locally and ran 75× slower in production.

**What the results do not establish.** The stage-to-stage improvements are not directly comparable,
which is why Table 4.1 names each row's evaluation set. The Stage IV oversampling result is
weakened by its own methodology, since duplicating validation and test images changed the
evaluation distribution in the direction of the intervention. **`manhole` is untested** — zero
validation and zero test instances means we cannot state, in any form, how well the system detects
manhole covers. And **no user has used it**: every usability claim rests on design reasoning, not
evidence.

**The recall problem, and why it is the central tension.** For a safety system a false negative is
more dangerous than a false positive: an unreported staircase can cause a fall, while a spurious
warning about a bench costs a moment of annoyance. That argues for maximising recall — lower the
threshold, widen the high-risk list, alert more. But the audio channel is serial, low-bandwidth,
and shared with the user's own hearing, which is itself a primary navigation sense. A system that
speaks continuously does not merely annoy; it **masks the environmental sounds the user relies
on**, and trains the user to ignore it. Alert fatigue is a safety failure with the same end state
as a missed detection.

SeeSense resolves this by moving the filtering out of the detector and into the logic layer. The
detector runs at a moderate confidence threshold so recall is preserved in the *data*; the decision
about what to *say* is then made by three independent filters — is the object actually approaching,
is it close or fast enough to matter, and has the user already been told about this specific
tracked object. The visual channel receives everything; the audio channel receives only changes.
This is the design contribution of the project we would defend most strongly, and it is also, at
present, unvalidated by any real user — which is precisely why it heads the future-work list.
