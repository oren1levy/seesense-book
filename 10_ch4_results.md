# Chapter 4 — Results and Analysis

## 4.1 Experimental Setup

| Training environment | | Deployment environment | |
|---|---|---|---|
| Platform | Google Colab | Hosts | **GCP VM + NVIDIA T4 GPU** (primary) · Railway container, CPU-only (baseline) |
| GPU | NVIDIA A100-SXM4-40GB, 40,441 MiB | Serving | Uvicorn under systemd, port 443, single origin (VM) · Docker (Railway) |
| | (Stage III oversampling ran on a Tesla T4) | Database | MongoDB Atlas |
| Frameworks | PyTorch 2.11.0 + CUDA 12.8, Ultralytics 8.4.115 | Model | `seesense_model.pt`, ~19 MB, **17 classes** |
| Storage | Google Drive, `save_period=1` | Input size | 640×640 (clamp 160–640; 512 in the CPU era) |
| Seeds | 42, for both the split and training | Confidence | 0.4, overridden per sensitivity profile |

The most important line is the **two deployment targets**. The optimisation campaign of §4.4.1
was fought and measured on the CPU-only container, where every millisecond had to be found in
software; the delivered system then moved to a GPU VM, which changed the operating point
wholesale — including reverting one CPU-era optimisation that had become a GPU-era bottleneck.
The latency story in §4.4 therefore names its era for every figure.

## 4.2 Dataset Results

| Property | Value |
|---|---|
| Classes / images / boxes | 17 · **91,139** · **964,837** |
| Splits | train 74,096 / val 11,086 / test 5,957 (≈ 81/12/7) |
| Sources merged | 10 (one merged academic base of three corpora + 9 Roboflow) |
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

**Table 4.1 — Comparative evaluation across all four stages**

| Stage | Model | Evaluated on | P | R | mAP@0.50 | mAP@0.50:0.95 |
|---|---|---|---:|---:|---:|---:|
| I | Custom ResNet18 + hand-written head | Original validation | n/c | n/c | n/c | n/c |
| II | YOLOv8n, 10 classes | Original test (3,699 img / 86,381 inst.) | 0.669 | 0.409 | 0.4528 | 0.2847 |
| III a | YOLOv8n fine-tuned, 14 classes | New-class test (784 img / 1,548 inst.) | 0.862 | 0.818 | 0.853 | 0.618 |
| III b | Stage III a + weak-class oversampling | Oversampled test (1,082 img / 2,448 inst.) | 0.857 | 0.843 | 0.866 | 0.630 |
| III c | Attempted combined dataset (**invalid**) | Old-only test (merge failed) | 0.671 | 0.404 | 0.4372 | 0.2750 |
| **IV** | **YOLO26s, 17 classes** | **Final validation (11,086 img / 180,736 inst.)** | **0.7542** | **0.5872** | **0.6356** | **0.4581** |

`n/c` = not computed; the custom detector was evaluated qualitatively and by loss, and inventing
metrics retrospectively would be dishonest. Stage I's conclusion: loss improved but validation
stalled at epoch 4, and detections were unreliable and pole-biased. Stage II: fast and usable, but
recall far too low for a safety system. Stage III a: strong — on a smaller, easier, differently
distributed set, and bought at the price of catastrophic forgetting of the original ten classes.
Stage III b: recall and mAP improved, but the evaluation methodology was compromised because the
duplication routine was applied to the validation and test splits as well as to training.
Stage III c: **invalid**, trained on 10-class data under a 14-class YAML. Stage IV is the delivered
model. The three Stage III rows are the three negative results documented in §3.3.4.

🔴 **The Stage IV row is validation, not test.** The frozen 5,957-image test split has never been
evaluated against the delivered checkpoint; every Stage IV figure in this chapter is measured on
the 11,086-image validation split. One `model.val(split='test')` call on the final weights would
put this row on the same footing as Stages II and III.

**The rows are not directly comparable, and that is the most important thing to understand about
this table.** Stage II and the combined run used the same large, dense, heavily-imbalanced
ten-class test set. Stages III a and III b used much smaller sets built from single-class sources,
where a typical image contains one clearly-photographed object against an uncluttered background. A
0.853 mAP@0.50 on 784 such images is genuinely good *on that data*; it is not evidence the model
became twice as good at the original task. Only **Stage II and Stage IV** were evaluated over their
complete class vocabularies on large, dense scenes, which is why they are the only pair in the
table compared directly (§3.3.6).

### 4.3.1 Final model results

The delivered detector is the Stage IV YOLO26-small checkpoint: 260 layers and 9,961,022
parameters during training, fusing at inference to 122 layers, 9,471,759 parameters and
20.8 GFLOPs. It was trained for 80 epochs at 640×640, batch 64, seed 42, with mosaic augmentation
closed over the final ten epochs, and the shipped weights are the best checkpoint by validation
mAP@0.50:0.95 rather than the final epoch — over the last twenty epochs classification loss fell
from 1.255 to 1.096 while mAP@0.50 moved only within 0.631–0.636, so the objective and the
capability had visibly decoupled (§3.3.5).

**Validation over 11,086 images and 180,736 instances:** Precision **0.7542**, Recall **0.5872**,
mAP@0.50 **0.6356**, mAP@0.50:0.95 **0.4581**. The full per-class breakdown is Table 3.10 (§3.3.5)
and is not repeated here.

Read by class, the result falls into three regimes. The **navigation vocabulary this project set
out to add is the strongest group** — `scooter` at 0.950 mAP@0.50 and 0.853 mAP@0.50:0.95,
`crosswalk` at 0.957 and 0.774, `bollard` at 0.864 and 0.631, `trash_can` at 0.832 and 0.684. That
these sit above the general urban classes, and that their strict-IoU scores are high rather than
merely their lenient ones, is the clearest single piece of evidence that the data campaign of §3.2
did what it was built to do. The **classic urban objects sit in the middle** — `fire_hydrant`
0.655, `car` 0.644, `motorcycle` 0.555, `person` 0.553 — every one of them improved over the
Stage II baseline. The **weakest group is unchanged in composition since Stage II**: `pole` at
0.256 with recall 0.205, `curb` at 0.289 with recall 0.206, and `traffic_light` at 0.445. Poles
alone account for 90,122 of the 180,736 validation instances — half the entire object population,
and still the hardest class in the taxonomy. Poles and kerbs together are what holds overall recall
at 0.587, and their persistence across every architecture tried in this project marks their
difficulty as a property of thin, low-contrast, ambiguously bounded geometry rather than of model
capacity or data volume.

🔴 **`manhole` has no measurement.** The validation split contains zero manhole instances, and the
0.4581 the framework prints against that class is its placeholder for unevaluated classes — a value
identical, character for character, to the overall mAP@0.50:0.95 (§3.3.5). It is reported
throughout this book as *not measured*, never as a result.

> **Figure 4.1** — Training curves for the Stage IV run: box, classification and L1 losses falling
> while precision, recall and both mAP measures rise. YOLO26 reports an **L1** regression term
> where YOLOv8 reported distribution focal loss, on a numerically different scale — order 10⁻³
> rather than order 1 — so the loss axes are not comparable with any earlier stage.
> `[[FIGURE: results.png]]`

> **Figure 4.2** — Normalized confusion matrix over the 17-class **validation** split.
> `[[FIGURE: confusion_matrix_normalized.png]]`

> **Figure 4.3** — Per-class precision–recall and F1-versus-confidence curves, validation split.
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
pretrained weights (§3.3.5), plus the step up from YOLOv8-nano to the
YOLO26-small architecture and its roughly threefold increase in capacity; built-in augmentation;
recovery of 6,199 polygon-labelled images adding real, non-synthetic signal; and a verified,
defect-free dataset trained in one campaign rather than assembled by sequential fine-tuning.

**Factors that depress performance, and would be expected to:** **augmentation inflation**, since
the gap-fill sources are ~3× Roboflow-augmented and augmented copies are far less informative than
independent scenes; **domain shift** between clean academic imagery and messy real street
photographs; **limited model capacity**, since even at the small tier this is a 9.5 M-parameter
network separating 17 classes, with heavy classes dominating the gradient and suppressing
rare-class learning; **class heterogeneity by design** in `construction`; and **label noise** from
merging ten independently-annotated sources with differing conventions — what counts as a "kerb"
in one dataset may not in another.

### 4.3.3 From trained model to deployed model

The model evaluated here is the model the product runs: the Stage IV 17-class YOLO26-small
weights, selected on validation mAP@0.50:0.95 and exported alongside a fixed-shape ONNX artefact,
shipped with the deployment — baked into the Railway image, committed to the repository for the
GPU VM — rather than downloaded at startup. Deploying a retrained model touches four places that
must move together or the system will report classes it cannot detect and fail to report classes
it can: the weights, the server's class list and default high-risk set, the client's Hebrew
class-name maps, and the settings class grid. The server and client halves were, at the time of
the code review in §3.5.10, still out of sync with the model — a filtering and labelling defect
rather than a detection defect, but exactly the drift one shared list per side would prevent
(§6.3).

## 4.4 Runtime and Latency Results

### 4.4.1 The optimisation log

**Table 4.2 — Performance optimisations with measured effect**

| # | Change | Measured effect |
|---|---|---|
| 1 | Per-frame HTTP POST → **WebSocket streaming** | Removed per-frame handshake and header overhead; server-side cost ~33 ms at the 640-pixel input in use at the time (rows 6 and 8 change the configuration these figures refer to, so the column is not a single monotone series) |
| 2 | Per-frame settings read → **in-memory cache** | **71 ms → 0 ms** per frame |
| 3 | **DB write off the hot path** (pre-generated ObjectId) | `record_id` returned before the insert; no I/O in the request path |
| 4 | **Inference to a worker thread** (`asyncio.to_thread`) | Event loop free; `/health` keeps answering, so the watchdog stops falsely reporting instability |
| 5 | **Global inference lock** | Prevents concurrent forward passes |
| 6 | **Input size 640 → 512** (CPU era) | The single biggest CPU-era win — smaller uploads *and* faster inference → **~41 ms/frame** 🔴; reverted to 640 once the GPU made the full size affordable |
| 7 | Client `toDataURL` → **`toBlob`** | Async encode, no base64 round trip; quality checks moved onto the resized image |
| 8 | **Thread-pool capping at 8** (CPU era) | Container reported 48 cores while limited to 8 vCPU; 16 threads measured **4× slower** than 8. **Reverted on the GPU VM**, where the same cap cost 26–36 → 149 ms/frame (§5.3) |
| 9 | **ONNX Runtime port** | ~1.5× faster locally, **75× slower on the deployment target (2,323 ms/frame)** — reverted and removed |
| 10 | **Trend-based motion + per-track alert dedup** | Not raw speed, but removed a flood of redundant alerts and work |
| 11 | Per-frame write threads → **batched writer** | One daemon thread flushing once a second replaced ~80 threads and 80 round trips per second at 40 FPS; event-loop starvation and a frozen overlay disappeared |
| 12 | **CPU container → GPU VM** | Inference ~200 ms → tens of milliseconds; end-to-end ~130 ms, of which ~90–100 ms is network distance (Warsaw↔Israel). Sessions disagree on the server figure — see the spread in §4.4.2 🔴 |
| 13 | **In-flight depth retuned 4 → 6** against measurement | Closed the gap between server-capacity FPS and client FPS at the cost of one queue slot of delay (§4.4.2) |

Items 8 and 9 share a root cause (§5.3) and are the most valuable entries precisely because they
are negative results: the local benchmark was not merely optimistic, it pointed in the opposite
direction from the truth — and item 8's reversal on the GPU target repeated the lesson in the
other direction.

### 4.4.2 Latency and throughput

Two operating points bracket the project. **CPU era (Railway, input 512):** server processing
approximately **41 ms** per frame, dominated by the `inference` stage; a ~131 ms mobile round
trip; ~22 FPS at ~216 ms end-to-end at depth 4. **GPU era (GCP VM, input 640, measured
6 August 2026):** server time approximately **16.4 ms** per frame — a ~61 FPS ceiling — with a
no-queue round trip R₀ ≈ **120 ms**, of which ~90–100 ms is Warsaw↔Israel network distance; at
the shipped depth of 6 the pipeline sustains roughly **50 FPS at ~120–130 ms end-to-end**. One
warm-up artefact is known: the first frames after a fresh process start include a one-time CUDA
initialisation spike (a ~13 s outlier was observed once), which contaminates maxima but not
steady-state averages.

🔴 **The GPU-era server figure is not yet settled, and the book should not pretend otherwise.**
Three measurement sessions survive in the repository and they disagree: the deployment log of
5 August records a **28 ms** average with ~130 ms end-to-end; the thread-cap experiment records a
healthy band of **26–36 ms** per forward pass against 149 ms with the cap in place; and the client
configuration's note of 6 August, from which the shipped pipeline depth was chosen, records
**16.4 ms**. The last is the most recent and the only one accompanied by a depth sweep, so it is
the figure used above, but a spread of roughly two-to-one across three sessions is not a measured
result. `[[TODO: reset the persisted stats on the GPU deployment, run one clean session, and
replace this paragraph with the per-stage server and client averages from GET
/get_system_status.]]`

The network is now the largest term by far — roughly three-quarters of the end-to-end budget is
distance, not compute. This is the fundamental argument for the on-device model in future work,
and in the nearer term for migrating the VM to a Tel Aviv region when GPU capacity appears: no
amount of server optimisation can remove a term not spent on the server.

**Table 4.3 — Throughput versus in-flight depth** (GPU deployment: S ≈ 16.4 ms, R₀ ≈ 120 ms)

| Depth | Throughput | Per-frame latency | Note |
|---:|---|---|---|
| 6 (deployed) | ~50 FPS | ~120 ms | Just below the crossover depth ≈ R₀/S ≈ 7 |
| 7 | ~53 FPS | ~133 ms | At the crossover |
| 10 | ~60 FPS | ~166 ms | Ceiling reached; queueing begins |
| 20 | ~60 FPS | ~332 ms | Same throughput, double the delay |

> **Figure 4.4** — Latency breakdown per pipeline stage (client, network, server), and throughput
> versus pipeline depth showing the crossover at depth ≈ 7.
>
> `[[FIGURE: two-panel — stacked latency bar, and dual-axis depth curve from Table 4.3]]`

This is the classic bandwidth–delay shape: throughput rises with depth until it saturates at the
server's service rate (1/S), after which additional depth buys only queueing delay. The deployed
depth of 6 deliberately sits below the crossover, keeping the pipeline in the regime where
latency ≈ R₀ — because for a safety system, latency rather than frame rate is the number that
matters.

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
becoming visible and the user hearing the warning — approximately **120–130 ms** at the deployed
GPU operating point, and ~216 ms in the earlier CPU deployment. Both columns are shown, because
the reaction-distance arithmetic is the strongest argument for the migration.

| Moving party | Speed | Distance in 130 ms | in 216 ms |
|---|---|---:|---:|
| Walking pedestrian (the user) | 1.4 m/s | **0.18 m** | 0.30 m |
| Running person | 3.0 m/s | 0.39 m | 0.65 m |
| Cyclist / e-scooter | 6.0 m/s | 0.78 m | 1.30 m |
| Urban car | 13.9 m/s (50 km/h) | **1.81 m** | 3.00 m |
| Fast urban car | 19.4 m/s (70 km/h) | 2.52 m | 4.19 m |

A user walking at normal pace advances about 18 cm between a hazard appearing and being warned
about it — a fraction of a stride, a comfortable margin against a static obstacle detected at the
"Close" threshold. A car at 50 km/h closes 1.8 m during the same delay (3 m at the CPU-era
latency), which is why the alert logic escalates a fast-approaching high-risk object to `high` at
**any** distance rather than waiting for the box to grow (§3.4.4): with a vehicle, the distance
at the moment of detection is the entire safety margin.

Frame rate determines how often the situation is *re-assessed*; latency how stale each assessment
is. At **7 FPS** the gap between assessments is 143 ms, during which a pedestrian moves 20 cm and
a car 2 m — adequate for static obstacles, marginal for vehicles. At the deployed **~50 FPS** the
gap is 20 ms — 3 cm of pedestrian motion, 28 cm of car motion — and the 0.8-second approach
window (§3.4.3) holds around forty samples, ample for the trend fit to separate genuine approach
from box jitter; even at a degraded 10 FPS it still holds the minimum five.

**Conclusion: 7 FPS is sufficient for walking-pace hazards but not comfortable for vehicles;
~50 FPS at ~120–130 ms end-to-end is a comfortable operating point for urban pedestrian use.**
Against the original POC criterion — an alert within 200–300 ms under stable network conditions —
the delivered system is well inside the band, and even the earlier CPU deployment's ~216 ms met
it.

Of that budget, ~90–100 ms is network distance and ~16 ms server processing. **The network is the
largest term and the one we cannot optimise from the server.** Three routes would reduce it:
**on-device inference**, removing the network term entirely — our own characterisation estimated
80–150 ms inference on an Apple Neural Engine, giving 100–180 ms total; **edge deployment
geographically closer to users** — now demonstrated concretely, since the VM sits in Warsaw only
because Tel Aviv had no GPU capacity, and migrating would remove most of the remaining latency;
and **lower input size**, reducing both upload and inference at a measurable cost in small-object
recall — a knob the user could in principle be given as "prioritise responsiveness" versus
"prioritise detail".

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
| Continuous | Yes | Yes | Yes | No — on demand | **Yes, ~50 FPS** |
| Latency | Instant | Instant | Instant | Seconds | **~130 ms** |
| Motion awareness | No | Yes | No | No | **Yes — tracked approach and speed** |
| Cost / extra hardware | Very low / cane | Very high / dog | Low / device | Low / none | **Low / none** |
| Training required | Weeks | Months, both parties | Minimal | Minimal | **Minimal** |
| Works offline | Yes | Yes | Yes | Partly | **No** — current limitation |

The sharpest comparison, however, is not with any of those columns but with the alternative
architecture SeeSense itself rejected. A browser-based on-device detector — the same detector
family running in TensorFlow.js on the phone — buys independence from the network outright: no
round trip, no connectivity dependency, and camera frames that never leave the device. It pays
for that with model capacity and frame rate, since a phone browser cannot run a 9.5 M-parameter
network at 640 pixels and tens of frames a second. SeeSense made the opposite call, buying
capacity and frame rate at the cost of a hard network dependency (§2.7).

Both calls are defensible, and which is correct depends on the user rather than on the
engineering. For someone moving fast through variable coverage, uninterrupted operation is worth
more than fine discrimination. For a pedestrian at walking pace whose hazards are small,
low-contrast and easily confused with one another — a kerb against a pavement, a bollard against
a pole — the larger model and the higher re-assessment rate matter more, and a dropped connection
is an inconvenience rather than a danger, because the cane is still in their hand. The correct
end state is the hybrid of §6.3: the server model while the network is good, an on-device model
when it is not.

## 4.8 Discussion of Findings

**What the results establish.** A self-trained small-tier detector on a self-assembled dataset is
sufficient to drive a usable assistive system, and the trajectory — from a hand-written detector
that could not generalise, through a baseline with 0.409 recall, to a 17-class model on 91,139
verified images — demonstrates that the limiting factor at every stage was **data, not
architecture**. The single largest accuracy improvement came from taking `crosswalk` from 33 images
to 3,511, not from changing any model. **The systems engineering matters as much as the model**:
the difference between 33 ms and 41 ms of server latency is small, but the difference between an
alert on every frame and an alert on every *change* is the difference between a product and an
unusable demo — and the difference between 2,323 ms and 41 ms per frame came from a thread-count
environment variable. And **measurement on the deployment target is not optional**: the ONNX
experiment benchmarked 1.5× faster locally and ran 75× slower in production, and the thread cap
that saved the CPU deployment had to be removed on the GPU one, where it cost 5× by itself.

**What the results do not establish.** The stage-to-stage improvements are not directly comparable,
which is why Table 4.1 names each row's evaluation set. The Stage III oversampling result is
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
