## 3.3 Model Development and Training

Model development progressed through five stages, each a response to a limitation discovered in
the previous one. This section documents all five, including the two that failed, because the
failures determined the shape of the final system more than the successes did.

| Stage | Approach | Outcome |
|---|---|---|
| I | Custom detector: frozen ResNet18 + hand-written YOLO-style head | Trained, but stopped generalising at epoch 4; unreliable, pole-biased detections. **Abandoned.** |
| II | YOLOv8-nano, transfer learning, 10 classes | P 0.669 / R 0.409 / mAP@0.50 0.4528. Usable, recall far too low. |
| III | YOLOv8 fine-tune on a 14-class expanded dataset | P 0.862 / R 0.818 / mAP@0.50 0.853 on the new-class split. |
| IV | Targeted oversampling of the two weakest classes | Recall 0.818 → 0.843; evaluation methodology compromised. |
| V | YOLO11-nano on the final 17-class, 91,139-image dataset, 3 seeds | **The delivered model.** |

An additional attempted stage — merging the old and new datasets — failed silently and is
documented in §3.3.6 as a negative result.

### 3.3.1 Why a detector had to be built before one could be chosen

The supervisor's instruction in the first review meeting was explicit and repeated: train the
objects yourself, do not take a library somebody else has already trained. It would have been
faster to open Ultralytics on day one; it would also have meant graduating from a deep-learning
specialisation without ever having written a loss function balancing four competing terms.
Stage I had two goals — understand every component of a detection pipeline by implementing it, and
establish whether a compact custom detector could meet the requirement. It succeeded at the first
and failed at the second, which is precisely the information that justified moving on.

### 3.3.2 Stage I — Custom ResNet18-based detector

A pretrained ResNet18 [15] served as the backbone with its classification head replaced by a
custom convolutional detection head. Predictions were made on a 20×20 grid with two candidate
boxes per cell, each carrying objectness, four coordinates and ten class scores — an output tensor
of shape `[batch, 20, 20, 2, 15]`. The backbone was **frozen**, reducing computation and
protecting pretrained features at the cost of preventing adaptation to street scenes. Ground-truth
boxes were encoded into grid targets, and we ran a **grid-collision analysis**, because a
fixed-grid detector with two boxes per cell structurally cannot represent three objects whose
centres fall in one cell — a real constraint on a crowded pavement.

The loss combined four terms with hand-tuned weights: objectness 1.0, no-object 0.5 (down-weighted
because empty cells vastly outnumber occupied ones), box regression 5.0 (up-weighted because
coordinate errors are small in magnitude and would otherwise be drowned out) and classification
1.0. A pre-training sanity check gave total loss 4.5647 — objectness 1.1239, box 0.1551,
classification 2.6651, the large classification term being expected from a randomly-initialised
head.

Ten epochs were scheduled with `ReduceLROnPlateau` and early stopping at patience 3, batch size 8,
LR 0.001, images at 640×640.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train loss | 1.3272 | 1.2316 | 1.1899 | 1.1642 | 1.1375 | 1.1184 | 1.0728 |
| Val loss | 1.3594 | 1.3151 | 1.3147 | **1.2759** | 1.2883 | 1.2883 | 1.2840 |
| | save | save | save | save | 1/3 | 2/3 | **stop** |

> **Figure 3.4** — Custom ResNet18 detector: training loss falls monotonically while validation
> loss reaches its minimum at epoch 4 and then stalls.
>
> `[[FIGURE: line plot of the table above with the early-stop point marked]]`

The signature is textbook overfitting. Qualitatively, before training the model produced no
detections at confidence 0.6 — correct for an untrained head; after training one sample produced
six detections and another 54 after NMS, most labelled `pole`. The model had learned to emit
boxes, but the volume, duplication and overwhelming bias toward `pole` — the class with 314,120 of
the base dataset's annotations — showed weak confidence calibration, class bias driven directly by
the imbalance of §3.2.6, and inadequate duplicate suppression.

Three compounding causes: the **frozen backbone** could not adapt; the **simplified grid head**
cannot represent dense scenes; and the **imbalance** made predicting poles the cheapest way to
reduce loss. The conclusion presented at the POC meeting was that a from-scratch CNN struggles
with dense street scenes and that the correct response was a mature single-stage detector. This
was the project's first significant experimental result.

### 3.3.3 Stage II — YOLOv8-nano baseline on ten classes

YOLOv8 supplies everything Stage I had to hand-build and get right: multi-scale heads, anchor-free
label assignment, CIoU and distribution focal losses, mixed precision, augmentation, checkpointing
and standard metrics.

**Configuration:** `yolov8n.pt` COCO-pretrained; 10 classes; 100 epoch target; 640×640; batch 16;
patience 20; auto-selected optimizer at lr 0.01, momentum 0.9; NVIDIA A100-SXM4-40GB;
3,012,798 parameters and 8.2 GFLOPs. The pretrained checkpoint transferred 319 of 355 parameter
tensors, the mismatch being the detection head, necessarily reinitialised for a different class
count. Training used 25,892 images, validating against 7,399 images with 172,375 instances.

The first epochs were not monotonic — validation mAP@0.50 fell from 0.300 at epoch 1 to 0.226 at
epoch 3 before recovering, a normal consequence of a freshly initialised head disturbing
pretrained features, and a useful reminder not to judge a run by its first epochs.

**Test results (3,699 images, 86,381 instances):** Precision **0.669**, Recall **0.409**,
mAP@0.50 **0.4528**, mAP@0.75 0.2954, mAP@0.50:0.95 **0.2847**; inference 0.8 ms/image with
0.6/1.0 ms pre/post-processing on the A100.

Precision of 0.669 means two thirds of the model's claims were correct. Recall of 0.409 means it
missed nearly six of every ten real objects. For a safety system that asymmetry is the wrong way
round: a false alarm costs a moment of irritation, a missed obstacle can cost a fall. The low
recall was driven substantially by the rare classes — `stairs` with 836 annotations competing for
gradient against `pole` at 314,120 — and became the central motivation for the entire
data-expansion effort of §3.2.

### 3.3.4 Stage III — Expansion to 14 classes and fine-tuning

Seven external single-class datasets were extracted and remapped into a common label space. Each
used class ID 0 internally; labels were rewritten so ID 0 became the correct global class, and
filenames were given source-specific prefixes to prevent collisions.

| Source | → class | Train / val / test annotations |
|---|---|---|
| `bicycle_extracted` | bicycle | 1,582 / 446 / 230 |
| `dogs_extracted` | dog | 3,630 / 620 / 300 |
| `stairs_extracted` | stairs | 1,687 / 200 / — |
| `bollard_extracted` | bollard | 3,051 / 879 / 434 |
| `crosswalk_extracted` | crosswalk | 961 / 291 / 134 |
| `pothole_extracted` | pothole | 3,277 / 377 / 399 |
| `scooter_extracted` | scooter | 1,411 / 62 / 51 |

The intermediate dataset held 8,176 train / 1,598 val / 784 test images, with 106 background
images in train and 18 in val and zero corrupt images. The best Stage II checkpoint was fine-tuned
for 30 epochs at 640×640, batch 16, on an A100, with a deliberately **low initial learning rate of
0.0003** — adapting the detector to the expanded label space without destroying what Stage II had
learned.

| Split | Images | Instances | P | R | mAP@0.50 | mAP@0.50:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 1,598 | 2,875 | 0.872 | 0.803 | 0.850 | 0.599 |
| **Test** | **784** | **1,548** | **0.862** | **0.818** | **0.853** | **0.618** |

**An important caveat.** These numbers are dramatically better than Stage II's — and they are *not
comparable to them*. They were measured on a different, smaller, more specialised test set built
around the newly-collected classes, where a single-class source image typically contains one
clearly-photographed instance of one object. The original test set contained 86,381 instances
across 3,699 dense street scenes. Reporting 0.853 as though it superseded 0.4528 would be a
straightforward misrepresentation, which is why Table 4.1 names each row's evaluation set.

**Instance-count analysis**, a cheap diagnostic that reveals tendencies mAP hides:

| Class | Ground truth | Predicted @ 0.25 | Difference |
|---|---:|---:|---:|
| bollard | 434 | 519 | **+85** over-predicted |
| crosswalk | 134 | 133 | −1 |
| pothole | 399 | 329 | **−70** under-predicted |
| scooter | 51 | 51 | 0 |

Bollards were over-predicted, likely confusion with poles and other narrow vertical street
furniture; potholes under-predicted, being low-contrast, irregular and usually at an oblique
angle; and `scooter`'s perfect count came from only 51 test instances, too small a sample to
conclude anything from.

### 3.3.5 Stage IV — Weak-class oversampling

`pothole` and `scooter` were selected as the weakest classes, and every image containing either
was duplicated twice in addition to the original, raising train from 8,176 to 11,384 images. The
model was initialised from the Stage III checkpoint and fine-tuned for 15 epochs at LR 0.0001.

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Precision | 0.862 | 0.857 | −0.005 |
| **Recall** | 0.818 | **0.843** | **+0.025** |
| mAP@0.50 | 0.853 | 0.866 | +0.013 |
| mAP@0.50:0.95 | 0.618 | 0.630 | +0.012 |
| Test images / instances | 784 / 1,548 | 1,082 / 2,448 | +298 / +900 |

Recall — the metric that matters most for a safety system — rose 0.025 at the cost of 0.005
precision, the expected trade.

**Methodological limitation, stated plainly.** The oversampling was applied to **all three
splits**. The "after" test set is not the same test set as the "before" one, and it over-weights
precisely the classes the intervention targeted. A rigorous experiment oversamples training only.
The improvement is therefore encouraging but not cleanly measured — one of the specific errors
that motivated the "freeze the test set" discipline of §3.6.2. A second limitation is more
fundamental: duplicating an image adds frequency but not diversity. The model sees the same
pothole three times, not three potholes, which is why the genuine fix is what §3.2 spent its
effort on.

### 3.3.6 The failed combined-dataset experiment

The final legacy objective was merging the original ten-class dataset with the new-class dataset
into one 14-class corpus and fine-tuning for 50 epochs. What the stored notebook output shows: the
old splits copied successfully; the new train, validation and test splits **all skipped because
their folders were missing**, with the script printing `Skipping new train - missing folder` and
continuing; a "combined" dataset containing exactly the original counts; and a generated YAML
declaring 14 classes, four of which had no data at all.

| Metric | Original YOLOv8 test | "Combined" run |
|---|---:|---:|
| Precision | 0.669 | 0.671 |
| Recall | 0.409 | 0.404 |
| mAP@0.50 | 0.4528 | 0.4372 |
| mAP@0.50:0.95 | 0.2847 | 0.2750 |

The run performed slightly *worse* than the baseline it was meant to improve — exactly what one
expects from training a 14-class head on 10-class data. **This is not a valid combined-dataset
result and must not be read as one.**

**Root cause:** the pipeline depended on a path in Colab's ephemeral local storage, reclaimed
between the session that created the intermediate dataset and the session that consumed it.

**Why the failure was valuable.** It cost a 50-epoch GPU run and produced a scientifically
meaningless number, but it generated the four requirements that shaped the final pipeline:
persist intermediate artefacts to Drive, never `/content`; **fail loudly** — a missing required
input must raise, not print and continue; **verify the artefact, not the script** (§3.2.10 is this
lesson made routine); and never let the class count and the data disagree, because a 14-class YAML
over 10-class data produced a run that looked valid, reported plausible metrics, and meant
nothing.

### 3.3.7 Stage V — The final YOLO11 model

The pipeline moved from `yolov8n.pt` to `yolo11n.pt`. Because the two share an identical label
format this required no data conversion whatsoever — only the base model filename changed. The
previous YOLOv8 checkpoint could not resume YOLO11 training, so training started from pretrained
YOLO11 weights with the v8 result retained as a comparison baseline.

**Strategy: train N times, keep the best.** A single run is sensitive to random initialisation,
augmentation sampling and data ordering, so quoting one number gives no sense of variance. We
trained **three times with different seeds** and automatically selected the highest validation
mAP@50-95:

```python
N_RUNS, EPOCHS, IMG, BATCH = 3, 80, 640, 16
SEEDS = [0, 1, 2]

results = []
for i in range(N_RUNS):
    model = YOLO("yolo11n.pt")
    model.train(data=yaml, epochs=EPOCHS, imgsz=IMG, batch=BATCH,
                seed=SEEDS[i], patience=20, verbose=False)
    results.append((float(model.val(data=yaml).box.map), SEEDS[i], best_pt))

results.sort(reverse=True)              # keep the best-performing seed
best_score, best_seed, best_path = results[0]
```

**Table 3.3 — Final training configuration**

| Parameter | Value | Rationale |
|---|---|---|
| Base model | `yolo11n.pt` | Lightweight, real-time capable, newest in the family |
| Learning strategy | Transfer learning | Far faster convergence than from-scratch on 17 classes (§2.3) |
| Epochs / patience | 80 / 20 | Early stopping if validation stalls |
| Image size | 640×640 letterbox | Identical to the server's preprocessing |
| Batch size | 16 | Fits available Colab GPU memory |
| Runs | 3 (seeds 0, 1, 2) | Robustness against an unlucky initialisation |
| Selection metric | Validation **mAP@50-95** | The strict metric; rewards accurate localisation |
| Augmentation | Mosaic, flip, scale, HSV (defaults) | Generalisation and implicit rare-class enrichment |

> **Figure 3.5** — Sample augmented training batches showing mosaic composition, HSV jitter and
> random scale/flip with YOLO boxes overlaid.
>
> `[[FIGURE: train_batch0.jpg and train_batch1.jpg from the winning run]]`

The three seed scores give a small empirical spread, which is a far more honest statement of
performance than a single number: `[[TODO: report the three per-seed validation mAP@50-95 values
and the selected seed]]`.

**Infrastructure and reproducibility.** All training ran on Google Colab (A100-SXM4-40GB for the
legacy stages; `[[TODO: confirm the GPU for the final runs]]`) with PyTorch [41] and Ultralytics
[4]. Reproducibility measures adopted after the failures of §3.2.7 and §3.3.6: fixed seeds (42 for
the split, 0/1/2 for training); Drive persistence, never `/content`; a resumable, idempotent
dataset builder; explicit verification of the built artefact before any run; and a single
`data.yaml` with relative paths so the dataset is portable between machines. Full parameters for
all five stages are in Appendix C.
