## 3.3 Model Development and Training

Model development progressed through four stages, each a direct response to a limitation measured
in the previous one. This section documents all four, including the negative results, because the
failures determined the shape of the final system more than the successes did. The trajectory was
not a planned tour of architectures; it was a sequence of responses to specific, measured failures.

| Stage | Approach | Outcome |
|---|---|---|
| I | Custom detector: frozen ResNet-18 + hand-written single-scale head, 10 classes | Early-stopped at epoch 7, best validation loss 1.2759 at epoch 4. Collapsed to predicting `pole` almost everywhere — and **no detection metric was ever computed**, so the collapse was invisible. **Abandoned.** |
| II | YOLOv8-nano transfer learning, 10 classes, 100 epochs | P 0.670 / R 0.411 / mAP@0.50 0.458 / mAP@0.50:0.95 0.289 on 7,399 validation images. A trustworthy baseline; recall far too low for a safety system. |
| III | Vocabulary extension to 14 classes: fine-tuning on seven auxiliary sources, weak-class oversampling, and an attempted merge | Excellent numbers on a narrow benchmark, obtained at the cost of **catastrophic forgetting** of the original ten classes, a contaminated evaluation split, and a silently corrupted 50-epoch run. Three negative results, all load-bearing. |
| IV | YOLO26-small on the verified 17-class, 91,139-image dataset | **The delivered model.** P 0.7542 / R 0.5872 / mAP@0.50 0.6356 / mAP@0.50:0.95 0.4581 over 11,086 validation images and 180,736 instances. |

Complexity across these stages did not rise monotonically — it **peaked at Stage I and fell
thereafter**. Stage I required hand-implementing the target encoder, the composite loss, the
decoder, the IoU computation, non-maximum suppression and all visualisation, and produced no
evaluation harness despite that effort. Stage II replaced every line of it with framework calls.
Complexity then reappeared one level up, in Stage III's custom data-handling code, which is
precisely where the project's most expensive failure occurred. Stage IV re-invested that complexity
into verification and infrastructure. The pattern is worth stating once: **effort spent
re-implementing solved algorithmic problems produced an unevaluable model; effort spent verifying
data and securing infrastructure produced a deployable one.**

### 3.3.1 Why a detector had to be built before one could be chosen

The supervisor's instruction in the first review meeting was explicit and repeated: train the
objects yourself, do not take a library somebody else has already trained. It would have been
faster to open Ultralytics on day one; it would also have meant graduating from a deep-learning
specialisation without ever having written a loss function balancing four competing terms.
Stage I had two goals — understand every component of a detection pipeline by implementing it, and
establish whether a compact custom detector could meet the requirement. It succeeded at the first
and failed at the second, which is precisely the information that justified moving on.

### 3.3.2 Stage I — Custom ResNet18-based detector

A pretrained ResNet-18 [15] served as the backbone with its classification head replaced by a
custom convolutional detection head: a 3×3 convolution reducing 512 channels to 256, batch
normalisation, ReLU, then a 1×1 convolution to the output channels. A 640×640 input yields a 20×20
stride-32 feature map, and predictions were made on that grid with two candidate boxes per cell,
each carrying objectness, four coordinates and ten class scores — an output tensor of shape
`[batch, 20, 20, 2, 15]`, verified in the notebook as `torch.Size([2, 20, 20, 2, 15])`. The
backbone was **frozen**, confining all learning to roughly 1.2 million head parameters over
features never adapted to street scenes.

The data was the base ten-class set of §3.2.2: 25,892 training, 7,399 validation and 3,699 test
images covering person, car, bicycle, motorcycle, bench, fire hydrant, traffic light, stairs, pole
and dog. Its training-split annotation counts were already severely skewed — `pole` 314,120,
`car` 124,449, `person` 78,872, `traffic_light` 56,978, then a long tail down to `bicycle` 8,752,
`motorcycle` 8,656, `fire_hydrant` 2,740, `bench` 2,152, `dog` 2,136 and `stairs` 836. Poles alone
were roughly 51 % of every annotation the model would ever see.

The loss combined four terms with hand-tuned weights: objectness 1.0 (binary cross-entropy with
logits), no-object 0.5 (down-weighted because empty cells vastly outnumber occupied ones), box
regression 5.0 (mean squared error, up-weighted because coordinate errors are small in magnitude
and would otherwise be drowned out) and classification 1.0 (cross-entropy). A pre-training sanity
check gave total loss 4.5647 — objectness 1.1239, box 0.1551, classification 2.6651, the large
classification term being expected from a randomly-initialised head. Augmentation was limited to
colour jitter (brightness, contrast and saturation 0.2, hue 0.05) with ImageNet normalisation.

Ten epochs were scheduled with Adam at learning rate 0.001, `ReduceLROnPlateau` at factor 0.5 and
patience 1, early stopping at patience 3, batch size 8, images at 640×640.

**Table 3.3 — Custom detector training and validation losses**

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train loss | 1.3272 | 1.2316 | 1.1899 | 1.1642 | 1.1375 | 1.1184 | 1.0728 |
| Val loss | 1.3594 | 1.3151 | 1.3147 | **1.2759** | 1.2883 | 1.2883 | 1.2840 |
| Learning rate | 0.001 | 0.001 | 0.001 | 0.001 | 0.001 | 0.0005 | 0.0005 |
| | save | save | save | save | 1/3 | 2/3 | **stop** |

> **Figure 3.4** — Custom ResNet-18 detector: training loss falls monotonically from 1.3272 to
> 1.0728 while validation loss reaches its minimum at epoch 4 and then plateaus near 1.28.
>
> `[[FIGURE: line plot of the table above with the early-stop point marked]]`

The signature is textbook overfitting, and it was the *only* signal the run produced. Qualitative
decoding was far more revealing. Before training, decoding at confidence 0.25 yielded zero
detections — correct for an untrained head. After training, decoding at confidence 0.55 followed by
NMS at IoU 0.5 produced **six surviving detections on a sample image, every one of them labelled
`Pole`**, with confidences between 0.568 and 0.725 and several boxes degenerate, spanning as little
as three pixels in width. Lowering the threshold to 0.50 on another sample produced 54 detections,
showing a confidence distribution concentrated in a narrow band where small threshold changes
caused order-of-magnitude changes in detection count.

**Four failure mechanisms** were identified, and each maps onto a design decision taken later.

The first was **grid capacity saturation**. A 20×20 grid with two predictions per cell can
represent at most two objects whose centres fall in the same 32×32-pixel region. Against 314,120
pole annotations — frequently receding rows of near-collinear vertical structures — and 124,449
cars in dense traffic, ground-truth objects were systematically discarded during target encoding.
The model was trained against targets that were themselves incomplete, and no amount of
optimisation recovers information destroyed before the forward pass.

The second was **class collapse under extreme imbalance**. With poles at roughly 51 % of all
annotations, the cross-entropy term was minimised most efficiently by predicting `Pole` almost
everywhere. That every surviving detection carried the pole label is the direct empirical
expression of this collapse, and it is the same imbalance quantified in §3.2.6.

The third was **inadequate box regression**. Mean squared error on sigmoid-activated width and
height optimises a quantity only loosely correlated with intersection over union. The box loss fell
from 0.0438 to 0.0421 across the entire run — roughly 4 % — meaning the localisation branch learned
almost nothing. The three-pixel boxes are the visible consequence.

The fourth, and by far the most consequential, was the **absence of a detection metric**. The
experiment measured loss, and loss alone: no mean average precision, no per-class precision or
recall, no confusion matrix. Consequently the run *appeared* to be succeeding — training loss fell
monotonically — while the detector was collapsing to a single class. **A pipeline that cannot
detect its own failure is not a research instrument.** To these must be added the frozen backbone
and the single stride-32 prediction scale, which left the model structurally incapable of
localising small objects — the majority of the navigation-relevant vocabulary.

**Why it was abandoned rather than repaired.** Repair would have required implementing multi-scale
prediction over a feature pyramid, replacing fixed-slot assignment with dynamic label assignment,
substituting an IoU-family loss for MSE, adding class-balanced sampling, and building a complete
mAP evaluation harness. Each is a component a mature framework already provides in tested,
optimised form. The engineering judgement was that scarce effort belonged on dataset quality and
deployment characteristics rather than on re-implementing solved problems. Three lessons carried
forward: no run is *evaluated* in the absence of standard detection metrics; extreme class
imbalance is a first-order design constraint, not an incidental property of the data; and
target-encoding schemes must be validated against the actual object density of the dataset before
training begins, because information discarded at encoding time is unrecoverable downstream.

### 3.3.3 Stage II — YOLOv8-nano baseline on ten classes

YOLOv8 was selected because it resolves all four Stage I failures *structurally*: a feature-pyramid
neck with three detection scales answers the single-scale limit; anchor-free dynamic label
assignment eliminates fixed-slot target collisions; distribution focal loss with a complete-IoU
objective replaces MSE with an objective aligned to localisation quality; and the integrated
validator produces per-class precision, recall, mAP@0.50 and mAP@0.50:0.95 automatically, ending
the evaluation blindness that had made Stage I uninterpretable. The **nano** variant was chosen
deliberately: the detector had eventually to run on constrained hardware, and establishing the
baseline at the smallest capacity point yields the most honest picture of what the domain demands.

**Configuration.** Initialised from the COCO-pretrained `yolov8n.pt`; the head automatically
reconfigured from 80 to 10 classes while backbone and neck retained pretrained weights. The
instantiated model comprised **130 layers, 3,012,798 parameters and 8.2 GFLOPs**, fusing at
inference to **73 layers, 3,007,598 parameters and 8.1 GFLOPs**. Training ran 100 epochs at
640×640, batch 16, early-stopping patience 20, on an NVIDIA A100-SXM4-40GB. The optimizer was left
in automatic mode and resolved to **MuSGD at learning rate 0.01, momentum 0.9** across three
parameter groups. Default augmentation was retained — mosaic with closure over the final ten
epochs, HSV jitter, horizontal flip, scale and translation, plus light Albumentations transforms.
Each epoch comprised 1,619 iterations; the complete run required **6.243 hours**.

The first epochs were not monotonic — validation mAP@0.50 fell from 0.300 at epoch 1 to 0.226 at
epoch 3 before recovering, a normal consequence of a freshly initialised head disturbing pretrained
features, and a useful reminder not to judge a run by its first epochs. Thereafter improvement was
smooth but sharply diminishing: mAP@0.50 reached 0.362 by epoch 10, 0.409 by epoch 20, 0.431 by
epoch 30, 0.443 by 50, 0.449 by 70 and 0.458 at epoch 100. Roughly 94 % of the final mAP was
reached in the first 30 of 100 epochs.

**Final validation (7,399 images, 172,375 instances):** Precision **0.670**, Recall **0.411**,
mAP@0.50 **0.458**, mAP@0.50:0.95 **0.289**.

**Table 3.4 — YOLOv8n per-class validation results (Stage II)**

| Class | Images | Instances | P | R | mAP@0.50 | mAP@0.50:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| Person | 4,406 | 22,771 | 0.713 | 0.444 | 0.513 | 0.305 |
| Car | 5,106 | 35,833 | 0.758 | 0.546 | 0.617 | 0.420 |
| Bicycle | 1,209 | 2,632 | 0.662 | 0.337 | 0.401 | 0.225 |
| Motorcycle | 1,062 | 2,342 | 0.648 | 0.460 | 0.504 | 0.294 |
| Bench | 358 | 666 | 0.616 | 0.176 | 0.207 | 0.148 |
| Fire hydrant | 705 | 788 | 0.779 | 0.454 | 0.516 | 0.379 |
| Traffic light | 2,469 | 16,669 | 0.684 | 0.333 | 0.389 | 0.198 |
| Stairs | 174 | 219 | 0.595 | 0.543 | 0.581 | 0.374 |
| **Pole** | 3,586 | **89,785** | 0.617 | **0.179** | **0.222** | **0.107** |
| Dog | 505 | 670 | 0.625 | 0.636 | 0.630 | 0.436 |

**Test results (3,699 images, 86,381 instances):** Precision **0.6692**, Recall **0.4086**,
mAP@0.50 **0.4528**, mAP@0.75 0.2954, mAP@0.50:0.95 **0.2847**; inference 0.8 ms/image with 0.6 ms
pre- and 1.0 ms post-processing on the A100, from a 6.3 MB checkpoint. The validation-to-test gap
of 0.005 in mAP@0.50 and 0.004 in mAP@0.50:0.95 is evidence of genuine generalisation rather than
memorisation of the validation split.

Two readings matter. Precision of 0.670 means two thirds of the model's claims were correct; recall
of 0.411 means it missed nearly six of every ten real objects. **For a safety system that asymmetry
is the wrong way round**: a false alarm costs a moment of irritation, a missed obstacle can cost a
fall. And the per-class table shows the deficit is not uniform — it is concentrated in `pole`
(recall 0.179 over 89,785 instances), `bench` (0.176 over 666) and `traffic_light` (0.333). Note
that these two groups fail for opposite reasons: `bench` and `stairs` are starved of data, while
`pole` is the most abundant class in the dataset and *still* the worst. That distinction — scarcity
versus geometry — directed everything that followed. The scarcity half became the data-expansion
effort of §3.2; the geometry half is still unsolved and still visible in the delivered model.

### 3.3.4 Stage III — Vocabulary extension, and what it cost

Stage III set out to add navigation-specific classes the base dataset lacked. It produced three
negative results in a row, and those three results are the reason Stage IV is designed the way it
is.

**Seven auxiliary single-class datasets** were extracted and remapped into a common label space.
Each had been annotated independently with its target object at class index 0, so all required
rewriting so that ID 0 became the correct global class; filenames were given source-specific
prefixes to prevent collisions.

**Table 3.5 — Auxiliary dataset composition (annotation counts)**

| Source | → class ID | Train | Val | Test |
|---|---:|---:|---:|---:|
| `bicycle_extracted` | 2 (bicycle) | 1,582 | 446 | 230 |
| `stairs_extracted` | 7 (stairs) | 1,687 | 200 | — |
| `dogs_extracted` | 9 (dog) | 3,630 | 620 | 300 |
| `bollard_extracted` | 10 (bollard) | 3,051 | 879 | 434 |
| `crosswalk_extracted` | 11 (crosswalk) | 961 | 291 | 134 |
| `pothole_extracted` | 12 (pothole) | 3,277 | 377 | 399 |
| `scooter_extracted` | 13 (scooter) | 1,411 | 62 | 51 |

The resulting intermediate dataset held **8,176 train / 1,598 val / 784 test** images over a
14-class vocabulary. The best Stage II checkpoint was fine-tuned for 30 epochs at 640×640, batch
16, on an A100, with a deliberately low initial learning rate of **3×10⁻⁴** — the model now
comprising 3,008,378 parameters and 8.1 GFLOPs, 511 iterations per epoch, completing in
**0.421 hours**.

The results were, on their face, excellent: validation over 1,598 images and 2,875 instances gave
precision 0.870, recall 0.804, mAP@0.50 0.850 and mAP@0.50:0.95 0.600; the test split of 784 images
and 1,548 instances gave 0.862 / 0.818 / 0.853 / 0.618, in close agreement.

**Table 3.6 — Validation results after fine-tuning on the expanded dataset (Stage III)**

| Class | Images | Instances | P | R | mAP@0.50 | mAP@0.50:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| Bicycle | 394 | 446 | 0.969 | 0.980 | 0.988 | 0.770 |
| Stairs | 183 | 200 | 0.898 | 0.900 | 0.918 | 0.671 |
| Dog | 266 | 620 | 0.906 | 0.921 | 0.945 | 0.660 |
| Bollard | 367 | 879 | 0.882 | 0.974 | 0.947 | 0.726 |
| Crosswalk | 220 | 291 | 0.919 | 0.869 | 0.934 | 0.705 |
| Pothole | 124 | 377 | 0.767 | 0.515 | 0.647 | 0.328 |
| Scooter | 26 | 62 | 0.749 | 0.468 | 0.571 | 0.338 |

**Negative result one: catastrophic forgetting.** Only *seven* of the fourteen declared classes
appear in that table. The validator's instance-count array records **zero instances** for person,
car, motorcycle, bench, fire hydrant, traffic light and pole — because the expanded dataset was
assembled exclusively from the auxiliary single-class sources and contains none of the original
imagery. The consequence is twofold. First, mAP@0.50 of 0.850 is computed over seven comparatively
easy classes on a dataset averaging fewer than two objects per image, whereas Stage II's 0.458 was
computed over ten classes on scenes averaging more than twenty. The two numbers measure different
things and **their comparison is meaningless** — which is why every row of Table 4.1 names its own
evaluation set. Second, and far more seriously, thirty epochs of gradient descent on data in which
person, car, pole and traffic light never appear is a continuous training signal that those classes
are absent from every image; the classification head was actively driven to suppress them.

Diagnostic inference confirmed it. Running prediction over the expanded test set and tallying class
assignments yielded detections for **bollard (519), crosswalk (133), pothole (329) and scooter (51)
alone** — none of the original ten-class vocabulary appeared at all. The model had become an
excellent seven-class detector at the cost of the ten-class capability Stage II had spent 6.243
GPU-hours acquiring. This is catastrophic forgetting in its classical form, and it established the
requirement that governed the rest of the project: **vocabulary extension must be performed on data
containing all classes simultaneously, never by sequential fine-tuning on disjoint subsets.**

**Negative result two: a contaminated evaluation split.** Within the expanded vocabulary two
classes were clearly deficient — `pothole` at 0.647 mAP@0.50 with recall 0.515, and `scooter` at
0.571 with recall 0.468, the latter measured on just 62 validation instances. Acquiring more
annotated data was not available within the project's constraints, so duplication-based
oversampling was applied: every image whose label file contained class 12 or 13 was written three
times under distinct filenames, raising the training split from 8,176 to **11,384** images.
Fine-tuning ran 15 epochs from the previous checkpoint at batch 16.

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Precision | 0.862 | 0.857 | −0.005 |
| **Recall** | 0.818 | **0.843** | **+0.025** |
| mAP@0.50 | 0.853 | 0.866 | +0.013 |
| mAP@0.50:0.95 | 0.618 | 0.630 | +0.012 |
| Test images / instances | 784 / 1,548 | 1,082 / 2,448 | +298 / +900 |

The targeted classes moved as intended — `scooter` from mAP@0.50 0.718 to 0.774 and mAP@0.50:0.95
0.507 to 0.568 with recall 0.686 → 0.766; `pothole` from 0.616 to 0.619 with recall 0.476 → 0.526.
But **the duplication routine was applied uniformly across training, validation *and* test**, so
the evaluation sets were themselves enlarged — validation 1,598 → 1,898 and test 784 → 1,082 — with
duplicated copies of precisely the images containing the targeted classes. The "after" figures are
not measured on the "before" benchmark. The scooter gain is corroborated by a genuine per-instance
recall improvement and is probably real, but it is not cleanly measured, and the correct
methodology — oversample the training split alone, hold the evaluation splits fixed — was recorded
as a protocol defect to be corrected in the final stage. It is one of the specific errors behind the
"freeze the test set" discipline of §3.6.2.

Two smaller observations from the same run are worth recording because both were visible only in
the log. The run executed on a **Tesla T4 rather than the A100**, completing in 0.875 hours with
per-image inference rising from about 0.5 ms to 2.1 ms — a practical illustration of hardware
variability in shared cloud environments. And **the automatic optimizer mode overrode the requested
learning rate of 1×10⁻⁴ and selected its own**, so the intended reduction was never applied.

The wider lesson has a low ceiling written into it: repeating the same 62 scooter instances three
times supplies no new visual variation, it merely reweights the loss. Duplication moves a badly
under-represented class from unusable to marginal; it cannot substitute for collection.

**Negative result three: the silently failed merge.** The forgetting diagnosis pointed to an
unambiguous remedy — train on the union of the original ten-class dataset and the expanded set, so
all fourteen classes appear in every epoch. A merge routine copied both sources into a unified
directory under distinguishing prefixes. On execution it reported the three old splits copied
successfully and **all three new splits skipped, with the message that the folder was missing**: the
oversampled dataset had been produced in an earlier session and no longer existed in the ephemeral
working environment. The routine printed a notice and continued. Verification then reported a
"merged" dataset of 25,892 / 7,399 / 3,699 images — figures identical to the original.

A fifty-epoch run was therefore launched on a dataset believed to hold fourteen classes and in fact
holding ten. The configuration declared `nc: 14`, so the model was built with fourteen outputs, four
of which had no instances anywhere. Training proceeded from the previous checkpoint at 2×10⁻⁴ and
consumed **3.046 hours on the A100 without raising a single error**.

**Table 3.7 — Failed combined run versus the Stage II baseline (identical test split)**

| Metric | Stage II baseline | "Combined" run | Change |
|---|---:|---:|---:|
| Precision | 0.6692 | 0.6709 | +0.0017 |
| Recall | 0.4086 | 0.4041 | −0.0045 |
| mAP@0.50 | 0.4528 | 0.4372 | −0.0156 |
| mAP@0.50:0.95 | 0.2847 | 0.2750 | −0.0097 |

Three GPU-hours produced a model measurably *worse* than the baseline it started from, having in
the interim also lost the four new classes acquired in the preceding experiments. The regression is
consistent with a model fine-tuned away from the ten-class distribution and then partially
retrained back toward it, arriving at a compromise inferior to either endpoint. **This is not a
valid combined-dataset result and must not be read as one.**

**Four lessons, which between them specify Stage IV.** The failure was *silent, not loud* — every
component behaved as designed: the copy routine reported the missing directory, the trainer accepted
a configuration declaring more classes than the data contained, the validator reported metrics for
the classes present, and no exception was raised at any point. The defect was not in any component
but in the **absence of a verification gate between data preparation and training**. Second,
**declared class count must be validated against observed class content** — counting the distinct
class indices in the label files and comparing against the configuration is cheap, and disagreement
should be a hard error. Third, **temporary storage is a correctness hazard, not an inconvenience**;
any artefact a later step requires must be persisted and its presence asserted before use. Fourth,
sequential fine-tuning on disjoint class subsets is not a viable vocabulary-extension strategy at
all — the only reliable path to a seventeen-class detector was a single unified dataset trained in
one campaign.

### 3.3.5 Stage IV — YOLO26-small on seventeen classes

Stage IV is the delivered model, and it is the implementation of everything the first three stages
established: one unified dataset containing every class, audited before a single epoch was
permitted to run, trained in a single campaign under an explicitly serialised configuration with
per-epoch checkpoints on durable storage.

**The dataset** is the verified 17-class corpus of §3.2 — 74,096 train / 11,086 val / 5,957 test
= 91,139 images carrying 693,146 + 180,752 + 90,939 = **964,837 objects**, 2.9× the Stage II
imagery, distributed as a single 11.7 GB archive. Its per-class composition is Table 3.2 (§3.2.6).
The imbalance identified in Stage I persists and in absolute terms has intensified — `pole` now
accounts for 453,239 objects, roughly 47 % of all annotations, against `manhole`'s 722, a ratio
beyond 600:1. But the distribution is materially healthier at the low end: `scooter`, which had 62
validation instances in Stage III, now has 10,876 objects across 6,634 images, and `bollard` has
grown from 879 validation instances to 6,597 objects. **The classes oversampling had propped up
artificially are now genuinely represented.** `manhole`, appearing exactly once per image across
722 images, remained the clear weak point (§3.2.8).

**The audit gate.** Before any training was permitted, the dataset was subjected to the systematic
verification of §3.2.10 — the direct institutional response to the merge failure. Structural
discovery confirmed all six required directories and matching image/label counts in each split.
Five defect categories were then checked: empty annotation files, images lacking a label, labels
lacking an image, invalid annotations, and corrupted images. Annotation validation examined **every
line of every label file across all 91,139 files**, verifying exactly five fields, a class index
parsing to an integer inside the valid range, four normalised coordinates parsing as floats within
[0, 1], and no box of zero or negative width or height. Image integrity was verified by opening and
validating every file. **The audit returned 0 defects in every category**, followed by visual
inspection of twelve randomly sampled training images with boxes and labels rendered. This is the
substantive difference between Stage IV and everything preceding it: training began from *verified*
data integrity rather than assumed integrity.

**Table 3.8 — Stage IV training environment and configuration**

| Parameter | Value | Rationale |
|---|---|---|
| Base model | YOLO26-small, pretrained | End-to-end NMS-free head; small tier chosen over nano for capacity (§2.3) |
| Framework | Ultralytics 8.4.115 | Per-class metrics, checkpointing, early stopping, export |
| Hardware | A100-SXM4-40GB (40,441 MiB) | Enables batch 64; asserted at startup rather than falling back to CPU |
| Runtime | PyTorch 2.11.0 + CUDA 12.8, 12 CPU cores, 83.5 GB RAM | Colab |
| Image size | 640×640 letterbox | Identical to the server's preprocessing |
| Epochs / patience | 80 / 20 | Early stopping if validation stalls — never triggered |
| Batch size | 64 | Fourfold the Stage II baseline, made possible by A100 memory |
| Optimizer | auto → **MuSGD**, lr 0.01, momentum 0.9 | 114 unregularised / 126 weight-decayed / 126 bias groups |
| Mixed precision | Enabled, verification check passed | Throughput |
| Dataloader | 8 workers, caching disabled | Dataset too large to cache |
| Augmentation | Mosaic, closed over the final 10 epochs; HSV, flip, scale | Generalisation and implicit rare-class enrichment |
| Seed / determinism | **42**, deterministic execution | Reproducibility |
| Checkpointing | `save_period=1` to mounted Drive | Session volatility — see below |
| Config record | Serialised to JSON alongside the artefacts | Configuration is an experimental result |

The instantiated network comprised **260 layers, 9,961,022 parameters and 22.8 GFLOPs during
training, fusing at inference to 122 layers, 9,471,759 parameters and 20.8 GFLOPs**. All **708
items transferred** from the pretrained checkpoint. The detection head was configured for seventeen
classes over feature maps of 128, 256 and 512 channels, with **end-to-end operation enabled**.

One documentation discrepancy is recorded here for accuracy rather than buried: the run directory
and the configuration variable were both named for the *medium*-scale variant, while the model
actually trained — as confirmed by the architecture summary and parameter count in the training log
— was the **small**-scale variant. Every metric below corresponds to the small model.

A second difference from Stage II is visible in the training log itself. Where the YOLOv8 run
reported box, classification and **distribution focal** loss terms, the YOLO26 run reports box,
classification and **L1** loss, reflecting the revised regression formulation. The L1 term operates
on a numerically different scale — order 10⁻³ rather than order 1 — which makes cross-architecture
loss comparison meaningless and is a further reason to compare on metrics rather than losses.

**Checkpointing and resumption.** The single most important infrastructural change in this stage was
persistent checkpointing. Output went to mounted Google Drive rather than ephemeral session storage,
and the save period was set to one, writing a complete checkpoint after every epoch: eighty
checkpoints `epoch0.pt` through `epoch79.pt` plus `best.pt` and `last.pt`, each ≈20.3 MB after
optimizer stripping, ≈1.6 GB for the run. The notebook tests for `last.pt` in the run directory and,
if found, loads it and resumes; otherwise it initialises from pretrained weights. This makes the
training cell **idempotent across session boundaries** — an essential property, because Colab
sessions terminate on a schedule shorter than the total training time required. The mechanism was
exercised in practice: the executed session located the existing checkpoint and **resumed from epoch
61 to the 80-epoch target, completing the remaining twenty epochs in 2.501 hours**, the first sixty
having been completed in prior sessions, implying a total cost of roughly ten GPU-hours. Without
per-epoch persistence they would have been lost exactly as the oversampled dataset was lost in
Stage III. GPU memory during the resumed phase ran at ≈19.6 GB, peaking at 19.9 GB.

**Table 3.9 — YOLO26 validation metrics, the complete final window (epochs 61–80)**

| Epoch | box_loss | cls_loss | l1_loss | P | R | mAP@0.50 | mAP@0.50:0.95 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 61 | 1.700 | 1.255 | 0.00705 | 0.752 | 0.584 | 0.634 | 0.456 |
| 62 | 1.705 | 1.260 | 0.00703 | 0.752 | 0.587 | 0.635 | 0.457 |
| 63 | 1.704 | 1.256 | 0.00705 | 0.753 | 0.588 | 0.635 | 0.457 |
| 64 | 1.693 | 1.247 | 0.00700 | 0.749 | 0.589 | **0.636** | **0.458** |
| 65 | 1.689 | 1.251 | 0.00697 | 0.754 | 0.587 | **0.636** | **0.458** |
| 66 | 1.684 | 1.238 | 0.00690 | 0.754 | 0.587 | **0.636** | **0.458** |
| 67 | 1.677 | 1.227 | 0.00686 | 0.753 | 0.589 | **0.636** | **0.458** |
| 68 | 1.672 | 1.223 | 0.00681 | 0.757 | 0.586 | 0.635 | 0.458 |
| 69 | 1.671 | 1.218 | 0.00676 | 0.759 | 0.586 | 0.635 | 0.458 |
| 70 | 1.661 | 1.206 | 0.00672 | 0.752 | 0.589 | 0.635 | 0.457 |
| 71 | **1.711** | 1.187 | 0.00688 | 0.753 | 0.587 | 0.635 | 0.458 |
| 72 | 1.699 | 1.170 | 0.00673 | 0.756 | 0.587 | 0.634 | 0.458 |
| 73 | 1.688 | 1.158 | 0.00663 | 0.753 | 0.589 | 0.634 | 0.458 |
| 74 | 1.678 | 1.147 | 0.00654 | 0.756 | 0.587 | 0.633 | 0.458 |
| 75 | 1.673 | 1.138 | 0.00644 | 0.758 | 0.586 | 0.633 | 0.457 |
| 76 | 1.662 | 1.130 | 0.00636 | 0.758 | 0.586 | 0.633 | 0.457 |
| 77 | 1.652 | 1.117 | 0.00628 | 0.757 | 0.587 | 0.632 | 0.457 |
| 78 | 1.642 | 1.108 | 0.00621 | 0.759 | 0.585 | 0.631 | 0.456 |
| 79 | 1.639 | 1.101 | 0.00615 | 0.761 | 0.585 | 0.631 | 0.456 |
| 80 | 1.631 | 1.096 | 0.00609 | 0.763 | 0.584 | 0.631 | 0.456 |

Three things in that window matter more than the numbers themselves. First, **mAP@0.50 ranges over
only 0.005 across twenty epochs (0.631–0.636) and mAP@0.50:0.95 over only 0.002, while
classification loss falls steadily from 1.255 to 1.096** — the loss was still improving long after
the metrics had stopped. Optimising the objective and improving the capability had decoupled,
which is the practical case for selecting checkpoints on metrics rather than on loss. Second, the
single visible perturbation is **mosaic closure at epoch 71**, where box loss steps from 1.661 to
1.711 as the augmentation distribution changes; over the final ten epochs precision rose
0.753 → 0.763 while recall fell 0.587 → 0.584 and mAP@0.50 drifted 0.635 → 0.631, so closing the
augmentation traded a little recall for a little precision and cost slightly more of the headline
metric than it returned. Third, the strict metric plateaus at 0.458 from epoch 64 to epoch 74 and
then decays. **The best checkpoint was therefore selected from mid-window rather than from the
final epoch**, and early stopping never triggered — the run completed its full schedule. Stage IV
was the most stable run in the project by a wide margin.

> **Figure 3.5** — Sample augmented training batches showing mosaic composition, HSV jitter and
> random scale/flip with YOLO boxes overlaid, and the corresponding label-versus-prediction pair.
>
> `[[FIGURE: train_batch*.jpg and val_batch*_labels/pred.jpg from the Stage IV run]]`

**Best-checkpoint validation (11,086 images, 180,736 instances):** Precision **0.7542**, Recall
**0.5872**, mAP@0.50 **0.6356**, mAP@0.50:0.95 **0.4581**. (The instance count is 16 below the
180,752 validation boxes of §3.2 because Ultralytics removed 16 duplicate labels while scanning
the split — a discrepancy worth naming so it is not mistaken for an error.)

**Table 3.10 — YOLO26 per-class validation results (the delivered model)**

| Class | Images | Instances | P | R | mAP@0.50 | mAP@0.50:0.95 |
|---|---:|---:|---:|---:|---:|---:|
| Person | 5,083 | 24,474 | 0.706 | 0.495 | 0.553 | 0.340 |
| Car | 5,615 | 37,057 | 0.744 | 0.590 | 0.644 | 0.449 |
| Bicycle | 1,300 | 2,790 | 0.698 | 0.398 | 0.456 | 0.268 |
| Motorcycle | 1,129 | 2,459 | 0.726 | 0.488 | 0.555 | 0.342 |
| Bench | 665 | 1,201 | 0.749 | 0.402 | 0.463 | 0.328 |
| Fire hydrant | 924 | 1,018 | 0.819 | 0.597 | 0.655 | 0.489 |
| Traffic light | 2,547 | 16,924 | 0.685 | 0.414 | 0.445 | 0.237 |
| Stairs | 555 | 677 | 0.795 | 0.728 | 0.769 | 0.591 |
| **Pole** | 3,806 | **90,122** | 0.670 | **0.205** | **0.256** | **0.128** |
| Dog | 615 | 809 | 0.739 | 0.700 | 0.733 | 0.550 |
| **Curb** | 76 | 238 | 0.588 | **0.206** | **0.289** | **0.156** |
| **Crosswalk** | 169 | 190 | 0.900 | 0.937 | **0.957** | 0.774 |
| **Scooter** | 400 | 584 | 0.904 | 0.913 | 0.950 | **0.853** |
| **Bollard** | 198 | 769 | 0.812 | 0.821 | 0.864 | 0.631 |
| **Trash can** | 267 | 397 | 0.785 | 0.754 | 0.832 | 0.684 |
| Construction | 459 | 1,027 | 0.747 | 0.748 | 0.748 | 0.509 |
| `manhole` | 0 | 0 | — | — | — | **not measured** |

**On the missing row.** `manhole` does not appear because the validation split contains no manhole
instances at all (§3.2.8). The per-class summary printed by the notebook reports 0.4581 for
`manhole` — a value identical, character for character, to the overall mAP@0.50:0.95, because the
framework fills unevaluated classes with the global mean. It is a placeholder, not a measurement.
It is reported here as *not measured* precisely so that a number which looks like a result cannot be
mistaken for one.

The remaining sixteen classes fall into three clear regimes.

The **strongest group is the newly introduced navigation vocabulary**: `scooter` at mAP@0.50 0.950
with mAP@0.50:0.95 0.853, `crosswalk` at 0.957 and 0.774, `bollard` at 0.864 and 0.631, `trash_can`
at 0.832 and 0.684. These are exactly the classes the project set out to add, and they are detected
substantially more reliably than the general urban classes. Their high mAP@0.50:0.95 values show
the localisation is genuinely good rather than merely adequate at a loose IoU threshold. The
comparison with Stage III is the argument of this whole chapter in miniature: `scooter`, which
oversampling lifted only to 0.774 mAP@0.50 and 0.568 mAP@0.50:0.95 on a duplicate-inflated
benchmark, now reaches **0.950 and 0.853 on a clean split with genuine data**. Real examples
accomplish what duplication can only approximate.

The **middle group is the classic urban objects**: `fire_hydrant` 0.655, `car` 0.644, `motorcycle`
0.555, `person` 0.553. Each improved over Stage II, and `person` remains limited by recall (0.495),
reflecting how many pedestrians in urban imagery are heavily occluded or distant.

The **weakest group is unchanged in composition from Stage II** — `pole` at mAP@0.50 0.256 and
recall 0.205, `curb` at 0.289 and 0.206, `traffic_light` at 0.445 and 0.414. Poles are 90,122 of
180,736 validation instances, **half the entire validation object population, and still the hardest
class in the vocabulary**. Kerbs, at 238 validation instances, suffer both scarcity and intrinsic
difficulty: a kerb is a low-contrast linear boundary of genuinely ambiguous extent, and an
axis-aligned box is a poor representation for that geometry. These two classes are the primary
constraint on overall recall, and their persistence across *every* architecture tried in this
project establishes their difficulty as a property of the object geometry rather than of any model
— a 44 % increase in absolute pole annotations between Stage II and Stage IV moved pole mAP@0.50 by
0.034.

**Inference performance.** The standalone validation pass over the full split recorded 0.6 ms
preprocessing, 1.4 ms inference and 0.1 ms post-processing per image. A field-test pass over 100
randomly selected test images at confidence 0.25 — a qualitative inference run only, with no
metrics computed, so the test split's freeze (§3.6.2) is intact — recorded 1.8 / 1.1 / 0.2 ms,
about **3.1 ms per image end to end**, corresponding to roughly 900 FPS for the inference stage
alone on the A100. The
post-processing figure deserves emphasis: the YOLOv8 baseline required 1.0–1.2 ms per image,
comparable to or exceeding its own inference time, because NMS cost scales with the number of
candidate boxes and therefore with scene density. YOLO26 emits a fixed-size output tensor of shape
`(1, 300, 6)` with no suppression stage, so its post-processing cost is an order of magnitude lower
and, far more importantly for a safety device, **effectively constant regardless of how crowded the
scene is**. A detector whose latency is stable between an empty pavement and a busy intersection is
materially more suitable here than one whose worst case coincides with its most demanding scenario.
The field test also confirmed behaviour across the vocabulary: detections spanned all seventeen
classes with plausible scene composition — dense scenes returning combinations such as thirteen
cars, seventeen traffic lights and seven poles in a single frame, alongside isolated images
returning one construction marker or one trash can. Two of the hundred images returned no detections
at 0.25.

**Export.** The best checkpoint was exported to ONNX at **opset 20 with onnxslim optimisation,
producing a 36.4 MB artefact from the 19.4 MB PyTorch checkpoint in 5.3 seconds**, accepting input
`(1, 3, 640, 640)` and producing output `(1, 300, 6)` — three hundred candidate detections, each
four box coordinates, a confidence and a class index, with no post-processing required. This
fixed-shape, framework-independent artefact is directly deployable to ONNX Runtime, TensorRT and
mobile inference engines, and it is what makes the edge-inference path of §6.3 a configuration
change rather than a re-engineering effort. The complete artefact set was persisted to Drive: the
ONNX export, `best.pt` and `last.pt`, all eighty per-epoch checkpoints, the serialised training
configuration, a metrics summary in JSON, and the generated diagnostic plots — confusion matrices
raw and normalised, precision, recall, F1 and precision–recall curves, and paired
label-versus-prediction batch visualisations.

### 3.3.6 What the four stages established

Stage II and Stage IV are the only two full-vocabulary models evaluated over their complete class
sets under equivalent protocols, so they are the only pair worth comparing directly. Precision
improved from 0.670 to 0.754, a relative gain of **12.5 %**; recall from 0.411 to 0.587, a relative
gain of **42.8 %**; mAP@0.50 from 0.458 to 0.636; and mAP@0.50:0.95 from 0.289 to 0.458, a relative
gain of **58.5 %**. Both directions of movement are ordinarily in tension — adding seven classes
usually depresses per-class performance through inter-class confusion, and improving recall usually
costs precision — so achieving both simultaneously indicates a genuine capability gain rather than
a repositioning along an existing trade-off curve. That mAP@0.50:0.95 improved *more* than
mAP@0.50 is itself informative: because it averages over IoU thresholds up to 0.95 it is far more
sensitive to localisation precision, so the model is not merely finding more objects but placing
tighter boxes around them. The full stage-by-stage comparison, with each row's evaluation set named,
is Table 4.1 (§4.3).

**Table 3.11 — The ten shared classes: Stage II versus Stage IV validation mAP**

| Class | II mAP@0.50 | IV mAP@0.50 | II mAP@0.50:0.95 | IV mAP@0.50:0.95 |
|---|---:|---:|---:|---:|
| Person | 0.513 | 0.553 | 0.305 | 0.340 |
| Car | 0.617 | 0.644 | 0.420 | 0.449 |
| Bicycle | 0.401 | 0.456 | 0.225 | 0.268 |
| Motorcycle | 0.504 | 0.555 | 0.294 | 0.342 |
| **Bench** | 0.207 | **0.463** | 0.148 | 0.328 |
| Fire hydrant | 0.516 | 0.655 | 0.379 | 0.489 |
| Traffic light | 0.389 | 0.445 | 0.198 | 0.237 |
| **Stairs** | 0.581 | **0.769** | 0.374 | 0.591 |
| **Pole** | 0.222 | **0.256** | 0.107 | 0.128 |
| Dog | 0.630 | 0.733 | 0.436 | 0.550 |

The validation sets differ in size and composition, so these figures indicate direction and
magnitude rather than a controlled ablation. **Every shared class improved.** The largest gains
were `bench`, which more than doubled from 0.207 to 0.463, and `stairs`, 0.581 → 0.769 — precisely
the two classes whose training representation grew most in the unified dataset, from 2,152 to
10,703 annotations and from 836 to 5,295 respectively. The smallest gain was `pole`, 0.034, despite
its already being the most abundant class. The pattern is consistent across the table: **classes
limited by data availability improved markedly; classes limited by object geometry improved
marginally.** That single sentence is the most useful predictive statement the chapter produces.

Five lessons emerged with enough force to be the chapter's durable contribution.

**A training pipeline must be able to detect its own failure.** Metrics that measure the objective
being optimised are not a substitute for metrics that measure the capability being sought. The
Stage I collapse was invisible precisely because only the former were recorded.

**Data verification must precede training and must be a hard gate.** The combined-dataset run
consumed three GPU-hours and produced a regression solely because a missing directory was *reported*
rather than *raised*, and a declared class count was accepted without being checked against the
classes present.

**Class imbalance is a first-order design constraint that scales with the dataset rather than being
solved by it.** Poles were the dominant class and the worst-performing class in every stage of this
project; increasing their absolute count by 44 % moved their mAP@0.50 by 0.034.

**Vocabulary extension by sequential fine-tuning on disjoint class subsets causes catastrophic
forgetting.** The only reliable method is a unified dataset in which every class is present in
every epoch.

**Infrastructure is not separable from research.** Temporary storage corrupted an experiment, and
the eighty-epoch run that produced the delivered model existed only because per-epoch checkpoints
were written to durable storage behind an automatic resume path.

The delivered detector meets its functional requirements, and its limitations are characterised
rather than merely suspected. Overall recall of 0.587 means a substantial proportion of annotated
objects is still missed, and the deficit is concentrated in poles and kerbs, whose difficulty has
been shown to be a property of thin, low-texture, ambiguously bounded geometry rather than of any
architecture or of data scarcity. The `manhole` class, at 722 objects with no validation
representation, remains unmeasured. That these constraints can be stated precisely, with the
evidence that establishes them, is the direct product of the evaluation discipline adopted after the
first stage failed silently — and it is, in the end, the most consequential outcome of the process.
Full parameters for all four stages are in Appendix C.
