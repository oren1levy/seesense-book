## 3.2 Data Collection, Preparation and Verification

Every other component of SeeSense began from something that already existed — a detector
architecture, a web framework, a database. The dataset did not: no public corpus is organised
around the objects that injure a pedestrian who cannot see them. This section documents how one
was assembled from ten sources that disagreed with each other about almost everything, and how it
was verified.

### 3.2.1 Defining the class taxonomy

Which objects the system should detect is not a decision any existing dataset makes for you.
General benchmarks such as COCO [5] are built around objects that appear in photographs people
take; driving datasets such as Cityscapes [8] and Mapillary Vistas [7] around what matters from a
car. Neither is built around what injures a pedestrian who cannot see.

We derived the taxonomy from the hazard: what causes falls, what causes collisions, and what a
blind pedestrian needs to locate deliberately. It grew from 4 classes (person, car, stairs, pole)
to 10, then to the final 17.

| ID | Class | Hazard rationale |
|---|---|---|
| 0 | `person` | Collision risk; the most common dynamic object in any street scene |
| 1 | `car` | Highest-consequence hazard; consolidates truck and bus |
| 2 | `bicycle` | Fast, quiet, frequently on pavements |
| 3 | `motorcycle` | Fast, often parked across pavements |
| 4 | `bench` | Waist-height fixed obstacle the cane may miss |
| 5 | `fire_hydrant` | Low fixed obstacle, trip hazard |
| 6 | `traffic_light` | Both an obstacle and a navigation landmark |
| 7 | `stairs` | Highest-consequence fall hazard |
| 8 | `pole` | Most numerous fixed obstacle in an urban street |
| 9 | `dog` | Unpredictable dynamic obstacle |
| 10 | `curb` | Most common cause of trips at pavement transitions |
| 11 | `crosswalk` | A navigation *target*, not an obstacle — the only "go here" class |
| 12 | `scooter` | Electric scooters parked across pavements are a recent, severe hazard |
| 13 | `bollard` | Waist-height, narrow, hard, easy for a cane to pass beside |
| 14 | `trash_can` | Frequently moved and rarely where it was yesterday |
| 15 | `manhole` | An open or raised cover is a fall hazard |
| 16 | `construction` | Barriers, cones, scaffolding — one "not walkable" signal |

An intermediate **14-class** configuration also existed during development — the ten base classes
plus `bollard`, `crosswalk`, `pothole` and `scooter` — and is what Stage III was trained
on (§3.3.4). It was superseded by the 17-class taxonomy, which dropped `pothole` and added `curb`,
`trash_can`, `manhole` and `construction`. The deployed system runs the 17-class model.

Two consolidation decisions are worth recording. `truck` and `bus` were mapped to `car`: from the
perspective of a pedestrian's safety decision all three are "large fast metal object", and
splitting them would divide training signal for no behavioural benefit. And `construction`
deliberately collapses barricades, barriers, cones, machinery and scaffolding into one class,
because the required user response — do not proceed this way — is identical for all of them.

### 3.2.2 The base ten-class dataset

The base dataset was built in a dedicated Colab notebook from three academic sources, fully
scripted so the build is reproducible from a clean environment.

**MS COCO 2017** [5] supplied eight classes. Rather than downloading the full 25 GB image set, the
pipeline downloads only the annotation JSON (~252 MB), filters for the target category IDs,
samples a per-class quota of image IDs, and downloads *only those images* by URL. COCO boxes are
absolute `[x_min, y_min, w, h]`; conversion to YOLO's normalised centre form must be clamped,
because a small number of COCO boxes extend marginally outside the image.

**Open Images V7** [6] supplied `stairs` and `pole`, neither of which exists as a COCO category,
and supplemented `traffic_light`. Open Images identifies classes by machine identifiers rather
than names, so the pipeline resolves each target by searching the class-description CSV for
synonyms — `pole` matches utility pole, telegraph pole and lamp post — then streams and filters
the bounding-box CSV. Open Images stores boxes as already-normalised `XMin, XMax, YMin, YMax`,
requiring a *different* conversion from COCO's — exactly the kind of detail that silently corrupts
a dataset if assumed rather than checked.

**Mapillary Vistas** [7] supplied street-level scenes from a *pedestrian viewpoint*, which was the
specific requirement the supervisor set in the first meeting: training data must resemble what the
camera will actually see, not what a car-mounted camera sees. **Cityscapes** [8] was integrated as
an optional source but contributed little.

All sources were then unified through four steps: **letterbox resizing to 640×640**, preserving
aspect ratio and padding with grey (114) — chosen so that training-time and inference-time
preprocessing are the same operation, with box coordinates transformed by the same scale and
offset; **stratified 70/20/10 splitting** on each image's primary class with a fixed seed of 42;
**`data.yaml` emission**; and **quality validation** — per-class counts per split, box size and
aspect-ratio distributions, and visual inspection of sampled images with boxes drawn, because a
numeric check cannot tell you your coordinate conversion has flipped the y-axis, and a picture
can.

The result was **36,990 images** (train 25,892 / val 7,399 / test 3,699) with a distribution that
already showed the imbalance that would dominate the rest of the project: `pole` 314,120
annotations, `car` 124,449, `person` 78,872, `traffic_light` 56,978, down to `bench` 2,152, `dog`
2,136 and `stairs` 836 — a 375:1 ratio, before we had added a single new class.

### 3.2.3 Expanding to seventeen classes

The seven new classes exist in no single academic dataset, so they were assembled from **Roboflow
Universe** [35]. Nine sources were selected, evaluated and merged with the base set.

**Table 3.1 — The ten data sources**

| Source | Type | Role |
|---|---|---|
| `seesense_data/processed` | COCO + Open Images + Mapillary | Classes 0–9 |
| OOD (`ood-kllke`) | Detection | person, car, curb, dog, bollard, trash_can |
| Obstacle Finder | Detection | scooter, manhole, bollard, curb |
| Outside Objects | Detection | crosswalk, curb, trash_can, stairs |
| University-Outdoor | Detection | stairs, trash_can |
| Construction Site Annotations V2 | Detection / segmentation | construction |
| Benches (v11) | Detection | bench (gap-fill) |
| cross (`cross-vgnt6`, v11) | Detection | crosswalk (gap-fill) |
| scooter (v11) | Detection | scooter (gap-fill) |
| stairs (v3, v11) | **Segmentation** | stairs (gap-fill) |

Two candidates were **excluded deliberately**: the raw Mapillary archive, already baked into the
base set, whose re-addition would have duplicated tens of thousands of images and leaked
near-duplicates across the train/val/test boundary; and `accesibility street`, whose labels
describe accessibility *conditions* rather than physical objects and are incompatible with a
detection schema.

### 3.2.4 Struggle: ten incompatible taxonomies

Every contributed dataset defines its own class names and numeric ordering. The same object
appears as `bike` and `bicycle`; `Trashcan`, `Trash Can` and `waste_container`;
`spherical_roadblock` and `warning_column`, both of which are bollards on inspection. Case
matters. Several sources contain classes we actively do not want — `bus`, `tree`, `sidewalk`,
`cane`, `stop_sign`.

No automatic method is safe: fuzzy matching would merge `pole` with `pothole`. The solution was an
**explicit per-source mapping dictionary** in which every source class name maps either to a
unified class or to `None` (drop) — for instance the construction source collapses `Barricade`,
`Barrier`, `Cone`, `Heavy`, `Machine`, `Scaffolding` and `Scaffolding Pole` into `construction`
while dropping `Truck`. Critically, **any class not present in the map is reported as UNMAPPED and
dropped, with a count**. The builder never guesses. A silent default would inject mislabelled
boxes with no trace, and mislabelled boxes are worse than missing ones because the model actively
learns from them.

### 3.2.5 Struggle: sources with meaningless class names

The four gap-fill datasets declared class names such as `['19']` (scooter) and
`['0', '1', 'object']` (cross). Name-based mapping is impossible, so these were mapped by
**numeric class ID** after counting the actual ID distribution in the label files. For `cross` the
distribution was `{1: 3863, 0: 313, 2: 8}`; since the whole dataset is about crosswalks and ID 2
accounts for 8 boxes of 4,184, we mapped **IDs 0 and 1 → `crosswalk`** and dropped ID 2.

The ID-0 decision is an **assumption**, flagged rather than buried: those 313 boxes were assumed
to be crosswalks from the dataset's overall subject, not from visual verification. Worst case it
introduces ~7 % label noise in `crosswalk`, depressing that class's precision without being
catastrophic. It is listed as an open item in §6.2 and Appendix A.

### 3.2.6 Struggle: severe class imbalance

After the first 17-class merge several classes were not merely thin but unlearnable: `car` 32,307
images, `person` 28,790, `pole` 20,303 (with **453,239 boxes**) against **`crosswalk` at 33
images**, `manhole` at 722 with one box each, `scooter` 1,838 and `stairs` 2,311. A class with 33
images cannot be learned by any method.

The response was two-level. **At the data level**: dedicated gap-fill datasets, which took
`crosswalk` from 33 to 3,511 images and reinforced `scooter`, `stairs` and `bench`; **multi-source
merging** so each minority class draws on several independent datasets and varies in camera,
geography, lighting and annotation style; an explicit **~3,000-image floor** used as the decision
rule for "does this class still need collection?"; and **recovery of silently-dropped images**
(§3.2.7), which added 6,199. **At the training level**, for the imbalance collection cannot
remove — there genuinely are more poles than manhole covers, and every photograph containing a
manhole also contains poles — YOLO's built-in augmentation, availability of class weighting and
focal-style loss, and monitoring **per-class AP rather than only the global mean** so a collapsing
class is visible instead of averaged away.

**Table 3.2 — Per-class counts, final 17-class dataset (verified from the built dataset)**

| # | Class | Images | Boxes | # | Class | Images | Boxes |
|---|---|---:|---:|---|---|---:|---:|
| 0 | person | 28,829 | 128,464 | 9 | dog | 2,981 | 3,809 |
| 1 | car | 32,307 | 192,524 | 10 | curb | 2,258 | 4,118 |
| 2 | bicycle | 8,071 | 15,647 | 11 | crosswalk | 3,511 | 4,223 |
| 3 | motorcycle | 7,905 | 16,679 | 12 | scooter | 6,634 | 10,876 |
| 4 | bench | 5,921 | 10,703 | 13 | bollard | 2,699 | 6,597 |
| 5 | fire_hydrant | 4,752 | 5,286 | 14 | trash_can | 2,126 | 3,006 |
| 6 | traffic_light | 12,733 | 82,892 | 15 | manhole | 722 | 722 |
| 7 | stairs | 4,260 | 5,295 | 16 | construction | 9,490 | 20,757 |
| 8 | pole | 20,303 | 453,239 | | **Total** | **91,139 unique** | **964,837** |

The "Images" column counts images containing at least one instance of that class; it sums to more
than 91,139 because images are multi-label — a street photograph routinely contains people, cars,
poles and a crosswalk.

> **Figure 3.2** — Composition of the final dataset by source, and per-class image and box counts
> on a log scale showing the residual 628:1 ratio between `pole` and `manhole`.
>
> `[[FIGURE: two-panel — stacked bar of source contributions, horizontal log-scale bar of Table 3.2]]`

### 3.2.7 Struggle: segmentation labels silently dropped — the hardest bug

This produced no error message of any kind.

**Symptom.** After the merge the dataset contained 84,940 images, but the `stairs` gap-fill set of
~1,950 images had contributed exactly **one**. Not zero — which would suggest a path or permission
problem — but one, which looks like a partially-working integration.

**Diagnosis.** We opened an actual label file rather than trusting the source's `data.yaml`. A
YOLO *detection* label is five numbers per line — `class x_center y_center width height`. The
stairs labels looked like `0 0.193 0.517 0.191 0.529 0.199 0.544 …`, dozens of numbers: **YOLO
segmentation (polygon) format**, one vertex pair per point around the outline. The merge parser
contained `if len(p) != 5: continue`, so every polygon line was silently skipped, and images whose
annotations were *all* polygons ended up with zero valid boxes and were never copied.

**Fix.** Detect the format by counting coordinates and, for polygons, compute the tight enclosing
box:

```python
coords = p[1:]
if len(coords) == 4:                                  # detection: x y w h
    cx, cy, w, h = map(float, coords)
elif len(coords) >= 6 and len(coords) % 2 == 0:       # segmentation polygon
    xs = [float(v) for v in coords[0::2]]
    ys = [float(v) for v in coords[1::2]]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    w,  h  = xmax - xmin,       ymax - ymin
if w > 0 and h > 0:
    new_lines.append(f"{uid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
```

> **Figure 3.3** — Polygon-to-bounding-box conversion: a segmentation outline and its tight
> enclosing box.
>
> `[[FIGURE: one stairs image with polygon overlaid and the derived box]]`

**Payoff.** Re-running the resumable builder recovered not only the 1,949 stairs images but also
polygon-annotated images in **Construction (+3,185)** and **bench (+846)**, disappearing for the
same reason, with the balance spread thinly across the other polygon-bearing sources. The dataset
grew from **84,940 to 91,139** — 6,199 recovered samples, 6.8 % of the final dataset, that had
been vanishing without a single warning.

**Lesson.** "Image count equals label count" is a necessary but wholly insufficient check: a
silent format mismatch drops images and their labels *together*, so every consistency check still
passes. The only reliable defence is to **open a raw label file from every new source by hand**
before integrating it — two minutes per source, which would have saved several days here.

### 3.2.8 Struggle: a val/test split with no manhole examples

Computing the per-split class distribution — rather than assuming the split had behaved — revealed
that `manhole`, at 722 images with one box each, had **zero examples in both validation and
test**. All 722 landed in train.

Training runs fine; *measurement* fails. There is no validation or test AP for `manhole`, so we
cannot say how well the model detects them, and the class contributes nothing to model selection.
A class invisible to the evaluation is an untested feature shipped into a safety system. The
documented mitigation is to move a stratified handful into val and test and collect ~2,000 more
before relying on the class. The general rule adopted afterwards: when adding data later,
distribute it ~70/20/10 rather than appending it all to train, or validation and test metrics stop
being meaningful for exactly the classes you just added.

### 3.2.9 Struggle: disk space and duplication

The raw archives totalled ~35 GB, dominated by a 27 GB Mapillary archive already baked into the
base set, and naive copying would have duplicated the entire base set. Three measures kept the
build feasible: **hardlinks** (`os.link()`) for the 36,990 base images, so they cost approximately
zero additional disk while appearing as normal files; **copying only the nine Roboflow sources**,
whose labels are rewritten; and **scratch cleanup** afterwards, reclaiming ~45 GB. The builder is
also **resumable**, skipping any output file already written — which mattered enormously during
the polygon fix, because re-running only had to process the previously-dropped files rather than
rebuilding 85,000 images.

### 3.2.10 Verification protocol

This protocol exists because of the Colab merge failure of §3.3.4, in which a training run
proceeded happily on data that was not what it claimed to be.

| Check | Result |
|---|---|
| Size | 91,139 images — train 74,096 / val 11,086 / test 5,957 (≈ 81/12/7) |
| Image↔label pairing | **0 orphans** in all three splits |
| Label format | All 91,139 checked: 5-column, coordinates in `[0,1]`, no zero-area boxes — including every polygon-converted label |
| Class IDs | All within `0–16` |
| Images | 0 zero-byte files; valid JPEG/PNG headers across all ten source types |
| `data.yaml` | `nc: 17`, relative paths → portable and self-contained |

### 3.2.11 Known data limitations

Recorded here because a dataset card listing only strengths is not a dataset card. **`manhole` is
under-represented and unmeasured** (722 images, no val/test presence). **Augmentation inflation**:
the four gap-fill sources are Roboflow-augmented at ~3×, so 6,634 `scooter` images are not 6,634
independent scenes. **The `crosswalk` ID-0 mapping is an assumption.** **Structural imbalance
remains** and requires loss reweighting or copy-paste augmentation, not more downloading. And
**domain shift**: the dataset is largely well-lit, well-composed photography, whereas real frames
from a phone held by a walking blind user contain motion blur, extreme angles, glare and partial
occlusion at a rate no benchmark reproduces — which is why the qualitative evaluation on
self-captured street photographs (§4.6) is the actual generalisation test, not decoration.
