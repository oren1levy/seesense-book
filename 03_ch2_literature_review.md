# Chapter 2 — Literature Review

## 2.1 Assistive Technology for Blind Navigation

Surveys of assistive technology for visually-impaired users consistently identify independent
outdoor mobility as the highest-impact and least well-solved need [37], [38]. The barrier is not
route planning — GPS solves that — but *local perception*: knowing what physical objects are in
the immediate path and which of them constitute a hazard.

Electronic travel aids fall into three families. **Range-sensor devices** (ultrasonic, infrared,
laser) are cheap, low-power and low-latency, but semantically blind: they report a distance,
never an object identity, so they cannot tell the user whether the thing two metres ahead is a
wall, a parked scooter, or a person about to move. **GPS and mapping applications** operate at a
spatial resolution three orders of magnitude coarser than the hazards that cause injury.
**Camera-based recognition applications**, including scene-description services and
remote-sighted-assistant services, provide rich semantic output but typically operate on demand,
on a single photograph, with response times measured in seconds — a mode suited to "what is
written on this sign?" rather than "is anything about to hit me?"

SeeSense targets the gap between the second and third families: continuous, semantic, real-time
analysis of a live video stream, delivered through the non-visual channels a blind user can
consume while walking.

Designing for a user who cannot see the screen imposes constraints that shaped almost every
decision in Chapter 3:

- **The interface is audio and haptic; the screen is secondary.** A visual HUD is useful for a
  sighted companion and indispensable during development, but cannot be the primary channel.
- **False negatives are worse than false positives — but only up to a point.** A missed obstacle
  is the more dangerous error, arguing for high recall. But an audio channel is serial and
  low-bandwidth, and it competes with the environmental hearing the user navigates by. A system
  that speaks constantly masks that hearing and trains the user to ignore it. **Alert fatigue is
  a safety failure too**, with the same end state as a missed detection.
- **Every state change must be announced**, and setup must not require sight: large tap targets,
  a single-tap SOS with no gesture to learn, screen-reader-compatible markup.

## 2.2 Object Detection as a Real-Time Task

Object detection is the joint task of classification ("what?") and localisation ("where?"),
producing for each object a class, a confidence and a bounding box. It is strictly harder than
classification because the number of objects is unknown, objects occlude one another, and the
same object may occupy 5 or 500 pixels in the same image.

**Two-stage detectors** such as Faster R-CNN [14] first generate region proposals and then
classify and refine each; they are accurate but run a second network over many proposals per
image. **Single-stage detectors** predict boxes and classes directly from the feature map in one
pass — substantially faster, and with modern architectures no longer meaningfully less accurate.
For a live stream on a budget of tens of milliseconds per frame, single-stage is the only
realistic choice.

**Metrics.** A prediction is a true positive if its Intersection over Union (IoU) with an
unmatched ground-truth box of the same class exceeds a threshold. **Precision** is the fraction
of claims that were real; **Recall** the fraction of real objects found; **AP** the area under
the precision–recall curve for one class; and **mAP** the mean of AP over classes. **mAP@0.50**
uses a single lenient IoU threshold; **mAP@0.50:0.95** averages over ten thresholds from 0.50 to
0.95 and is therefore sensitive to localisation quality, not merely classification. COCO [5]
popularised the latter, and it is our model-selection metric.

Two properties mattered practically. mAP is a **mean over classes**, so a collapsed rare class
can be hidden by strong common ones — per-class AP must be inspected separately, which is what
revealed our weak classes. And a class with **no instances in validation or test contributes
nothing at all**, the situation we found with `manhole` (§3.2.8).

"Real time" here is not a frame rate but a closed-loop deadline: the time from a hazard becoming
visible to the user being warned, and the distance travelled meanwhile. This demands **bounded
per-frame latency** rather than high average throughput, **bounded queueing** (a naive
send-every-frame client builds an unbounded backlog the moment the server slows, and the user
starts hearing about a scene from five seconds ago), and **graceful degradation** with the user
told when quality drops.

## 2.3 The YOLO Family

YOLO [1] reframed detection as a single regression problem: the image is divided into a grid,
and each cell predicts boxes with objectness scores and class probabilities in one forward pass.
The loss is multi-component — localisation, objectness and classification — and balancing those
terms against the enormous imbalance between occupied and empty cells is the central difficulty
of implementing such a head, as we found when we implemented one ourselves (§3.3.2).

Successive versions refined the formula [2], [3], [44]: **multi-scale prediction** through a
feature pyramid, dramatically improving small-object detection — directly relevant to bollards,
kerbs and manhole covers; **anchor-free prediction** in v8 and later, removing a hyperparameter
set that had to be retuned per dataset; **decoupled heads** for classification and regression;
**task-aligned dynamic label assignment** replacing fixed IoU matching; **built-in augmentation**
including mosaic [3], HSV jitter, random scale and flip; and **Distribution Focal Loss with CIoU**
for box regression, replacing naive coordinate regression.

The Ultralytics implementations of YOLOv8 and YOLO11 [4], [47] package this behind one training
and validation API with mixed precision, checkpointing, early stopping, per-class metrics and
confusion matrices. Two properties made the family correct for SeeSense. The **nano variant is
genuinely small** — approximately 3.0 M parameters, 8.2 GFLOPs, a ~6 MB weights file — which is
what makes CPU inference in a cloud container feasible within tens of milliseconds and an
eventual on-device deployment plausible. And **YOLOv8 and YOLO11 share an identical label
format**, so migrating between them required changing exactly one string, with no dataset
conversion — which mattered because our dataset was already built and verified when we upgraded.
The trade-off is capacity: a nano model separating 17 classes, several of them visually similar
street furniture, has limited headroom (§4.8).

**Transfer learning versus training from scratch.** Our POC document originally specified
training from scratch for full control over class definitions. In practice we used transfer
learning from COCO-pretrained weights, because the low-level features a detector needs are
domain-general and relearning them consumes epochs producing no class-specific benefit, and
because the supervisor's requirement was that *we* train the detector on *our* classes and data
and diagnose the process — which initialising a backbone from public weights does not circumvent.
We also built and trained a detector genuinely from scratch (§3.3.2) and report its results, so
both objectives are satisfied.

## 2.4 Multi-Object Tracking and Motion Analysis

A detector is memoryless: it has no notion that the car in frame 100 is the car from frame 99.
For SeeSense this is disqualifying, because the central question — *is this object approaching
me?* — is about a trajectory, not a frame.

In **tracking-by-detection**, each frame's detections are associated with existing tracks.
**SORT** [10] uses a Kalman filter plus the Hungarian algorithm [12] on an IoU cost matrix;
**DeepSORT** [11] adds a learned appearance embedding. **ByteTrack** [9] made a simple
observation with large consequences: conventional trackers discard low-confidence detections as
noise, but an object that becomes briefly occluded or blurred produces exactly a low-confidence
detection — so discarding them kills the tracks of precisely the objects hardest to follow.
ByteTrack associates in two stages, first high-confidence detections to tracks, then the
remaining low-confidence ones to still-unmatched tracks. Continuity improves substantially at
essentially no computational cost and with no appearance model.

SeeSense implements a ByteTrack-inspired tracker (§3.4.3) with two design points arising from the
application rather than the literature. **Windowed rather than frame-to-frame motion**, because
boxes jitter by a few pixels every frame even on a static object, so consecutive-frame comparison
makes the "approaching" flag flicker. And **growth in box area as a proximity proxy**: because
the user is themselves walking, an object stationary in the world still grows in frame as the
user approaches it — which is the correct interpretation for a collision warning, and comes free
from the same measurement.

## 2.5 Estimating Distance from a Single Camera

The original characterisation specified **monocular depth estimation** — a per-pixel depth map
from a single RGB image — using models in the MiDaS [16], Monodepth2 [17] or PackNet [18]
families. We did not use them, for three reasons worth stating because the decision is a real
trade-off rather than an omission. **Cost**: depth estimation is a second dense-prediction network
per frame, unaffordable on a budget where the detector alone costs ~41 ms. **Scale ambiguity**:
monocular depth is ambiguous up to scale, and metric distance requires camera intrinsics plus a
ground-plane assumption — every assumption another failure mode on a real pavement. And **the
decision we need is coarse**: the alert logic needs Close / Medium / Far and "growing or
shrinking", both obtainable from the box we already have.

SeeSense therefore uses **bounding-box area ratio** as a proximity proxy with per-profile
thresholds (§3.4.4). Its weaknesses are well known — it conflates a large distant object with a
small near one, and is sensitive to occlusion and truncation at the frame edge — but it is free,
robust, and sufficient for a three-way decision. Crucially its *derivative over time*, which is
what the danger logic actually keys on, is far more reliable than its absolute value. Metric
depth is listed as future work (§6.3).

## 2.6 Class Imbalance in Detection Datasets

Detection datasets are imbalanced along two axes. **Foreground–background imbalance** is
inherent: most candidate locations contain no object. Focal Loss [13] addresses this by
down-weighting easily-classified negatives; modern YOLO training incorporates equivalent
mechanisms. **Inter-class imbalance** is a property of the data as collected: in our final
dataset `pole` accounts for 453,239 boxes and `manhole` for 722, a ratio of 628:1, and a model
minimising an unweighted loss can reduce it more by improving on poles than by learning manhole
covers at all.

Four families of mitigation exist, and we used or evaluated all four. **Data-level collection**
is the only technique adding genuine information, and is what we did most of — `crosswalk` went
from 33 images to 3,511. **Resampling** (§3.3.5) improved recall by +0.025 and mAP@0.50:0.95 by
+0.012, limited by the fact that duplicating an image adds frequency but no visual diversity.
**Loss reweighting** through class weights or focal-style losses raises rare-class gradient
contribution. **Synthetic augmentation** — standard mosaic/flip/scale/HSV, and more powerfully
**copy-paste augmentation** [19], which pastes rare-class instances into other scenes — is the
most promising untried technique for our remaining imbalance.

The literature is also clear about a trap we fell into and documented: resampling must be applied
to the **training split only**. Our first oversampling experiment duplicated validation and test
images too, changing the evaluation distribution and making the comparison less rigorous than it
appeared (§3.3.5).

## 2.7 Deployment Architectures and Real-Time Transport

**Edge (on-device) inference** has decisive advantages in principle: zero network latency, no
connectivity dependency, no per-frame bandwidth, and complete privacy — camera frames never leave
the device, which for a system continuously filming public space is a substantive ethical
property. Our characterisation specified a CoreML model quantised to 8-bit (15–30 MB) on the
Apple Neural Engine, estimating 80–150 ms inference and 100–180 ms end-to-end. The costs are
equally real: constrained compute forces a smaller model and lower resolution, and continuous
camera plus NPU use creates thermal and battery load — our estimate was 3–5 hours of continuous
operation, which is why a hard start/stop control is a functional requirement. The reference
project we were given, RoadXpert, took this route with TensorFlow.js in the browser, reporting
5–10 FPS on mid-range Android and 15–20 FPS on desktops.

**Cloud inference** removes the compute ceiling and centralises model updates, at the cost of
round-trip latency, bandwidth, a hard connectivity dependency and the privacy implications of
transmitting a continuous camera feed. **Hybrid with failover** was our POC target: run against
the server while the network is good, monitor RTT and bandwidth, and on degradation (latency
> 300 ms, bandwidth < 1 Mbps, or packet loss) fail over to a compressed on-device model with a
"safety ping" SMS carrying the last GPS position to an emergency contact.

SeeSense as delivered implements the **cloud** architecture with the *monitoring* half of the
hybrid fully built — the watchdog measures RTT continuously, warns in Hebrew at 100 ms and
150 ms, and stops scanning after three consecutive pings above 200 ms — but with the on-device
failover model deferred. This is stated plainly rather than blurred, because the difference
between designing and shipping a hybrid system is exactly the kind of claim a project book should
not soften.

**Transport.** UDP, used by conventional video streaming, was rejected explicitly in our POC
document: a corrupted JPEG header does not degrade an image, it makes it undecodable and crashes
inference, so integrity was prioritised over raw throughput and **TCP** mandated. **HTTP POST per
frame** was the initial implementation — simple and stateless, but paying request and response
header overhead on every frame. **WebSocket** [26] upgrades one HTTP connection to a persistent
bidirectional channel, after which each frame costs a few bytes of framing; migrating to it was
our first and largest transport optimisation. A related issue is **flow control**: a client
sending unconditionally builds an unbounded queue whenever the server is slower than the capture
rate. Bounded in-flight depth gives

```
per-frame latency ≈ network_RTT + depth × server_processing_time
throughput        ≈ min(1 / server_time, depth / network_RTT)
```

Depth 1 wastes the network; large depth adds queueing delay to every frame. §3.5.2 documents our
operating point and §4.4 the measured curve.

## 2.8 Related Systems and the Gaps SeeSense Addresses

Mature **commercial applications** address adjacent problems — scene description, text reading,
and remote human assistance by video call. Both are complementary rather than competitive: they
operate on demand with human-scale response times. **Research prototypes** [37], [38] frequently
combine a camera with depth sensors, an embedded board and a custom haptic vest, reporting strong
results on custom hardware the user must acquire, wear and maintain — reintroducing the cost and
adoption barrier that makes guide dogs inaccessible. **RoadXpert** shares our detector family and
web-delivery philosophy but targets a sighted rider at vehicle speed with a screen in front of
them, leading to fundamentally different alert semantics.

The gaps SeeSense addresses: **continuous rather than on-demand** analysis at ~22 FPS; a **class
taxonomy chosen for a blind pedestrian**, including `curb`, `bollard`, `manhole`, `construction`
and `crosswalk` — objects irrelevant in a driving dataset and absent from general benchmarks but
exactly what causes falls; **motion-aware alerting rather than presence-based alerting**, since a
parked car is not an event but a car whose box has grown 25 % in four frames is; **commodity
hardware only**, a browser URL with no wearable and no installation; **a complete product**
rather than a demo; and **measured, reported performance** on the actual deployment target,
including the experiments that failed.
