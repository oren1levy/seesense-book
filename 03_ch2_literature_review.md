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
nothing at all**: our final validation table reports 16 of the 17 classes, `manhole` being absent
because all 722 of its images landed in train (§3.2.8). Worse, the framework leaves no visible
gap — for a class it never evaluated, Ultralytics' per-class array returns the *overall* mAP, so
`manhole` prints as 0.4581, exactly the model's global score. A number that looks like a result
and is in fact a placeholder is a failure mode worth naming.

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
for box regression, replacing naive coordinate regression. YOLO26 [48] revises that last point
again, reporting an **L1** regression term in place of DFL; the term operates on a numerically
different scale — order 10⁻³ rather than order 1 — so training losses are not comparable across the
two generations, which is one reason every comparison in this book is made on metrics rather than
on losses.

The Ultralytics implementations [4], [47] — YOLOv8 and YOLO11 during development, YOLO26 in the
delivered system, all inside one package — put this behind a single training and validation API
with mixed precision, checkpointing, early stopping, per-class metrics and confusion matrices.
Two properties made the family correct for SeeSense. **The tiers are small, and the tier is a
one-token choice**: within YOLO26 the pretrained nano checkpoint is 5.3 MB and the small
checkpoint 19.5 MB, and moving between them means editing a filename. We train the **small**
tier, which fuses to 122 layers, 9,471,759 parameters and 20.8 GFLOPs and ships as a 19.4 MB
weights file — light enough for CPU inference in a cloud container within tens of milliseconds,
while leaving the nano tier's capacity ceiling behind. And **the label format is identical across
these generations**: our `data.yaml` still carries the header comment `17-class dataset
(YOLO11-ready)` and was handed to YOLO26 training unmodified, so the upgrade required no dataset
conversion — which mattered because the dataset was already built and verified when we changed
detector. The trade-off is that capacity is not the only ceiling: even at ~9.5 M parameters,
classes that are thin, repetitive and routinely truncated at the frame edge stay hard, which is
why `pole` reaches mAP@0.50:0.95 0.128 while `scooter` reaches 0.853 (§4.3.1).

**The NMS-free head is the property that matters most for a safety device.** YOLO26 [48] replaces
the conventional dense head followed by non-maximum suppression with an end-to-end formulation that
emits a fixed-size output tensor of shape `(1, 300, 6)` — three hundred candidate detections, each
carrying four box coordinates, a confidence and a class index — with no suppression stage at all.
The consequence is not merely that post-processing is cheaper, though it is: 0.1–0.2 ms per image
against the 1.0–1.2 ms the YOLOv8 baseline spent, a cost that had been comparable to that model's
own inference time. The consequence is that post-processing cost becomes **constant**. Classical
NMS scales with the number of surviving candidate boxes and therefore with scene density, so its
worst case coincides precisely with the crowded intersection where a blind pedestrian most needs a
timely warning. A detector whose latency is the same on an empty pavement and in dense traffic is
materially better suited to this application than one whose latency is merely lower on average. The
fixed output shape also makes the ONNX export a directly deployable artefact and removes the NMS
IoU threshold from the set of deployment parameters that have to be tuned (§3.4).

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
application rather than the literature. **Trend rather than frame-to-frame motion**: boxes jitter
by a few pixels every frame even on a static object, so consecutive-frame comparison makes the
"approaching" flag flicker. The tracker instead fits a least-squares line through apparent object
size over a 0.8-second window and asks both how much the fitted size grew and how cleanly — the
growth divided by the residual scatter — so uncorrelated jitter cannot tilt the verdict while a
slow, steady approach can. The flag is latched with hysteresis and must hold for 0.3 s before it
fires, and every threshold is expressed as a duration rather than a frame count, so the behaviour
is unchanged when the frame rate drops. And **growth in apparent size as a proximity proxy**:
because the user is themselves walking, an object stationary in the world still grows in frame as
the user approaches it — the correct interpretation for a collision warning, with the useful
property that relative growth per second equals the inverse of time-to-contact, so "fast" can be
defined as "will reach the user within three seconds" regardless of the object's size or distance.

## 2.5 Estimating Distance from a Single Camera

The original characterisation specified **monocular depth estimation** — a per-pixel depth map
from a single RGB image — using models in the MiDaS [16], Monodepth2 [17] or PackNet [18]
families. We did not use them, for three reasons worth stating because the decision is a real
trade-off rather than an omission. **Cost**: depth estimation is a second dense-prediction network
per frame, unaffordable on a budget where the detector alone costs tens of milliseconds even at
its best. **Scale ambiguity**:
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
from 33 images to 3,511. **Resampling** (§3.3.4) improved recall by +0.025 and mAP@0.50:0.95 by
+0.012, limited by the fact that duplicating an image adds frequency but no visual diversity.
**Loss reweighting** through class weights or focal-style losses raises rare-class gradient
contribution. **Synthetic augmentation** — standard mosaic/flip/scale/HSV, and more powerfully
**copy-paste augmentation** [19], which pastes rare-class instances into other scenes — is the
most promising untried technique for our remaining imbalance.

The literature is also clear about a trap we fell into and documented: resampling must be applied
to the **training split only**. Our first oversampling experiment duplicated validation and test
images too, changing the evaluation distribution and making the comparison less rigorous than it
appeared (§3.3.4).

## 2.7 Deployment Architectures and Real-Time Transport

**Edge (on-device) inference** has decisive advantages in principle: zero network latency, no
connectivity dependency, no per-frame bandwidth, and complete privacy — camera frames never leave
the device, which for a system continuously filming public space is a substantive ethical
property. Our characterisation specified a CoreML model quantised to 8-bit (15–30 MB) on the
Apple Neural Engine, estimating 80–150 ms inference and 100–180 ms end-to-end. The costs are
equally real: constrained compute forces a smaller model and lower resolution, and continuous
camera plus NPU use creates thermal and battery load — our estimate was 3–5 hours of continuous
operation, which is why a hard start/stop control is a functional requirement. Browser-based
on-device detection is feasible today through TensorFlow.js or WebGPU, but published figures for
that route sit in the range of single-digit to low-twenties frames per second on mid-range mobile
hardware, an order of magnitude below what an unconstrained server delivers.

**Cloud inference** removes the compute ceiling and centralises model updates, at the cost of
round-trip latency, bandwidth, a hard connectivity dependency and the privacy implications of
transmitting a continuous camera feed. **Hybrid with failover** was our POC target: run against
the server while the network is good, monitor RTT and bandwidth, and on degradation (latency
> 300 ms, bandwidth < 1 Mbps, or packet loss) fail over to a compressed on-device model with a
"safety ping" SMS carrying the last GPS position to an emergency contact.

SeeSense as delivered implements the **cloud** architecture with the *monitoring* half of the
hybrid built — the watchdog measures RTT continuously, warns in Hebrew at two degradation levels,
and flags the connection red after three consecutive slow pings — but with the on-device failover
model deferred, and the automatic stop on a red connection not currently wired to the flag
(§3.5.7). This is stated plainly rather than blurred, because the difference between designing
and shipping a hybrid system is exactly the kind of claim a project book should not soften.

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

Depth 1 wastes the network; large depth adds queueing delay to every frame. §3.5.1 documents our
operating point and §4.4 the measured curve.

## 2.8 Related Systems and the Gaps SeeSense Addresses

Mature **commercial applications** address adjacent problems — scene description, text reading,
and remote human assistance by video call. Both are complementary rather than competitive: they
operate on demand with human-scale response times. **Research prototypes** [37], [38] frequently
combine a camera with depth sensors, an embedded board and a custom haptic vest, reporting strong
results on custom hardware the user must acquire, wear and maintain — reintroducing the cost and
adoption barrier that makes guide dogs inaccessible. **Driver- and rider-assistance systems**
share much of the same stack — a YOLO-family detector, a web or embedded delivery path — but
target a sighted user at vehicle speed with a screen in front of them, which changes the alert
semantics fundamentally: a visual overlay is the primary channel, alerts can be triggered by a
fixed region of interest rather than by tracked motion, and the vocabulary is traffic furniture
rather than pavement hazards.

The gaps SeeSense addresses: **continuous rather than on-demand** analysis at tens of frames per
second; a **class taxonomy chosen for a blind pedestrian**, including `curb`, `bollard`,
`manhole`, `construction` and `crosswalk` — objects irrelevant in a driving dataset and absent
from general benchmarks but exactly what causes falls; **motion-aware alerting rather than
presence-based alerting**, since a parked car is not an event, but a car whose apparent size has
been growing steadily for most of a second is;
**commodity hardware only**, a browser URL with no wearable and no installation; **a complete
product** rather than a demo; and **measured, reported performance** on the actual deployment target,
including the experiments that failed.
