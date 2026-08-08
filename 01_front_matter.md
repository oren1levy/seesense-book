# SeeSense

## Real-Time Environment Recognition for Blind and Visually-Impaired Pedestrians

**Oren Levy**

**Liad Lati**

**Omer Helfer**

**Shir Yahav**

Approved by the supervisor: Dr. Moshe Butman

Submitted to the Computer Science Faculty of the College of Management
Rishon LeZion

August 2026

Source code: <https://github.com/OmerHelfer/SeeSense> [45]

---

# Acknowledgments

We would like to express our deepest gratitude to our supervisor, Dr. Moshe Butman, for
continuous guidance and insightful feedback throughout the project. Special thanks to the
Computer Science faculty and the SeeSense team members for their dedication and collaboration.

---

# Executive Summary

**SeeSense** is a real-time assistive navigation system for blind and visually-impaired
pedestrians. A standard smartphone, held upright, streams camera frames to a server that runs a
custom-trained object detector, tracks each detected object across frames, decides whether the
user is actually in danger, and returns a verdict fast enough for the phone to speak or vibrate a
warning while the obstacle is still ahead of the user.

The system is a full product, not only a model. The client is a mobile-first, right-to-left
Hebrew single-page application in React 19: it opens the rear camera, uses the device-orientation
sensor to check that the phone is pitched within fifteen degrees of vertical, streams JPEG frames
over a WebSocket while capping how many frames may be in flight at once, and turns the server's
verdict into Hebrew text-to-speech and vibration patterns — speech being the only channel on iOS,
which does not implement the Vibration API. The server is a Python FastAPI application performing
YOLO inference, ByteTrack-inspired tracking with Hungarian assignment, motion-first danger
classification, and the whole application backend — accounts, JWT auth, two administrative tiers
above the ordinary user, per-user settings and per-user high-risk classes, detection history, a
feedback and ticketing system, verified emergency contacts, an SOS flow, and a persistent
performance-metrics subsystem. No camera frame is ever stored: the history record holds only
derived scalars — class label, confidence, a coarse Close/Medium/Far band and the danger level.
The application runs in two parallel deployments against a managed MongoDB database: a Docker
image on Railway as the CPU-only baseline, and a GPU-backed Google Cloud VM on which the same
FastAPI process also serves the built client — one origin, one port, one certificate.

The deep-learning work went through four stages, and the negative results were as instructive as
the positive ones. A detector written from scratch — frozen ResNet18 backbone, hand-implemented
YOLO-style head, grid encoder, multi-component loss, decoder and NMS — trained successfully but
stopped generalising after four epochs and produced unreliable, pole-biased detections on dense
street scenes. A YOLOv8-nano pipeline on the same ten-class dataset gave Precision 0.669, Recall
0.409, mAP@0.50 0.4528 and mAP@0.50:0.95 0.2847 on a 3,699-image test set with 86,381 instances —
usable, but recall far too low for a safety application. Fine-tuning on an expanded 14-class
dataset reached 0.862 / 0.818 / 0.853 / 0.618 on the new-class split, and targeted oversampling of
the two weakest classes raised recall to 0.843. An attempted merge of the old and new datasets
failed silently because Colab's ephemeral storage had lost the intermediate folders, producing a
misleading "combined" run that trained on the old data only — the failure that motivated
rebuilding the entire data pipeline.

That rebuild produced the final dataset: **17 classes, 91,139 verified images** (train 74,096 /
val 11,086 / test 5,957) and 964,837 boxes, merged from ten sources — MS COCO, Open Images V7 and
Mapillary Vistas plus nine Roboflow datasets remapped into one unified taxonomy. Building it
surfaced the hardest bug of the project: several sources were annotated as YOLO **segmentation
polygons** rather than boxes, and the merge parser silently skipped every line without exactly
five fields. Converting polygons to their tight enclosing boxes recovered 6,199 images that had
been vanishing without a single error message. The detector now in production is a **YOLO26-small
model (about 10 M parameters)** fine-tuned from pretrained weights on all seventeen classes for 80
epochs at 640×640, batch 64, seed 42, with mosaic augmentation disabled for the final ten epochs.
It validates at Precision 0.754, Recall 0.587, mAP@0.50 0.636 and mAP@0.50:0.95 0.458 — measured
over the full seventeen-class taxonomy, a considerably harder benchmark than any earlier stage.
The alert vocabulary the server currently speaks is a thirteen-class subset of that taxonomy: the
network also predicts kerbs, manholes, bins and construction, but those detections are filtered
out before they reach the user.

On the systems side the project produced a measured optimisation log rather than a guess: HTTP
POST per frame replaced by a WebSocket stream; a 71 ms per-frame settings read replaced by an
in-memory cache costing 0 ms; per-frame database writes consolidated into a once-a-second batch
writer off the hot path; inference moved to a worker thread behind a single global lock so the
event loop kept answering health pings; and, on the CPU-only container, a reduction of the
detection input size from 640 to 512 pixels that brought server-side inference to roughly 41 ms
per frame. Two experiments failed usefully: an ONNX Runtime port that benchmarked 1.5× faster
locally ran **75× slower** on the deployment target because the container advertises the host's
core count and the runtime oversubscribed it — and the thread cap written to fix that later
crippled the GPU deployment in turn and had to be reverted, teaching the same lesson twice. The
delivered deployment runs the detector on a cloud GPU at the full 640-pixel input, at roughly
16 ms of server time per frame, sustaining about **50 FPS at ~120–130 ms end-to-end** against an
original success criterion of 200–300 ms. Alert quality needed the same discipline as latency:
motion timings were re-expressed in seconds rather than frame counts, approach detection became a
least-squares trend test with hysteresis and a confirmation streak, and alerts now fire only when
an object's danger level escalates — so a parked car no longer speaks on every frame.

SeeSense demonstrates that a phone, a commodity cloud container and a self-trained detector
suffice to deliver useful real-time obstacle warnings — and that in a
system of this kind the difficult engineering is not the neural network but everything around it:
label taxonomies that disagree, silent data loss, class imbalance no collection can remove, alert
fatigue, and the gap between a benchmark image and a real pavement.

---

# Table of Contents

**1. Introduction** — Background · Problem Statement · Objectives · Scope and Limitations ·
Methodology · Organization

**2. Literature Review** — Assistive Navigation Technology · Real-Time Object Detection · The
YOLO Family · Multi-Object Tracking · Distance from a Single Camera · Class Imbalance ·
Deployment Architectures and Transport · Related Systems and Gaps

**3. System Design and Implementation** — Architecture · Data Collection, Preparation and
Verification · Model Development and Training · Server Implementation · Client Implementation ·
Deployment and Evaluation Methodology

**4. Results and Analysis** — Experimental Setup · Dataset Results · Model Accuracy Across Stages
· Runtime and Latency · Real-Time Analysis · Qualitative Results · Comparison · Discussion

**5. Engineering Challenges and Lessons Learned** — Data · Training and Evaluation · Systems ·
User Experience · Process

**6. Conclusion and Future Work**

**7. References**

**Appendices** — A. Dataset Card · B. Configuration and API Summary · C. Reproducibility
Parameters · D. Project Timeline and Team Roles

---

# List of Abbreviations

| | | | |
|---|---|---|---|
| **AP** Average Precision | **FIFO** First In, First Out | **JPEG** Joint Photographic Experts Group | **POC** Proof of Concept |
| **API** Application Programming Interface | **FPS** Frames Per Second | **JSON** JavaScript Object Notation | **RTL** Right-to-Left |
| **CIoU** Complete Intersection over Union | **GPS** Global Positioning System | **JWT** JSON Web Token | **RTT** Round-Trip Time |
| **CNN** Convolutional Neural Network | **GPU** Graphics Processing Unit | **LR** Learning Rate | **TTL** Time To Live |
| **COCO** Common Objects in Context | **HSV** Hue, Saturation, Value | **mAP** mean Average Precision | **TTS** Text-To-Speech |
| **CORS** Cross-Origin Resource Sharing | **HTTP** Hypertext Transfer Protocol | **NMS** Non-Maximum Suppression | **WS** WebSocket |
| **CPU** Central Processing Unit | **HUD** Heads-Up Display | **NPU** Neural Processing Unit | **YAML** YAML Ain't Markup Language |
| **DFL** Distribution Focal Loss | **IoU** Intersection over Union | **ONNX** Open Neural Network Exchange | **YOLO** You Only Look Once |

---

# List of Figures

| Figure | Caption | § |
|---|---|---|
| 3.1 | SeeSense architecture and the life of one frame | 3.1 |
| 3.2 | Composition of the final 17-class dataset, and per-class counts | 3.2 |
| 3.3 | Polygon-to-bounding-box conversion for segmentation-annotated sources | 3.2 |
| 3.4 | Custom ResNet18 detector: training versus validation loss | 3.3 |
| 3.5 | Sample augmented training batches (mosaic, HSV jitter, scale, flip) | 3.3 |
| 3.6 | Motion-first alert classification decision tree | 3.4 |
| 3.7 | Dashboard during an active scan: detections, tracking badge, spirit level, SOS | 3.5 |
| 3.8 | Settings: menu, sensitivity, alert channel and intensity controls | 3.5 |
| 4.1 | Training curves for the final YOLO26 run | 4.3 |
| 4.2 | Normalized confusion matrix on the 17-class validation split | 4.3 |
| 4.3 | Per-class precision–recall and F1 curves | 4.3 |
| 4.4 | Latency breakdown and throughput versus pipeline depth | 4.4 |
| 4.5 | Qualitative detections on real street photographs, and failure cases | 4.6 |

---
