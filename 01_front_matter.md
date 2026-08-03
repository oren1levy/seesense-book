# SeeSense

## Real-Time Environment Recognition for Blind and Visually-Impaired Pedestrians

**Oren Levy**

**Liad Lati**

**Omer Helfer**

**Shir Yahav**

Approved by the supervisor: Dr. Moshe Butman

Project approver: Dr. Raz Lin

Specialization: Deep Learning · Project #503

Submitted to the Computer Science Faculty of the College of Management
Rishon LeZion

August 2026

Source code: <https://github.com/oren1levy/SeeSense>

---

# Acknowledgments

We would like to express our deepest gratitude to our supervisor, Dr. Moshe Butman, for his
continuous guidance and honest, demanding feedback throughout the year. From the very first
meeting he insisted that we train the detector ourselves rather than reuse a model somebody else
had already trained, and that we demonstrate one complete working pipeline before chasing
accuracy. Both instructions were uncomfortable at the time and both turned out to be the reason
this project produced something we actually understand.

We also thank Dr. Raz Lin and the Computer Science faculty of the College of Management for the
framework, the review meetings, and the academic standard they held us to; and our families and
friends for their patience during the long weeks of dataset building, failed training runs, and
evenings spent walking around Rishon LeZion pointing a phone at kerbs and manhole covers.

---

# Executive Summary

**SeeSense** is a real-time assistive navigation system for blind and visually-impaired
pedestrians. A standard smartphone, held upright, streams camera frames to a server that runs a
custom-trained object detector, tracks each detected object across frames, decides whether the
user is actually in danger, and returns a verdict fast enough for the phone to speak or vibrate a
warning while the obstacle is still ahead of the user.

The system is a full product, not only a model. The client is a mobile-first, right-to-left
Hebrew Progressive Web App in React 19: it opens the rear camera, verifies with the gyroscope
that the phone is held usably, streams JPEG frames over a WebSocket with bounded-depth
backpressure, and converts the server's verdict into Hebrew text-to-speech and haptic patterns.
The server is a Python FastAPI application performing YOLO inference, ByteTrack-style tracking,
motion-first danger classification, and the whole application backend — accounts, JWT auth, a
three-level admin system, per-user settings, detection history, a feedback and ticketing system,
verified emergency contacts, an SOS flow, and a persistent performance-metrics subsystem. It is
deployed as a Docker image on Railway against MongoDB Atlas.

The deep-learning work went through five stages, and the negative results were as instructive as
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
been vanishing without a single error message. The final model is a YOLO11-nano detector trained
by transfer learning for 80 epochs at 640×640, batch 16, run three times with different seeds and
selected by validation mAP@50-95.

On the systems side the project produced a measured optimisation log rather than a guess: HTTP
POST per frame replaced by a WebSocket stream; a 71 ms per-frame settings read replaced by an
in-memory cache costing 0 ms; the database write moved off the hot path; inference moved to a
worker thread so the event loop kept answering health pings; and reducing the detection input
size from 640 to 512 — the single biggest lever — bringing server-side latency to roughly **41 ms
per frame**. Two experiments failed usefully: an ONNX Runtime port that benchmarked 1.5× faster
locally ran **75× slower** on the deployment target because of container thread oversubscription.
With a measured ~131 ms network round trip and bounded in-flight depth, the pipeline sustains
roughly **22 FPS at ~216 ms end-to-end**, against an original success criterion of 200–300 ms.

SeeSense demonstrates that a phone, a commodity cloud container and a self-trained nano detector
suffice to deliver useful real-time obstacle warnings — and that in a system of this kind the
difficult engineering is not the neural network but everything around it: label taxonomies that
disagree, silent data loss, class imbalance no collection can remove, alert fatigue, and the gap
between a benchmark image and a real pavement.

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
| **AP** Average Precision | **E2E** End-to-End | **NMS** Non-Maximum Suppression | **RTT** Round-Trip Time |
| **API** Application Programming Interface | **FPS** Frames Per Second | **NPU** Neural Processing Unit | **SPA** Single-Page Application |
| **CNN** Convolutional Neural Network | **HUD** Heads-Up Display | **OID** Open Images Dataset | **TTL** Time To Live |
| **COCO** Common Objects in Context | **IoU** Intersection over Union | **PWA** Progressive Web App | **TTS** Text-To-Speech |
| **CORS** Cross-Origin Resource Sharing | **JWT** JSON Web Token | **REST** Representational State Transfer | **WS** WebSocket |
| **DFL** Distribution Focal Loss | **mAP** mean Average Precision | **RTL** Right-to-Left | **YOLO** You Only Look Once |

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
| 3.7 | Dashboard HUD: brackets, spirit level, detection overlay, alert overlay | 3.5 |
| 4.1 | Training curves for the final YOLO11 run | 4.3 |
| 4.2 | Normalized confusion matrix on the 17-class test set | 4.3 |
| 4.3 | Per-class precision–recall and F1 curves | 4.3 |
| 4.4 | Latency breakdown and throughput versus pipeline depth | 4.4 |
| 4.5 | Qualitative detections on real street photographs, and failure cases | 4.6 |

---
