# Chapter 1 — Introduction

## 1.1 Background

The World Health Organization estimates that at least 2.2 billion people live with a vision
impairment, roughly 43 million of them blind [36]. For this population the activity that most
directly determines independence is walking safely from one place to another: independent
mobility governs whether a person can work, study, shop or leave the house alone.

The available tools have barely changed in structure for decades. The **white cane** is cheap,
reliable and universally understood, but it is fundamentally a contact sensor reporting what lies
within about a metre of the ground directly ahead. It cannot report a car approaching from the
left, a bicycle at 20 km/h, a scaffolding pole at head height, or an open manhole two metres
away. The **guide dog** solves far more of the problem — a trained dog performs genuine route
judgement and intelligent disobedience — but costs tens of thousands of shekels, requires ongoing
care and long joint training, and creates a dependency that becomes acute when the dog is ill,
retires or dies. Between these poles sit **electronic travel aids**: ultrasonic and infrared
detectors that measure distance but never object identity, and GPS applications that guide at
street granularity while remaining blind to the immediate physical environment.

Meanwhile three technologies matured almost simultaneously. Single-stage detectors of the YOLO
family made it possible to localise and classify multiple objects in one forward pass, fast
enough for video. Smartphones acquired cameras, gyroscopes, GPS, haptics and speech synthesis in
one device that visually-impaired users already own and know how to operate. And commodity cloud
infrastructure made a fast container reachable from anywhere over a mobile network for a few
dollars a month. The gap between what a blind pedestrian can perceive and what a phone camera
plus a neural network can perceive became, for the first time, an engineering problem rather than
a research problem.

## 1.2 Problem Statement

Blind and visually-impaired pedestrians lack any continuous, semantic, real-time description of
the obstacles around them. Existing aids fail in one of three ways: **limited range with no
semantics** (a cane cannot distinguish a kerb from a bollard from a parked scooter — a
distinction that determines whether the correct action is to step up, step around, or stop);
**cost, maintenance and dependency** (guide dogs); or **no real-time analysis** (camera and GPS
applications describe a static photograph on demand or route between addresses, but never analyse
a live stream and warn about a hazard that is actively approaching).

A practical system must satisfy several requirements that actively conflict. It must be
**accurate** enough to be trusted, **fast** enough that a warning arrives while there is time to
react, **quiet** enough that it does not fire on every frame and become impossible to live with,
**cheap** enough to run on hardware the user already owns, and **accessible** enough to be
operated by somebody who cannot see the screen at all. Optimising one in isolation breaks the
others: a larger model is more accurate and too slow; a lower confidence threshold raises recall
and floods the user with false alarms; a per-frame alert is maximally responsive and maximally
unbearable. The problem SeeSense addresses is designing a complete pipeline that resolves these
conflicts to a working operating point, and measuring honestly where that point lies.

## 1.3 Objectives

1. **Build and train an object detector ourselves**, on a dataset we assemble ourselves, rather
   than integrating a pre-trained third-party detection product — an explicit and repeated
   instruction from the supervisor.
2. **Assemble a unified, verified obstacle dataset** covering the classes that matter to a blind
   pedestrian, not the classes an academic benchmark happens to contain.
3. **Demonstrate one complete end-to-end pipeline** — camera → preprocessing → network →
   inference → decision → feedback — working on a real phone before optimising accuracy.
4. **Convert raw detections into safety decisions**: estimate proximity, determine whether an
   object is approaching, decide whether the situation is dangerous, and decide whether this is
   *news* to the user.
5. **Deliver feedback through non-visual channels** — Hebrew speech and haptic patterns —
   configurable per user.
6. **Measure and optimise end-to-end latency and throughput** with real numbers on the real
   deployment target.
7. **Ship a complete product**, including accounts, settings, history, a feedback channel,
   emergency contacts and SOS — because an assistive system that cannot be configured or trusted
   is not an assistive system.

## 1.4 Scope and Limitations

**In scope:** a full client → server → model → logic → feedback pipeline over a live camera
stream at a fixed square resolution; a YOLO-family detector trained by us on our own dataset; a
FastAPI backend; an accessible mobile-first web client with haptic and audio feedback;
gyroscope-based alignment gating; multi-object tracking and motion analysis; and the surrounding
application (auth, settings, history, feedback, emergency contacts, SOS, administration).

**Explicitly out of scope:** on-device (edge) inference — designed and specified but not
implemented in the timeframe; metric monocular depth estimation, replaced by a bounding-box-area
heuristic classifying proximity as Close / Medium / Far (§2.5); hard real-time guarantees below
40 ms; personalisation through learning; indoor navigation; and formal clinical or user-study
validation with blind participants.

**Known limitations** are collected in §4.8, Chapter 5 and §6.2. The most significant are the
residual class imbalance (`pole` alone accounts for 453,239 of the dataset's 964,837 boxes), the
`manhole` class having no validation or test examples at all, the absence of any offline
capability, and degraded behaviour under motion blur and low light.

## 1.5 Methodology

The project ran from December 2025 to August 2026 with a supervisor review roughly every six
weeks. The methodology was deliberately **breadth-first**: present one working end-to-end
pipeline regardless of model quality, and only then optimise. In retrospect this was correct,
because almost every serious problem we hit was an integration problem rather than a modelling
problem. Work proceeded in six phases: characterisation and dataset search (Dec–Jan); POC
definition and infrastructure (Feb); first detection experiments and the CNN→YOLO pivot (Feb–Mar);
dataset expansion from ~37,000 to 91,139 images and from 10 to 17 classes (Apr–Jul); product
engineering and the measured optimisation campaign (Apr–Jul); and final training, evaluation and
documentation (Jul–Aug).

Two disciplines imposed by the supervisor cost us time and improved the result. First, **train it
yourself** — no reuse of an already-trained third-party detector. Second, **freeze the test set**:
once we were satisfied with the test split and began evaluating on it, the algorithm
configuration was frozen and no further changes were made on the basis of test results.

## 1.6 Organization of the Project Book

**Chapter 2** surveys assistive navigation technology, object detection and the YOLO family,
tracking, monocular distance estimation, class imbalance, deployment architectures and transport,
and positions SeeSense against existing systems. **Chapter 3** is the technical core: the
architecture, the complete data pipeline, the five stages of model development, the server and
client implementations, deployment and evaluation methodology. **Chapter 4** presents results —
dataset, accuracy, latency, real-time analysis, qualitative behaviour and comparison.
**Chapter 5** documents the engineering problems we actually hit, with root causes and fixes.
**Chapter 6** concludes and states future work. **Chapter 7** lists references, and the
**Appendices** contain the dataset card, configuration and API summary, reproducibility
parameters, and the project timeline and division of work.
