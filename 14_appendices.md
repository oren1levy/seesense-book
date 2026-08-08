# Appendices

## Appendix A — Final Dataset Card

**Name:** `Data-25-07` · **Built and verified:** 25 July 2026 · **Format:** YOLO detection
(`class x_center y_center width height`, normalised) · **Compatible with:** YOLOv8, YOLO11 and
YOLO26 (identical label format, no conversion required)

| Property | Value | | Property | Value |
|---|---|---|---|---|
| Classes | 17 | | Train / val / test | 74,096 / 11,086 / 5,957 |
| Unique images | 91,139 | | Image size | 640×640 letterbox, grey 114 padding |
| Bounding boxes | 964,837 | | Sources | 10 (academic base + 9 Roboflow) |

Per-class counts are in Table 3.2 (§3.2.6). Verification results — 0 orphans, 0 invalid labels,
0 out-of-range class IDs, 0 corrupt images — are in §3.2.10.

```yaml
path: .
train: images/train
val:   images/val
test:  images/test
nc: 17
names:
  0: person        1: car           2: bicycle      3: motorcycle
  4: bench         5: fire_hydrant  6: traffic_light  7: stairs
  8: pole          9: dog          10: curb        11: crosswalk
  12: scooter     13: bollard      14: trash_can   15: manhole
  16: construction
```

**Known issues.** (1) **`manhole` has no validation or test examples** — all 722 images are in
train, so the class receives no AP and contributes nothing to model selection. (2) Five classes
remain below the 3,000-image floor: manhole (722), trash_can (2,126), curb (2,258), bollard
(2,699), dog (2,981). (3) **Augmentation inflation** — the four v11 gap-fill sources are
Roboflow-augmented at ~3×, so raw counts overstate scene diversity. (4) **Unverified mapping** —
in the `cross` source, label IDs 0 (313 boxes) and 1 (3,863 boxes) were both mapped to `crosswalk`;
ID 0 should be spot-checked visually. (5) **Structural imbalance** — `pole` alone holds 453,239
boxes, 47 % of all annotations, which requires loss reweighting or copy-paste augmentation rather
than more collection. (6) **Polygon-derived boxes** — the stairs set and parts of the construction
and bench sets were annotated as polygons and converted to tight enclosing boxes, which for a
non-convex object such as a staircase is slightly larger than a hand-drawn detection box.

**Reproduction.**

```bash
python build_final_25_07.py          # resumable: re-running skips completed files
yolo detect train data=Final-Data-25-07/data.yaml model=yolo26s.pt imgsz=640 epochs=80 batch=64
```

**Manifest:** `data.yaml` (17-class config) · `build_final_25_07.py` (resumable builder;
hardlinks the base set, remaps and copies nine sources, converts polygons to boxes — all mapping
logic in the `MAPS` and `NEW_SETS` dictionaries) · `images/` and `labels/` for each split ·
`README_HANDOFF.md` · the Colab training notebook (resumable: it loads `last.pt` from mounted
Drive if present, otherwise initialises from pretrained weights).

## Appendix B — Configuration and API Summary

### B.1 Key server constants

| Constant | Value | | Constant | Value |
|---|---|---|---|---|
| `MODEL_MODE` | `custom` | | `DARK_IMAGE_THRESHOLD` | 25 |
| `TARGET_SIZE` | 640 | | `MIN_IMAGE_BYTES` | 1000 |
| `MIN`/`MAX_INPUT_SIZE` | 160 / 640 | | `BLUR_THRESHOLD` | 50.0 |
| `CONFIDENCE_THRESHOLD` | 0.4 | | `OVEREXPOSED_THRESHOLD` | 240 |
| `NMS_IOU_THRESHOLD` | 0.45 (legacy — unused by the NMS-free YOLO26 path, §2.3) | | `UNIFORM_STD_THRESHOLD` | 10 |
| `JWT_EXPIRATION_HOURS` | 24 | | `MIN_RESOLUTION` | 120 px |
| `MAX_EMERGENCY_CONTACTS` | 5 | | `MAX_CODE_ATTEMPTS` | 3 |

There is deliberately no frame-rate constant (`TARGET_FPS` removed August 2026 — the client's
in-flight depth is the only rate control) and no thread-pool cap (removed after the GPU
migration; §3.4.1).

**Tracker (all timings in seconds):** `HIGH_CONF_THRESHOLD` 0.5 · `IOU_THRESHOLD` 0.3 ·
`MAX_AGE_SECONDS` 1.2 · `APPROACH_WINDOW_SEC` 0.8 (min 5 samples) · hysteresis `ENTER_GROWTH`
0.045 with `ENTER_SNR` 2.2, `EXIT_GROWTH` 0.015 with `EXIT_SNR` 1.0 · `CONFIRM_SEC` 0.30 ·
`RELEASE_SEC` 0.25 · `RAPID_TIME_TO_CONTACT_SEC` 3.0 · `LATERAL_THRESHOLD` 15 px · `SMOOTH_N` 3
(median filter at each end of the window) · `MIN_HITS` 3 · history 48 samples · bbox EMA 0.4.
A `MOTION_WINDOW_SEC` 0.6 also exists in the file but is dead — the live window is
`APPROACH_WINDOW_SEC`. **Presence:** TTL 1.5 s · static confirm 0.8 s · alert-state TTL 5 s.
**Batch writer:** flush 1.0 s · max 5,000 pending, oldest dropped first.

**Client:** `COMPRESSION_PERCENT` 75 (JPEG quality 0.25) · `INPUT_SIZE` 640 · `MAX_INFLIGHT` 6 ·
capture poll 120 Hz · stale-entry prune 3,000 ms · reconnect delay 3,000 ms, max 5 attempts ·
alignment tolerance ±15° · health poll 5 s with 4 s timeout, thresholds 100/150/200 ms,
yellow/orange streak 2, RED streak 3, recovery streak 2 · TTS cooldown 3 s · haptic cooldown 2 s ·
danger re-announce 2 s · presence heartbeat 30 s · SOS location 12 s high-accuracy then 8 s
cached, then send without a position.

**Unwired (§3.4.7):** a stream-configuration service defines `input_size` 160–640 step 32,
`compression_percent` 0–95 step 5 and `max_inflight` 1–16 step 1, defaulting to 640 / 75 / 6 —
the same values the client compiles in — but nothing loads, exposes or consumes it.

**Sensitivity profiles** are in §3.4.4. **`HIGH_RISK_CLASSES` default (verified from
`core/config.py`):** car, motorcycle, bicycle, person, stairs, dog, bollard, pothole, scooter.
**Server class list:** still the legacy 14-class vocabulary — the ten base classes plus
`bollard`, `crosswalk`, `pothole`, `scooter` — so of the model's 17 classes, 13 pass the server's
name filter and `curb`, `trash_can`, `manhole` and `construction` are dropped (§3.4.4, §6.3).

### B.2 API surface

| Group | Endpoints |
|---|---|
| Health & status | `GET /` · `GET /health` · `GET /get_system_status` (admin L1+) · `POST /reset_system_status` (L2) |
| **Streaming** | **`WS /stream/ws?token=&input_size=`** · `GET /stream/session_status` |
| Inference | `GET /inference/get_supported_objects` · `POST /inference/pause_detection` · `POST /inference/resume_detection` |
| Settings | `GET /settings/get_settings` · `POST /settings/update_settings` · `GET /settings/available_classes` · `POST /settings/reset_settings` |
| Auth | `POST /users/register` · `POST /users/login` (10/min) · `POST /users/heartbeat` · `POST /users/logout` · `DELETE /users/account` |
| Passwords | `POST /users/change_password` · `POST /users/forgot_password` (3/min) · `POST /users/reset_password` |
| Profile & history | `GET`/`POST /users/profile[/update]` · `GET /users/history` · `DELETE /users/history[/{id}]` |
| Feedback (user) | `POST /users/feedback/{quick,from_history,general}` · `GET /users/feedback/{pending,all}` · `POST /users/feedback/{id}/{update,submit}` · `DELETE /users/feedback/{id}` · `GET`/`POST /users/feedback/responses/{unseen_count,seen}` |
| Contacts & SOS | `POST /users/contacts/{add,verify,resend_code}` · `DELETE /users/contacts/remove` · `GET /users/contacts` · `POST /users/emergency_alert` · `GET /users/emergency_alerts` |
| Admin | `GET /admin/{overview,admins,user}` · `POST /admin/user/{set_password,update,set_level}` · `DELETE /admin/user` · `GET /admin/feedback` · `POST /admin/feedback/{id}/{take,resolve,assign}` |

**WebSocket close codes:** 1000 clean (no reconnect) · 4001 missing token and 4003 invalid or
expired token (no reconnect — session expired) · anything else retries after 3 s, up to 5 attempts.

## Appendix C — Reproducibility Parameters

| Stage | Initialisation | Epochs | Image size | Batch | Initial LR | Early stopping |
|---|---|---|---:|---:|---|---|
| I — Custom detector | Pretrained ResNet18, frozen backbone | 10 planned, stopped at 7 | 640 | 8 | 0.001 | patience 3 |
| II — YOLOv8n baseline | `yolov8n.pt` | 100 target | 640 | 16 | auto (MuSGD, 0.01) | patience 20 |
| III a — 14-class fine-tune | `seesense_yolov8_best.pt` | 30 | 640 | 16 | 0.0003 | default |
| III b — Weak-class fine-tune | `best_new_classes.pt` | 15 | 640 | 16 | 0.0001 requested, overridden by auto mode | default |
| III c — Attempted combined | `/content/best.pt` | 50 | 640 | 16 | 0.0002 | default |
| **IV — Final YOLO26** | `yolo26s.pt`, pretrained | **80** | **640** | **64** | auto (MuSGD, 0.01) | **patience 20**, never triggered |

Stage IV additionally: seed 42 with deterministic execution; AMP enabled and verified; 8 dataloader
workers with caching disabled; mosaic closed over the final ten epochs alongside HSV jitter,
horizontal flip and random scale; `save_period=1` writing all eighty per-epoch checkpoints plus
`best.pt` and `last.pt` to mounted Google Drive behind an automatic resume path; and the full
configuration serialised to JSON beside the artefacts. The run spanned several Colab sessions, the
last of them resuming at epoch 61 and completing epochs 61–80 in 2.501 hours, for roughly ten
GPU-hours in total. Dataset split seed: 42, stratified on each image's primary class.

## Appendix D — Project Timeline and Team Roles

| # | Meeting | Date | Theme and outcome |
|---|---|---|---|
| 1 | Characterisation | 26 Dec 2025 | Architecture debate — camera resolution versus transmission cost. Dataset search begun with the requirement that data simulate a *walking pedestrian's* viewpoint and be validated against self-captured photographs. Supervisor: **train the objects yourself**. |
| 2 | Work plan and POC definition | 30 Jan 2026 | Architecture locked: Client → Server → Algorithm → Feedback, 640×640, streaming. Four sprints defined. Supervisor: present **one** working end-to-end pipeline; prioritise full flow over accuracy; start the server immediately. |
| 3 | POC presentation | 31 Mar 2026 | ~37,000-image pipeline in unified YOLO format. **The CNN → YOLOv8 pivot**: a from-scratch detector could not handle dense street scenes. |
| 4 | Progress review | 28 Apr 2026 | ~57,000 images. Decisions: **define and freeze the test set**; best achievable training on 16–17 classes; then deployment; a cloud model in parallel with a lightweight on-phone model. |
| 5 | Progress review | 24 Jun 2026 | 10 → 17 classes: ~37,000 new images and ~77,000 new boxes. Multi-seed training with best-by-mAP selection proposed (not carried through - see meeting 6). Next meeting designated the "scientific" one: real-time verification (are 7 FPS enough?), precise latency measurement, three alternatives if too slow, deployment architecture, confusion matrices. |
| 6 | Final review | 27 Jul 2026 | Final dataset: **17 classes, 91,139 images**, verified. Imbalance handling presented (crosswalk 33 → 3,511, multi-source merging, recovery of ~6,200 dropped images). Training plan as presented: YOLO11 transfer learning, 80 epochs, 640, batch 16, 3 seeds, best by mAP@50-95. 🔴 The delivered model departs from this plan: the final campaign was **YOLO26-small, 80 epochs, 640, batch 64, a single seed (42)**, with the best checkpoint chosen on validation mAP@0.50:0.95 (§3.3.5). The minute is reproduced as recorded; whether the change was agreed at this meeting or afterwards is not documented. Remaining: complete the book, produce loss and mAP graphs, capture 3 real street examples, submit. |

**Sprint plan (Feb–Apr 2026, managed in Trello):** Sprint 1 (1–14 Feb) infrastructure and data —
training data ready, server skeleton up, camera feed showing; Sprint 2 (15–28 Feb) core DL and
backend — model training, smart alignment, inference pipeline, logic layer; Sprint 3 (1–14 Mar)
full integration — client–server loop, haptic and audio feedback, latency optimisation, edge cases;
Sprint 4 (15 Mar–1 Apr) stabilisation — field testing, bug fixes, demo video, documentation.

**Supervisor:** Dr. Moshe Butman · **Project approver:** Dr. Raz Lin · **Project #503** ·
Repository: https://github.com/OmerHelfer/SeeSense · Trello:
https://trello.com/b/4ii1hYJx/seesense
