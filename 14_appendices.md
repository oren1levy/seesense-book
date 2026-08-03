# Appendices

## Appendix A — Final Dataset Card

**Name:** `Data-25-07` · **Built and verified:** 25 July 2026 · **Format:** YOLO detection
(`class x_center y_center width height`, normalised) · **Compatible with:** YOLOv8 and YOLO11
(identical label format, no conversion required)

| Property | Value | | Property | Value |
|---|---|---|---|---|
| Classes | 17 | | Train / val / test | 74,096 / 11,086 / 5,957 |
| Unique images | 91,139 | | Image size | 640×640 letterbox, grey 114 padding |
| Bounding boxes | 964,837 | | Sources | 10 (3 academic + 9 Roboflow) |

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
yolo detect train data=Final-Data-25-07/data.yaml model=yolo11n.pt imgsz=640 epochs=80 batch=16
```

**Manifest:** `data.yaml` (17-class config) · `build_final_25_07.py` (resumable builder;
hardlinks the base set, remaps and copies nine sources, converts polygons to boxes — all mapping
logic in the `MAPS` and `NEW_SETS` dictionaries) · `images/` and `labels/` for each split ·
`README_HANDOFF.md` · `train_best_colab.py` (three-seed train-and-select-best).

## Appendix B — Configuration and API Summary

### B.1 Key server constants

| Constant | Value | | Constant | Value |
|---|---|---|---|---|
| `MODEL_MODE` | `custom` | | `DARK_IMAGE_THRESHOLD` | 25 |
| `TARGET_SIZE` | 640 | | `MIN_IMAGE_BYTES` | 1000 |
| `MIN`/`MAX_INPUT_SIZE` | 160 / 640 | | `BLUR_THRESHOLD` | 50.0 |
| `TARGET_FPS` | 40 | | `OVEREXPOSED_THRESHOLD` | 240 |
| `CONFIDENCE_THRESHOLD` | 0.4 | | `UNIFORM_STD_THRESHOLD` | 10 |
| `NMS_IOU_THRESHOLD` | 0.45 | | `MIN_RESOLUTION` | 120 px |
| `JWT_EXPIRATION_HOURS` | 24 | | `MAX_EMERGENCY_CONTACTS` | 5 |
| `TORCH_NUM_THREADS` | 8 | | `MAX_CODE_ATTEMPTS` | 3 |

**Tracker:** `HIGH_CONF_THRESHOLD` 0.5 · `IOU_THRESHOLD` 0.3 · `MAX_AGE` 10 frames ·
`MOTION_WINDOW` 4 frames · `APPROACH_RATIO` 1.10 · `RAPID_APPROACH_RATIO` 1.25 ·
`LATERAL_THRESHOLD` 15 px · track history 10 frames.

**Client:** `COMPRESSION_PERCENT` 50 · `INPUT_SIZE` 512 · `MAX_INFLIGHT` 5 · stale-entry prune
3,000 ms · RTT FIFO cap 120 · reconnect delay 3,000 ms, max 5 attempts · alignment tolerance ±15° ·
health poll 5 s with 4 s timeout, thresholds 100/150/200 ms, RED streak 3, recovery streak 2 ·
TTS cooldown 3 s · presence heartbeat 30 s · capture poll multiplier 3×.

**Sensitivity profiles** are in Table §3.4.4. **Deployed classes (17)** match the dataset
(Appendix A). `[[TODO: verify the default HIGH_RISK_CLASSES list and these constants against the
deployed core/config.py.]]`

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
| II — YOLOv8n baseline | `yolov8n.pt` | 100 target | 640 | 16 | auto (0.01) | patience 20 |
| III — 14-class fine-tune | `seesense_yolov8_best.pt` | 30 | 640 | 16 | 0.0003 | default |
| IV — Weak-class fine-tune | `best_new_classes.pt` | 15 | 640 | 16 | 0.0001 | default |
| — Attempted combined | `/content/best.pt` | 50 | 640 | 16 | 0.0001 | default |
| **V — Final YOLO11** | `yolo11n.pt` | **80** | **640** | **16** | default | **patience 20** |

Stage V additionally: 3 runs with seeds 0, 1, 2; selection by validation mAP@50-95; Ultralytics
default augmentation (mosaic, horizontal flip, random scale, HSV jitter). Dataset split seed: 42,
stratified on each image's primary class.

## Appendix D — Project Timeline and Team Roles

| # | Meeting | Date | Theme and outcome |
|---|---|---|---|
| 1 | Characterisation | 26 Dec 2025 | Architecture debate — camera resolution versus transmission cost. Dataset search begun with the requirement that data simulate a *walking pedestrian's* viewpoint and be validated against self-captured photographs. Supervisor: **train the objects yourself**. |
| 2 | Work plan and POC definition | 30 Jan 2026 | Architecture locked: Client → Server → Algorithm → Feedback, 640×640, streaming. Four sprints defined. Supervisor: present **one** working end-to-end pipeline; prioritise full flow over accuracy; start the server immediately. |
| 3 | POC presentation | 31 Mar 2026 | ~37,000-image pipeline in unified YOLO format. **The CNN → YOLOv8 pivot**: a from-scratch detector could not handle dense street scenes. |
| 4 | Progress review | 28 Apr 2026 | ~57,000 images. Decisions: **define and freeze the test set**; best achievable training on 16–17 classes; then deployment; a cloud model in parallel with a lightweight on-phone model. |
| 5 | Progress review | 24 Jun 2026 | 10 → 17 classes: ~37,000 new images and ~77,000 new boxes. Multi-seed training with best-by-mAP selection. Next meeting designated the "scientific" one: real-time verification (are 7 FPS enough?), precise latency measurement, three alternatives if too slow, deployment architecture, confusion matrices. |
| 6 | Final review | 27 Jul 2026 | Final dataset: **17 classes, 91,139 images**, verified. Imbalance handling presented (crosswalk 33 → 3,511, multi-source merging, recovery of ~6,200 dropped images). Training: YOLO11 transfer learning, 80 epochs, 640, batch 16, 3 seeds, best by mAP@50-95. Remaining: complete the book, produce loss and mAP graphs, capture 3 real street examples, submit. |

**Sprint plan (Feb–Apr 2026, managed in Trello):** Sprint 1 (1–14 Feb) infrastructure and data —
training data ready, server skeleton up, camera feed showing; Sprint 2 (15–28 Feb) core DL and
backend — model training, smart alignment, inference pipeline, logic layer; Sprint 3 (1–14 Mar)
full integration — client–server loop, haptic and audio feedback, latency optimisation, edge cases;
Sprint 4 (15 Mar–1 Apr) stabilisation — field testing, bug fixes, demo video, documentation.

**Division of work**, as assigned in the characterisation and POC planning documents; the team
worked collaboratively across boundaries, particularly during integration.

| Member | Primary responsibilities |
|---|---|
| **Oren Levy** | Characterisation document and screen mockups; monorepo and Git; client application skeleton; model validation; latency optimisation; demo video |
| **Omer Helfer** | Feasibility and latency study; server environment; model training; server inference pipeline; latency optimisation |
| **Liad Lati** | Training environment and YOLO configuration; alert-condition logic definition; server logic layer; edge-case handling; documentation lead; submissions |
| **Shir Yahav** | Dataset search and evaluation against self-captured field images; data collection, filtering, class mapping and preprocessing into unified YOLO format; train/val/test splitting and data-quality validation; gyroscope smart-alignment client logic; client–server connection; feedback implementation; user-experience analysis |

`[[TODO: confirm this division against how the work actually ran, particularly for the later
phases — dataset expansion, server engineering, client build — which the original planning
documents do not cover.]]`

**Supervisor:** Dr. Moshe Butman · **Project approver:** Dr. Raz Lin · **Project #503** ·
Repository: https://github.com/oren1levy/SeeSense · Trello:
https://trello.com/b/4ii1hYJx/seesense
