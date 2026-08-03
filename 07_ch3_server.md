## 3.4 Server Implementation

The backend performs two quite different jobs. On the hot path it is a real-time inference
pipeline: one JPEG in, one verdict out, tens of times a second. Everywhere else it is a
conventional application backend. The design keeps these strictly separated, because anything
blocking the first makes the product unusable.

### 3.4.1 Structure and startup

```
Server/
├── main.py            app factory, lifespan, logging, CORS, rate limiting
├── core/              config.py · database.py · auth.py
├── api/               stream.py ⭐ · inference.py · settings.py · users.py · admin.py
├── services/          vision_service · logic_service · motion_tracker · session_service
│                      user_service · email_service · presence · perf_history
├── ml_engine/         model_loader.py · seesense_model.pt ⭐ (the fine-tuned weights)
├── schemas/           Pydantic request/response models
└── utils/metrics.py   live PerformanceTracker (sliding windows)
```

The order of operations in `main.py` is deliberate, and two steps exist because of production
bugs. **`YOLO_AUTOINSTALL=false` is set before any Ultralytics import**, because Ultralytics
otherwise attempts to `pip install` missing optional dependencies at runtime — unacceptable inside
a serving process. **Maths thread pools are capped before torch, NumPy or OpenCV are imported**
(`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, default 8), because a cgroup limits how much CPU a
container may *use* without changing the core count it *sees*: Railway reported
`os.cpu_count() == 48` while the replica was limited to 8 vCPU, so PyTorch spawned 48 threads for
8 vCPU of work — six times oversubscribed, burning most of its time on context switches. A
controlled sweep measured 16 threads running **four times slower** than 8 on a 16-core machine.
The cap must precede import because the libraries read these variables once, at import time.

Then: rotating-file logging; a lifespan startup connecting Mongo, running the admin-level
migration and loading the model into `app.state.model`; SlowAPI rate limiting; CORS with an
explicit origin list extensible at runtime through an environment variable; and the five routers.

### 3.4.2 The real-time WebSocket pipeline

The client connects to `ws://host/stream/ws?token=<JWT>&input_size=512`. No token closes with
code **4001**; an invalid or blacklisted token with **4003** — distinct codes so the client can
tell "you never sent credentials" from "your session expired" and react correctly to each.
`input_size` is a *request*, clamped to `[160, 640]`, used for both letterboxing and inference and
**echoed back**, so both sides agree on the coordinate space in which boxes are expressed; without
this, an overlay drawn at 640 over a frame processed at 512 would be misaligned by 25 %. User
settings are then fetched **exactly once** into the in-memory cache — the only database read at
connect time, and there are **zero settings reads per frame** thereafter. The server replies with
`session_id`, `target_fps` and the clamped `input_size`, from which the client configures its
whole capture loop, so one constant changed on the server retunes the pipeline without a client
release.

Client text messages carry telemetry (`rtt_report`, `fps_report`, `client_stage_report`, all
validated); binary messages are frames. The server's verdict carries `frame`, `record_id`,
`latency_ms`, `danger`, `danger_cleared`, `clearance_message`, `alert_is_new`, `alert_level`,
`distance` and a list of objects each with class, confidence, bbox, area ratio, distance,
position, alert level, message and motion block.

**Concurrency.** Inference runs inside `asyncio.to_thread` and is serialised behind a global lock.
Both halves are deliberate. *Off the event loop*, because a ~40 ms synchronous forward pass would
block the loop for 40 ms of every frame — which does not merely slow the stream but stalls
`/health`, at which point the client's watchdog concludes the connection is unstable and turns
RED, a self-inflicted false alarm. *Serialised*, because Ultralytics YOLO is not guaranteed
thread-safe for concurrent forward passes and the hardware cannot execute two efficiently anyway.
The result is many concurrent connections, one inference at a time, and a responsive event loop.

**Everything else off the hot path.** The `db_write` stage builds the document with a
**pre-generated ObjectId** and performs no I/O, which is what allows `record_id` to be returned
before anything is persisted — the client needs it immediately, to attach a one-tap feedback
report to this exact frame. After the response is sent, two daemon threads perform the writes;
neither can stall the stream and a failed write cannot crash it.

**Error handling.** A failed quality gate returns `{"type": "error", "detail": …}` with a
human-readable reason — and `danger_cleared` is still evaluated, so a user who has moved away from
a hazard still hears "path clear" even if the next frame is too dark to analyse. Other exceptions
are logged with a traceback and returned generically; failures are counted in both the live
tracker and the persisted history.

### 3.4.3 Vision service and motion tracking

`decode_image()` rejects payloads under 1,000 bytes, decodes with OpenCV, checks the original's
longest side is ≥ 120 px, **letterbox-resizes** to a square padded with grey 114 — byte-for-byte
the same preprocessing used to build the training set, which is why training and inference
distributions match — and applies four quality gates on the **resized** image, a real optimisation
since a Laplacian variance over a 2048×1536 photograph costs several times what it costs over a
512×512 one for an identical decision.

| Gate | Condition | Message to the user |
|---|---|---|
| Camera covered | grey std-dev < 10 | "Camera appears to be covered or blocked." |
| Too dark | mean intensity < 25 | "Image is too dark…" |
| Overexposed | mean intensity > 240 | "Image is overexposed…" |
| Blurry | Laplacian variance < 50 | "Image is too blurry…" |

These gates exist because of the user. A sighted user notices immediately that their thumb is over
the lens; a blind user has no way to know, and would otherwise receive a system that has silently
stopped detecting while appearing to work. Turning "no detections" into "your camera is covered"
is a safety feature.

The **tracker** is ByteTrack-inspired [9], one instance per user, cleared on disconnect. Each
track carries an auto-incrementing ID, class, confidence, bbox, lifecycle counters and a deque of
the last ten frames' box, area and centre. Association is two-stage: detections are split at
confidence 0.5; high-confidence detections are matched to existing tracks by building an IoU
matrix and solving with the Hungarian algorithm [12] (`scipy.optimize.linear_sum_assignment` on
`1 − IoU` [43]), accepting pairs with IoU ≥ 0.3; then the *remaining low-confidence* detections
are matched to still-unmatched tracks — the ByteTrack insight, which prevents a track dying merely
because the object was briefly occluded and its confidence dipped. Unmatched high-confidence
detections spawn tracks; tracks unseen for more than 10 frames are removed.

**Windowed motion analysis** compares the current frame against **4 frames back**, not the
previous frame. This is not a detail; it is the difference between a usable and an unusable
product. YOLO boxes jitter by a few pixels every frame even on a perfectly static object, so
frame-to-frame area comparison made the `approaching` flag flicker several times a second and
spammed the alert channel. A short window smooths the jitter: a static object reads `static`, and
only sustained growth reads `approaching`. Area ratio ≥ 1.10 gives `approaching` and `moderate`;
≥ 1.25 gives `fast`; ≤ 1/1.10 gives `moving_away`; a horizontal centre shift beyond 15 px gives
`left` or `right`.

### 3.4.4 Danger logic

`assess_danger()` converts tracked boxes into a safety verdict: apply the user's **sensitivity
profile** and drop detections below its confidence floor; compute `area_ratio = bbox_area /
frame_area` and map it to a **distance category**; derive **position** from the box centre by
splitting the frame into thirds; classify each object's **alert level**; compose a spoken
**message** ("car approaching fast on your right"); and roll up to a frame verdict — highest alert
level, closest distance, `danger = (highest == "high")`.

| Profile | Confidence | Close area ratio | Medium area ratio |
|---|---:|---:|---:|
| `low` | 0.70 | 0.40 | 0.25 |
| `medium` (default) | 0.50 | 0.15 | 0.05 |
| `high` | 0.35 | 0.08 | 0.03 |

The profile changes two things at once, which is what makes it a meaningful control rather than a
slider: a `low` user hears only about high-confidence objects filling 40 % of the frame — very
close, very certain — while a `high` user hears about lower-confidence objects at 8 % of frame
area, considerably further away.

**Motion-first alert classification** is the design decision that makes the application usable
rather than exhausting. If an object is **not approaching** (static or moving away): a high-risk
*non-vehicle* obstacle at `Close` gives `low`, a soft caution; vehicles and anything further give
`none`, completely silent. If it **is approaching**: high-risk and `fast` gives `high` at any
distance; high-risk at `Close` or `Medium` gives `high`; high-risk at `Far` gives `low`, an early
heads-up; not high-risk gives `high` at `Close`, `low` at `Medium`, else `none`.

> **Figure 3.6** — Motion-first alert classification decision tree.
>
> `[[FIGURE: decision tree of the rules above]]`

The net effect: parked cars, street furniture, a standing person, or the user sitting still all go
quiet. The red alert clears once relative motion stops and re-fires only on a genuinely new
approach. And because a user walking *toward* a stationary object makes its box grow, that also
registers as approaching — exactly correct, since a collision is equally likely regardless of
which party is moving.

`HIGH_RISK_CLASSES` defaults to the subset that can cause injury on contact — `car`,
`motorcycle`, `bicycle`, `person`, `stairs`, `dog`, `bollard`, `scooter`, `curb`, `manhole`,
`construction` — excluding landmarks and passive furniture: `bench`, `fire_hydrant`,
`traffic_light`, `pole`, `trash_can`, `crosswalk`. Each user can override the list, which matters:
a user whose route has a lot of scaffolding may want `pole` promoted, while a confident cane user
may want it demoted. `[[TODO: confirm the deployed default against core/config.py — the list above
is the 17-class equivalent of the documented 14-class default, not a verified copy.]]`

### 3.4.5 Sessions, caching and alert deduplication

**The in-memory cache** (`user_id → {paused, settings, session_id}`) is written on connect, on
pause/resume and on any settings update, and read on **every frame** instead of querying Mongo.
The measured difference recorded in the code is **0 ms versus 71 ms per frame** — at 22 FPS, 1.5
seconds of database work per second of walking.

**Session resume** reuses an existing active session, otherwise reactivates a stopped session
whose `stopped_at` is within **15 minutes**, otherwise creates a new one — so a dropped connection
in a lift, or a two-minute pause to cross a road, does not fragment one walk into six sessions.

**Danger-cleared detection** retains the previous frame's danger flag per user and returns true
exactly on a true→false transition. That single edge triggers the one-shot "path clear"
announcement — positive confirmation that the hazard has passed, which for a user who cannot see
the road is as important as the warning itself.

**Per-track alert deduplication** retains the last announced alert level **per track ID** and
returns true only when some object's level genuinely changed (`none → low`, `low → high`). Without
it, a car sitting at `low` for ten seconds would re-fire speech and vibration on roughly 200
consecutive frames. Tracks absent from the current frame are forgotten, so a real-world object
that leaves and returns *is* treated as new — the desired behaviour, since that is new
information. This mechanism, with the windowed motion analysis above, is what allows the visual
channel to update continuously while audio and haptics fire only on change.

### 3.4.6 Data model, authentication and the application backend

A single shared `MongoClient` serves the application; `connect()` runs once at startup, pings and
ensures indexes. Collections: `users` (with emergency contacts embedded), `settings`, `sessions`,
`detection_history` (one document per processed frame), `feedback`, `emergency_alerts`,
`blacklisted_tokens` (**TTL 24 h**), `reset_codes` (**TTL 15 min**) and `perf_history` (**TTL ~400
days**). TTL indexes mean expiry is enforced by MongoDB rather than application code, so revoked
tokens and expired codes disappear with no cleanup job to forget to write. E-mail addresses are
canonicalised (lower-cased and trimmed) on register, login, admin lookup and password reset,
closing a duplicate-account hole that is trivial to create and painful to fix afterwards.

**Authentication** uses JWT HS256 [33] with a 24-hour lifetime and bcrypt [34] password hashing;
hashes never leave the service layer. Logout is a **real blacklist** rather than a client-side
token discard, with a TTL index so the collection self-prunes. Every successful token verification
stamps presence, which is how "who is online" stays accurate for a user browsing without scanning.
**Three admin levels** are stored on the user document — 0 regular, 1 admin (view everything,
manage regular users, triage feedback), 2 super admin (delete users, grant levels, reassign
feedback, reset performance data) — enforced by composing guards, with a rule that a level-1 admin
can never act on a user with `admin_level ≥ 1`. Endpoints return the caller's level so the
interface can hide what it cannot use, while the server enforces regardless. **Lockout protection**
prevents demoting or deleting the last level-2 admin, or a super admin demoting themselves;
without these a single mis-click permanently orphans the system. WebSocket authentication is
separate, since FastAPI's dependency machinery is unavailable on a handshake.

**Settings** are validated on every write — unknown keys 400, `high_risk_classes` a subset of the
supported classes, sensitivity/alert-type/voice-gender from fixed sets, intensities in `[0,1]` —
and a successful write refreshes the in-memory cache, so **a live WebSocket picks up the change on
the very next frame** without reconnecting: a user can slide the sensitivity control while walking
and hear the effect immediately.

**The two-axis feedback system** is the most intricate application logic, and its two status
fields are easy to confuse. `status` is the *user's* axis (`pending` → `submitted`);
`handling_status` is the *administrator's* axis, layered on top and fully independent (`pending` →
`in_progress` → `resolved`). Only submitted feedback enters the admin queue, and once an admin
takes a report the user's edit endpoint returns **409** — the report is locked, and the client
renders a lock icon. Three entry points match three situations: a one-tap report filed during a
walk, linked to the last `record_id` and capturing a **detection snapshot** so the frame's context
survives even if the user later deletes that history record; a considered report written afterward
from a history record; and a standalone report. This exists because a deployed detector is wrong
in ways its test set cannot predict, and the only source of that information is the user.

**Emergency contacts** are embedded, capped at five, and carry a verification state machine: a
6-digit code is e-mailed to the **contact** (not the user) with an explanation of what SeeSense is
and what being an emergency contact means; verification allows three attempts and expires after 30
minutes; only verified contacts receive SOS alerts. `trigger_emergency` builds a Google Maps link,
persists the alert with the list of contacts notified so SOS history is auditable, and **sends
every e-mail on a background thread, returning immediately** — because of a production incident
(Table 5.1) in which blocking SMTP on the event loop froze the entire server.

### 3.4.7 Performance metrics and known constraints

Performance measurement is built into the product in two layers. **Live**: a global tracker with
100-sample sliding windows over server latency, client-reported RTT (plus a 60-point history for
the live chart), frame arrival times, client capture FPS, per-stage latency, and the untrusted —
therefore validated and clamped — client stage breakdown. Throughput is computed from completion
timestamps of *successful* frames over a rolling ten-second window as `(n−1)/(t_last − t_first)`,
so it reads correctly immediately with no warm-up and decays to zero when idle. Four FPS numbers
are reported deliberately, because they answer four questions (§4.4.2). **Persistent**: because
the live tracker resets on restart, metrics also accumulate into a minute-aligned bucket flushed
to `perf_history` on a daemon thread, keyed by minute so a re-flush cannot double-count, with ~400
days of retention.

Known constraints worth recording: **in-memory state assumes a single worker process** — the
settings cache, trackers, live metrics and presence map all live in process memory, so scaling to
multiple replicas would need Redis; this buys 71 ms per frame at the cost of horizontal
scalability, the correct trade at this stage and the wrong one for a production service. There is
**no automated test suite** — the interactive WebSocket harness is not assertions; the
highest-value additions would be unit tests for the alert classifier and the motion analyser,
both pure functions encoding the system's core safety rules. And the **JWT secret falls back to a
development default** if unset, which a production deployment should fail loudly on.
