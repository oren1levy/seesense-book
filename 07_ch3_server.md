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
│                      user_service · email_service · presence · perf_history · db_writer
├── ml_engine/         model_loader.py · seesense_model.pt ⭐ (the fine-tuned weights)
├── schemas/           Pydantic request/response models
└── utils/metrics.py   live PerformanceTracker (sliding windows)
```

The order of operations in `main.py` is deliberate, and its first lines exist because of
production bugs. **`YOLO_AUTOINSTALL=false` is set before any Ultralytics import**, because
Ultralytics otherwise attempts to `pip install` missing optional dependencies at runtime —
unacceptable inside a serving process. Immediately below it sits a warning that **maths thread
pools must *not* be capped** — the inversion of an earlier fix, and a lesson paid for twice. In
the CPU-only container era the cap was essential: a cgroup limits how much CPU a container may
*use* without changing the core count it *sees*, so Railway reported `os.cpu_count() == 48` while
the replica was limited to 8 vCPU, PyTorch spawned 48 threads for 8 vCPU of work, and a
controlled sweep measured 16 threads running **four times slower** than 8 on a 16-core machine
(§5.3). After the migration to a GPU VM the same cap became the bottleneck in the opposite
direction — YOLO went from 26–36 ms to 149 ms per frame with the cap in place — and it was
removed. The rule the file now records: treat the reported core count as something to observe,
not something to correct, and re-measure on every new deployment target.

Then: rotating-file logging; a lifespan startup that connects Mongo, runs the admin-level
migration, anchors the performance clock (below), loads the model into `app.state.model` and
starts the batch database writer; SlowAPI rate limiting; CORS with an explicit origin list
extensible at runtime through an environment variable; five routers; and — when a built client
bundle is present — static serving of the React application itself, with SPA fallback routing and
a path-traversal guard, which is what makes the single-origin GPU deployment of §3.6.1 possible.
Shutdown flushes the batch writer before closing the database, so a clean stop does not discard
the last second of history.

The performance-clock step is small but exists for a measurement reason. The "measured over"
span shown on the admin dashboard used to be derived from the oldest surviving metrics bucket, so
a scoped reset — or simply TTL expiry — silently moved the start of the recording period forward
and made the system look younger than it was. Startup now backfills a single `perf_meta` document
recording when measurement genuinely began, written so that the earliest writer wins and no later
reset can overwrite it.

### 3.4.2 The real-time WebSocket pipeline

The client connects to `ws://host/stream/ws?token=<JWT>&input_size=640`. No token closes with
code **4001**; an invalid or blacklisted token with **4003** — distinct codes so the client can
tell "you never sent credentials" from "your session expired" and react correctly to each.
`input_size` is a *request*, clamped to `[160, 640]`, used for both letterboxing and inference and
**echoed back**, so both sides agree on the coordinate space in which boxes are expressed; without
this, an overlay drawn at 640 over a frame processed at 512 would be misaligned by 25 %. User
settings are then fetched **exactly once** into the in-memory cache — the only database read at
connect time, and there are **zero settings reads per frame** thereafter. The server replies with
`session_id` and the clamped `input_size`. Deliberately absent is any frame-rate field: a
`TARGET_FPS` existed until August 2026 and was removed as a second, interacting limiter that held
the real rate below what the pipeline could sustain — the client's bounded in-flight window
(§3.5.1) is now the only thing that governs the send rate.

Client text messages carry telemetry, all validated and clamped as untrusted input: `rtt_report`
(which also carries the small health-ping RTT, so the frame round trip can be split into an
outbound and a return leg), `fps_report`, `lost_report` — frames that never came back, which only
the phone can count — and `client_stage_report`. Binary messages are frames. The server's verdict
carries `frame`, `record_id`, `latency_ms`, `danger`, `danger_cleared`, `clearance_message`,
`static_notice`, `alert_is_new`, `alert_level`, `distance` and a list of objects each with class,
confidence, bbox, area ratio, distance, position, alert level, message, a `watched` flag and
motion block.

**Concurrency.** Inference runs inside `asyncio.to_thread` and is serialised behind a global lock.
Both halves are deliberate. *Off the event loop*, because a synchronous forward pass of tens of
milliseconds would block the loop for that long on every frame — which does not merely slow the stream but stalls
`/health`, at which point the client's watchdog concludes the connection is unstable and turns
RED, a self-inflicted false alarm. *Serialised*, because Ultralytics YOLO is not guaranteed
thread-safe for concurrent forward passes and the hardware cannot execute two efficiently anyway.
The result is many concurrent connections, one inference at a time, and a responsive event loop.

**Everything else off the hot path.** The `db_write` stage builds the document with a
**pre-generated ObjectId** and performs no I/O, which is what allows `record_id` to be returned
before anything is persisted — the client needs it immediately, to attach a one-tap feedback
report to this exact frame. After the response is sent, the record and the session's frame count
are queued to a **batch writer** — a single daemon thread that flushes once a second. The first
implementation spawned a fresh thread per write, per frame: at forty frames a second that meant
roughly eighty threads and eighty database round trips a second, which starved the event loop and
froze the client's overlay. The writer buffers up to 5,000 pending records and drops the oldest
first if it ever falls behind, inserts detections unordered in one `insert_many`, and collapses
frame-count updates to the latest value per session before writing them — that last one because
every frame's update targeted the same session document, which MongoDB then had to serialise.
Batching keeps every write off the hot path without the thread cost, and a failed write still
cannot crash the stream. One consequence for the timing breakdown: the `db_write` figure the
dashboard reports is the writer's *amortised* cost per record, not this frame's own work, so it
is shown for visibility but deliberately excluded from the per-frame total.

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
640×640 one for an identical decision.

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
track carries an auto-incrementing ID, class, confidence, bbox, lifecycle counters and a
wall-clock-stamped history of recent boxes (up to 48 samples, sized to cover the motion window at
the deployed frame rate). Association is two-stage: detections are split at confidence 0.5; high-confidence
detections are matched to existing tracks by building an IoU matrix and solving with the
Hungarian algorithm [12] (`scipy.optimize.linear_sum_assignment` on `1 − IoU` [43]), accepting
pairs with IoU ≥ 0.3; then the *remaining low-confidence* detections are matched to
still-unmatched tracks — the ByteTrack insight, which prevents a track dying merely because the
object was briefly occluded and its confidence dipped. The matrix is additionally **class-gated**:
a track never absorbs a detection of another class, because letting a person's track swallow an
overlapping car detection once made its area history read as a 3× jump and fire a phantom "fast
approach". Unmatched high-confidence detections spawn tracks; tracks unseen for longer than
**1.2 seconds** are removed — a wall-clock bound, so track lifetime does not shrink when the
frame rate rises. The reported box is smoothed with an exponential moving average, which steadies
both the Close/Medium/Far classification and the client overlay against YOLO's jitter.

**Motion analysis is a trend test measured in seconds, not a frame comparison**, and two rules
govern it. First, *no threshold is expressed in frames*: the deployed frame rate swings from ~50
on Wi-Fi to under 10 on a congested cellular link, so a frame-count constant tuned at one rate
silently changes meaning at another. Every timing below is a duration. Second, *the test looks
for a consistent trend, not a large jump*. The first implementation asked whether the box had
grown ≥ 22 % between two moments — a magnitude test, and magnitude is distance-dependent: the
same walking speed grows a box 13 % at ten metres but 56 % at three, so the test was blind to
anything approaching from a distance, and once let a dog approach continuously for 5.2 seconds
before firing.

The current test fits a least-squares line through apparent *size* — √area, which is proportional
to inverse distance and therefore near-linear under constant closing speed — over a 0.8-second
window with at least five samples. Two quantities fall out of the fit: **growth**, the fitted
size change across the window relative to its mean, and **SNR**, that change divided by the
residual scatter around the fit. The second is what separates signal from noise: detection jitter
is large but uncorrelated, so it inflates the residual without tilting the line, while a slow,
steady approach tilts the line consistently even when each individual frame moves less than the
jitter. The `approaching` verdict is latched with hysteresis — entering at growth ≥ 4.5 % with
SNR ≥ 2.2, exiting only below 1.5 % growth or SNR 1.0 — and must hold for 0.30 s before latching
on and fail for 0.25 s before releasing, so no single noisy box can raise or clear an alert. A
track reports no motion at all until it has been detected three times.

Speed is graded by **time-to-contact** rather than raw growth: relative growth of apparent size
per second equals closing speed over distance — exactly the inverse of time-to-contact — so the
`fast` threshold reads directly as "will reach the user within three seconds", independent of the
object's size or range. A car 24 m away closing at 8 m/s and a dog one metre away at 0.33 m/s are
both three seconds out, and both deserve the same red alert — which is also what lets a fast
vehicle fire while it is still far away, rather than waiting for its box to look "close". Lateral
direction comes from the median-filtered shift of the box centre across the same window (±15 px).

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
rather than exhausting, and its final form is stricter than earlier versions. A class the user
has switched off **never alerts, at any distance** — it is still detected and drawn, but the
Settings screen's promise that unchecked classes are detected without being marked as danger is
kept literally (an earlier fall-through let an unchecked bench turn the screen red). An object
that is **not approaching** (static or moving away) also never alerts, no matter how close:
nothing is developing, and sitting beside one's own dog, or facing a parked car, must stay
silent. Only an object that **is approaching** alerts, escalating by speed and distance: `fast` —
time-to-contact under three seconds — gives `high` at any distance; approaching at `Close` or
`Medium` gives `high`; approaching from `Far` gives `low`, an early heads-up. Because "not
alerting" and "not there" are different things, each object also carries a `watched` flag, and
clearance is decided by presence rather than by alert level (§3.4.5).

> **Figure 3.6** — Motion-first alert classification decision tree.
>
> `[[FIGURE: decision tree of the rules above]]`

The net effect: parked cars, street furniture, a standing person, or the user sitting still all go
quiet. The red alert clears once relative motion stops and re-fires only on a genuinely new
approach. And because a user walking *toward* a stationary object makes its box grow, that also
registers as approaching — exactly correct, since a collision is equally likely regardless of
which party is moving.

**The server keeps its own class vocabulary, and it has not caught up with the model.** Detections
are filtered by **name, not by index**: the parser reads each detection's class name from the
model's own label map, normalises it, and drops anything not in the server's list. Two
consequences follow, and both are recorded here rather than smoothed over. `curb` (class 10) and
`trash_can`, `manhole` and `construction` (classes 14–16) are resolved correctly by the model and
then silently discarded, with no log line and no counter — so four of the seventeen trained
classes can never reach a user. And `pothole`, which the server still lists, is not a class the
delivered model can predict at all, so it is dead vocabulary that nevertheless appears in the
public class-list endpoints and in every new user's default high-risk set.

`HIGH_RISK_CLASSES` defaults (verified from the deployed `core/config.py`) to `car`,
`motorcycle`, `bicycle`, `person`, `stairs`, `dog`, `bollard`, `pothole` and `scooter`, excluding
landmarks and passive furniture: `bench`, `fire_hydrant`, `traffic_light`, `pole`, `crosswalk`.
Each user can override the list, which matters: a user whose route has a lot of scaffolding may
want `pole` promoted, while a confident cane user may want it demoted. The default is inherited
from the legacy 14-class configuration and has not been reconciled with the 17-class model: it
still contains `pothole`, which the deployed model does not predict, and cannot contain `curb`,
`trash_can`, `manhole` or `construction`, which the server's class list does not yet include
(Appendix B.1, §6.3).

### 3.4.5 Sessions, caching and alert deduplication

**The in-memory cache** (`user_id → {paused, settings, session_id}`) is written on connect, on
pause/resume and on any settings update, and read on **every frame** instead of querying Mongo.
The measured difference recorded in the code is **0 ms versus 71 ms per frame** — at the deployed
~50 FPS, three and a half seconds of database work for every second of walking.

**Session resume** reuses an existing active session, otherwise reactivates a stopped session
whose `stopped_at` is within **15 minutes**, otherwise creates a new one — so a dropped connection
in a lift, or a two-minute pause to cross a road, does not fragment one walk into six sessions.

**Clearance is decided by presence, not by alert level.** An early version announced "path
clear" on the danger flag's true→false transition — but a watched object that merely stops moving
also drops to `none`, so the system could declare the path clear while the user stood on a
collision course with a stationary person. For someone who cannot look up and check, that is the
most dangerous sentence the application can produce. The current logic tracks every watched
object that is close enough to matter — anything classified `Far` is ignored, as is any detection
without a stable track ID — and raises `danger_cleared` only when everything it had engaged with
has genuinely left the frame, unseen for 1.5 seconds, so "נתיב פנוי" means the path actually is
clear. The same mechanism produces the **static notice**: a watched object confirmed
motionless for 0.8 seconds is announced once per still episode, so a hazard that stops alerting
does not silently vanish from the user's world.

**Per-track alert deduplication** retains the last announced alert level **per track ID** and
returns true only when some object's level genuinely **escalated** (`none → low`, `low → high`).
Without it, a car sitting at `low` for ten seconds would re-fire speech and vibration on every
frame in that window — several hundred of them at the deployed rate. Escalation only, deliberately: an earlier any-change version also fired
on `high → low`, and an object oscillating across the approach threshold then alerted in both
directions, producing a continuous stream of speech for a single stationary object —
de-escalations now update the HUD without interrupting the user. Track state expires on a timer
rather than on a single missed frame, so a momentary confidence dip does not make a still-present
object "new". This mechanism, with the trend-based motion analysis above, is what allows the
visual channel to update continuously while audio and haptics fire only on change.

### 3.4.6 Data model, authentication and the application backend

A single shared `MongoClient` serves the application; `connect()` runs once at startup, pings and
ensures indexes. Collections: `users` (with emergency contacts embedded), `settings`, `sessions`,
`detection_history` (one document per processed frame), `feedback`, `emergency_alerts`,
`blacklisted_tokens` (**TTL 24 h**), `reset_codes` (**TTL 15 min**), `perf_history` (**TTL ~400
days**), `perf_meta` — a single untimed document recording when performance measurement began
(§3.4.1) — and `app_config`, which the unwired stream-configuration service declares and never
writes (§3.4.7). TTL indexes mean expiry is enforced by MongoDB rather than application code, so revoked
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
**no automated test suite** — the interactive WebSocket harness contains no assertions (and ships
a hard-coded developer JWT that should be removed from the repository); the highest-value
additions would be unit tests for the alert classifier and the motion analyser, both pure
functions encoding the system's core safety rules. A simulation harness for approach detection
does exist (`dev_tools/approach_harness.py`), generating physically plausible detection sequences
at several frame rates and running them through the real tracker and danger logic — evidence-based
tuning, though not a regression suite, and currently tied to one machine by an absolute path. The
**JWT secret falls back to a development default** if unset, which a production deployment should
fail loudly on. Some **dead code survives the rewrites** and is worth naming because it reads as
current: `check_danger_cleared()` still carries a confident docstring describing the alert-level
clearance rule that presence tracking replaced, and has no callers at all; two frame-count helpers
are likewise vestigial. Finally, a **runtime-tunable streaming configuration is half-built**: a
service exists that would persist a global input size, JPEG compression and pipeline depth in the
database, clamped to documented limits, and an administrative page exists to edit them — but
nothing loads the service at startup, no endpoint exposes it, the page is not routed, and the
client-side functions it calls were never written. It is scaffolding for a feature that was not
finished, described here as such rather than as a capability (§6.3).
