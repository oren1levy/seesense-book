## 3.5 Client Implementation

The client is a mobile-first, right-to-left, Hebrew-language web application in React 19 and
Vite 8, designed to be opened in a phone browser — no installation, no app store. It opens the
rear camera, verifies the phone is held correctly, streams frames, draws a live heads-up display,
and converts verdicts into Hebrew speech and haptic vibration.

For the primary user the visual layer is not the interface; the audio and haptic channels are.
Everything below that looks like polish — throttling, deduplication, spoken confirmations,
announce-once flags — exists because a non-visual interface is far less forgiving of noise.

Fifteen routes are defined: four public (login, register, forgot/reset password) and eleven
protected — the camera dashboard, settings, profile, emergency contacts, detection history, SOS
history, three feedback pages and three admin pages. Admin links are hidden unless the user is an
administrator, and the **server** enforces the actual permission; hiding a control is a courtesy,
never a security boundary.

### 3.5.1 Streaming configuration

Four documented numbers control the capture and upload pipeline. **`COMPRESSION_PERCENT = 50`**
sets JPEG quality. **`INPUT_SIZE = 512`** is the square capture and detection size and **the
biggest performance lever in the system**: the server runs YOLO at this size, so smaller means a
smaller upload *and* a faster forward pass. 640 gives most detail; 512, 416 and 320 are
progressively faster but the model sees less and may miss small or distant objects. The server
clamps and echoes the value so the two sides never disagree about the coordinate space.
**`MAX_INFLIGHT = 5`** is the pipeline depth — the number of frames sent but unanswered at any
moment — with the trade-off documented as an explicit model:

```
per-frame latency ≈ network_RTT + depth × server_processing_time
throughput        ≈ min(1 / server_time, depth / network_RTT)
```

At `INPUT_SIZE = 512` (~41 ms server-side) and a measured ~131 ms round trip, depth 1 gives ~7 FPS
at ~131 ms with the server idle between frames; depth 2 gives ~13 FPS at ~172 ms; depth 4 gives
~22 FPS at ~216 ms, about 96 % of the ~23.5 FPS ceiling. The reasoning recorded in the file is a
safety argument, not a performance one: 22 FPS is more than smooth enough for a pedestrian, and
216 ms of lag corresponds to roughly 3 m of reaction distance against a vehicle at 50 km/h.
Crucially this is a **bounded** queue, not fire-and-forget, so a slow server can never build an
unbounded backlog of stale frames.

### 3.5.2 `VisionStream` and backpressure

`VisionStream` wraps one session. The WebSocket URL is derived from `VITE_API_URL` — `https` →
`wss` — so there is no second URL to keep in sync. `_sendTimes` is a FIFO of timestamps for frames
sent but unanswered; `canSend` is true while its length is below `MAX_INFLIGHT`, after pruning
entries older than three seconds. That prune is a robustness measure: if a result is ever lost, its
FIFO entry would otherwise occupy a slot forever, and after five such losses the client would stop
sending and appear to hang.

Because results arrive in send order, each incoming message is paired with the oldest outstanding
timestamp to yield that frame's round-trip time. Both `result` **and** `error` messages record an
RTT — an error still means the frame is finished and its slot must be released. Every five
seconds, three small text messages carry telemetry off the hot path: average RTT, actual capture
FPS from the last 30 sends, and the aggregated client stage breakdown.

The **reconnection policy** distinguishes causes. Close code 1000 (clean) does not reconnect.
Codes 4001 and 4003 **deliberately do not reconnect**, because retrying would replay the same dead
token indefinitely; instead the client fires a session-expired notification so the application
logs out and redirects to `/login` with an explanation. Any other close retries after 3 seconds,
up to 5 attempts.

### 3.5.3 Camera, capture and overlay geometry

`getUserMedia` opens the rear camera at an ideal 1280×720; on denial the user receives a Hebrew
alert explaining that camera access is mandatory. **Pinch-to-zoom** is tracked through pointer
events with pointer capture, ranging 1–5×, attempting **native hardware zoom first** (clamped to
the device's reported capability) with a CSS transform as the universal fallback.

The capture pipeline begins with a **cheap early-out**: `shouldCapture()` is evaluated *before*
touching the canvas, so a frame the consumer would discard costs essentially nothing — no draw, no
encode. Then a **zoom-aware crop** samples a smaller central region at higher zoom so the captured
frame matches what the user is looking at; `drawImage` is timed as `capture` and
`canvas.toBlob(cb, 'image/jpeg', quality)` as `encode`. `toBlob` is used rather than `toDataURL`
because it is asynchronous and returns a Blob that goes straight onto the socket with no base64
conversion.

The client **polls about three times faster than it sends**. An in-flight slot frees the instant a
*result* arrives, and that moment never aligns with a fixed timer; polling only at the send rate
would leave each freed slot idle for up to a full interval — dead time capping throughput well
below the depth ceiling. The extra ticks are nearly free because of the early-out.

**Overlay geometry** is the hardest mathematics in the client: mapping a box in
`inputSize × inputSize` space onto screen pixels. The captured frame is the **centre square**
(side = `min(videoW, videoH)`) of the raw video; the `<video>` uses `object-fit: cover`, so it is
scaled by `coverScale = max(cW/vW, cH/vH)` and centred, making that square a centred square of
side `baseSize × coverScale` in container coordinates, with detection-to-container scale
`squareSide / inputSize`. Zoom is applied as an **identical CSS transform on both the video and
the SVG**, so the mapping is computed in pre-zoom space and boxes track the feed at any zoom;
stroke widths and font sizes are divided by the zoom factor to stay screen-constant. Boxes are
keyed by `track_id` so React reuses the same DOM node across frames and the box animates smoothly
instead of remounting and flickering.

### 3.5.4 Orientation and the alignment gate

`useOrientation` reads `DeviceOrientationEvent`; "aligned" means the phone is upright within
`|beta − 90| ≤ 15°`. While it is not aligned, **no frames are sent at all** — the system tells the
user to straighten the device rather than analysing a picture of the pavement or the sky.

The platform split is the whole reason this is a hook. On Android and desktop the listener
attaches immediately. On **iOS 13+**, `DeviceOrientationEvent.requestPermission()` must be called
from a user gesture, so the hook exposes `requestPermission()` and the dashboard awaits it inside
the "start scanning" handler, idempotently and handling denial.

The gate converts an invisible failure into an actionable instruction. A blind user holding the
phone at 40° receives no useful detections and, without the gate, no explanation; with it, they
hear "הטה את המכשיר" — straighten the device.

### 3.5.5 Feedback: haptics and Hebrew speech

The feedback service does three jobs. It is a **runtime settings store** — the single source of
truth for volume, vibration intensity, alert channel and voice gender, mirroring the database into
`localStorage` so values are available instantly on load and, crucially, are actually *applied*
when producing sound or vibration, with a pub/sub keeping the settings sliders and the floating
mute button in sync.

**Haptics** use named millisecond patterns: `start` `[60,30,60]`, `stop` `[80]`, `aligned` `[30]`,
`detection` `[100,50,100]`, `danger` `[200,100,200,100,400]`. The Vibration API cannot change
amplitude, only duration, so intensity **scales the vibrating pulses** while leaving the pauses
intact — a low-intensity danger alert keeps its recognisable rhythm rather than becoming a
different signal. A support check lets the settings page state plainly that iOS has no Vibration
API at all, rather than presenting a control that silently does nothing.

**Hebrew text-to-speech** maps every backend class name to Hebrew (מכונית, אופניים, קורקינט…).
The server's `alert_message` is English; the client always composes its own Hebrew utterance, so
presentation language is purely a client concern. Several details matter more than they would in a
visual application. **Voice selection**: the browser's voice list loads asynchronously, so it is
cached and refreshed on `voiceschanged`, filtered by a `he` language prefix, with gender matched
best-effort against name hints and falling back to "not the opposite gender" and then to any
Hebrew voice; the settings page reports how many exist so it can warn that the gender preference
may have no audible effect on this device. **Throttling**: a three-second cooldown, tracked
separately for arbitrary messages and object announcements, with the latter additionally keyed on
class name so a *different* object may interrupt — a car interrupting a bench announcement is
correct. **Mute is derived, not a separate flag**: muted means precisely "the audio channel
produces no sound", so unmuting restores volume and, if the channel was haptic-only, switches it
to both — avoiding the classic bug where a user unmutes and still hears nothing. And the mute
announcement bypasses both the cooldown and the audio gate, because "שמע כבוי" would otherwise be
inaudible at exactly the moment it is needed.

### 3.5.6 The dashboard

Starting a scan requests the iOS gyroscope permission, constructs a `VisionStream`, connects,
registers it as the active stream and starts the health watchdog; stopping reverses this. Either
transition fires a haptic pulse and a spoken confirmation.

`handleResult` runs per frame with a stable reference so it never causes a re-subscribe: ignore if
not scanning or paused; store `record_id` for one-tap feedback; update the overlay, alert level
and leading object's direction and Hebrew name (`render`); if `danger_cleared`, speak "נתיב פנוי"
once and return; otherwise **gate voice and haptics on `alert_is_new`** and fire the appropriate
pattern and announcement (`feedback`). That single condition is the difference between an
application that can be worn for an hour and one that cannot be tolerated for a minute.

The capture gate is `isScanning && isAligned && stream.isOpen && stream.canSend`, **re-checked at
send time** because alignment or the in-flight count may have changed during the asynchronous
encode.

HUD elements: corner brackets (grey → cyan when scanning → green when aligned); a status badge
(`IDLE` → `LIVE` → `TRACKING`); a **scan sweep shown only while frames are actually being sent** —
honest feedback rather than decoration; a **spirit level** driven by the gyroscope; a tilt warning;
a direction indicator colour-coded by alert level; an alert overlay with `role="alert"` and
`aria-live="assertive"` plus a pulsing red viewport border on high danger; a **health dot** whose
millisecond figure is shown **only to administrators**, because a number a regular user cannot act
on is noise; an always-visible **quick-report button** filing a wrong-detection report against the
last `record_id` with a 2.5 s cooldown; and an **SOS button** that is a single tap — deliberately
not a long-press or a gesture — falling back to coordinates `(0,0)` if the location fix fails so
the alert still goes out.

> **Figure 3.7** — Dashboard HUD: corner brackets, spirit level, detection overlay, alert overlay
> and health indicator.
>
> `[[FIGURE: annotated screenshot of the dashboard mid-detection]]`

### 3.5.7 Health watchdog

`healthService` polls `GET /health` every 5 seconds with a 4-second timeout.

| Level | Threshold | Behaviour |
|---|---|---|
| GREEN | < 100 ms | healthy, no feedback |
| YELLOW | ≥ 100 ms | speaks "החיבור לא יציב" **once** |
| ORANGE | ≥ 150 ms | speaks a recommendation to move, once |
| RED | ≥ 200 ms on **3 consecutive** polls | `danger` haptic, "החיבור אבד, הסריקה הופסקה", stops scanning |

Recovery needs **2 consecutive** polls below the red threshold. Announce-once flags reset on
recovery, so a genuinely new degradation is still announced. Requiring consecutive streaks rather
than single readings is what stops one unlucky ping from terminating a healthy session — a mobile
network produces occasional 400 ms outliers with no underlying problem.

This is the *monitoring* half of the hybrid failover architecture of §2.7: the measurement,
thresholds, user notification and automatic stop are implemented. What is missing is the on-device
model to fail over *to*, which is why the system currently stops rather than switching modes.

### 3.5.8 Client metrics, session handling and timezones

`clientMetrics` mirrors the server's stage breakdown for the on-device half — `capture`
(drawImage), `encode` (toBlob), `render` (overlay and HUD) and `feedback` (TTS and haptic
dispatch) — with the network round trip between `encode` and `render` measured separately. It is
zero-overhead by design: one array push onto a bounded 100-sample buffer, no timers, nothing
allocated on the hot path. The aggregate ships every five seconds piggy-backed on the RTT report,
so the full end-to-end breakdown — four client stages, network, five server stages — is visible in
one place for any live session.

The axios instance attaches the bearer token and, on a 401, notifies session expiry — but **only
for authenticated calls**; the public authentication paths are excluded, because a 401 there means
"wrong credentials", not "your session died", and logging someone out for mistyping a password
would be absurd. The notification is guarded by an `alreadyFiring` flag, because a dead token 401s
*every* in-flight request at once and a burst of 401s must not produce a burst of logouts; it also
writes a notice so `/login` can explain *why* the user is back there rather than showing a blank
form. On logout the vision stream is disconnected **first**, then the token blacklisted
server-side, then storage cleared.

**Timezones** were a real bug, fixed properly. The server writes `datetime.now().isoformat()`,
which on the UTC host produces a UTC timestamp **with no timezone marker**; browsers parse a
marker-less ISO string as *local* time, so every relative time was off by the viewer's UTC offset —
an event from moments ago displaying as "3 hours ago" in Israel during summer. The fix appends `Z`
when no marker or offset is present and formats pinned to `Asia/Jerusalem`, so a timestamp reads
identically for every viewer.

### 3.5.9 Design system and accessibility

The design system is a hand-written `global.css` of roughly 3,600 lines with no framework: dark
glassmorphism with a neon HUD, mobile-first. Tokens cover the palette, glass surfaces with
`backdrop-filter`, glow presets, a radius scale, and **safe-area insets** paired with
`viewport-fit=cover` so the HUD and tab bar clear the iPhone notch and home indicator. Global
rules set RTL direction, apply the `height: -webkit-fill-available` workaround for iOS Safari's
100vh behaviour, and keep a visible focus outline for keyboard accessibility.

**Accessibility is a functional requirement here, not a compliance exercise.** Every alert has an
audio and/or haptic channel, with the visual HUD secondary. Icon-only buttons carry `aria-label`,
toggles `aria-pressed`, the active tab `aria-current`, the danger overlay `role="alert"` with
`aria-live="assertive"`, the tilt warning `aria-live="polite"`, and decorative HUD layers
`aria-hidden` so a screen reader does not read the scan sweep. **Every** state change is spoken:
scan on/off, mute on/off, feedback sent, SOS sent or failed, connection degraded, lost and
restored, and "נתיב פנוי" when danger clears. Alert deduplication exists precisely so the audio
channel remains usable — an alert on every frame would be worse than no alert, because it would
mask the alerts that matter. Tap targets are large and the SOS is one tap with no gesture to learn.

### 3.5.10 Known client issues

From a review of the current code: a genuine **one-character bug** in `EmergencyContacts.jsx`
using assignment instead of comparison, so every contact renders as verified and pending contacts
lose their "verify now" button — the highest-value single-character fix in the client. **Hebrew
class-name maps are out of sync** across four pages, several page-local copies still listing
classes the deployed model no longer emits while omitting newer ones; the effect is cosmetic, but
it means every retrain requires editing five files instead of one shared map. **The timezone fix
is not applied everywhere** — three pages still call `new Date(ts)` directly. **The health
thresholds contradict the file's own header comment** and are tight relative to real mobile RTT,
so YELLOW fires often. And there is **no error boundary**: an exception inside a page unmounts the
React tree and leaves a blank screen, which for this user is a completely silent failure — the
most important missing robustness feature in the client.
