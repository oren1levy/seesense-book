## 3.5 Client Implementation

The client is a mobile-first, right-to-left, Hebrew-language web application in React 19 and
Vite 8, designed to be opened in a phone browser — no installation, no app store. It opens the
rear camera, verifies the phone is held correctly, streams frames, draws a live heads-up display,
and converts verdicts into Hebrew speech and haptic vibration.

For the primary user the visual layer is not the interface; the audio and haptic channels are.
Everything below that looks like polish — throttling, deduplication, spoken confirmations,
announce-once flags — exists because a non-visual interface is far less forgiving of noise.

Sixteen routes are defined: four public (login, register, forgot/reset password) and twelve
protected — the camera dashboard, settings, profile, emergency contacts, detection history, SOS
history, three feedback pages and three admin pages. Admin links are hidden unless the user is an
administrator, and the **server** enforces the actual permission; hiding a control is a courtesy,
never a security boundary.

### 3.5.1 Streaming configuration

Three documented numbers control the capture and upload pipeline. **`COMPRESSION_PERCENT = 75`**
sets JPEG compression, mapping to a canvas quality of 0.25 — the one knob for trading image
sharpness against upload size. **`INPUT_SIZE = 640`** is the square capture and detection size
and **the biggest performance lever in the system**: the server runs YOLO at this size, so
smaller means a smaller upload *and* a faster forward pass. During the CPU-only era the client
ran at 512 to hold latency down (§4.4.1); on the GPU deployment the full 640 is affordable,
restoring detail on small and distant objects. The server clamps and echoes the value so the two
sides never disagree about the coordinate space. **`MAX_INFLIGHT = 6`** is the pipeline depth —
the number of frames sent but unanswered at any moment — and it is deliberately **the only
control on the send rate**: a separate `TARGET_FPS` existed until August 2026 and was removed as
a second, interacting limiter. The client sends the next frame the moment a reply frees a slot,
so the rate settles at whatever the network and the server can sustain, governed by:

```
throughput λ = min(1/S, depth/R₀)        S  = server time per frame
latency    W = depth / λ                 R₀ = round trip with no queueing
```

Below the crossover depth ≈ R₀/S the server idles between frames and added depth is free
throughput; above it the server never idles and every added frame only queues. Measured on
6 August 2026 against the GPU deployment (S ≈ 16.4 ms, a ~61 FPS ceiling; R₀ ≈ 120 ms): depth 6
gives ~50 FPS at ~120 ms, depth 7 ~53 FPS at ~133 ms, depth 10 ~60 FPS at ~166 ms, and depth 20
the same ~60 FPS at ~332 ms — identical throughput for double the delay. The shipped depth of 6
sits just below the crossover (≈ 7), and the reasoning recorded in the file is a safety argument,
not a performance one: latency is the safety number — at 50 km/h a car covers 1.4 m per 100 ms —
and above ~25 FPS extra frames buy almost nothing, since the 0.8-second approach window is
already oversampled. Crucially this is a **bounded** queue, not fire-and-forget, so a slow server
can never build an unbounded backlog of stale frames.

All three are **compile-time constants**, changed by editing the file and rebuilding. Only the
input size has any server involvement, and only in one direction: the client requests its own
value on every connect and the server echoes back the value it will actually use, which the
client then adopts for the capture canvas and the overlay coordinate space. Making all three
adjustable at runtime by an administrator was started and not finished (§3.4.7).

### 3.5.2 `VisionStream` and backpressure

`VisionStream` wraps one session. The WebSocket URL is derived from `VITE_API_URL` — `https` →
`wss` — so there is no second URL to keep in sync. `_sendTimes` is a FIFO of timestamps for frames
sent but unanswered; `canSend` is true while its length is below `MAX_INFLIGHT`, after pruning
entries older than three seconds. That prune is a robustness measure: if a result is ever lost, its
FIFO entry would otherwise occupy a slot forever, and after six such losses the client would stop
sending and appear to hang.

Because results arrive in send order, each incoming message is paired with the oldest outstanding
timestamp to yield that frame's round-trip time. Both `result` **and** `error` messages record an
RTT — an error still means the frame is finished and its slot must be released. Every five
seconds, four small text messages carry telemetry off the hot path: average RTT (together with
the latest health-ping RTT, so the server can split the round trip into an outbound and a return
leg), actual capture FPS from the last 30 sends, a count of frames that never came back — which
only the phone can know — and the aggregated client stage breakdown.

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

The capture timer **polls at a fixed 120 Hz, far faster than any achievable send rate** — and
deliberately does not set the frame rate; `MAX_INFLIGHT` does. An in-flight slot frees the
instant a *result* arrives, and that moment never aligns with a fixed timer; polling only at the
send rate would leave each freed slot idle for up to a full interval — dead time capping
throughput well below the depth ceiling. The extra ticks are nearly free because of the early-out.

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
hear "הטה את המכשיר" — straighten the device. One caveat is recorded in §3.5.10: the hook's
initial state assumes an upright phone, so a device that never fires an orientation event — no
gyroscope, or a declined iOS motion permission — reads as permanently aligned and the gate
silently becomes a no-op.

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
The server's `alert_message` is English and the client never reads it; the client composes every
utterance itself from the structured fields, so presentation language is purely a client concern.
Phrases are built from where an object *is* rather than where it is going — "סכנה קרובה, מכונית
לפניך" for an approaching threat, "אדם לפניך, אין תנועה" for a watched object confirmed
motionless — because a spoken bearing is only useful if it tells the user where to turn. Several details matter more than they would in a
visual application. **Voice selection**: the browser's voice list loads asynchronously, so it is
cached and refreshed on `voiceschanged`, filtered by a `he` language prefix, with gender matched
best-effort against name hints and falling back to "not the opposite gender" and then to any
Hebrew voice; the voice-info API reports how many Hebrew voices exist, and the settings page uses
it to disable the gender choice when it can have no audible effect on this device.
**Throttling**: five speech paths with deliberately
different rules. The ordinary path applies a three-second cooldown and cancels whatever is
speaking, with an explicit *priority* flag that bypasses the cooldown for utterances that must
never be swallowed — "path clear", the repeated danger warning, and the stop confirmation. A
second path is used for status and lifecycle lines and deliberately neither cools down nor
cancels, so paired announcements ("scanning on, connecting" then "connected") queue in order
instead of cutting each other off. Object announcements keep their own cooldown keyed on class
name so a *different* object may interrupt — a car interrupting a bench announcement is correct.
A voice-preview path and the mute announcement bypass the throttle entirely. Haptics have their
own two-second floor on the alert patterns. **Mute is derived, not a separate flag**: muted means precisely "the audio channel
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
once and return. While a red-level danger is present and still approaching, the leading threat is
re-announced every two seconds — a deliberate bypass of the novelty gate, because an ongoing
threat must not fall silent for having been mentioned once. A `static_notice` from the server is
phrased in Hebrew and spoken once per still episode. Otherwise **voice and haptics are gated on
`alert_is_new`** (`feedback`). That last condition is the difference between an application that
can be worn for an hour and one that cannot be tolerated for a minute.

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
not a long-press or a gesture. The location fix behind it makes two attempts, a high-accuracy one
with a twelve-second budget and then a fast cached one with eight, before giving up and sending
the alert with no position at all: the system never fabricates a location, because an early
`(0,0)` fallback produced a Google Maps link to "Null Island" in the Atlantic, which is worse
than an honest "location unavailable". Administrators additionally see a live server-and-client
frame-rate readout beside the health dot, sampled once a second and written so that a steady
stream causes no re-render.

> **Figure 3.7** — Dashboard HUD: corner brackets, spirit level, detection overlay, alert overlay
> and health indicator.
>
> `[[FIGURE: annotated screenshot of the dashboard mid-detection]]`

### 3.5.7 Health watchdog

`healthService` polls `GET /health` every 5 seconds with a 4-second timeout.

| Level | Threshold | Behaviour |
|---|---|---|
| GREEN | < 100 ms | healthy, no feedback |
| YELLOW | ≥ 100 ms on **2 consecutive** polls | speaks "החיבור לא יציב" **once** |
| ORANGE | ≥ 150 ms on **2 consecutive** polls | speaks a recommendation to move, once |
| RED | ≥ 200 ms on **3 consecutive** polls | `danger` haptic and a spoken "connection lost" |

The dot reacts to a single reading but the *voice* waits for a streak, so the visual indicator
stays responsive without the audio channel commenting on every fluctuation. Recovery needs **2
consecutive** polls below the red threshold and names the state it recovered *to* rather than
just announcing that something changed. Announce-once flags reset on recovery, so a genuinely new
degradation is still announced, and the all-clear only speaks if a degradation was announced in
the first place. One wording detail is deliberate: the red announcement says only that the
connection was lost, because the previous text also claimed scanning had stopped — and it has
not (below). Telling a blind user their scan has stopped while it is still running is the worst
possible direction for that error to point. Requiring consecutive streaks rather
than single readings is what stops one unlucky ping from terminating a healthy session — a mobile
network produces occasional 400 ms outliers with no underlying problem.

This is the *monitoring* half of the hybrid failover architecture of §2.7: the measurement,
thresholds and user notification are implemented. Two gaps remain. The **automatic stop on a red
connection is not currently wired** — the dashboard's disconnect callback only logs, so scanning
continues through a red state (§3.5.10). And there is no on-device model to fail over *to*, which
is why degradation today produces warnings rather than a mode switch.

### 3.5.8 Client metrics, session handling and timezones

`clientMetrics` mirrors the server's stage breakdown for the on-device half — `capture`
(drawImage), `encode` (toBlob), `render` (overlay and HUD) and `feedback` (TTS and haptic
dispatch) — with the network round trip between `encode` and `render` measured separately. It is
zero-overhead by design: one array push onto a bounded 100-sample buffer, no timers, nothing
allocated on the hot path. The aggregate ships every five seconds piggy-backed on the RTT report,
so the full end-to-end breakdown — four client stages, network, six server stages — is visible in
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

The design system is a hand-written `global.css` of roughly 3,700 lines with no framework: dark
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
using assignment instead of comparison (`c.status='verified'`), so every contact renders as
verified and pending contacts lose their "verify now" button — the highest-value
single-character fix in the client. **The alignment gate fails open**: `useOrientation`
initialises `beta` to 90°, so a device that never fires an orientation event — no gyroscope, or a
denied iOS motion permission — reads as permanently aligned and is never told to straighten the
phone. **Hebrew class-name maps exist in six copies** — a complete 14-class map in the feedback
service, four page-local 12-class copies still carrying `bus` and `truck` (classes the server
never emits) while missing `bollard`, `crosswalk` and `scooter`, and a 10-class map in Settings —
so several screens render newer classes as raw English, and every retrain requires editing six
files instead of one shared map. **The timezone fix is not applied everywhere** — three pages
still call `new Date(ts)` directly. **The health service's body comments still describe a stale
250/400/600 ms scheme** although its header and constants are now 100/150/200 ms, and **the red
state does not stop scanning** — the watchdog's automatic stop is not wired to anything, which
the code now states honestly rather than announcing a stop that does not happen. **Admin routes
are guarded only server-side**: the client hides the links but any authenticated user can open
`/admin/*` URLs — harmless, since the server enforces, but untidy. **An administrative page for
the runtime streaming configuration exists as a file but is unreachable** — it is not routed, no
navigation links to it, and the three service functions it imports were never written, so it
would fail on load; it is the client half of the unfinished feature described in §3.4.7.
Similarly, service functions for **changing a password and clearing all history** exist with no
interface calling them. And there is **no error boundary**: an exception inside a page unmounts
the React tree and leaves a blank screen, which for this user is a completely silent failure —
the most important missing robustness feature in the client.
