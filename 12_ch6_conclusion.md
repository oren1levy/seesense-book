# Chapter 6 — Conclusion and Future Work

## 6.1 Project Achievements

SeeSense set out to determine whether a phone the user already owns, a self-trained detector and a
commodity cloud container are enough to give a blind pedestrian useful, real-time warnings about
the obstacles around them. The answer is yes, with the qualifications §6.2 states plainly.

**A dataset.** A unified, verified 17-class obstacle-detection dataset of **91,139 images and
964,837 bounding boxes**, merged from ten sources — MS COCO, Open Images V7 and Mapillary Vistas
plus nine contributed Roboflow datasets — into a single taxonomy chosen for pedestrian hazards
rather than for whatever a benchmark happened to contain. Every image and label was verified: zero
orphans in any split, every label five-column with coordinates in range, every class ID valid.
Building it required unifying ten disagreeing taxonomies, recovering 6,199 images being silently
destroyed by a format mismatch, and raising `crosswalk` from 33 images to 3,511.

**A model, and four instructive failures on the way to it.** We built a detector from scratch —
frozen ResNet18 backbone, hand-written YOLO-style head, grid encoder, four-component loss, decoder
and NMS — and established experimentally that it could not cope with dense street scenes, and that
a pipeline recording only loss cannot detect its own collapse. We established a YOLOv8n baseline at
P 0.669 / R 0.409 / mAP@0.50 0.4528 on 3,699 images and 86,381 instances; fine-tuned to 14 classes
reaching 0.862 / 0.818 / 0.853 / 0.618 on the new-class split, and diagnosed the catastrophic
forgetting of the original ten classes that came with it; improved recall to 0.843 with targeted
oversampling, and recorded the split contamination that made the gain imprecise; produced, and
correctly identified as invalid, a combined-dataset run that had trained on the wrong data; and
delivered a YOLO26-small model trained in a single campaign on the final verified 17-class dataset,
validating at P 0.754 / R 0.587 / mAP@0.50 0.636 / mAP@0.50:0.95 0.458 over 11,086 images and
180,736 instances.

**A complete product.** Not a notebook and a demo video, but a deployed client–server system: a
mobile-first Hebrew RTL web app with camera capture, gyroscope alignment gating, bounded-depth
WebSocket streaming, a live detection HUD, Hebrew text-to-speech, haptic patterns, a connection
watchdog, and a full account area — profile, per-user sensitivity and class preferences, session-
grouped detection history, a three-entry-point feedback system, verified emergency contacts and an
SOS flow — plus three administrative pages behind a three-level permission system. On the server: a
six-stage timed inference pipeline, a ByteTrack-inspired per-user tracker, motion-first danger
logic, per-track alert deduplication, presence-based clearance, JWT authentication with real
revocation, thirteen transactional e-mail templates, and a two-layer performance-metrics
subsystem.

**A measured performance story.** On the CPU-only container, server latency was brought to
approximately 41 ms per frame through a documented sequence of changes: HTTP → WebSocket, a 71 ms
per-frame database read eliminated, per-frame writes batched off the hot path, inference moved to
a worker thread, and input size reduced from 640 to 512. The delivered system then moved to a
GPU-backed VM, restoring the full 640-pixel input at roughly 16 ms of server time per frame and
sustaining about **50 FPS at ~120–130 ms end-to-end** — comfortably inside the original POC
criterion of a 200–300 ms alert, with most of the remaining budget being network distance rather
than compute. Two experiments failed usefully and are reported as such: an ONNX port 1.5× faster
locally and 75× slower in production, and a thread cap that saved the CPU deployment and had to
be reverted on the GPU one.

**A design contribution.** The most transferable result is not a number. It is the observation that
for a non-visual safety interface the hard problem is **suppression, not detection**. Running the
detector at a moderate threshold to preserve recall in the data, then gating what is *said* through
three independent filters — is it approaching, is it close or fast enough to matter, and has this
specific tracked object already been announced — is what separates a system that can be worn for an
hour from one that is intolerable within thirty seconds.

## 6.2 Limitations

Stated plainly, because a book listing only achievements is not a project book.

**`manhole` has never been evaluated** — 722 images, all in train, zero in validation and test.
**Structural class imbalance remains**: `pole` accounts for 47 % of all annotations, `manhole` for
0.07 %, and collection cannot close that gap. **Augmentation inflation**: four gap-fill sources are
~3× Roboflow-augmented, so raw counts overstate true scene diversity. **One unverified label
mapping**: in the `cross` source, class ID 0 (313 boxes) was assumed to be `crosswalk` on the basis
of the dataset's subject rather than visual inspection. **No offline capability** — the hybrid
failover is designed and its monitoring half fully implemented, but there is no on-device model to
fail over to, so loss of connectivity means loss of the system. **Domain shift is not addressed**:
training data is well-lit, well-composed photography while real input contains motion blur, glare
and extreme angles at a rate no benchmark reproduces. **Single-worker architecture**: caches,
trackers, metrics and presence all live in process memory, so horizontal scaling needs Redis. **No
automated tests** — the interactive harness is not a test suite. **No error boundary in the
client**: an exception unmounts the React tree and leaves a blank screen, which for a user who
cannot see the screen is a completely silent failure. **Known open bugs and unfinished work**:
the `EmergencyContacts` assignment-instead-of-comparison bug; the alignment gate failing open on
devices that never report orientation; six drifted Hebrew class-name maps; the timezone fix not
applied on three pages; health-service comments describing stale thresholds, and values tighter
than measured mobile RTT justifies; the watchdog's automatic stop on a red connection not being
wired; a runtime streaming-configuration feature built on both sides but connected on neither;
and service functions for changing a password and clearing history that no interface calls.

**And the most important one: no blind user has used this system.** Every usability claim in this
book is a design argument, not an empirical finding. SeeSense is a research prototype — not
certified, not clinically validated, and not suitable as a sole navigation aid. It is designed to
complement a white cane, never to replace one.

## 6.3 Future Work

**Immediate — completing what is built.** Reconcile the class vocabulary end to end: the 17-class
model is deployed, but the server still filters through the legacy 14-class list — so `curb`,
`trash_can`, `manhole` and `construction` never reach the user — and the client carries six
drifted Hebrew name maps; one shared server list and one shared client map would make a future
retrain touch two files rather than eight. Redistribute `manhole` across the splits and
re-evaluate, so the class stops being unmeasurable. Fix the known bugs, the one-character
contacts bug and the fail-open alignment gate first, and either re-wire the watchdog's automatic
stop or record the decision not to. Finish or delete the half-built runtime streaming
configuration: the service layer, the limits and the administrative page all exist, and what is
missing is a startup load, two endpoints, three client service functions and a route — an hour of
work that would turn input size, compression and pipeline depth into things an administrator can
tune against a live deployment instead of constants requiring a rebuild. Add an error boundary with a spoken failure announcement, so
a crash is audible rather than invisible. Unit-test the two pure safety functions — the alert
classifier and the motion analyser. Recalibrate the health thresholds against the measured
~120 ms baseline and correct the stale comments. And visually verify the `cross` ID-0 mapping on
a sample of images.

**Evaluation — the missing evidence.** A **user study with blind and visually-impaired
participants** is the single highest-value piece of future work, and the questions it must answer
are specific: is the alert cadence tolerable over a 30-minute walk; are the Hebrew announcements
intelligible over street noise; is the Close/Medium/Far abstraction actionable; do users trust the
"path clear" signal; and does the system change behaviour in a way that is safe. Beyond that: field
testing across night, rain, glare and crowds with success and failure documented per condition;
measured battery drain and thermal behaviour over continuous sessions on several devices; an
evaluation of the delivered checkpoint on the frozen 5,957-image test split, which has still never
been run; and a controlled comparison against at least one alternative detector architecture on
that same split, so the choice of YOLO26s rests on measurement rather than reasoning alone.

**Model and data.** Address residual imbalance with technique rather than collection: class
weighting, focal-style loss, and **copy-paste augmentation** [19] to synthesise rare-class
instances into varied scenes — the most promising untried avenue for `manhole`, `curb` and
`trash_can`. Add **motion-blur and low-light augmentation** plus a training subset captured *while
walking*, attacking the domain shift directly. Step up model size (`yolo26m`/`l`) for the
server-side model where the compute budget can absorb it — the delivered model is already the
*small* tier — keeping a nano or quantised variant for the eventual on-device path, and measure the
accuracy/latency trade rather than assuming it. Collect more data for the
five classes below the floor, distributed properly across splits. And publish a **model card**
documenting intended use, measured per-class performance, failure modes and explicit non-use cases.

**The hybrid architecture.** An **on-device model** quantised to 8-bit and exported to CoreML or
TensorFlow Lite removes the ~100 ms network term that dominates the latency budget, and is the
only route to operation without connectivity. **Complete the failover**: the watchdog already
detects degradation, so connect it to a mode switch rather than a warning, with a spoken
explanation of the reduced capability. Implement the **safety-ping protocol** from the
characterisation document — on losing connectivity, send the last known GPS position to a
verified contact, so a user who falls in a dead zone has already been located. And **migrate the
GPU VM to a Tel Aviv region** when capacity appears — the current deployment sits in Warsaw only
because no Israeli GPU capacity was available, and the ~90–100 ms of distance it adds is the
single largest removable term in the latency budget.

**Product and capability.** A **native application** (or a Capacitor wrapper) for true background
execution, reliable haptics on iOS and deeper camera control — trading the frictionless web
distribution that motivated the current choice for capability, ideally offered alongside rather
than instead of the web client. **Metric depth estimation** to replace the box-area heuristic,
converting "Close" into "two metres" and enabling genuinely distance-aware alert timing.
**Speed-aware alert timing** using the device's own motion sensors, so alert distance scales with
how fast the user is actually moving. **Learned personalisation**: the feedback system already
collects labelled corrections from real users, with a snapshot of each frame's context — that is a
training signal, and closing the loop from user feedback to per-user threshold adaptation and
eventually periodic retraining is the natural extension of infrastructure that already exists.
**Horizontal scalability** via Redis-backed shared state. And **multi-language support**, which is
cheap because the announcement layer is cleanly separated — the server emits English facts and the
client composes the utterance, so adding a language is one map and one voice-selection path.

---

SeeSense demonstrates that practical, real-time obstacle warning for blind pedestrians can be
delivered on hardware people already own, with a model a student team can train themselves, at a
latency that leaves usable reaction time. It also demonstrates, at some length, that the difficult
engineering in a system of this kind is not the neural network. It is the ten label taxonomies that
disagree with each other, the six thousand images that vanish without an error message, the
container that lies about how many cores it has, the e-mail send that freezes the whole server, and
the twenty alerts per second that make a technically correct system unusable by a human being.

Those are the parts we learned the most from, and they are why this book devotes a chapter to them.
