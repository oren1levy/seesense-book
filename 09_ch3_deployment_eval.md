## 3.6 Deployment and Evaluation Methodology

### 3.6.1 Deployment

The server ships as a Docker image from `python:3.11-slim`, with `libgl1` and `libglib2.0-0`
installed explicitly because OpenCV's wheel links against them and the slim base does not include
them. `requirements.txt` is copied and installed **before** the project source so Docker's layer
cache is not invalidated by every code change. The container starts with

```
sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"
```

in **shell form**, so the platform-injected `$PORT` is expanded at runtime rather than passed as a
literal string. One `.gitignore` detail matters operationally: `*.pt` is ignored **except** the
deployed weights, which must ship inside the image — a container that downloads its model at
startup fails in exactly the circumstances where you most need it to start.

The system runs in **two parallel deployments** against the same MongoDB Atlas database,
configured entirely by environment variable (`SECRET_KEY`, `MONGODB_URI`,
`EMAIL_ADDRESS`/`EMAIL_PASSWORD`, optional extra `CORS_ORIGINS`, the injected `PORT`). The
**Railway container** is the stable baseline: CPU-only, built from the Dockerfile above,
auto-deploying on every push, with the client build pointing at it and deriving the WebSocket URL
from the API URL. The **primary deployment is a Google Compute Engine VM** with an NVIDIA T4
GPU — provisioned in `europe-central2` (Warsaw) because the Tel Aviv region had no GPU capacity
at the time, a fact with a measurable cost: roughly 90–100 ms of end-to-end latency is
Warsaw↔Israel network distance that a local region would remove. The VM runs no container: a
`systemd` unit runs Uvicorn directly on port 443 with a real Let's Encrypt certificate, obtained
without purchasing a domain by deriving a hostname from the VM's IP through `nip.io`. It is a
**single-service architecture** — the same FastAPI process serves the built React client, so
there is one origin, one port, one certificate and no CORS in production; the client is built in
a dedicated mode whose empty API URL means "use my own origin". The model weights travel with
`git clone`, since the deployed checkpoint is committed to the repository; secrets live in a
gitignored `.env` recreated per machine. A pair of deploy scripts makes an update a one-command
operation from any teammate's machine: pull, rebuild the client only if it changed, reinstall
dependencies only if the requirements changed, restart the service. One provisioning lesson is
worth recording: `pip install torch` installs the CPU-only build by default, and the resulting
system *works* — at ~200 ms per frame instead of tens of milliseconds, silently. The GPU build must be
installed explicitly, and "is it actually on the GPU" became a deployment check rather than an
assumption.

**Device testing requires HTTPS.** `getUserMedia`, `DeviceOrientationEvent` and `geolocation` all
require a secure context; a phone cannot reach a laptop's `localhost`, and a plain-HTTP LAN
address does not qualify. Real-device testing therefore ran through an **ngrok HTTPS tunnel** with
the hostname registered in Vite's allowed hosts. This small operational detail has a large
consequence: **nothing about the camera, gyroscope or SOS location can be tested in a desktop
browser on localhost**, so every capture-path change required a real phone and a tunnel — an
iteration loop substantially slower than a normal web project's, worth planning for from day one.

### 3.6.2 Evaluation methodology

The system was evaluated along three axes, because "does it work?" decomposes into three questions
with three different kinds of answer.

**Quantitative model evaluation** with `model.val()` on held-out splits: Precision and Recall
overall and per class; **mAP@0.50** as the lenient metric; **mAP@0.50:0.95** as the strict
model-selection metric used to select the best checkpoint from the run; **per-class AP** inspected
separately and deliberately, because the mean hides a collapsed rare class; **confusion matrices**
both normalised and count-based, since the count-based version reveals frequency effects that
normalisation removes; **precision-, recall- and F1-versus-confidence curves**, used to choose the
operating threshold rather than accepting a default; and **ground-truth versus predicted instance
counts** per class, a cheap diagnostic revealing systematic over- or under-prediction that mAP
alone does not surface (§3.3.4).

**Test-set discipline.** Following the supervisor's instruction at the fourth review meeting, once
we were satisfied with the composition of the test split and began evaluating on it, **the
algorithm configuration was frozen**; hyperparameters, augmentation, thresholds and architecture
were selected on the *validation* split only. This was adopted specifically because of the
Stage III mistake (§3.3.4), where oversampling was applied to the test split and thereby changed
the distribution the comparison was measured against. Repeatedly evaluating on a test set and then
tuning against the result is leakage: the number stops being an estimate of generalisation and
becomes an estimate of how much the test set has been memorised through the experimenter. Three
rules followed: the test split is touched once, at the end; data added later is distributed across
all three splits; and any change after a test evaluation invalidates it and requires a re-run,
reported as such. For Stages II and III that final touch happened and is reported in Table 4.1;
for the delivered Stage IV model it has not yet happened, which is the book's single largest open
item (§4.3).

**Qualitative evaluation.** Numbers on a held-out split from the same distribution answer a
narrower question than they appear to: our test split is drawn from the same academic and
contributed sources as the training data — well-lit, well-composed, deliberately photographed —
while the real input is a frame from a phone held by a walking person who cannot see what they are
pointing it at. The protocol therefore has two parts: side-by-side ground truth and predictions on
sampled validation images, to check that boxes are on the right objects and to observe behaviour
under occlusion and shadow; and, per the supervisor's instruction from the very first meeting,
**real street photographs captured by the team** in the actual deployment environment. That is the
genuine generalisation test, and it is where domain shift shows up.

**Systems evaluation** was continuous and in production rather than benchmarked once: per-stage
latency on both sides (four client, six server stages), round-trip time measured both by pairing
results with send timestamps and independently by the health watchdog, four distinct FPS measures
(§4.4.2), throughput over a rolling ten-second window, and success and failure counts persisted
per minute for up to 400 days. Crucially all of it was measured **on the deployment target**, not
on a development machine — the most expensive lesson of the project (§5.3) being that the two
can differ by a factor of 75.

**What we did not evaluate**, stated explicitly because the gap matters: **no user study with
blind or visually-impaired participants**. Every usability claim in this book — that alert
deduplication makes the system tolerable, that the Hebrew announcements are clear, that a
single-tap SOS is the right affordance — is a design argument supported by reasoning, not an
empirical finding. There is likewise **no formal safety or clinical validation**: SeeSense is a
research prototype, not certified and not suitable as a sole navigation aid, designed to
complement a cane rather than replace one. And there is **no controlled comparison against an
alternative detector** on the same test split; YOLO26 was selected on architectural and practical
grounds rather than by a bake-off.
