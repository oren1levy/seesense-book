# AMBIGUOUS — things that don't add up, for Shir to decide

Everything here was found while reconciling the book against the code (2026-08-06, re-scanned
2026-08-07 after commit `6d22728`). These are not missing numbers (that's `TODO_NUMBERS.md`) —
they are contradictions or open decisions. The book currently takes the position noted in
*italics*; change the text if you decide otherwise.

## From the 7 August re-scan — read these first

0a. **Three commit messages describe fixes that are not in the code.** `85747b7 "Fix
    assignment-instead-of-comparison hiding the verify now button"` — `EmergencyContacts.jsx:212`
    is still `{c.status='verified' ? (`, unchanged. `d4ea3cc "Wire up change-password and
    clear-history, which had no UI at all"` — the two `userService` functions were added but no
    page imports either of them. *The book still lists all three as open bugs.* Worth checking
    whether work was lost in a merge, because the intent clearly existed.

0b. **A whole feature is built on both sides and connected on neither.**
    `Server/services/stream_config_service.py` (persist input size / compression / pipeline depth,
    with limits and clamping) and `Client/src/pages/AdminStreamConfig.jsx` (a complete Hebrew
    admin UI with live-vs-default display and a level-2 permission gate) both exist. But nothing
    calls `load_stream_config()` at startup, there is no endpoint, the page has no route, and the
    three `adminService` functions it imports were never written. *The book describes it as
    unfinished scaffolding in §3.4.7, §3.5.8 and §6.3.* Decide before the defense: finish it
    (roughly an hour), or leave it and keep the honest description. Do not describe it as working.

0c. **The GPU server-latency figure is unsettled — three sessions, roughly 2:1 apart.**
    `DEPLOYMENT.md` (5 Aug): 28 ms average, ~130 ms end-to-end. `main.py` comment: healthy band
    26–36 ms. `streamConfig.js` (6 Aug): 16.4 ms with the depth sweep the shipped depth-6 was
    chosen from. *The book uses 16.4 ms as the operating point and now states the spread openly in
    §4.4.2 with a 🔴.* One clean measurement settles it — this is TODO B.3.

0d. **Both repo docs are now behind the code.** `SERVER.md` still documents the frame-based
    tracker, per-frame write threads, the superseded danger rules, five stages, twelve e-mail
    templates and Railway-only deployment. `CLIENT.md` still documents 512 / 50 % / depth 5 and
    the old 22 FPS / 216 ms table. On every one of those points **the book is now more correct
    than the repo docs** — do not "correct" the book back against them. If anyone regenerates
    these docs before submission, re-check the book against the code, not against the docs.

## Decisions needed

1. **Which deployment is "the system"?** `DEPLOYMENT.md` calls Railway "the submitted baseline /
   safety net" and the GCP GPU VM the platform "for thesis measurements". *The book now presents
   the GPU VM as primary and Railway as the baseline.* Confirm this is the story you want to
   defend — and which URL, if any, goes in the book.
2. **Headline runtime numbers have no saved artefact.** ~16.4 ms/frame, R₀ ~120 ms, ~50 FPS,
   ~120–130 ms end-to-end all come from a code comment dated 2026-08-06 and `DEPLOYMENT.md`
   prose. *Used, but flagged 🔴 in §4.4.2.* Capture the dashboard measurement (TODO items 3–5).
3. **The red watchdog state no longer stops scanning.** `healthService` says so explicitly; the
   dashboard's disconnect callback only logs. Deliberate, or a regression while rewiring? *Book
   states monitoring-only and lists the stop as unwired.* If you re-wire it, update §2.7, §3.5.7,
   §3.5.8, Table 5.1, §6.2, §6.3.
4. **Sustained-danger re-announce every 2 s** (Dashboard) bypasses the `alert_is_new` gate. It
   looks deliberate (an ongoing threat shouldn't fall silent) but it partially contradicts the
   "audio fires only on change" thesis. *Book describes it as a deliberate exception.* Confirm.
5. **Server vocabulary vs model vocabulary.** `config.py` still carries the legacy 14-class list
   (including `pothole`, never trained) and `HIGH_RISK_CLASSES` includes `pothole` while lacking
   `curb`/`manhole`/`construction`. Fix the code before submission, or keep the book's "13 of 17
   announced" framing? *Book documents the mismatch as a known limitation.* Note the demo
   implication either way: nothing the model finds as curb/trash_can/manhole/construction will
   ever be announced during the defense.
6. **Defense timing.** `DEPLOYMENT.md` says defense Friday 2026-08-07 and submission Sunday
   2026-08-09 — i.e. tomorrow and in three days. If that's right, items 1–2 above and TODO A.1
   (the test-split run) are the only ones that plausibly fit in the time.

## Contradictions found and how they were resolved

7. **Two different motion-tracker descriptions existed in the book** (a 4-frame window with
   1.10/1.25 ratios in §3.4.3 vs a 0.6 s window with 1.22×/1.08× hysteresis in §2.4). The code
   is a third thing: least-squares trend on √area over 0.8 s, growth+SNR hysteresis, 0.30 s
   confirm. *Both sections rewritten from the code.*
8. **Thread caps.** The book said "capped at 8 before import"; `main.py` now says the opposite
   (caps removed after the GPU migration, with measured numbers). *Book tells both eras.*
9. **`(0,0)` SOS fallback** was in the book; the code deliberately sends no coordinates and its
   comment mocks the Null-Island link. *Book corrected.*
10. **Per-frame daemon-thread DB writes** were in the book; the code batches via `db_writer`
    (1 s flush) and documents the old way as the bug. *Book corrected.*
11. **Client constants**: book said 50 % compression / 512 / depth 5; code says 75 % / 640 /
    depth 6. *Book corrected; the 512 story kept as CPU-era history.*
12. **Val instances 180,736 vs 180,752 boxes**: Ultralytics silently removed 16 duplicate labels.
    *Explained in §3.3.5 — informational, no action.*
13. **Hebrew class maps**: book said four drifted copies missing the new 17-class names; reality
    is six copies, none of which (nor the server) knows the four new classes, and four pages
    still carry dead `bus`/`truck` entries. *Book corrected.*

## Smaller oddities (no book change made — fix in code or ignore)

14. `MOTION_WINDOW_SEC = 0.6` in `motion_tracker.py` is dead — the real window is
    `APPROACH_WINDOW_SEC = 0.8`. Delete the dead constant to avoid future confusion.
15. `tests/test_ws.py` ships a real JWT with a real email address. Rotate/remove before the repo
    is submitted.
16. `POST /settings/reset_settings` writes defaults but doesn't refresh the live cache, unlike
    every other settings write — a live session won't see a reset until reconnect.
17. `vite.config.js` hardcodes a personal ngrok hostname in `allowedHosts`.
18. Admin routes are not gated client-side (`ProtectedRoute` checks login only); Settings hides
    the link via `is_admin` while Dashboard checks `admin_level >= 1`.
19. The notebook's final report prints `Model: yolo26m.pt` and the run folder is named
    `YOLO26m_640_Baseline`, but the trained network is YOLO26s — already explained in §3.3.5;
    just don't screenshot that report cell into the book.
20. The dataset zip is named `dataSetV26.zip` while the folder is `Data-25-07` — harmless, but
    worth one consistent name if it appears in the book.
21. The title page used to say `github.com/OmerHelfer/SeeSense` while the README, reference [45]
    and Appendix D say `github.com/oren1levy/SeeSense`. *Unified to oren1levy* — confirm that is
    the repo you're submitting.
