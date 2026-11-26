# Measuring the controller

Every change to recognition is measured on the same recorded clips before and
after the change. Only numbers produced this way go into the README, the
changelog or any write-up. If a change has not been measured, describe what
was changed, not that the problem is solved.

## What is measured

| Metric | Definition | From |
|---|---|---|
| Tracking loss % | Frames with no hand detected ÷ frames in which the hand should be visible (the whole clip minus any `absent` intervals) | replay |
| Dropouts / min | Separate runs of lost frames, per minute | replay |
| Idle false triggers / min | Swipes detected in *idle* clips, where no command was intended, per minute | replay |
| Scripted hit rate % | Intended gestures detected with the right direction **and** finger count ÷ intended gestures | replay |
| Wrong loop / direction | A swipe fired in place of an intended one but with the wrong finger count or direction | replay |
| Missed gestures | Intended, nothing fired | replay |
| Scripted extras / min | Fired in a scripted clip with nothing intended | replay |
| Software latency (ms) | From reading the camera frame that triggered an immediate loop start to the `play()` call | live log |

A **swipe** is what `GestureEngine.detect_swipe` emits. It requires a target
(1–5) that held for 0.12 s while the hand moved slower than swipe speed; the
target is fixed when the stroke reaches swipe speed. The stroke needs observed
horizontal travel of at least 9% of image width within a 0.45 s window,
smoothed speed of at least 0.35 image widths/s, at least 0.06 s of motion with
multiple moving samples, and the cooldown to have elapsed. A stroke emits at
most once. Every emitted swipe is counted, whether or not it changed a loop.

Since 2026-11-26 the 0.8 s cooldown starts only after a swipe that started,
queued, stopped or cancelled a loop, so a no-op swipe (such as a wind-up in the
opposite direction) does not block the swipe that follows. A stroke that began
during the cooldown never fires, however long it lasts. A left swipe on a
queued loop cancels its start; a right swipe on a stopping loop keeps it
playing. Counts can therefore depend on which loops were playing. The original
baseline used the instantaneous finger count, a 45 px/frame threshold and a
cooldown after every swipe.

Scripted clips are scored by aligning the intended sequence with the detected
one (edit distance). Timing is not checked, only order.

## What these numbers do not cover

- **Replay is not live.** Replay processes every recorded frame. Live, the app
  reads frames only as fast as it can process them, so a slower pipeline can
  score the same in replay and worse live. For every change also check the live
  frame interval (`bench.py latency` prints it).
- **Clips are compressed** (`.mp4`, mp4v). `.avi` records MJPG at higher quality
  and larger size. Before and after see the same files, so comparisons stay fair,
  but replay can differ slightly from what the camera delivered live.
- **Software latency is not gesture-to-sound latency.** It leaves out:
  camera capture (exposure → frame delivered to the app); audio output
  (`play()` → sound from the speakers: pygame's 512-sample buffer is about
  11.6 ms at 44.1 kHz, plus the device); and the time a swipe takes before its
  smoothed velocity crosses the threshold. Since 2026-11-26 an immediate start is
  played in the frame that triggered it; before, it was played on the next
  frame's queue check, which added about one frame interval.
- **Queued loops** wait for the master-loop boundary by design; latency does not
  apply to them.

## Recording protocol

Keep the laptop, camera, its position and the resolution (default 1280×720)
the same for every session. Record in demo mode; sound is not needed.

```bash
python main.py --demo --record recordings/s1-normal-idle.mp4    # press q to stop
```

Each recording writes `NAME.mp4` and `NAME.frames.csv` (capture time of every
frame). Replay uses those times, so cooldowns and loop boundaries see the
recorded timing. `recordings/` is git-ignored: the clips show your face.

**Clip kinds** (about 60 s each)

- `idle` — hand in view, doing things that are not commands: holding up
  fingers without swiping, pinching, drifting slowly, turning
  the hand, reaching for the keyboard, nodding along. Every detected swipe is a
  false trigger.
- `scripted` — the fixed sequence `R1 R2 R3 L1 L2 L3 R4 R5 L4 L5 R1 L1`
  (R/L = swipe right/left, digit = fingers shown), one gesture about every 4 s,
  at a natural speed. If you perform a gesture wrongly, either re-record or set
  `expected` to what you actually did.
  With R1, hold the fingers briefly until `TARGET: LOOP N` shows the intended
  number, then swipe. Keep this preparation consistent when recording new clips.

**Conditions**

- `normal` — usual room light.
- `lowlight` — main lights off, one dim source (for example the screen only).
  Write a light reading (phone lux meter) in `notes`.
- `fast` — the scripted sequence as quick flicks, normal light (motion blur).
- `occlusion` — hand partly hidden: holding an object (phone, drumstick), the
  other hand crossing in front, or the hand partly out of frame at the edge.
  Put any stretch where the hand is fully out of frame in `absent`
  (`start-end` seconds from clip start, `;`-separated).

Record the full set in at least three separate sessions (different days or
times of day). Name clips `s<N>-<condition>-<kind>.mp4`.

**Manifest** — copy `bench/clips.example.csv` to `recordings/clips.csv` and
list every clip: `clip,condition,kind,expected,absent,notes`.

Once the baseline is measured, do not edit or delete clips. If you add clips,
re-run the baseline commit on the enlarged set before comparing.

## Commands

```bash
# Baseline: the commit that added this tooling, before any recognition change
git tag measure-baseline <commit>
python bench/bench.py replay --label baseline

# After each change (commit first, so the summary records the commit)
python bench/bench.py replay --label lowlight-v1
python bench/bench.py compare results/baseline results/lowlight-v1

# Re-run the baseline later on a larger clip set
git worktree add ../gmc-baseline measure-baseline
python ../gmc-baseline/bench/bench.py replay --label baseline \
    --manifest recordings/clips.csv --out results/baseline

# Software latency, live (separate run from recording; encoding adds time)
python main.py --log recordings/live-s1.jsonl
#   start a loop from silence (swipe right), stop it, wait for the clock to
#   reset, repeat ~20 times
python bench/bench.py latency recordings/live-s1.jsonl
```

`results/<label>/summary.json` records the commit, whether the tree was dirty,
per-clip numbers and per-condition totals. Per-frame logs stay in
`results/<label>/logs/` (git-ignored).

## End-to-end gesture-to-sound latency (not done yet)

Software latency misses the camera and the speakers. To measure the whole
chain, film the hand and the laptop with a phone (60 fps video with sound),
use a loop whose first sample is a sharp attack, and do about 20 immediate
starts as above. For each, take the first frame where the wrist starts the
swipe and the first frame where the loop's attack shows in the audio track.
Report the median and the resolution (±17 ms at 60 fps).

## R1: failed baseline attempt and development diagnostics

On 2026-11-06 the user reported that swipes could not be triggered reliably,
palm-facing finger counts were wrong, and the number disappeared during fast
movement. The scripted sequence could not be completed. The user requested
evidence preservation and an immediate first repair instead of completing the
full baseline protocol first. See [REPAIR-R1.md](REPAIR-R1.md).

The two frozen clips are **unlabelled development diagnostics**. Do not fill in
the standard 12-action sequence just because one file is named `scripted`, or
report hit rates from it. Do not count every no-hand frame as tracking loss:
the hand is genuinely absent in parts and no complete `absent` annotation exists.

```bash
# Existing local frozen copies; each output directory must be new.
python bench/diagnose.py \
  --source-root recordings/repair-r1-evidence/before-source \
  --out results/r1-before-repeat \
  --clip recordings/repair-r1-evidence/clips/*.mp4
python bench/diagnose.py --out results/r1-after-repeat \
  --clip recordings/repair-r1-evidence/clips/*.mp4
```

This saves source/clip hashes, capture spans, event lists and per-frame landmarks
without assigning ground truth. Counts from these runs are diagnostic evidence,
not a substitute for the formal measurement protocol above. Logs and private
video remain git-ignored. The hand-only numeric regression fixtures are derived
from visually inspected development frames, so they are not independent test data.

R1 replaces screen-direction finger counting with joint geometry, and uses
timestamped, width-normalized wrist motion. A stable target is held for at most
0.22 s of tracking loss. No command fires without a detected hand, and motion
history restarts on reacquisition: a swipe that happens entirely while tracking
is lost can still be missed. Low-light, fast-motion and occlusion performance,
live frame intervals and latency still need fresh measurement.

The R1 volume follow-up locks volume by default. Press `v` for explicit pinch
adjustment; swipes are suppressed in that mode, and loss of tracking locks the
last committed volume. Raw video and frame timestamps do **not** record keyboard
events: standard replay tests the default locked-volume loop-control path. Test
the active volume mode separately with a live `--log` run and the volume regression
tests. New frame logs record `volume`, `volume_adjusting`, source hash and Python
environment so volume changes and the running version can be checked.
