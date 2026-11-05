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

A **swipe** is what `GestureEngine.detect_swipe` emits: at least one finger
extended, smoothed wrist velocity beyond the threshold, cooldown elapsed. It is
counted whether or not it changed a loop, so the counts do not depend on which
loops happened to be playing.

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
  smoothed velocity crosses the threshold. In the current code an immediate start
  is played on the *next* frame's queue check, so software latency includes about
  one frame interval.
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
  fingers without swiping, pinching to change volume, drifting slowly, turning
  the hand, reaching for the keyboard, nodding along. Every detected swipe is a
  false trigger.
- `scripted` — the fixed sequence `R1 R2 R3 L1 L2 L3 R4 R5 L4 L5 R1 L1`
  (R/L = swipe right/left, digit = fingers shown), one gesture about every 4 s,
  at a natural speed. If you perform a gesture wrongly, either re-record or set
  `expected` to what you actually did.

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

## Open questions for the first baseline

- **Thumb direction.** `count_fingers` treats a right-hand thumb as extended
  when its tip is to the right of the joint below it, in the mirrored image.
  For a palm facing the camera that may be reversed. The scripted clips'
  wrong-loop count will show whether finger counts are off by one.
- **Frame-rate dependence.** The swipe threshold is in pixels per frame. Webcams
  often drop frame rate in low light, which makes the same hand speed cover more
  pixels per frame. Compare the `fps` column of low-light and normal clips.
