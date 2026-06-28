---
title: "The policy keeps grasping to the right of the object — data diversity and memorization"
date: 2026-06-21
authors: ratel
excerpt: "The grasp always missed in the same direction. The cause was zero positional diversity in the collected data — the policy had memorized a fixed grasp pose. Diagnosed with a 0.059 spread + added a dataset_gate @check."
tags:
  - SmolVLA tuning
  - SmolVLA
  - dataset
  - evaluation
---

> This is a series documenting the tuning process I went through bringing SmolVLA onto a real SO-101 robot (9/9). I've kept what I confirmed through measurements and commits separate from my own opinions, sentence by sentence.

> **Terms in one line each**
> - **policy**: a learned model that takes camera and joint state as input and outputs the next action. Here it refers to SmolVLA.
> - **grasp pose**: the full arm joint configuration at the exact moment the cube is grasped.
> - **shoulder_pan**: joint 1 of the 6. It's the base joint that rotates the arm left/right (azimuth), and it's the only axis that distinguishes left from right for the cube.
> - **RTC**: an inference structure that swaps action chunks (50 steps per inference) asynchronously. Covered in part 8.

---

## TL;DR

- Even right after inference (RTC async) succeeded smoothly and quickly, the gripper kept missing the cube's center, always in the same direction (off to the right), trying to grasp one side of the cube. The first clue was that this was systematic, not random.
- The cause was neither the inference code nor the calibration. The positional diversity of the cube in the collected data (`so101_cube_v0`, 51 episodes) was essentially zero. The grasp-time value of `shoulder_pan`, the joint that distinguishes left from right, was clustered at a single 8°-wide point within the dataset's full range of motion (about 71°).
- When positional diversity is zero, the policy doesn't need to learn "see the cube and track it" — it just memorizes a fixed grasp pose. So at deployment it always went to the memorized azimuth (≈ −22°), and whenever the cube drifted even slightly from that spot, the grasp landed off to one side.
- The response had three branches. The diagnostic tool `demo_replay_rerun.py` automatically detects and visualizes the grasp segment; the regression-prevention gate `dataset_gate.py` gained a `@check("grasp_pose_diversity")`; and I decided to re-collect the data with the cube spread across the whole workspace.
- Key numbers, before and after: the median grasp-pose spread `arm_pose_spread_median` was **0.059** (gate WARN), and the target is ≥0.12 PASS.

---

## 1. Symptom — what wasn't working

Earlier in this series I covered putting the SmolVLA policy on an SO-101 arm and having it do pick-and-place. Through part 8 I refined the inference pipeline (the RTC structure that swaps action chunks asynchronously) so the rollout ran smoothly and quickly (the queue never starved, `queue_starved=0`). The motion was smooth, but the results, once I looked at them, were off.

The gripper tried to grasp the right side of the cube rather than its center. Not once or twice but repeatedly, and always in the same direction. That was the key clue. If the miss were random I'd suspect noise, but when the direction is consistently (systematically) skewed to one side, something is casting a bias shadow.

Another thing that threw me off was that the offline metrics looked good. Numbers like the residual against the training data or the out-of-distribution (oos) measure were fine. Yet online, the moment the object's position drifted even slightly outside the training distribution, it missed.

In my view, when you see "the motion is fine but the result is skewed to one side," it's often already not an inference-speed or smoothness problem. Smoothness and accuracy are different axes, the way I see it.

---

## 2. Cause — why it happened

First I nailed down what it wasn't. Half the diagnosis was settled right here.

I ruled out two causes. One was calibration. The follower calibration file (2026-06-11) covers both collection (2026-06-20) and the current deployment. Since the same file is used at collection and at deployment time, a left/right offset can't originate there. The other was the inference code. The part-8 RTC optimization was about action-chunk coordinate frames, latency, and the queue, and is unrelated to grasp position. The rollout ran normally with `queue_starved=0`.

That leaves the data. The only joint that distinguishes left from right is `shoulder_pan` (id 1, base azimuth). I aggregated the grasp-time `shoulder_pan` value across all 51 episodes.

The grasp-time shoulder_pan distribution was as follows.
```
mean -21.9°   std 4.2°   min -26.0°   max +6.1°
→ 50/51 grasps clustered at one point in -26~-18° (8° wide)
→ the dataset's full pan range of motion is -40~+32° (about 71°)
```

The single joint that distinguishes left from right was pinned to almost one value at the moment of the grasp. 50 of the 51 grasps fell within an 8°-wide band, which is only a small fraction (about 8% per the original note) of the dataset's full pan range (about 71°). Extending this to the other positional joints gave the same picture. The median pose spread at the grasp event (each joint's std ÷ range) was 0.059, meaning the joints each moved through only 4–9% of their full range of motion.

To sum up, the 51 demos were collected in a single session with the cube placed in nearly the same spot every time. At the moment of grasp the whole arm was in almost the same pose each time, and positional diversity was essentially absent.

![Comparison of cube positions clustered in one spot versus spread across the whole workspace](/diagrams/09_grasp_diversity.svg)

*On the left, the cube was collected almost entirely from one cell (spread 0.059 · WARN); on the right is the re-collection target of spreading evenly across the whole grid (spread ≥0.12 · PASS).*

Put simply, if the cube is always in the same spot, the policy has no reason to check its position with the camera. Since "send the arm to one memorized pose and close" already matches all of the training data, the policy memorizes the fixed grasp pose wholesale instead of learning to track position. So at deployment, when you place the cube in a new spot, it only goes to the memorized spot and grasps one side.

### Mechanism — memorization vs. visual tracking

From here on this is my own inference. When positional diversity is zero, the policy doesn't need to learn "look at the cube with the camera and track toward it (visual tracking)." Memorizing the single, nearly fixed grasp pose "go to pan ≈ −22° and close" already matches all the training data. From the training loss's point of view, memorization is the perfect answer.

So at deployment the policy always goes to the memorized azimuth (≈ −22°). If you place the cube even slightly off the training spot, the gripper still goes to −22° and lands off to one side of the cube. When the cube is on the home (+) side of −22°, the gripper bites the far side (= the right side) of the cube. That, I think, is why the direction is always the same. Aiming at a fixed azimuth means that no matter which way the cube drifts, the grasp consistently lands on the opposite side.

The good offline metrics are explained by this same mechanism. Within the training distribution, memorization is the right answer, so the residual comes out small. The failure only shows out-of-distribution — online, when you place the cube in a new position. Generalization, after all, is only verified outside the original distribution.

---

## 3. What I changed

I responded along two branches. A diagnostic tool to confirm this cause, and a gate to prevent a recurrence in the next collection. The commit is `a3521b1`.

### (a) Diagnostic tool — `demo_replay_rerun.py`

It replays the demo dataset as a rerun `.rrd`, but auto-crops to just the pick segment. The core is detecting the pick event from the gripper signal (action idx 5).

- It finds the pick segment from the gripper open/close sequence. The flow is: while approaching, a first opening peak appears (first_open, hovering over the table), followed by a close (grasp). The code detects it with an open threshold of `lo + 0.6·(hi−lo)` and a close threshold of `lo + 0.35·(hi−lo)`.
- There's one trap. The global argmax where the gripper opens widest is when it places onto the tray (release). You must not take that as the grasp, so it explicitly avoids it by taking only the first close after first_open as the grasp.
- Only the detected segment is logged with both cameras (top/wrist) and the `shoulder_pan`/`gripper` traces, with an event marker stamped on the grasp frame. On the top camera, `draw_guides` composites a vertical center line (yellow) and 1/3 guide lines so you can judge by eye "whether a person is grasping the cube's center or its side."

A small cube (2cm) gets occluded by the gripper at shallow camera angles, so static frame extraction kept failing; routing around that via the motion context (the event) is the point of this tool.

### (b) Regression-prevention gate — `dataset_gate.py`'s `@check("grasp_pose_diversity")`

I added one check to the gate that automatically inspects a dataset before handing it off to training. The design principle is that it reads joint names and DoF from `info.json` and works without hardcoding.

```python
@check("grasp_pose_diversity")
def c_grasp_div(ctx):
    # auto-detect the gripper joint by histogram end-mass
    # — don't assume we know which joint is the open/close one
    masses = [end_mass(k) for k in range(A_all.shape[1])]
    grip_j = int(np.argmax(masses))
    if masses[grip_j] < GATE["grasp_event_endmass"]:   # = 0.45
        yield _f("PASS", "no clear open/close event — check N/A (non-grasp task)")
        return
    # ... gather positional joint poses from each episode's grasp frame ...
    med = float(np.median(arm))   # = arm_pose_spread_median
    if med < GATE["grasp_spread_warn"]:                # = 0.12
        yield _f("WARN", "fixed grasp pose memorized → misses at variable positions, "
                         "recommend re-collecting spread across the whole workspace", arm_pose_spread_median=med)
    else:
        yield _f("PASS", "sufficient object position diversity", arm_pose_spread_median=med)
```

The above is an excerpt of just the core structure. The thresholds and branching logic match the source, but it's not the full function.

The design intent is as follows.

- It only goes as far as WARN and does not FAIL. For a fixed-position task, low grasp diversity is normal. FAILing would block it and become a false alarm, so whether to block is left to human judgment.
- It only runs when the open/close signal is clear. If the end-mass `end_mass` (how strongly values cluster in the histogram's two extreme bins) is below 0.45, it treats the gripper's 2-state signal as ambiguous and skips with N/A. This is so it doesn't apply to non-grasp tasks, which keeps the gate from assuming a form factor.
- If fewer than 3 episodes have a detected grasp event, that too is treated as WARN.

### (c) Direction of the fix — re-collection

I re-collect with the cube spread across the whole workspace. Left/center/right × near/far, randomly placing the cube in a different cell each episode on roughly a 5cm grid (~25×20cm). The condition for proceeding to training is that after re-collection the gate's `arm_pose_spread_median` reaches ≥0.12 (PASS, ideally ≥0.20).

---

## 4. Before and after

| Metric | before (cube_v0) | after (re-collection target) |
|---|---|---|
| grasp-time shoulder_pan (mean ± std) | **−21.9° ± 4.2°** (min −26.0, max +6.1) | spread across the whole workspace (not measured) |
| width of grasp cluster / full range of motion | **8°-wide point (50/51 grasps) / 71° total** | left/center/right × near/far spread target |
| median grasp-pose spread `arm_pose_spread_median` | **0.059** (gate WARN) | **≥0.12 PASS**, ideally ≥0.20 |
| `dataset_gate` grasp_pose_diversity verdict | **WARN** @ 0.059 < 0.12 | **PASS** after re-collection is the condition for proceeding to training |

What's quantitatively confirmed is the distribution numbers (−21.9° ± 4.2°, spread 0.059) and the ruled-out causes (calibration, inference code). The causal claim "misses to the right → fixed-azimuth memorization" is inference, and is labeled as an opinion in the original note. The actual PASS number after re-collection, and a quantification of the online miss (offset distance / failure rate), are still unmeasured; only visual observation was recorded. I also couldn't fully rule out another path, "the camera moved at deployment so the cube ended up offset from the training spot" — but either way the conclusion that the fix is the data is the same. For reference, the share of the range the grasps used is written differently across sources, "about 8%" in the original note §3 and "about 6%" in the commit/gate comments, so in the body I just wrote "8°-wide point" instead of committing to a figure.

---

## 5. If you applied this at a company

Even outside of robots, these are checkpoints you can carry over to any pipeline that runs a policy/model on training data.

1. A systematic, consistent directional bias online is not noise but a shadow of the data distribution. Rather than touching the inference code or post-processing first, it's better to aggregate the collection diversity first. In particular, look at the input position diversity at the moment the model actually makes its decision (here, the moment of grasp).

2. Good offline metrics are no reason to relax. If input diversity is zero, the model memorizes a fixed output instead of generalizing from the input. Within the training distribution memorization is the right answer, so the metrics come out good. Generalization is only verified out of distribution, so you have to test separately at new positions and new conditions to see it.

3. When static analysis is blocked, route around it with events. Here, because of the small object and shallow camera angle, pulling the grasp-moment frame as an image kept failing. Instead, picking just the pose at that moment via the gripper open/close event and computing statistics quickly surfaced the cause. In general ML terms, this is aggregating only the input distribution at the decisive moment rather than looking at all of the input.

4. A failure type you've found once is best locked into a gate. If you stop at diagnosis, it'll bite you again next time. Adding one `@check` to the data gate so it auto-verifies the next collection turns one-off debugging into a reusable asset. But set the blocking strength carefully. Some tasks legitimately have low diversity (fixed-position work), so leave it as WARN rather than FAIL and leave the final call to a human.

---

## (End) Series nav

- Previous: [It was smooth but slow — 3 RTC speed tunings (including a half-speed bug)](./smolvla_tuning_08_rtc_speed.md)
- Next: **None — this is the last post in the series (9/9).**
- Series map: [The SmolVLA real-robot tuning journey (series map)](./smolvla_tuning_01_intro.md)

Original note: `docs/learning/m7w5_grasp_diversity_recollect.md` · diagnostic tool `workbench/so101/scripts/demo_replay_rerun.py` · gate `workbench/so101/scripts/dataset_gate.py`
