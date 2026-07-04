---
title: "Why pretrained SmolVLA couldn't even lift the arm — a normalization frame mismatch"
date: 2026-06-21
authors: ratel
excerpt: "The normalization statistics were baked into a different robot's (SO-100) coordinate frame, so coordinates jumped silently. Swapping one set of stats moved the chunk start-point residual from 232.7 to 105.8."
homeCover: "/diagrams/02_norm_frame.svg"
metric: "232.7 → 105.8"
tags:
  - SmolVLA tuning
  - SmolVLA
  - normalization
  - SO-101
---

> This is a series documenting the tuning process I went through bringing SmolVLA onto a real SO-101 robot (2/9). I've kept what I confirmed through measurements and commits separate from my own opinions, sentence by sentence.

## TL;DR

I ran the public pretrained policy `lerobot/smolvla_base` zero-shot on an SO-101 real-robot dataset we collected ourselves, and the arm barely moved. In the offline chunk comparison the error was concentrated in just two joints (shoulder_lift, elbow_flex), with MAE 203° and 127° respectively, and in the online rollout the elbow hyperextended into a region absent from the demonstrations, driving the forearm down below the table plane (SR 0/1, stopped by the user).

The cause was that the action normalization statistics (mean/std) were baked into a different robot's (SO-100) coordinate frame. The policy output is a value in normalization space, so the statistics used to recover it are exactly what define the robot coordinate frame. If the statistics belong to a different rig, the same output silently jumps coordinates when decoded into ours.

The fix was simple. I replaced the unnormalize statistics, using our own dataset's statistics instead of the pretrained ones (so100-blue), via the flag `UNNORM=ours`. The weights and inputs were left untouched.

By the numbers, the chunk start-point residual L2 dropped **232.7 → 105.8**, shoulder_lift MAE dropped **203° → 76°**, and wrist_roll MAE dropped 21.6° → 4.9°. The elbow, however, only went from 127° → 105.6° with the mean correction alone, leaving a residual, because of the std ratio (ours is about 3x wider).

---

## 1. Symptom — what didn't work

To set the background in one line: SmolVLA is a policy that takes camera images and robot joint state and emits 50 steps of action at once. "zero-shot" means using the public weights as-is, with no additional training on our robot data.

We collected 5 pick-and-place episodes (`so101_pnp_tray_v0`) with the SO-101 arm by hand (teleop). Treating that as ground truth (teleop GT), we fed the same inputs into pretrained SmolVLA and compared the output chunks (offline replay, robot not actuated). Then, with a safety guard attached, we also ran it on the actual arm (online rollout).

The offline comparison results (`smolvla_real_so101_pnp_replay.json`) were as follows.

- The chunk start-point (i0) residual L2 was **232.65**. That's about 2.2x the comparison value (107.5, note §1) we measured the same way in M2 on a public set close to the SO-100 training distribution.
- The failure was asymmetric, concentrated in two joints. The per-joint MAE was 203.2° for shoulder_lift and 127.1° for elbow_flex, while the remaining arm joints (shoulder_pan 14.4°, wrist_flex 14.1°, wrist_roll 21.6°) stayed at the 14–21° level.

Here is what happened in the online rollout (`smolvla_rollout_so101_v0.json` + notes §5·§6).

With our-stats remap plus a relative-target clamp (5°/step), 15Hz, and a condition of executing only the first 10 of the 50-step chunk before re-predicting, it completed 45 chunks / 450 steps, and at first it looked like a collision-free "hover" by the encoder metrics (`v0` JSON). But the next day (2026-06-12), comparing the `.rrd` command time series against the GT envelope revealed that the elbow had hyperextended to 139° (GT max 83°+56°), a "ground-digging" where the forearm drove down below the table plane (note §6). SR was 0/1, and as the motors repeatedly stalled against the table, the risk of damage led the user to stop the experiment.

> The 139° command maximum and the per-joint command ranges are descriptive values from note §6, extracted from the `.rrd` (an untracked video log). The tracked rollout JSON does not store the command time series, so I could not directly re-verify them from the JSON.

Before seeing it on screen — that is, when looking only at the encoder metrics — I misread this behavior as "hover (hesitation)." It was actually "ground-digging." In my opinion, visual observation should have come before the encoder metrics.

---

## 2. Cause — why it happened

### Normalization space and the recovery formula

SmolVLA's `predict_action_chunk` emits a tensor `(B, 50, 6)` in normalization space. The formula to convert this back into actual robot angles is simple.

```
robot coordinate value = normalized_output * std + mean
```

Here `mean`/`std` are the statistics of the data used during training. In other words, this mean/std defines which robot coordinate frame you're in. The normalized output is just a unitless number on its own, and which coordinates it resolves to depends entirely on the statistics used for recovery.

### The statistics belonged to a different robot

In note §2, directly comparing the action means of three sources, `smolvla_base` had been trained by normalizing actions to the SO-100 rig distribution, and the recovery statistics were baked into `so100-blue.buffer.action` (confirmed by the script output `unnorm_stats_key="so100-blue.buffer.action"`). Those statistics had a lift mean of +125.7°, an elbow mean of +125.4°, and a roll mean of -106.5°.

But in our SO-101 demonstrations, the actual operating center of those joints was lift -27.6°, elbow +22.4°, roll -4.6°. Because the recovery formula forces the SO-100 means, the lift/elbow results are off by more than 100° versus GT. The other joints happened to have similar frames across the two rigs and stayed fine at 14–21°, which is why the failure came out asymmetric, concentrated in just two joints.

The point I want to make here is that this is not "was our calibration wrong?" Our rig's operating range (lift -27.6°, elbow +22.4°) was in the same region as the SO-101 community datasets (note §2 comparison). In my view, the non-standard side is not our setup but the frame of the statistics the model carries.

The output a policy produces is, on its own, a unitless number, so which mean/std you multiply to recover it is exactly what fixes the robot coordinate frame. So even a single identical output decodes to a wrong joint angle when resolved with SO-100 statistics, and back to its correct place when resolved with our data statistics. The figure below shows this split at a glance.

![Coordinate jump caused by the difference in normalization recovery statistics](/diagrams/02_norm_frame.svg)

*Even for the same normalized output, recovering with SO-100 statistics makes coordinates jump and the start-point residual becomes 232.7, while recovering with our data statistics brings it back to place and the residual drops to 105.8.*

### Moving the mean doesn't fix everything (the secondary cause)

Our elbow std is 53.6°, about 3x the pretrained rig's elbow std (18.2°, a descriptive value in note §6). A linear mean/std remap matches the mean (first moment) but cannot fix the variance ratio. An output the model intended as "about z ±2" decodes into a ±107° region when recovered with our std, so the z-scale gets over-amplified.

> The pretrained rig elbow std of 18.2° is a value described in the note body. I could not directly re-verify it against a separate statistics file in the repo. The roughly 3x ratio against our 53.6° is consistent with the note, and this post's conclusion (that the variance ratio is the secondary cause) depends only on that ratio.

There are tertiary causes too — joint correlations and the visual domain gap (the demonstration manifold itself differs, note §9) — but this post focuses on the primary and secondary.

---

## 3. What I changed

I didn't touch the weights or the inputs. I changed just one set of statistics used for recovery.

The file is `workbench/so101/scripts/svla_real_replay.py`, and the flag is the environment variable `UNNORM=ours`.

Instead of the `so100-blue.buffer.action` statistics baked into `postprocess.steps`, it reads mean/std from our dataset's `meta/stats.json["action"]` for recovery (the output key changes to `OURS:so101_pnp_tray_v0`).

```python
# When UNNORM=ours: recover with our dataset statistics
stats = json.load(open(DS_ROOT / "meta/stats.json"))["action"]
action_mean = torch.tensor(stats["mean"], ...)
action_std  = torch.tensor(stats["std"],  ...)
pred = pred * action_std + action_mean   # our frame instead of SO-100 statistics
```

I also added two safety guards.

- `ENV_CLIP=1` clips predictions to the demonstration min~max (the action min/max in stats.json) and records the per-joint clip activation rate (`clip_fraction_per_joint`). This becomes the "out-of-support fraction" metric in the next section.
- For the online rollout (`svla_rollout.py`), on top of `unnorm=OURS` I added a relative-target clamp (`max_relative_target=5°/step`, as of v0), 15Hz, executing only the first 10 of the 50-step chunk before re-predicting, and a 40-second auto-shutoff.

Commit `bf9ca5e` (2026-06-12) added three safeguards to `svla_rollout.py`. (1) GT envelope clip, (2) `Present_Load` overload auto-stop, and (3) torque-off verification at shutdown. These guards are a prerequisite before trusting real-robot output, so I kept them in place from this normalization validation all the way through. The detailed design of the safety guards — how they work and the deg/s invariant and so on — is covered in [part 6](./smolvla_tuning_06_safety_clamp.md).

> Keeping the contexts separate: the teleop pan ~25° skew was a separate problem where the homing pose from the initial calibration was recorded with one arm rotated ~25°off (note/setup §9). It was fully fixed by recalibration and is unrelated to this normalization issue.

---

## 4. Before/after comparison

With the statistics swap (remap) alone, the joints whose means then matched recovered sharply. The elbow still had the std-ratio distortion left, so it took adding the envelope clip to halve it. The numbers below are values directly cross-checked from the three report JSONs (`_replay` / `_unnorm_ours` / `_envclip`).

| Metric | Before | After | Source |
|---|---:|---:|---|
| Residual L2 i0 (chunk start) | 232.7 | **105.8** | `_replay.json` → `_unnorm_ours.json` |
| Residual L2 i49 (chunk end) | 266.0 | 173.3 | same |
| shoulder_lift MAE (°) | 203.2 | **76.1** | same (per-joint) |
| wrist_roll MAE (°) | 21.6 | **4.9** | same (-77%) |
| elbow_flex MAE (°) — remap only | 127.1 | 105.6 | same (-17%, residual) |
| elbow_flex MAE (°) — remap + envelope clip | 105.6 | **51.3** | `_unnorm_ours.json` → `_envclip.json` (-51%) |
| Residual L2 i49 — remap → remap+clip | 173.3 | 127.9 | same (-26%) |

To quantify the risk: in the envelope-clip report (`_envclip.json`), elbow_flex's clip activation rate (out-of-support fraction) was 0.834. That is, 83.4% of the elbow predictions were outside the demonstration range (`clip_fraction_per_joint`). This is the quantification of the extrapolation behavior produced by the roughly 3x std amplification.

The same picture showed up in the online rollout (note §6, extracted from the `.rrd` command time series). The GT elbow range was -51~83°, but the rollout commands swung as far as -46~139° (exceeding the GT max by +56°).

i0 (the start) recovered well, but i49 (the end, late open-loop) and the real-robot collision remained. Good offline metrics are no guarantee that online is safe, in my view.

> The per-joint MAE, clip_fraction, residual L2, and unnorm_stats_key values in the §4 table were directly cross-checked from the three report JSONs above (the aggregate residual L2 means were confirmed to match each report's chunk_records mean). Things like the online rollout's latency or the clamp activation counts are values from the note body, so I don't assert them in this post (see original note §5).

---

## 5. If I were applying this at a company

These are the checkpoints for evaluating/deploying a pretrained policy trained on a different form factor/rig.

1. Check the source frame of the action statistics first. In my view, mean/std is not mere normalization metadata but a first-class variable that defines the robot coordinate frame. The same checkpoint swings between SR 0 and meaningful behavior over a one-line statistics difference (SO-100 vs our dataset). If you only report SR, a frame mismatch gets mis-logged as a "model failure." Remapping to the target rig's statistics is the top priority.

2. For residuals the mean (remap) doesn't resolve, suspect the variance ratio. A linear mean/std correction only moves the mean and cannot fix the scale distortion. Our elbow's std is about 3x, so 105.6° remained even after the remap. Look at the std ratio (z-scale) alongside.

3. Adopt 'out-of-support action fraction' as a standard diagnostic metric. It refers to the fraction of predictions outside the demonstration min~max. Even two policies both at SR 0 can be distinguished by this metric into a 'dangerous extrapolating type (elbow 83% OOS)' and an 'in-range underperforming type,' I think. The response is completely different depending on which one it is.

4. Pre-install the safety guards before going to real hardware. Make the envelope clip, the relative-target clamp (5°/step), and the load (`Present_Load`) auto-stop the defaults. And prioritize visual observation over the encoder metrics. After all, our 'hover' misreading was actually 'ground-digging.'

I view cross-embodiment zero-shot failure as not a defect in our setup but a structural limitation of the whole field. The standard solution is small-scale fine-tuning (e.g. 50ep) on the target rig, so I'm handing off the elbow z-scale residual that remap and clip couldn't fix to the next (part 4) fine-tuning path.

> The statement that cross-embodiment zero-shot ≈0% is a field consensus is a citation of external literature (SmolVLA paper, VLA survey 2508.13073, VLA-Pilot 2511.14178, etc.), not an internal repo measurement (note §7.1, based on a 2026-06-20 web search).

---

## Series nav

- Previous: [The SmolVLA real-robot tuning journey — from a policy that couldn't even lift the arm to smooth pick-and-place (series map)](./smolvla_tuning_01_intro.md)
- Next: [Trained with two cameras, yet the policy froze — the camera slot trap](./smolvla_tuning_03_camera_slot.md)
