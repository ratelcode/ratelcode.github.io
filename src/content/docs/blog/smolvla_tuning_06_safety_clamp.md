---
title: "Safety guards for a real robot — relative-target clamp + overload auto-abort"
date: 2026-06-21
authors: ratel
excerpt: "Sending a policy's raw output straight to the motors is dangerous. A three-layer guard — envelope clip + per-step clamp + Present_Load overload abort — plus a deg/s-invariant design."
homeCover: "/diagrams/06_safety_guards.svg"
metric: "3-layer guard"
tags:
  - SmolVLA tuning
  - SmolVLA
  - safety
  - SO-101
---

> A series documenting the tuning process I went through while putting SmolVLA on a real SO-101 robot (6/9). I've kept what I confirmed through measurements and commits separate from my own opinions, sentence by sentence.

## TL;DR

- On the real SO-101, the commands SmolVLA produced went far outside the motion range it had been demonstrated. Instead of lifting, the arm drove down below the table plane and the motor went into stall; the risk of damage was visible, so the user stopped the online experiment immediately (2026-06-12).
- The cause was that before the policy's raw output reached the motors, there was none of (a) blocking out-of-distribution extrapolation, (b) limiting per-step movement, or (c) overload auto-abort — none of these three guards existed.
- So in commit `bf9ca5e` I added the three guards: GT envelope clip, per-step relative-target clamp, Present_Load overload auto-abort, plus re-verification that torque is actually released on shutdown.
- In offline pre-validation, **83.4%** of elbow commands were outside the demonstrated range, so the clip fired hardest, and elbow MAE dropped from 105.6° to 51.3°. The subsequent FT-checkpoint real-robot rollout (2026-06-21) terminated safely with `aborted=null` and `torque_off_verified=true`.

---

## 1. Symptom — what went wrong

Let me start with some background. SmolVLA is a policy that takes camera images and joint state (obs) and emits the next 50 steps of joint commands (an action chunk) all at once. Streaming those commands directly to the motors of the real 6-joint arm (SO-101) is what an "online rollout" is.

The problem was that the commands the policy produced went well outside the motion range of the data a person had collected by demonstrating directly.

In the demonstration data, the range the elbow_flex joint actually reached was -51~83°. But in the zero-shot rollout before any clip was applied, the policy commands went all the way to **-46~139°**. That's hyperextension exceeding the demonstrated maximum of 83° by 56° (`m7w2_smolvla_real_zeroshot.md` §6 table).

The result was that the forearm drove down below the table plane. It looked like digging into the ground instead of lifting the arm, and the motor went into stall against the table. The risk of damage was clear, so the user stopped the online experiment immediately (2026-06-12).

To summarize, there was not a single layer of guard verifying whether the policy output could be trusted as-is.

---

## 2. Cause — why it happened

The primary cause of the commands leaving the range is the cross-rig statistics mismatch covered in part 2 of this series.

Our data's elbow standard deviation (std) is 53.6°, while the elbow std of the rig SmolVLA was pretrained on was 18.2°. About a 3x difference. When the model intends "move by z=±2 from the mean," in our coordinate frame where the std is 3x larger that decodes into a ±107° swing. That's why extrapolation went all the way to 139°, which never appears in the demonstrations (`m7w2` §6).

A linear mean/std remap — correcting coordinates by mean and variance — only matches the first moment (the mean); it does not correct the variance-ratio (z-scale) distortion.

On top of that, because there were no guards at all, the following three things were all unprotected.

1. There was nothing to block out-of-distribution extrapolation. Even if a command went to a coordinate the demonstrations had never visited, there was nothing to stop it.
2. There was nothing to limit jumps in the step target. Jumping a large angle in one step puts a torque spike on the motor.
3. There was no overload abort and no shutdown verification. Even if a motor was blocked by something and the load spiked, there was nothing to stop it, and on shutdown there was no check of whether the torque had actually been released.

In addition — this is a part I realized later while reinforcing it in M7W4 — if you hardcode the per-step movement limit as a constant in "degrees per step (°/step)," you get a structural flaw where it's tied to the control frequency (Hz). Change the Hz and the effective speed ceiling (°/s) silently shifts. For example, leaving 8°/step in place and raising from 15Hz to 30Hz relaxes the safety ceiling 2x, from 120°/s to 240°/s.

---

## 3. What I changed

Commit `bf9ca5e` added a three-layer guard right before send in `workbench/so101/scripts/svla_rollout.py` (this commit modifies that one file only). Order matters. It runs in the order denormalize → envelope clip → per-step clamp → overload monitor.

![Three-layer safety guard between policy output and the motors](/diagrams/06_safety_guards.svg)

*Caption: The raw command the policy emits is passed to the motors only after going through envelope clip → per-step clamp → overload monitor in turn.*

The three guards block different things. The envelope clip cuts off coordinates the demonstrations never visited, the per-step clamp cuts off commands that jump too large an angle in one step, and the overload monitor cuts off situations where a motor is blocked by something and the load spikes. These three stages are not a gate you pass through once; they re-run in the same order on every control step.

### (a) GT envelope clip — block out-of-distribution extrapolation

Right after denormalization, the command is clipped independently per joint to the min~max from the demonstration dataset's `stats.json`. Since a command extrapolated from the pretraining distribution is clipped to "a coordinate the demonstrations actually visited," physical collisions like penetrating the table plane can be prevented.

```python
# build per-joint min/max tensors from stats.json
a_min = torch.tensor(STATS["min"], dtype=torch.float32, device="cuda")
a_max = torch.tensor(STATS["max"], dtype=torch.float32, device="cuda")
...
chunk = chunk * a_std + a_mean             # denormalize
if env_clip:
    chunk = torch.clamp(chunk, a_min, a_max)   # disable with ENV_CLIP=0
```

### (b) per-step relative-target clamp — limit per-step movement

lerobot's `ensure_safe_goal_position` (`lerobot/src/src/lerobot/robots/utils.py:91`) limits `|goal - present|` per joint to a cap. If a step target jumps far, it moves only by the cap and defers the rest to the next step, preventing torque spikes.

```python
# synchronous script: constant (sweep variable)
MAX_REL = float(os.environ.get("MAX_REL", "5.0"))   # °/step
SOFollowerRobotConfig(max_relative_target=MAX_REL, ...)
```

Later, in M7W4's RTC script (`svla_rollout_rtc.py:74-75`), I changed this cap to be derived from angular velocity. With `MAX_REL = MAX_VEL_DPS / CONTROL_HZ` (default `MAX_VEL_DPS=120`°/s), the speed ceiling stays fixed at 120°/s even if you raise the Hz.

### (c) Present_Load overload auto-abort + shutdown verification

On every step it reads the max load via `sync_read`, and if the raw value exceeds 500 (about 50%) for 5 consecutive steps (about 0.33s for the 15Hz synchronous script), it aborts and breaks out of the loop. On shutdown it turns off torque and verifies it actually went off by re-reading.

```python
OVERLOAD_RAW = 500    # Present_Load raw (~50%)
OVERLOAD_STEPS = 5    # abort if exceeded for 5 consecutive steps
overload_streak = overload_streak + 1 if max_load > OVERLOAD_RAW else 0
if overload_streak >= OVERLOAD_STEPS:
    ...  # abort + break (record reason/load/t_s meta)
...
robot.bus.disable_torque()
tq = robot.bus.sync_read("Torque_Enable", normalize=False)
out["torque_off_verified"] = not any(tq.values())
```

In the M7W4 efficiency review, there was a proposal to "shorten the critical path by reducing the Present_Load read to every N steps," but I deliberately rejected it. Slack should be reclaimed only where the safety cost is zero (jpeg compression, CPU denormalization); load and anomaly-monitoring reads are right to keep on every control step.

---

## 4. Before/after comparison

The figures below were measured in offline replay pre-validation (2026-06-12, smolvla_base zero-shot, `so101_pnp_tray_v0` 5ep). This is not a real-robot run; it's the result of comparing the policy output against the demonstration data.

| Metric | before (remap only) | after (remap+clip) |
|---|---|---|
| elbow_flex clip fire rate (fraction of predicted steps outside the demonstration envelope) | 83.4% | clipped to demonstration min~max |
| elbow MAE | 105.6° | 51.3° |
| residual L2 i25 | 165.0 | 125.9 |
| residual L2 i49 | 173.3 | 127.9 |
| elbow_flex command range vs demonstration envelope (zero-shot, before clip) | -46~139° (56° over GT -51~83°) | clipped within demonstration min~max by envelope clip |
| per-step clamp derivation method (M7W4 deg/s invariant) | MAX_REL constant 5.0°/step (Hz dependent) | MAX_REL = 120°/s ÷ CONTROL_HZ (Hz independent) |

The run where the three guards were actually confirmed as a real safe success on hardware is the later RTC rollout based on the FT checkpoint (`smolvla_cube_frugal`).

| Metric (RTC 24Hz real-robot, 2026-06-21) | before (no guards) | after |
|---|---|---|
| safe-shutdown result | user-stopped due to stall/damage risk | `aborted=null`, `torque_off_verified=true`, `queue_starved_steps=0` |

Sources: `smolvla_real_so101_pnp_replay_envclip.json`, `smolvla_rollout_rtc_so101_ft_v0.json`, `m7w2_smolvla_real_zeroshot.md` §6·§8.

### Caveat

- Figures like elbow MAE 105.6→51.3 are offline replay pre-validation values, not real-robot ones.
- The 83.4% clip fire rate and the MAE improvement are for the base model + tray dataset. After FT, out-of-distribution (OOS) commands are nearly zero, so the clip effectively never fires and remains as a secondary safety net (`svla_rollout.py` docstring).
- `OVERLOAD_RAW=500` (~50%) and `OVERLOAD_STEPS=5` are thresholds I set, not values tuned against measurements. Among the confirmed reports there is no record of an actual overload abort being triggered (`aborted=null`).

---

## 5. If applied at a company

This is a checkpoint you can carry over as-is to any pipeline that sends the output of a policy — or any learned model — to a real actuator.

1. Put guards in multiple layers, not just one.
   - Use a distribution-based clip to cut off output beyond the min~max of the demonstration/training data. It's a cheap secondary safety net that partially corrects, downstream, the variance-ratio distortion a statistics remap can't catch.
   - Use a per-step rate limit to absorb outliers or runaways with a single-step cap.
   - Load-based auto-abort is the last line of defense against unknown failures.

2. Bind safety invariants per-second, not per-step, and derive them from the control period.
   - In my view, if you define ceilings in physical units (°/s, etc.) for velocity, acceleration, and torque, and derive `cap = ceiling / Hz`, then the safety ceiling stays automatically fixed even when you change the control frequency. Hardcode it as a constant (°/step) and safety silently relaxes when you raise the period.

3. Do not trade safety-monitoring frequency against performance optimization.
   - Personally, I think critical-path slack should be reclaimed only where the safety cost is zero (logging, compression, denormalization). Keep load and anomaly-monitoring reads on every control step.

4. For the shutdown path, don't rely on "it probably happened" — actually re-read and verify.
   - After releasing torque, I re-read `Torque_Enable` to confirm it was 0 (`torque_off_verified`). I don't assume disconnect took care of it on its own.

---

## Series navigation

- Previous: [Why the arm kept reaching but never lifted — action-chunk execution horizon](./smolvla_tuning_05_exec_horizon.md)
- Next: [Removing the hitch at chunk boundaries — RTC async inference structure](./smolvla_tuning_07_rtc_async.md)
