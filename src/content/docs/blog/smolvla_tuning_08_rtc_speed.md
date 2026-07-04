---
title: "Smooth but slow — three RTC speed tunings (including the half-speed bug)"
date: 2026-06-21
authors: ratel
excerpt: "Control Hz was half the recorded fps, giving 0.5x playback, plus an under-designed delay and a real_delay-always-0 bug. Reached 24Hz by deriving cadence, delay, and queue from dataset fps and measured p95."
homeCover: "/diagrams/08_rtc_speed.svg"
metric: "0.5x → 1.6x"
tags:
  - SmolVLA tuning
  - SmolVLA
  - RTC
  - performance
---

> This is a series on the tuning process I went through bringing SmolVLA up on a real SO-101 robot (8/9). I've kept what I confirmed through measurements and commits separate from my own opinions, sentence by sentence.

## TL;DR

- Turning on async inference (RTC) made the motion smooth, but the trajectory progressed noticeably slower than the demos.
- There were three causes. First, the control period was `CONTROL_HZ=15` while the dataset was `fps=30`, so it played back at exactly half speed. Second, RTC inference is heavier than synchronous inference, but I'd set the queue and delay too small. Third, there was a latent bug where `real_delay` was measured at the wrong point and always came out 0.
- I fixed it by redesigning the control period, delay, queue threshold, and safety clamp to all derive from two inputs: dataset fps and runtime-measured delay.
- Pulling out just the key before/after numbers: playback speed went from 0.5x to 1.0x (native), and to 1.6x at the default 24Hz; `queue_starved_steps` is 0; `measured_delays` went from a dead column of zeros to real values (~5–6); and `pred_ms_p95` went from a bootstrap assumption of 450 to a measured **251ms**.

---

## 1. Symptoms — what wasn't working

RTC (Real-Time Chunking), covered in an earlier post in this series, is an async inference setup where a background thread builds the next action chunk ahead of time and the main loop pulls actions out of a queue one at a time. To recap the background briefly: a sequence policy emits 50 future steps' worth of actions at once in a single inference (`predict_action_chunk`). The synchronous approach only infers the next chunk after the current one is fully consumed, so the robot stutters during that inference time. RTC builds the next chunk ahead of time and removes this stutter.

Turning RTC on was a half success.

- The success: the stutter at chunk boundaries went away and the motion became smooth.
- The failure: the overall trajectory progressed noticeably slower than the demos. The user's request was simple: "make it a bit faster."

The smoothness came from replaying the native trajectory (the actually recorded demo trajectory) as-is, and the slowness came from consuming that trajectory too slowly. The point of this post is that both came from the same cause.

---

## 2. Causes — why it happened

### Cause 1: control Hz was half the recorded fps (the half-speed bug)

The sequence policy's action chunk returns all 50 steps without trimming the time axis (`predict_action_chunk`). And the queue (`ActionQueue.get()`) pops one action per call and increments the index by +1. So "one control step = one action consumed" holds.

Here's where a quiet trap appears: the control cadence (period) is the playback speed.

The dataset `so101_cube_v0` was recorded at 30fps. So each pose in the 50-pose chunk is a future target spaced `1/30s ≈ 33.3ms` apart in the demo. But the rollout emitted one pose per step at `CONTROL_HZ=15` (66.7ms/step). That amounts to replaying the same trajectory over twice the time, which is exactly half speed (0.5x).

This isn't extrapolation or a model error. The trajectory the policy produced was correct; the consumption speed was simply halved.

![Half speed from the mismatch between control Hz and dataset fps](/diagrams/08_rtc_speed.svg)

*Caption: when the same chunk is consumed at different control Hz, the time it takes to use up one chunk (horizontal length) is the playback speed. At 15Hz, by the time native finishes you've only gotten halfway (pose 3), giving 0.5x; at 30Hz=fps it finishes at the same point, giving 1.0x native.*

Put simply: the poses in a chunk are "target coordinates" sampled 33.3ms apart in the demo, and each control step emits one of those coordinates. If the control period is 66.7ms (15Hz), you hold each coordinate twice as long, so you walk the same path in twice the time. That's why halving the Hz halves the speed exactly.

### Cause 2: RTC inference is heavier than synchronous inference → delay under-designed

RTC guidance runs not just a forward but also a backward pass on every denoise step (computing the correction term via `torch.autograd.grad`, `modeling_rtc.py`). The flow-matching loop repeats this every step, so the measured inference time is heavier than a plain offline `predict`.

The original note (`m7w4_rtc_async_efficiency.md`) describes the offline `predict` baseline as 285ms and the RTC-on measurement as "about 2x." That said, no figure directly comparing these two values within the same run remains in the rollout JSON. The JSON only outputs the measured `pred_ms_p95=251ms`. So take "about 2x" as the note's description.

The problem is the consequence. If you set a static delay using a synchronous rollout's delay intuition (offline predict time), the real measurement is heavier, so the queue and delay end up under-designed. Then a new chunk can re-execute a prefix that has already passed, risking a backward-stutter at the boundary.

### Cause 3: a latent bug where `real_delay` was always 0

`real_delay` is the number of actions the main loop actually consumed while inference was running. You need this value to correctly discard the already-passed prefix in a new chunk.

But the prediction thread was reading this value after `merge`. `merge` resets the queue's internal index to 0 (`action_queue.py`). Measuring the consumed amount after the reset naturally always yields 0.

So `measured_delays` was a meaningless column of zeros, and merge only ever received the static fallback value. At high Hz, this treats 0 already-passed actions as discarded and can lead to a backward-stutter that re-executes them.

---

## 3. What I changed

This redesign had a single principle: derive cadence, delay, queue threshold, and safety clamp all from two inputs (dataset fps and runtime-measured delay). If you don't hard-code constants, "half-speed"-class bugs can't structurally recur. The relevant file is `workbench/so101/scripts/svla_rollout_rtc.py`.

First, I made control Hz derive from fps. I removed the old constant `CONTROL_HZ=15` and derive it from the dataset fps. The default is a conservative 24Hz (1.6x vs. current), and you can raise it via env to 30Hz (=fps, 1.0x native).

```python
DS_FPS = int(json.load(open(_DS_DIR / "meta/info.json")).get("fps", 30))
# Cadence = dataset native fps. Must consume at the same fps for 1.0x playback.
# (The old CONTROL_HZ=15 was a bug that played 30fps demos at 0.5x.)
CONTROL_HZ = int(os.environ.get("CONTROL_HZ", "24"))
```

Next, I made the delay self-calibrate from the measured p95. Instead of a static `RTC_DELAY`, it tracks the inference-time p95 (`pred_p95`) at runtime and computes the delay from it. Only the bootstrap initial value is an estimate (env `PRED_MS`, default 450); after that, the measurements take over. To prevent queue starvation, the refill threshold is set to `drain_p95 + EXEC_HORIZON`.

```python
def derive_delay(pred_ms: float, hz: float) -> int:
    # Number of actions consumed during inference + 1 for margin
    return math.ceil(pred_ms / 1000.0 * hz) + 1
```

I fixed the `real_delay` bug by measuring the consumed amount before `merge` and passing it through directly.

```python
# Must read before merge resets last_index to 0 (previously read after merge, so always 0)
consumed = 0 if first else max(0, queue.get_action_index() - idx_at_start)
queue.merge(norm_chunk, norm_chunk, real_delay=consumed,
            action_index_before_inference=(None if first else idx_at_start))
```

Finally, I changed the safety clamp to derive from angular velocity (deg/s). The per-step clamp value is computed as `MAX_VEL_DPS / CONTROL_HZ`. The 120°/s speed ceiling is invariant, so raising the Hz doesn't loosen safety (if you keep per-step a fixed value, doubling the Hz also doubles deg/s). In this 24Hz run it came out to `120 / 24 = 5.0°/step`.

There were some side cleanups too. I throttled CPU denormalization and image logging to about 10Hz (`IMG_LOG_EVERY`) to clear the control thread's critical path, removed the `policy.reset()` footgun, and exposed a `NUM_STEPS` knob.

I deliberately left the safety-monitoring frequency alone. The review suggested cutting the overload-monitoring `Present_Load` read from every step to every N steps, but that read is the watchline for "stop immediately on a damage signal," so I don't think you can trade its response time for performance. I reclaimed critical-path headroom only where the safety cost is zero (jpeg compression, CPU denormalization).

---

## 4. Before/after comparison

All "after" numbers come from one 24Hz, 40-second real-robot run (`smolvla_rollout_rtc_so101_ft_v0.json`) and the training notes.

| Metric | Before | After |
|---|---|---|
| Playback speed multiplier | 0.5x (`CONTROL_HZ=15`, half-speed on 30fps demos) | 1.0x native (Hz=fps); 1.6x at default 24Hz |
| `queue_starved_steps` | starvation risk (static refill) | 0 |
| `measured_delays` (real_delay) | always 0 (dead column of zeros, measured after merge) | `[0, 6, 5, 6, 5, ...]` ~5–6 (42 values, now real) |
| `pred_ms_p95` (RTC inference delay) | bootstrap assumption 450ms (static) | measured 251ms (self-calibrated) |
| `inference_delay_final` / `refill_threshold_final` | static / fixed | 8 / 25 (p95 self-calibrating) |
| Safety clamp unit | fixed per-step value (deg/s loosens as Hz↑) | `MAX_VEL_DPS/Hz` = 120°/s invariant (24Hz→5.0°/step) |
| `last_pred_ms` / `chunks_predicted` / `steps_executed` | — | 242ms / 42 / 959 |
| Safe shutdown | — | `aborted=null`, `torque_off_verified=true` |

The felt result was "still smooth, now faster," and the speed target was met at the 24Hz default. The multi-agent review ran four perspectives (cadence, inference delay, critical path, queue) in parallel and adopted 25 of 26 findings (commit `f104447`), cross-checking each finding against the lerobot source. In the process, one wrong finding (a misread of weight direction) was rejected.

---

## 5. If you applied this at a company

These are the checkpoints worth keeping in mind when porting this case to a general ML/robotics pipeline. This section is my opinion.

1. When replaying a sequence (action-chunk) policy on real hardware, it's better not to hard-code the control period as a constant, because the playback cadence is the playback speed. If control Hz and recorded fps don't match, you get not a model error but a quiet speed bug. Deriving cadence from dataset fps gives 1.0x native when Hz=fps.
2. For the delay and queue design of an async inference pipeline, it's better to self-calibrate from runtime-measured p95 rather than from estimates. Heavy inference with a backward pass on every step, like RTC, is slower than synchronous-inference intuition, so a static delay easily ends up under-designed. It's safer to let the measurements drive it.
3. It's better to anchor safety invariants in per-second units rather than per-step. If you keep the speed ceiling in absolute units like deg/s, changing the control period won't automatically loosen safety.
4. A multi-agent (multi-perspective) review should pair each finding with source cross-verification. With verification attached, you can filter out both the latent bug (the always-0 because measured after merge) and the plausible wrong answer (the misread weight direction) in one pass. A review without verification tends to let convincing wrong answers through.
5. I think safety-monitoring frequency should be off-limits to optimization. Don't lower the frequency of a monitoring read like immediate overload shutdown just to shorten the critical path.

One more thing: to push further to 30Hz (2x), the measured p95 of 251ms makes chunk availability tight. So you'd have to cut inference first by lowering `NUM_STEPS`, and I think passing offline fidelity (residual/oos) with `compare_models` before deploying to hardware is a prerequisite. For reference, a separate symptom was observed in this rollout where the grasp skewed to the right of the cube; this turned out to be a data diversity problem, not inference or speed, and is covered in the next post. RTC speed optimization itself wraps up here.

---

## (End) Series nav

- Previous: [Killing the Chunk-Boundary Stutter — An RTC Async Inference Setup](./smolvla_tuning_07_rtc_async.md)
- Next: [The policy keeps grasping to the right of the object — data diversity and memorization](./smolvla_tuning_09_data_diversity.md)
