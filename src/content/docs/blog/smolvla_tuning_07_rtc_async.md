---
title: "Killing the Chunk-Boundary Stutter — An RTC Async Inference Setup"
date: 2026-06-21
authors: ratel
excerpt: "Synchronous inference stalls for hundreds of ms at every chunk boundary. Background prediction + queue consumption + boundary blend removes the hitch, and the single serial bus belongs to the main loop only."
tags:
  - SmolVLA tuning
  - SmolVLA
  - RTC
  - SO-101
---

> This is a series on the tuning process I went through bringing SmolVLA up on a real SO-101 robot (7/9). I've kept what I confirmed through measurements and commits separate from my own opinions, sentence by sentence.

## TL;DR

- A synchronous rollout uses up a whole chunk (50 actions), then stops in place to infer the next one. While inference takes hundreds of ms, the robot sits still, so it visibly stuttered at every chunk boundary.
- So I switched to an RTC (Real-Time Chunking) async setup. A background thread builds the next chunk ahead of time and fills the queue, and the main control loop pulls actions out of the queue one at a time and runs them without interruption. When stitching in a new chunk, flow-matching guidance smoothly blends the boundary so the target doesn't jump.
- Once I went async, two threads touching the single serial bus at the same time caused a `Port is in use` crash.
- I kept the bus exclusive to the main loop, and reworked the prediction thread so it never touches the bus directly and only reads a cache (`latest_obs`).
- The result: the chunk-boundary stutter is gone, and there's no bus crash (`aborted=null`, `torque_off_verified=true`).

> This post covers the structure (async + bus). The speed problem that surfaced separately after going async (half speed, under-designed delay, the `real_delay` bug) gets its own dedicated treatment in [part 8](./smolvla_tuning_08_rtc_speed.md).

---

## 1. Symptom — the synchronous-inference "hitch"

You don't need to have read the earlier posts. A sequence policy like SmolVLA returns target poses for 50 future steps (= an action chunk) all at once per inference. The robot pulls one action out of this chunk per step and runs it, and when the chunk is used up it infers again to get a new one.

The original synchronous rollout (`svla_rollout.py`) had a stop-per-chunk structure. Once a chunk was consumed, it stopped in place to infer the next one, and only moved again after inference finished. Inference takes hundreds of ms, so the robot visibly stuttered at every chunk boundary. The motion itself was correct, but it broke up at each chunk boundary.

The goal was simple: remove the stall time at the boundary so the motion is smooth.

---

## 2. The fix — RTC async inference

The basic idea is to decouple inference from execution (commit `3ea5609`, new file `svla_rollout_rtc.py`). It breaks into three parts.

![Comparison of the synchronous-inference hitch and RTC async queue consumption](/diagrams/07_rtc_async.svg)

*Figure: synchronous inference stops in place every time a chunk is used up to infer the next one, but RTC async pulls actions one at a time from a queue that a background thread has pre-filled, so it doesn't stop even at the boundary. The point where a new chunk is stitched in is joined smoothly by a boundary blend, and only the main loop touches the single serial bus.*

If you build chunks ahead of time and stack them in the queue, the main loop doesn't have to wait for inference to finish at the moment it needs the next action. Because it pulls an already-prepared action straight from the queue, the robot doesn't stop even at the instant the chunk changes.

(a) The background prediction thread. When the actions remaining in the queue drop below a threshold, it infers the next chunk in the background and fills the queue, without stopping main.

(b) The main control loop. It pulls one action per step with `ActionQueue.get()` and runs it. Since it doesn't wait for inference to finish, there's no stall window. Even at a chunk boundary, the next action is already sitting in the queue.

(c) The boundary blend (flow-matching guidance). If you just swap a pre-built new chunk into the empty slot, the target pose can suddenly jump at the boundary (because the end of the previous chunk and the start of the new one don't line up). When RTC generates a new chunk, it feeds the leftover portion of the previous chunk (`prev_chunk_left_over`) in as guidance, steering the denoise so the overlapping region connects smoothly. That way the target doesn't jump at the boundary.

> The smoothness that came from killing the hitch is, in my view, produced by (b) uninterrupted consumption and (c) the boundary blend together. With (b) alone, even without stalls, the target can jump at the boundary and judder.

This is how the chunk-boundary stutter disappeared. But going async did surface a new problem that hadn't come up before.

---

## 3. The trap — single-bus concurrent-access crash

The first real-robot run crashed. It was `ConnectionError: Failed to sync read 'Present_Load' ... Port is in use` (commit `c3efc0b`).

The cause is in the hardware structure. On the SO-101, the Dynamixel motors all hang off a single serial bus, so all read/write goes through one port.

When I went async, the background prediction thread called `robot.get_observation()`, which internally reads the bus. At the same time, the main loop reads/writes the same bus with `send_action()` and the `Present_Load` sync_read used for overload monitoring. With two threads touching the same serial port at once, the port-ownership conflict caused the crash.

At first I tried to serialize it with a lock (`pred_lock`), but that lock didn't cover the entire bus-access path (commit `c3efc0b` message). If even one path is outside the lock, the conflict happens all the same.

---

## 4. What I changed — single ownership of the bus

I made it so only the main loop accesses the bus (`send_action` / `get_observation` / `Present_Load`). The prediction thread doesn't touch the bus directly; it only reads the `latest_obs` cache that main updates every step under `obs_lock`. The `pred_lock` I'd used for serializing bus access was replaced with `obs_lock` for protecting the cache.

```python
# Main loop: write the obs read from the bus into the cache (only main touches the bus)
with obs_lock:
    latest_obs["obs"] = obs

# Prediction thread: doesn't touch the bus, only reads the cache
with obs_lock:
    obs = latest_obs["obs"]
```

> Given the RTC delay design, I think it's fine for prediction to see a one-step-stale obs (staleness). The inference result is used in a future window anyway. That's what let me simplify it to "one owner for the bus, everyone else uses the cache." Personally, I think reducing the number of access owners to one from the start is more robust than straining to cover every path airtight with locks.

---

## 5. Before/after

| Metric | Before (sync) | After (RTC async) | Source |
|---|---|---|---|
| Chunk-boundary behavior | stops then resumes at each boundary ("hitch") | no interruption (prefetch + boundary blend) | `svla_rollout.py` → `svla_rollout_rtc.py` (`3ea5609`) |
| Bus concurrent access | `ConnectionError: Port is in use` crash | no crash | commit `c3efc0b` |
| Safe shutdown | — | `aborted=null`, `torque_off_verified=true` | `reports/...rtc_so101_ft_v0.json` |

> Playback speed, delay, and queue-related quantitative numbers (`pred_ms_p95`, `measured_delays`, `queue_starved_steps`, etc.) are a separate problem that surfaced after going async, so they're covered in [part 8](./smolvla_tuning_08_rtc_speed.md). The results in this post stop at "hitch removed + crash removed."

---

## 6. If I applied this at a company

When you asynchronously split a real-time sequence policy into an "inference thread + execution loop," I think these are checkpoints that carry over even if you change domains.

1. Push heavy inference into the background and let the execution loop consume from a buffer. If you let inference stall execution, you get a stop at every output boundary. Build ahead and fill a queue, and have the execution loop pull from the queue, and the stops disappear.

2. When you stitch together results pre-built asynchronously, you need a boundary blend. If you just swap a new result into the empty slot, you get a discontinuity at the boundary. You need a mechanism that smoothly joins the overlapping region (flow-matching guidance here) for "no stops" to also mean "no judder."

3. Let only one owner access a single shared resource. For a one-of-a-kind resource like a serial bus, socket, or file handle, it's best to have one owner touch it and everyone else use a cache. If you try to serialize with a lock but miss some access path, you get an ownership conflict. Reducing the number of access owners itself is more robust.

---

## Series nav

- Previous post: [Real-robot safety guards — relative-target clamp + automatic overload abort](./smolvla_tuning_06_safety_clamp.md) (6/9)
- Next post: [Smooth but slow — three RTC speed tunings (including the half-speed bug)](./smolvla_tuning_08_rtc_speed.md) (8/9)
- Series map: [The SmolVLA real-robot tuning journey (series map)](./smolvla_tuning_01_intro.md)
