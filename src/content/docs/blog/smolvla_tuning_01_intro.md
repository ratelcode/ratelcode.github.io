---
title: "SmolVLA Real-Robot Tuning Journey — From a Policy That Couldn't Even Lift the Arm to Smooth Pick-and-Place (Series Map)"
date: 2026-06-21
authors: ratel
excerpt: "Dropping public SmolVLA onto SO-101 as-is, the arm wouldn't even lift — normalization, camera slot, execution horizon, safety, RTC, data diversity: the full map of a 9-part tuning journey."
tags:
  - SmolVLA tuning
  - SmolVLA
  - SO-101
  - VLA
---

> This is a series documenting the tuning process I went through while putting SmolVLA on the SO-101 real robot (1/9, series map). I've kept what I verified through measurement and commits separate from my own opinions at the sentence level.

## TL;DR
Dropping the public policy smolvla_base onto the SO-101 as-is, the arm couldn't even lift. The cause wasn't a single layer but spread across several: normalization frame, camera slot, rig scale, execution horizon, speed, and diversity. After applying, in order, a normalization-stats swap, filling camera3, a 30ep FT, an execution horizon of 10→40, a 3-layer safety guard, and RTC, elbow OOS dropped **from 74% to 0.1%**, SR rose from 0 to a completed pick-and-place, and RTC recovered from half speed up to 24Hz. My take is that for online failures it's better to suspect the input, normalization, and execution horizon before the weights, because the pattern of offline being fine while only the real robot fails kept repeating.

## The Full Journey
At zero-shot the arm couldn't even lift (Part 2). After that, going through a normalization swap (2), filling camera3 (3), FT (4), execution-horizon adjustment (5), and safety guards (6), it completed reach, grasp, lift, and place; after removing the stutter with RTC async (7), I tuned the speed (8). The remaining task is data diversity (9).

![SmolVLA SO-101 tuning journey flowchart](/diagrams/01_journey.svg)

*Caption: The public policy smolvla_base had SR 0 at zero-shot, and after going through the 6 tuning steps in the middle in order, it reached the completed pick-and-place on the right and RTC 24Hz.*

The far left of the figure is the starting state, the six blue boxes in the middle are the fixes applied in order, and the green box on the far right is the destination. No single step solved it; only after normalization, camera slot, fine-tuning, execution horizon, safety, and speed had stacked up did the arm carry the object all the way. The orange box at the bottom points to the task that remains even after completion (data diversity).

> This series is grouped by topic, not by time. For example, Part 6 on safety guards was actually introduced right after zero-shot, but grouping by topic placed it toward the end. It also moves between two datasets. `so101_pnp_tray_v0` (tray task, 30ep) is the basis for measuring the FT effect in Part 4, and `so101_cube_v0` (white-cube benchmark, 51ep) is the deployment pipeline for Parts 3, 7, 8, and 9. The cube 51ep is a round that repeated the same techniques on larger data after the tray 30ep FT.

## Problem / Cause / Fix Table
| Part | Symptom | Cause | Fix | Before/After |
|---|---|---|---|---|
| 2 | arm won't lift | unnorm stats in SO-100 frame | swap to our stats | SR0→large motion |
| 3 | freeze | camera3 not filled | zeros + .rrd repro | →completion |
| 4 | remap limits | rig z-scale gap | 30ep FT | OOS74%→0.1% |
| 5 | repeats reach only | only first 10 executed | EXEC 10→40 | one-chunk completion |
| 6 | motor damage | raw sent directly | clip + step + overload stop | damage 0 |
| 7 | boundary stutter | sync inference stalls | RTC async queue | stutter removed |
| 8 | half speed | Hz15 vs 30fps | derive from data | →24Hz |
| 9 | same side only | diversity 0, memorized | diagnosis + @check | 8° span within 71° (0.059) |

## Eight-Part Contents (each part reads independently)
2. [Normalization frame mismatch](./smolvla_tuning_02_norm_frame.md) — the stats were baked into the SO-100 frame, so coordinates jumped. A one-line swap brought SR 0 back to life.
3. [The camera-slot trap](./smolvla_tuning_03_camera_slot.md) — the freeze was caused by not filling camera3. I confirmed it with an offline reproduction using .rrd.
4. [Closing the frame and scale gap with fine-tuning](./smolvla_tuning_04_finetune.md) — FT closed the elbow extrapolation but didn't solve diversity. It left a warning that, since evaluation doubles as training, a low residual might just reflect fidelity to the data.
5. [Why the arm only repeated reach — the action-chunk execution horizon](./smolvla_tuning_05_exec_horizon.md) — it was because only the first 10 steps were executed. I raised EXEC_STEPS from 10 to 40.
6. [Real-robot safety mechanisms — clamp + overload stop](./smolvla_tuning_06_safety_clamp.md) — I built a 3-layer guard with distribution clipping, step limiting, and overload stop.
7. [Removing the chunk-boundary stutter — RTC async](./smolvla_tuning_07_rtc_async.md) — while removing the stutter, I cleared up a bus crash, half speed, and a real_delay=0 bug.
8. [Smooth but slow — RTC speed tuning](./smolvla_tuning_08_rtc_speed.md) — I caught the Hz constant, the unmeasured delay, and the always-zero bug from the data.
9. [It only grabs the same side — data diversity and memorization](./smolvla_tuning_09_data_diversity.md) — diversity was 0, so it memorized a fixed pose. It only covered an 8° span within 71° of pan (0.059).

## Applying This at Work (opinion)
Personally, if I were to move this into production work, I'd make sure to cover the following.

- Keep normalization, camera slot, scale, and execution horizon as a checklist.
- Don't trust offline metrics alone, because if evaluation doubles as training, a low metric may end up being memorization.
- I think the 3-layer guard and RTC are portable regardless of form factor.
- Catch collection diversity with @check.

> Next: [Part 2 — Normalization frame mismatch](./smolvla_tuning_02_norm_frame.md)
