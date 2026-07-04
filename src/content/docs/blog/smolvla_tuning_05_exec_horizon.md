---
title: "Why the arm only repeated reach and never lifted — the action-chunk execution horizon"
date: 2026-06-21
authors: ratel
excerpt: "Executing only the first 10 steps of a 50-step chunk pinned the arm in the reach pose. Bumping EXEC_STEPS from 10 to 40 ran reach→grasp→lift to completion within a single chunk."
tags:
  - SmolVLA tuning
  - SmolVLA
  - action chunk
  - SO-101
---

> This is a series documenting the tuning process I went through while putting SmolVLA on a real SO-101 robot (5/9). I've kept what I confirmed through measurements and commits separate, in the text, from my own opinions.

## TL;DR

- I ran the first real-robot rollout with the fine-tuned (FT) checkpoint, and the arm kept repeating the approach pose (reach) toward the object without ever lifting it. It did end safely, with torque released normally and no overload abort, but the task itself was never completed.
- The policy emits a 50-step action chunk at a time, and the cause was that the config (EXEC_STEPS=10) executed only the first 10 steps of the chunk before immediately re-running inference. At every re-inference point the obs was still the "lowered reach pose," so the policy again predicted from the front of the chunk (=reach), and the back of the chunk (indices 10–49), which corresponds to grasp/lift, was never executed.
- The fix was to raise `EXEC_STEPS` from 10 to 40 (nearly full-chunk open-loop) and make it adjustable via env.
- As a result, the shoulder_lift pose recovered from `-82.5° → -14.2°` (stuck in reach) to `-85.5° → 47.8°` (lift progressing), and the gripper from `1.0→1.2` (effectively not moving) to `1.2→29.3` (working). The code is `svla_rollout.py` L59, commit `9b802d8`.

---

## 1. Symptom — what wasn't working

Let me set the background in one paragraph. SmolVLA is a policy that takes images and joint states as input and outputs robot actions. Instead of one step per inference, this policy returns a 50-step bundle of actions (an action chunk) at once. The chunk holds, in time order, "approach the object (reach) → grasp → lift."

In the previous post (part 4), fine-tuning closed the normalization-frame and scale gap, so the policy output was now within the range of our SO-101 robot. So I ran the first real-robot rollout (`smolvla_rollout_so101_ft_v0`).

Here's what happened.

- It ran to the end safely with no abort. The overload auto-stop never triggered, and torque release at shutdown was confirmed normal (`torque_off_verified: true`). So the safety goal from the posts through part 4 was met.
- But the arm only repeated the approach pose and never lifted the object. With "twitching + arm not lifting," the task was incomplete.
- From the start of the rollout to the end, the shoulder_lift pose (the joint that raises the shoulder) only moved `-82.5° → -14.2°`, and the gripper went `1.0 → 1.2`, effectively not working (start_pose/end_pose in `smolvla_rollout_so101_ft_v0.json`).

The most decisive evidence was the per-chunk command range (`lift_cmd_range`). In the v0 JSON, across 39 chunks, chunks 0–32 (33 of them) had the shoulder_lift command stuck in the -80 to -84° band (the reach pose). Only at chunk 33 did it start to loosen up to -78 to -66°, rising to -13.6° in the final chunk. In other words, most of the rollout time was spent on reach commands, and the commands that lead into grasp/lift were barely executed.

## 2. Cause — why it happened (the mechanism, simply)

The concept to focus on here is the execution horizon — that is, "how many steps of a received chunk you execute as-is (open-loop), without looking at the next obs."

The setting at the time was `EXEC_STEPS = 10`. That means only the first 10 steps of the 50-step chunk were executed before re-running inference immediately with a new obs. This is where the problem arises.

1. The first 10 steps of the chunk are the reach approach segment. Executing them takes the arm to the end-of-reach pose and then stops.
2. At that point it re-runs inference. But the input obs is still "the pose that just reached."
3. Seeing the same initial state again, the policy again predicts from the front of the chunk (=reach).
4. Again only the first 10 steps (reach) run, it ends up in the reach pose again, it predicts reach again… and this loop repeats.

As a result, the system gets stuck at a single point (an attractor): the reach pose. It barely reaches the back of the chunk (indices 10–49), which corresponds to grasp/lift. That's because the closed loop (re-inference) runs too often and chops off the late-chunk motion every time.

![Reach lock-in when executing only the first 10 steps, compared with executing 40 steps](/diagrams/05_exec_horizon.svg)

*Caption: EXEC_STEPS=10 repeatedly executes only the chunk's front reach segment and stays in place, while EXEC_STEPS=40 runs reach→grasp→lift to completion in a single chunk.*

Put simply, if you execute only the first 10 cells and stop, the robot restarts each time from "the pose it just approached into." The policy then only emits the action that fits that pose — that is, "approach" again — so it never reaches the grasp-and-lift back portion and just circles the same spot. This is the "attractor lock-in" I mentioned above.

In the v0 JSON, the lift_cmd_range staying at -80 to -84 throughout chunks 0–32 is direct evidence of this lock-in. In my view, this isn't a policy-weight or training-data problem but a pure execution-schedule problem. Before suspecting the weights, I should have looked at the horizon first.

There's a side effect too. Every re-inference introduces an inference stall. Measured from the v0 JSON, the first chunk takes about `1147ms` as warmup, and subsequent re-inferences take roughly `325~373ms` (median about 353ms) (it's bf16, so it's independent of dtype). The more often you re-run inference, the more often this stall interrupts, and the worse the twitching gets. On top of that, frozen_frames (static frames during the demonstration) at 36.5% also contributed to the reach lock-in (a data limitation, which is the topic of part 9).

## 3. What I changed (specifics: file/flag/value)

I raised the `EXEC_STEPS` default in `workbench/so101/scripts/svla_rollout.py` from 10 to 40 and made it adjustable via env.

```python
# before
EXEC_STEPS = 10  # number of steps to execute out of the 50-step chunk

# after (svla_rollout.py L59, commit 9b802d8)
EXEC_STEPS = int(os.environ.get("EXEC_STEPS", "40"))
```

The execution loop structure is unchanged. It receives a chunk, runs `robot.send_action(...)` for the first `EXEC_STEPS` steps, then re-runs inference via `policy.predict_action_chunk`. With `EXEC_STEPS=40`, it executes almost to the end of the 50-step chunk in one go (open-loop), so reach→grasp→lift proceeds to completion within a single chunk. In the same commit I also moved `DURATION_S` to env.

I'd recommend a value of 40–50 (nearly full-chunk open-loop). Setting it too long reduces the chances to incorporate obs through re-inference, but at this task length I think 40 is the balance point between action completion and responsiveness.

"Isn't running open-loop for a long stretch dangerous?" — that worry naturally follows. But the safety monitor (envelope clip, per-step movement rate limit, overload abort) runs at every `send_action` step, independent of chunk length. So even with longer execution, per-step safety is the same (safety design is the topic of part 6).

## 4. Before/after comparison

Comparing v0 (EXEC_STEPS=10) and v1 (EXEC_STEPS=40).

| Metric | before (v0, EXEC_STEPS=10) | after (v1, EXEC_STEPS=40) | Source |
|---|---|---|---|
| EXEC_STEPS (steps executed out of the 50-step chunk) | 10 | 40 | `svla_rollout.py` L59 / commit 9b802d8 |
| shoulder_lift pose (start→end, °) | -82.5 → -14.2 (stuck in reach, no lift) | -85.5 → 47.8 (lift progressing) | v0.json / v1.json start_pose·end_pose |
| gripper (start→end) | 1.0 → 1.2 (effectively not moving) | 1.2 → 29.3 (working) | v0.json / v1.json |
| shoulder_lift command range (whole run, °) | chunks 0–32 stuck at -80 to -84 (loosens from chunk 33, end -13.6) | -83.7 ~ 51.8 (full reach→lift span) | v0.json / v1.json lift_cmd_range |
| re-prediction inference stall (per re-prediction) | ~325–373ms (median ~353, first-chunk warmup 1147) | same (re-prediction frequency ↓ → twitching reduced) | v0.json chunks[].pred_ms |

A note on how to read this. Even when you increase EXEC_STEPS, the stall time per re-inference (around 350ms) stays the same, but executing each chunk for longer reduces the number of re-inferences, which in turn reduces the twitching (v0 ran 39 chunks, v1 ran 13). It's a structure where action completeness and smoothness improve at the same time.

Let me also clearly separate out the remaining limitations. Even after fixing the horizon restored the motion, the problem of the grasp being biased to the right of the object remained. In a follow-up v2 where I loosened the rate-clamp (MAX_REL) from 5 to 10, this bias was the same. So in my view, the remaining grasp inaccuracy separates out not as a control-parameter issue but as a limitation of the training data (30 ep, frozen_frames 36.5%). What the EXEC_STEPS fix addressed reaches "safety" and "motion progression"; grasp precision is a separate-round task (part 9).

One thing worth noting, though: v0 and v1 are different rollout runs, so the starting poses differ slightly (start lift -82.5° vs -85.5°). It's not a perfectly controlled comparison. Even so, the fact that v0's lift command was stuck at -80 to -84 across chunks 0–32 is itself direct evidence of the EXEC_STEPS lock-in.

## 5. If you're applying this at a company (transferable lessons)

These are checkpoints for deploying a policy that returns a sequence/chunk at once (action-chunk, flow matching, VLA, etc.) on real hardware / online.

1. The execution horizon is a first-class hyperparameter. "How far into the chunk you execute open-loop" governs action completeness. It's invisible during the model-selection and training stages, but it's the first value you should tune in the deployment loop.
2. If the horizon is too short, suspect attractor lock-in. If every re-inference meets the same initial state again, only the front of the chunk (the approach segment) runs repeatedly and the back (the goal-achieving segment) is never reached. The symptom is "approach repeating, no finishing motion."
3. Check the horizon before suspecting the weights. This looks like a policy-weight or data problem, but it's actually a pure execution-schedule problem. Retraining and re-collecting data are expensive, so it's better to start by changing one line — the horizon.
4. Increasing the horizon kills two birds. Re-inference carries a cost (the inference stall, around 350ms in this case). Increasing the horizon reduces the number of re-inferences, improving action completeness and twitching at the same time.
5. Design the per-step safety monitor independently of chunk length. If you keep the envelope/rate/load checks running every step, per-step safety holds even when you increase the horizon and the open-loop span gets longer.
6. After fixing the horizon, isolate the cause of any remaining precision problem. As in this case, loosening a control parameter (rate clamp) and checking whether the result is the same lets you separate whether the remaining error is due to control or to data.

---

**Series nav**
← Previous: [Part 4 — Closing the frame/scale gap with fine-tuning](./smolvla_tuning_04_finetune.md)
→ Next: [Part 6 — Real-robot safety guards: relative-target clamp + overload auto-stop](./smolvla_tuning_06_safety_clamp.md)
