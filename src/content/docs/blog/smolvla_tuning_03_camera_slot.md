---
title: "Why the policy froze after training with two cameras — the camera slot trap"
date: 2026-06-21
authors: ratel
excerpt: "Even when you train with two cameras, the config expects three slots: camera1/2/3. The missing camera3 was the real cause of the freeze — filling it took shoulder_lift +4°→+87°."
tags:
  - SmolVLA tuning
  - SmolVLA
  - camera
  - SO-101
---

> This is a series documenting the tuning process I went through while putting SmolVLA on a real SO-101 robot (3/9). I've separated what I verified through measurements and commits from my own opinions in the text.

## TL;DR

- I ran a real-robot rollout with a SmolVLA that I re-fine-tuned (FT) on the cube dataset, and the arm barely lifted — it just twitched in place before stopping (freeze). Yet the offline fidelity metrics looked fine.
- I only connected two cameras (top/wrist), but the model config expected three image slots: camera1/2/3. The cause was that I didn't supply the camera3 slot at inference time, which broke the input schema. It wasn't a problem inside the model; one slot of the input tensor was simply missing.
- I fixed it to read `policy.config.input_features`, auto-detect any slot other than camera1/2, and fill it with `torch.zeros`. I removed the manual flag.
- With camera3 left out, shoulder_lift only reached **+4°** and the freeze reproduced; with it filled in, it reached **+87°** and lifted normally. After the fix, the real robot completed the entire reach→grasp→lift→place sequence.

---

## 1. Symptom — what didn't work

Let me give a bit of background first. SmolVLA is a policy that takes several camera images and the robot's joint state (state) as input (obs) and predicts the next 50 steps of motion (an action chunk) in one shot. I collected data of an SO-101 arm robot picking up and moving a cube (`so101_cube_v0`, 51 ep) and re-fine-tuned this policy on it.

When I ran a real-robot online rollout with that checkpoint, the result was strange.

The arm barely moved; it twitched in place and then stopped. In particular, `shoulder_lift`, the joint that raises the shoulder, effectively didn't change from its starting pose. This wasn't a case of motion being slightly off; it was a freeze with no motion at all.

There was one decisively contradictory signal. The offline chunk fidelity of the same checkpoint (`svla_compare_ft`) was fine. The residual between predicted and ground-truth actions (residual, the difference between the two) was low, and the ratio outside the demonstration range (OOS, out-of-support — the fraction of actions the training data never showed) was nearly 0.

The offline metrics were normal, but only the online rollout froze. This gap was a textbook signal that the model itself wasn't broken and that I should look at the difference between the online and offline input paths (`m7w4_camera3_rollout_fix.md` §1).

## 2. Cause — why it happened

At first I suspected something inside the model. Seeing the freeze, I assumed it had been trained on data with many still frames so that the average action was pushed toward 0 (idle-bias), and I even built a trimmed version with a lower fraction of still frames (`so101_cube_v0_trim`, frozen 37% → 7.5%) and re-FT'd on it. It still froze the same way. Probing the output distribution directly showed it wasn't no-motion, so the idle-bias hypothesis was disproven. I also ruled out candidates like a denormalization bug, physical camera movement, and the starting state being OOD (out-of-distribution, outside the training distribution), one by one.

What remained was the construction of the input tensor fed into inference itself.

This is where the trap was. SmolVLA is trained to take every image slot defined in the config's `input_features` as input. Even if I mapped only two real cameras during training, the model config inherits the base and keeps all three slots `camera1 / camera2 / camera3` as they are. In other words, the config is the source of truth, and camera3 was still alive in it.

- On the training path, mapping only two cameras worked (handled via rename_map). So it was easy to misread this as "I guess I don't need camera3."
- But on the inference path, the model still expects the camera3 slot. If you don't put the `observation.images.camera3` key in the obs batch, the expected input schema breaks and the model produces a near-no-motion output. This was the direct cause of the freeze.

![camera3 missing vs filled with zeros among the three camera slots](/diagrams/03_camera_slot.svg)

*Caption: the model expects three image slots; leaving camera3 empty makes it stop (+4°), while filling that slot with zeros makes it lift the arm normally (+87°).*

Seen as a diagram, the trap is simple. The model always expects three image slots as input, so when you connect only two cameras, the third slot is empty and the shape of the input itself is off. Just filling that empty slot with zeros of the same size makes the shape match, and the policy moves again.

So why did the offline fidelity look fine? Because the offline diagnostics that came out normal (the probe and model-comparison tools) were automatically filling the camera3 slot with zeros when constructing the input. That path filled all three slots and was fine, and the gap arose because only the online rollout dropped a slot and failed. That said, the fact that the same bug was also hidden in the FT path of the comparison script `svla_compare_ft` only came to light later, in the audit in §3.

The cause wasn't an inside-the-model hypothesis like idle-bias, denormalization, or state OOD, but a mismatch between the training schema and the inference schema, namely one missing image slot (`m7w4_camera3_rollout_fix.md` §2·§3).

### Reproducing the online failure offline — pinning down causation with numbers

The core of this debugging effort was nailing down "camera3 is the cause" with a controlled experiment rather than a guess. From the `.rrd` recording saved during the real-robot rollout (the input/output log file that rerun leaves behind), I extracted the live top/wrist frames the robot actually saw and replayed that same input offline. Then I compared with the camera3 slot toggled on and off.

Since there's no need to run the robot again, each hypothesis is verified with a single replay, without wear or safety concerns.

## 3. What I changed

The core of the fix was removing the structure where a person had to remember the number of camera slots and turn it on with a flag. Instead, it reads the expected slots directly from the model config and fills them automatically.

The target file is `workbench/so101/scripts/svla_rollout.py`.

- I removed the old manual `ADD_CAM3` environment-variable flag.
- In `main()`, I made it read `policy.config.input_features` and auto-detect slots that start with `observation.images` but aren't camera1/camera2 as `EXTRA_CAMS`.
- In `obs_to_batch()`, it fills each key in `EXTRA_CAMS` with `torch.zeros_like` of the same shape as camera1.

The two key pieces look like this.

```python
# main(): auto-detect the expected image slots from the model config (everything except camera1/2 = EXTRA_CAMS)
EXTRA_CAMS = [k for k in getattr(policy.config, "input_features", {})
              if k.startswith("observation.images")
              and k not in ("observation.images.camera1", "observation.images.camera2")]

# obs_to_batch(): automatically satisfy the detected extra slots (camera3, etc.) with zeros
for k in EXTRA_CAMS:
    out[k] = torch.zeros_like(out["observation.images.camera1"])
```

I propagated the same bug-class fix to other entry points too.

- In `svla_compare_ft.py`, I replaced the `add_cam3` flag with an `expected_image_keys()` helper. It turned out the FT path of this script was also dropping camera3, so I invalidated the FT numbers from the earlier report.
- For the newly added `svla_rollout_rtc.py` (RTC async inference), I applied the same `EXTRA_CAMS` auto-fill from the start.

The relevant commits are `91f1ec8` (svla_rollout: remove ADD_CAM3 → auto-detect and zeros-fill) and `3ea5609` (compare_ft same-bug fix + new rollout_rtc + training notes) (commit messages, `m7w4_camera3_rollout_fix.md` §5).

## 4. Before and after

Here are the results of the controlled experiment toggling only the camera3 slot using the live input frames captured in the `.rrd`, along with the real-robot results after the fix.

| Metric | camera3 missing (before) | camera3 filled / after fix (after) | Source |
|---|---|---|---|
| shoulder_lift change (.rrd offline reproduction) | **+4°** (exactly reproduces the real-robot freeze) | **+87°** (normal lift) | note §4 + commit 91f1ec8 |
| real-robot pick-and-place completion | freeze — no motion (shoulder_lift unchanged from starting pose) | completed the entire reach→grasp→lift→place sequence (shoulder_lift −68→+41→−73), safe with no abort | note §1·§6 |
| trimmed-version frozen frame ratio (idle-bias check) | 37% | 7.5% (re-FT on the trim still froze the same → idle-bias disproven) | note §1·§2 |

Leaving out camera3 reproduces offline exactly the no-motion the real robot showed, and filling it restores the lift. This causally confirmed that the missing camera3 was the cause.

Two side findings remain. First, with the lights off, the grasp misses to the right of the cube. Because the training data was collected with the lights ON, servoing wobbles when the lighting is out of distribution. That said, this is a lighting OOD case; the main cause of the systematic rightward bias in the grasp itself is a lack of positional diversity in the data, which I'll analyze quantitatively in part 9. Second, the per-step motion clamp `MAX_REL=5` fired on almost every step and created execution lag, so I'm considering raising it to `MAX_REL=8`. The safety premises (envelope clip + relative-target rate limit + Present_Load overload abort + .rrd) were maintained throughout (note §6).

One caveat: the auto-fill in the newly added `svla_rollout_rtc.py` has only passed static verification against the lerobot code. I haven't yet verified it in a first real-robot closed-loop run (marked ⚠ in commit 3ea5609).

## 5. Applying this at a company

This case isn't limited to SO-101. I think it's a trap anyone runs into when putting a multimodal policy whose input is made up of several slots (cameras/sensors) into production. Let me lay out the transferable takeaways.

- In my view, even when the offline metrics are good, an online failure should first be suspected to come from how the input tensor is constructed. loss, residual, and OOS are only fidelity given the correct input; they're no guarantee that the online path constructs the same input. Before digging into the model's internals (denormalization, distribution bias, OOD), it's better to first check that the training schema and the inference schema match.
- Personally, I think the structure where a person remembers the slot count and turns it on with a flag is itself a breeding ground for bugs. Since the model config is the source of truth, it's better to add a guard at the inference and evaluation entry points that verifies all input slots the config requires are actually filled in the obs, and either auto-fills missing slots with zeros or at least warns. This way the code follows along even when the number of cameras changes, so it becomes independent of form-factor / N-camera.
- I think the "reproduce the online failure offline" flow is worth adopting as a diagnostic standard. If you record the live input as-is during real-robot or production rollouts (here, the `.rrd`), you can isolate the cause with a controlled experiment by toggling only the input construction, without restarting the system. One offline replay per hypothesis is enough, with no robot wear or safety burden.

---

### Series navigation

- Previous: [Why pretrained SmolVLA couldn't even lift the arm — normalization frame mismatch](./smolvla_tuning_02_norm_frame.md)
- Next: [Closing the frame and scale gap with fine-tuning — what gets solved and what doesn't](./smolvla_tuning_04_finetune.md)
- Series map: [The SmolVLA real-robot tuning journey (series map)](./smolvla_tuning_01_intro.md)
