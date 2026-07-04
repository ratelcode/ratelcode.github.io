---
title: "Closing the frame and scale gap with fine-tuning — what it fixes and what it doesn't"
date: 2026-06-21
authors: ratel
excerpt: "30ep FT cut elbow out-of-support from 74% to 0.1%. But eval == training data, so the low residual reflects in-distribution fidelity, not generalization — read with caution."
tags:
  - SmolVLA tuning
  - SmolVLA
  - fine-tuning
  - SO-101
---

> A series on the tuning process I went through bringing SmolVLA onto a real SO-101 robot (4/9). I've kept what I verified through measurements and commits separate from my own opinions in the writing.

## TL;DR

- I ran the pretrained `smolvla_base` on our SO-101 robot as-is, and the elbow joint predictions landed far outside the range a human had demonstrated (out-of-support 74%). The motion amplitude ballooned to about 3x, and on the real robot the arm dug into the table.
- The reason is that `smolvla_base` is a policy trained on a different distribution (a different robot frame + a different camera viewpoint). It was misaligned with our data's normalization, joint scale, joint combinations, and viewpoint, and per-joint linear correction couldn't constrain the "joint combinations" part.
- So we fine-tuned (FT) `smolvla_base` on our 30 episodes (ep). A single retraining aligned normalization, scale, joint coupling, and viewpoint all to our distribution.
- In numbers, elbow out-of-support dropped 0.738 → **0.001**, elbow MAE 99.55° → **4.71°**, and overall residual (residual_l2) i0 99.17 → **3.27**. Training loss went 0.177 → 0.037, and it took about 1 hour 6 minutes on an RTX 5070 Ti.
- There's a part that didn't get solved, though. The 30 ep used for evaluation are the very data used for training, so the low residual is closer to in-distribution fidelity (partial memorization included) than to "generalization." The lack of data diversity (a bias where grasps skew to the right side of the object) was not closed by FT.

---

## 0. Background — so this post stands on its own

SmolVLA is a policy that takes camera images and robot joint state (obs) and predicts the next 50 steps of joint commands (action chunk) in one shot. I was trying to apply this policy's pretrained weights, `smolvla_base`, to SO-101, a 6-joint robot arm.

The problem is that the pretrained policy has never seen data captured with our robot and our cameras. So zero-shot (applied directly with no additional training), the predictions came out wrong. This post is the story of closing that gap with fine-tuning. Earlier posts partially addressed the normalization frame and camera slot issues, but a gap still remained, and that's the starting point for this one.

> Let me note the dataset baseline up front. The FT and measurements in this post were done on the first dataset (`so101_pnp_tray_v0`, the tray task, **30ep**). The cube benchmark (`so101_cube_v0`, **51ep**) used in series posts 3, 7, 8, and 9 is the next round, where the same FT technique was repeated with more data (it's the realization of the "30→55~60ep" plan in §5 of this post).

---

## 1. Symptom — what went wrong

Running the pretrained `smolvla_base` zero-shot on the real SO-101, the elbow_flex (elbow bend) predictions extrapolated outside the human demonstration range. Quantitatively, in the offline chunk comparison, 73.8% of predictions were outside the demonstration envelope (the min~max range observed in demonstrations) (out-of-support fraction = 0.738). As a result the elbow motion amplitude ballooned, and on the real robot the arm collided into the table.

The "out-of-support fraction" metric matters here. Literally, it measures how far the predicted action strays into regions a human never demonstrated, on a 0~1 scale. 0 means predictions are always inside the demonstration range; 1 means always outside. Commanding the robot into regions never seen on the real robot means it's unsafe, so this metric is a direct signal of collision risk.

This gap wasn't a single layer but three.

1. Coordinate frame offset — a difference in reference point from being trained on a different robot frame (SO-100 vs SO-101) (partially mitigated by statistical remap)
2. z-scale amplification — motion amplitude ballooned to about 3x (partially mitigated by per-joint envelope clipping)
3. Joint correlation + visual domain gap — the very combination of how joints move together is misaligned + camera viewpoint difference (not fixable by linear/per-joint correction)

I could address ① and ② per joint with add/clip (offset/clip), but ③ I couldn't. Even if you clip each joint individually so each one stays in range, the pose that multiple joints make together can still be wrong.

---

## 2. Cause — why it happened (mechanism)

`smolvla_base` is a policy trained on a distribution different from ours (a different robot frame + the pretraining camera viewpoint). So it's misaligned with our SO-101 demo distribution in four ways: ① normalization ② joint scale ③ joint combination (correlation) ④ camera viewpoint.

Of these, I see ① and ② as gaps you can address per joint. Things like shoulder +5°, elbow ÷3. But ③ joint combination is different. Inter-joint coupling like "how much does the shoulder follow down when the elbow bends" can't be expressed by per-joint clipping that cuts joints one at a time. So even with clipping applied, poses that dug into the ground still came out on the real robot.

Fine-tuning absorbs this problem all at once. FT re-optimizes the policy weights to our demo distribution, so it pulls the joint combinations and scales the base was extrapolating back inside the demonstration envelope. ①②③④ get aligned to our data simultaneously, not separately.

As a result the predicted actions stayed inside the demonstration min~max, the out-of-support fraction converged to 0, and the safety condition "predictions don't leave the demonstration range" was satisfied. The collision risk of digging into the ground on the real robot disappeared.

> There's a trap in the mechanism. The data used for evaluation is the same as the data used for training. So the lowered residual reflects "fits the learned distribution well (in-distribution fidelity, partial memorization included)," not "does well in new situations too (generalization)." The base has never seen this data, so this comparison isn't a fair generalization comparison; it only shows that "FT fits the training distribution and base doesn't." The detailed implications are covered in §4 and §5.

---

## 3. What I changed (files, flags, values)

I fine-tuned `smolvla_base` with `workbench/so101/scripts/train_smolvla.py` (a `lerobot-train` wrapper). The commits are `e39ac52` (camera mapping fix) + `d174214` (FT success + comparison measurement).

### 3-1. The camera slot mapping is the crux

`smolvla_base`'s config specifies `observation.images.camera1/2/3` for input and sets `empty_cameras=0`. Our data has only 2 cameras, `top`/`wrist`, so fine-tuning as-is leaves the internal `prepare_images` with 0 matching camera keys, causing an "All image features missing" crash. It's not a silent failure; the error drops immediately.

I solved this with lerobot's `--rename_map`, matching the key names to the slots at runtime.

```text
--rename_map='{"observation.images.top":"observation.images.camera1",
               "observation.images.wrist":"observation.images.camera2"}'
```

camera3 was left out of the mapping, so we effectively trained with 2 real cameras. Since image normalization is `VISUAL=IDENTITY`, the camera key-statistics mismatch is irrelevant, and only state/action are normalized with `MEAN_STD`, where the keys match.

### 3-2. Training settings

The training settings are as follows (`m7w3_smolvla_finetune.md` §1).

| Item | Value |
|---|---|
| trainable / total parameters | 99.9M / 450M (`freeze_vision_encoder=True` + `train_expert_only=True`, SmolVLA default) |
| batch_size | 8 (about 11.7GB / 16GB VRAM) |
| steps | 20,000 (about 5.9 epoch) |
| save_freq | 2,000 |
| seed | 1000 |
| other required | `--policy.device=cuda`, `--policy.push_to_hub=false`, `--wandb.enable=false` |
| training time | about 1 hour 6 minutes (RTX 5070 Ti) |
| loss | 0.177 → 0.037 (monotonic convergence, no NaN) |

The comparison was done with `workbench/so101/scripts/svla_compare_ft.py`. It measures chunk fidelity offline, without a robot. For both base and FT, unnormalize is unified to dataset statistics, and both automatically fill the image slots the model config expects (`camera1/2/3`) (missing slot = zeros) to satisfy the input tensor contract.

---

## 4. Before/after comparison

![How far predictions stray from the demonstration range before and after fine-tuning](/diagrams/04_finetune_oos.svg)

*Caption: The gray band is the motion range a human showed through demonstration. Before fine-tuning (top), predictions (red dots) spilled far outside the band, but after 30ep fine-tuning (bottom) predictions (green dots) mostly gathered inside the band.*

Here, "out-of-support (outside the demonstration range)" means the policy's predictions strayed into a motion region a human never once demonstrated. Commands go out for poses the robot has never been in, so the more dots outside the band, the higher the collision risk, like digging into the table. Fine-tuning is exactly the work of pulling these dots back inside the band.

The numbers below are all measured from `workbench/reports/svla_compare_base_vs_ft.json` (30 ep × 5 chunk start points, chunk_size 50, offline).

| Metric | base (zero-shot) | FT | source key |
|---|---|---|---|
| **elbow_flex out-of-support fraction** | 0.738 | **0.001** | `out_of_support_fraction.elbow_flex` |
| out-of-support fraction (overall mean) | 0.217 | **0.013** | `oos_overall` |
| shoulder_lift OOS fraction | 0.27 | **0.034** | `out_of_support_fraction.shoulder_lift` |
| wrist_roll OOS fraction | 0.129 | **0.001** | `out_of_support_fraction.wrist_roll` |
| elbow_flex MAE (deg) | 99.55 | **4.71** | `per_joint_mae.elbow_flex` |
| shoulder_lift MAE (deg) | 78.09 | **3.61** | `per_joint_mae.shoulder_lift` |
| shoulder_pan MAE (deg) | 19.62 | **1.19** | `per_joint_mae.shoulder_pan` |
| residual_l2 i0 (chunk first step) | 99.17 | **3.27** | `residual_l2.i0` |
| residual_l2 i49 (chunk last step) | 173.68 | **12.11** | `residual_l2.i49` |
| elbow_flex predicted motion range (deg) | 129.37 | **21.92** | `pred_motion_range_mean.elbow_flex` |
| training loss | 0.177 | **0.037** | `m7w3_smolvla_finetune.md §1` |
| inference latency mean (ms, chunk predict) | 239.4 (base 3cam) | 284.4 (ft 2cam) | `latency_ms_mean` |

Here's how to read the table.

- OOS converging to 0 means predictions almost never leave the demonstration range, which means the safety condition is met. The elbow dropping from 74% to 0.1% is the biggest change.
- MAE and residual_l2 plummeting means the predictions track the human demonstration trajectory precisely. But because of the trap in §5, this should be read as "fit to the training distribution," not "generalization."
- The elbow predicted motion range shrinking from 129° to 22° is a signal that the ballooned z-scale was normalized back to the demonstration scale.
- Latency actually went up (239 → 284ms). base is a 3-cam config (including empty slots) and FT is a 2-cam config, so it's not a clean comparison, but I think the fact that FT didn't make inference faster is worth recording.

---

## 5. What didn't get solved — a methodological warning

This is the part I most want to emphasize in this post.

The 30 ep used for comparison evaluation are the very same 30 ep used for FT training. This time there was no separate held-out split; everything was used for training. So FT's low residual is in-distribution fidelity (partial memorization included), not generalization, and it's not a fair generalization comparison against base.

Even so, there's a reason I think this measurement is meaningful. The out-of-support fraction converging to 0 is a safety metric unrelated to memorization. "Predictions don't leave the demonstration range" is, regardless of whether the training data was memorized, a direct signal that the real-robot collision risk has disappeared. So I trust this one thing.

What FT closed is only the in-distribution gaps (normalization, scale, joint coupling, viewpoint). The lack of data diversity — the out-of-distribution gap — was not closed. In the actual follow-up real-robot rollouts, a bias remained where the grasp target systematically skewed to the right, and it was the same even when I loosened the control rate. This is not control but a bias in the learned grasp target itself, that is, a textbook signal of insufficient BC (behavior cloning) data.

My plan is to address this bias with the next data round (30 → 55~60 ep) and a train/val split (the topic of series post 9). The point is that "gaps FT solves" and "gaps only data solves" are different kinds.

> Real generalization is judged only by real-robot rollouts at new object positions. No matter how low the offline residual is, if it's not held-out, it's not evidence of generalization.

---

## 6. If you apply this at a company (transferable checkpoints)

I've put together the items to check when porting this technique to an ML/robot pipeline.

1. Never mix evaluation data with training data. If you evaluate on the same data, low loss/residual is just in-distribution fidelity (partial memorization), not generalization. Separate train/val from before FT, and judge real generalization only with held-out (here, real-robot rollouts at new object positions).

2. Measure "safety" and "generalization" as separate metrics. Whether predictions stay within the training/demonstration range (in-distribution metrics like out-of-support fraction) and whether they succeed in new situations (held-out performance) are separate things. The former can be trusted regardless of memorization; the latter can't be known without held-out.

3. Check the pretrained policy's input tensor contract directly in the config before FT. When a policy forces `camera1/2/3` slots + `empty_cameras=0` like `smolvla_base`, even if the camera count differs you have to match the slots with `rename_map` and fill missing slots with zeros. Otherwise you get an "All image features missing" crash or an inference freeze.

4. Distinguish "gaps FT solves" from "gaps only data solves." In-distribution gaps (normalization, scale, joint coupling, viewpoint) are closed by FT all at once. But a lack of data diversity (systematic bias and other out-of-distribution issues) is not solved by FT and is resolved only by a data collection round. If the residual isn't failing to drop but is consistently wrong in a particular direction, that's not a training problem but a data problem.

---

## Series map

- Previous post: [Why the policy froze after training with 2 cameras — the camera slot trap](./smolvla_tuning_03_camera_slot.md)
- Next post: [Why the arm kept only reaching and never lifted — the action chunk execution horizon](./smolvla_tuning_05_exec_horizon.md)

> Original training note: `docs/learning/m7w3_smolvla_finetune.md` · raw measurements: `workbench/reports/svla_compare_base_vs_ft.json`
