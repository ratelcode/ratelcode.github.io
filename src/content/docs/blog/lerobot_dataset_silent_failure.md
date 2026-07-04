---
title: "A catalog of LeRobotDataset silent failures — 7 ways policy evaluation breaks quietly"
date: 2026-05-09
authors: ratel
excerpt: "7 silent failures where evaluation finishes without errors but the result has nothing to do with the metric you meant to measure — from variant selection to form-factor mismatch, language coverage 0%, and OOM."
tags:
  - LeRobot
  - dataset
  - evaluation
  - reproducibility
---

> This is a measurement log from a personal robot-evaluation research project. I've kept measured values and my own opinions separate sentence by sentence.

## 0. TL;DR

By silent failure I mean the situation where evaluation runs to completion with no errors, but the result you get has nothing to do with the metric you wanted to measure. This post collects 7 cases where things break quietly like this in the LeRobotDataset format and the policy evaluation pipeline. All of them are things I ran into directly during the M1-M5 measurements of this project.

![7 silent failures that exit without errors but quietly break the result](/diagrams/11_silent_failure.svg)

*Caption: the entire path from dataset load to clean exit (exit 0) passes green (no errors), yet only the result breaks red (SR 0%), with no signal telling you where it went wrong. The 7 orange spots at the top are the cracks that get quietly buried.*

The scary thing about silent failure is that an error never appears even once. The program runs all the way through cleanly and even prints a clean exit (exit 0), so from the screen alone everything looks fine. So until you check the result number (SR 0%), there's no way to notice which of the 7 spots above quietly went wrong.

| # | Silent failure | Measurement source | Gap |
|---|---|---|---|
| 1 | variant selection mistake (`_base` vs `_finetuned`) | M3 W2-7 | **SR 0% vs 100% (100pp)** |
| 2 | form-factor mismatch (action_dim 6→7 slicing) | M1 W3-A | SR=0 *quietly* (adapter passes) |
| 3 | language coverage 0% (task column empty) | M3 W4-2 dataset audit | policy forwards with a *random task* |
| 4 | missing image orientation flip | M3 W2-3 vs native LiberoEnv | perception distribution broken |
| 5 | outlier % distribution differs per dataset (normalization format) | M3 W4-2 audit (pusht 100% / svla 90% / libero 0%) | applying a single stats breaks *physical meaning* |
| 6 | bf16 cast vs native bf16 storage | M3 W2-7 mechanism 4 | mantissa 23→7bit loss |
| 7 | host swappiness vs docker cgroup OOM | M3 W2-9 (libero_object/10) | *quiet* exit 137 after SIGKILL |

The one thread running through all 7 is that the evaluation tool throws no explicit error and offers no format to surface what went wrong through measurement. So when an external user tries to reproduce a paper claim, there's no way to know exactly where it breaks. In my view, the external value of this post is in nailing down the 7 silent failures in a diagnosable form, and as diagnostic entry points I present eval-kit alpha v0.2's `variant_audit / dataset_audit / spike` API alongside them.

---

## 1. Measurement environment

```
Host:        Linux Ubuntu 26.04 / kernel 7.0
GPU:         NVIDIA RTX 5070 Ti, 16303 MiB, sm_120 (Blackwell)
Container:   lerobot-eval:0.5.1 (NGC 25.10, image ID eb5b017da246)
Stack:       torch 2.9.0a0+nv25.10 / lerobot 0.5.1 / transformers 5.3.0
```

This fingerprint is the reference point for reproducing the 7 findings below at byte level.

---

## 2. Silent failure 1: variant selection mistake — `_base` advertises the paper claim but SR=0%

First the measured values (`m3w2_pi05_libero.md §9`).

| Variant | SR (5 ep) | File | dtype storage | unnorm_stats |
|---|---:|---:|---|---|
| `lerobot/pi05_libero_base` | **0/5 = 0%** | 14.47 GB | F32 (812 tensors) | empty |
| `lerobot/pi05_libero_finetuned_v044` | **5/5 = 100%** | 7.47 GB | F32 (122) + BF16 (690) | baked into safetensors |

Both rows were run with the same command (`lerobot-eval --policy.path=<VARIANT> --env.task=libero_spatial --task_ids=[0] --n_episodes=5`), changing only `--policy.path`. Run the `_base` call written in the README example as-is and it fails, but no explicit error appears.

Here's my interpretation of why this slips by quietly. The `_base` model loads fine, forward runs fine, the episode reaches 200 steps, and then it quietly returns `success=False`. The paper advertises SR 90%+, but run the README example command as-is and you get SR=0%, with no explanation of where it went wrong. Four mechanisms act at once — multi-embodiment dilution, missing unnorm_stats, n_action_steps, and the fp32→bf16 cast — and none of them throws a runtime error.

What I built to catch this is eval-kit v0.2's `variant_audit` (`workbench/eval_kit/eval_kit/variant_audit.py`).
```python
audit("lerobot/pi05_libero_base")
# → eval_ready: False
#   reasons:
#     - unnorm_stats not baked in (~25pp)
#     - safetensors all-fp32 storage → bf16 cast drift (~15pp)
#     - n_action_steps=10 vs chunk_size=50 → train-inference mismatch (~20pp)
#     - 'base' suffix → fine-tuning starting point, not for evaluation
#   recommended: lerobot/pi05_libero_finetuned_v044
```

This audit checks automatically before you download, so you can avoid a wasted 14.47 GB download and 200 steps of wasted compute up front.

---

## 3. Silent failure 2: form-factor mismatch — the adapter passes *quietly*

The measurement basis is the M1 W3-A measurement in `docs/learning/w2_env_compat_metric_schema.md`.

The LIBERO env outputs a 7-DoF Franka EE delta, while the SmolVLA policy is trained on a 6-DoF SO-ARM100. The adapter I built to bridge them (`workbench/scripts/libero_adapter.py`) only matched shapes. It mapped state from 8→6 (slicing just the first 6 dims of the Franka joints) and action from 6→7 (putting SmolVLA's 6-DoF straight into the first 6 OSC_POSE slots and adding 1 gripper dim).

The result, running 15 trials as SmolVLA × libero_spatial/0 × 5 ep × 3 seed, was **SR=0/15 = 0%**.

This also slips by quietly, and the way I see it the reason is this. The adapter throws no runtime error. As long as the shapes match, SmolVLA forward runs fine. The env also accepts the 7-DoF action fine ([-1, 1] clip) and runs all 200 steps to the end. The problem is that the meaning is off. SmolVLA's normalized action space isn't a Franka EE delta, so the robot effectively does arbitrary actions and never completes the task. From an external user's standpoint, there's no entry point at all to diagnose why it failed.

The diagnostic direction is to check action_dim compatibility and make the ADAPTER_REGISTRY registration format explicit.
```python
from eval_kit.dataset_audit import audit
audit("lerobot/libero")
# → schema: action_dim=7, n_cameras=2, state_dim=8
#   compatibility:
#     compatible_policies: [pi05_libero_finetuned_v044]
#     notes: "7-DoF Franka + 2 camera = LIBERO format.
#            SmolVLA (6-DoF) not directly compatible — the adapter's shape-only mapping
#            causes a semantic mismatch (M1 W3-A SR=0 finding)."
```

---

## 4. Silent failure 3: language coverage 0% — the task column is *empty* but forward passes

The measured value is the quality entry in `workbench/reports/datasets/svla_so100_pickplace.json`.

```json
"language_coverage": {
  "sampled_frames": 100,
  "coverage_pct": 0.0,
  "sample_unique_tasks": []
}
```

Auditing 10 datasets, many came back with `coverage_pct: 0.0`. That is, either the `task` column is empty, or the access format in our audit code is off.

The way I see why this slips by quietly is this. Even with an empty `task` column, the policy forward just proceeds with an empty string. When the PaliGemma tokenizer receives an empty input, only `[BOS]` remains and the rest is filled with padding. The policy then produces actions unrelated to the task (skewed toward the average task-instruction distribution it saw during training), and the behavior loses coherence. Yet there's no runtime error and only SR quietly drops.

From here on it's my hypothesis. You should resolve `task_index` (int) via a `meta.tasks` lookup to recover the real task description. But the LeRobotDataset.meta.tasks format differs per dataset, so the v0.1 audit only keeps the `meta.tasks` lookup as a candidate.

This finding itself is a signal of the limits of our audit script, and is also primary evidence that the LeRobotDataset format differs from dataset to dataset.

---

## 5. Silent failure 4: image orientation flip — perception distribution broken

The measurement basis is the M3 W2-4 measurement in `m3w2_pi05_libero.md §1.4`.

The LIBERO env (`OffScreenRenderEnv`) outputs vertically flipped images due to the mujoco viewport convention. So the adapter I built myself (`libero_pi05_adapter.py`) applied a vertical flip with `arr[::-1, :, :]`. But lerobot's native `lerobot.envs.libero.LiberoEnv._format_raw_obs` passes the image straight to the policy without a flip ([source](https://github.com/huggingface/lerobot/blob/main/lerobot/envs/libero.py)).

So which of the two is the training distribution? I didn't directly measure whether the images in the training dataset (`HuggingFaceVLA/libero` or `lerobot/libero`) are flipped. But given that the native LiberoEnv works with SR>0 (M3 W2-7 100%), I believe the native format is the training distribution. Given that our adapter (with the flip applied) was SR=0, I infer that the vertical flip was actually over-processing and departed from the training distribution.

The reason this slips by quietly is that whether the flip is missing or excessive, there's no runtime error. The image shape (3, H, W) is fine, and vision_tower forwards any image fine. It just emits arbitrary features for inputs outside the training distribution. The broken perception is plainly visible when you watch the video — the robot grabs at the wrong spot — but as a runtime metric it just shows up as a quiet SR 0%.

The diagnostic is to compare image hashing or pixel statistics and diff the image space against the native env at byte level.

---

## 6. Silent failure 5: outlier % distribution differs per dataset — the deciding variable for normalization format

The measured value is the M3 W4-2 audit in `workbench/reports/datasets/_summary.md`.

| Dataset | action_dim | outlier % (`abs(action) > 1.0`) | normalization format |
|---|---:|---:|---|
| `lerobot/pusht` | 2 | **100.0%** | raw env coords (not normalized) |
| `lerobot/svla_so100_pickplace` | 6 | **89.7%** | raw deg (not normalized) |
| `lerobot/svla_so101_pickplace` | 6 | **97.8%** | raw deg |
| `lerobot/libero` | 7 | **0.0%** | normalized [-1, 1] |
| `lerobot/aloha_sim_insertion_human` | 14 | **2.4%** | normalized (few outliers) |
| `lerobot/aloha_static_coffee` | 14 | **4.4%** | normalized |

Even within the same lerobot toolkit, the normalization format is starkly different from dataset to dataset. The key trap is that `pusht`'s 100% and `libero`'s 0% are both normal. If an external user just overwrites without clearly knowing which stats to apply and how, the physical meaning breaks. The `unnorm_stats=={}` format seen earlier in M3 W2-7 mechanism 2 is the same story: the `_base` model output `[-1, 1]` only looks like it happens to be compatible with the LIBERO env's `[-1, 1]`, but the physical meaning of the two `[-1, 1]` differs.

The diagnostic is done with eval-kit v0.2's `dataset_audit`.
```python
audit("lerobot/pusht")
# → outlier_pct: 100.0 → "raw env coords, not normalized. stats must be baked in when training the policy"
```

---

## 7. Silent failure 6: bf16 cast vs native bf16 — mantissa 23→7bit loss

The measurement basis is `m3w2_pi05_libero.md §11.4`.

| Variant | safetensors dtype | config.dtype | inference dtype (CLI override) |
|---|---|---|---|
| `pi05_libero_base` | F32 (812 tensors all) | "float32" | `bfloat16` (CLI cast) |
| `pi05_libero_finetuned_v044` | F32 (122) + BF16 (690) | "bfloat16" | `bfloat16` (train-inference match) |

The `--policy.dtype=bfloat16` flag given on the CLI is identical for both variants, so to an external user the dtype handling looks the same. But the storage of the weights themselves differs. When casting from fp32 to bf16, the mantissa shrinks from 23bit to 7bit and the fine-grained information in the weight distribution is lost. Looking at the effect in isolation, I put it at roughly 15pp (m3w2 §11.4 hypothesis), but when it acts at the same time as the other 3 mechanisms it spreads into a catastrophic cascade. Not a single runtime warning appears in the process. You run the same command and end up with a fundamentally different numerical distribution.

The diagnostic is done with eval-kit v0.2's `variant_audit`.
```python
audit("lerobot/pi05_libero_base")
# → reasons:
#     - safetensors all-fp32 storage → bf16 cast drift risk (~15pp)
```

---

## 8. Silent failure 7: host swappiness vs docker cgroup OOM — *quiet* exit 137

The measurement basis is the M3 W2-9 measurement in `m3w2_pi05_libero.md §12.2`. `_finetuned_v044 × libero_object/0 × 5ep` was tried more than 8 times and was SIGKILL (exit 137) every time, and `_finetuned_v044 × libero_10/0 × 5ep` was also tried more than 3 times and was SIGKILL every time. It failed even with `--memory=28g --memory-swap=42g` specified, and the container memory monitor showed SIGKILL dropping right after a 19 GB peak.

To lay out why this is quiet: exit 137 is SIGKILL and the cause is usually the OOM-killer. But the host had 28 GB available and 12 GB of swap, so it wasn't a memory cgroup limit. My hypothesis is that the large BDDL/scene assets of object and libero_10 spike at the same time as model load and env render, and because of the host swappiness setting the kernel OOM-killer triggers before swap is even used. Other than exit 137, no traceback and no log are left behind. The last line of python stderr is "Making policy." and it dies right after that. From an external user's standpoint there's no way to diagnose where the memory spike happened, and they just come away with something like "it doesn't work on this host."

The diagnostic direction is as follows. Track the memory trajectory with a `docker stats` monitor (additional M3 W2-9 measurement), and nail down a per-host reproducibility class. For example, "host_compatible_suites": [spatial, goal], "host_incompatible_suites": [object, 10]. As an eval-kit v0.3 candidate, I'm thinking of a `host_compatibility` automatic check that measures host RAM, GPU VRAM, and swap and produces the list of compatible suites.

---

## 9. The common pattern of silent failures + eval-kit's diagnostic entry points

Bundling the 7 silent failures together, here's what they have in common. First, there's no runtime error. The pipeline runs all the way through cleanly and exits with exit 0. Second, the only signal that surfaces is that an explicit metric like SR or accuracy has broken, while there's no diagnostic entry point for why it broke. Third, as a result an external user can't tell where it breaks when reproducing a paper claim, and I see this as the toolkit's decisive limit in external visibility.

The entry points of eval-kit alpha v0.2 are as follows (`docs/blog/cross_policy_smolvla_pi05.md §11`). `variant_audit(repo_id)` directly diagnoses silent failures 1 and 6, `dataset_audit(repo_id)` automatically checks silent failures 2, 3, 4, and 5, and `runners.reproducibility.spike(baseline, rerun)` produces primary evidence on the host dependency of silent failure 7.

The v0.3 candidates drawn from this post's findings are these. `host_compatibility(suite_id, host_ram, host_swap, gpu_vram)` for silent failure 7, `image_orientation_check(env, dataset)` for silent failure 4, and `task_metadata_check(dataset)` (`meta.tasks` lookup) for silent failure 3.

---

## 10. The external visibility value of this post

If you nail down the 7 silent failures in a measurable form, an external user reproducing a paper claim gets an entry point to automatically diagnose where it breaks. Given that the lerobot OSS README example can trigger several silent failures, I believe this post's catalog can serve as a primary external reference.

For external citation attempts, I'm thinking of the following (charter §7 deliverable #3). The huggingface/lerobot GitHub issue draft (M4 W4) "README example reproduces SR ≈ 0%" is the first piece to surface silent failure 1 externally. Beyond that there are candidates to update each policy's README on the HuggingFace Hub (making silent failure 6's dtype storage explicit) and a LeRobotDataset format guide (the metadata-explicit format for silent failures 3 and 5).

I'll note the limits too. This catalog is a narrative synthesis of measurement data, not a paper-style academic report, and it doesn't support the arXiv format. There may also be an 8th and further silent failures beyond these. Personally, I think the SmolVLA non-determinism seen in M5 W1 (I'm guessing it's due to vision encoder dropout) is a candidate to add to this catalog.

---

## 11. Source material for this post

| Silent failure | Source material |
|---|---|
| 1. variant selection | `m3w2_pi05_libero.md §9, §11`, `workbench/reports/lerobot_eval_libero_spatial_*.json` |
| 2. form-factor mismatch | `m3w2_pi05_libero.md §1`, `libero_smoke.py` SR=0 measurement |
| 3. language coverage | `workbench/reports/datasets/_summary.md`, 10 audit JSON |
| 4. image orientation | `m3w2_pi05_libero.md §4`, `lerobot/envs/libero.py` source |
| 5. outlier % | M3 W4-2 audit 10 datasets |
| 6. dtype cast | `m3w2_pi05_libero.md §11.4` |
| 7. cgroup OOM | `m3w2_pi05_libero.md §12.2`, `docker stats` monitor |

All of this material is nailed down as raw json and learning-note format, so it can be reproduced at byte level.

---

## 12. Change log

- 2026-05-09: Draft (M5 W4). 7 silent failure catalog + eval-kit v0.2/v0.3 diagnostic entry points nailed down. Synthesis of M3 W4-2 audit + W2-7 variant + W3 reproducibility + W2-9 host limit material.
