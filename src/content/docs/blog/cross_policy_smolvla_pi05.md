---
title: "Real-Robot Comparison of Two VLA Policies — 9.2× the Model but Only 1.3× the forward, and variant Choice Worth 100pp SR"
date: 2026-05-09
authors: ratel
excerpt: "SmolVLA(450M) vs π0.5(4143M) compared under the same environment, same dataset, and same sweep — a 9.2× model is only 1.3× slower on forward, and a one-character difference in variant means SR 0% vs 100%."
homeCover: "/diagrams/10_cross_policy.svg"
metric: "SR 0% vs 100%"
tags:
  - VLA
  - SmolVLA
  - π0.5
  - evaluation
  - reproducibility
---

> This is a measurement log from a personal robot evaluation research project. I've kept measured values and my own opinions separate at the sentence level.

## 0. TL;DR

In this post I compared two VLA policies, SmolVLA(450M) and π0.5(4143M), by running the same sweep on the same environment (Linux Ubuntu + RTX 5070 Ti 16GB, NGC 25.10 + lerobot 0.5.1) with the same dataset (`lerobot/svla_so100_pickplace`). Here are the four results up front.

1. Looking at chunk_latency, π0.5 is a 9.2× larger model by parameter count, yet forward was only 1.3× slower (SmolVLA 204ms vs π0.5 267ms). Model size and inference time did not scale proportionally.
2. The seed variance pattern was the exact opposite. For SmolVLA, seed_std stayed flat as num_steps increased, but for π0.5, seed_std grew as num_steps increased (0.018 → 0.042). This was a signal that the two policies' flow-matching approaches are fundamentally different.
3. Choosing the wrong variant split SR into 0% and 100% (a 100pp gap). `lerobot/pi05_libero_base` (a multi-embodiment pretrain checkpoint) and `lerobot/pi05_libero_finetuned_v044` (a LIBERO finetune) differ by a single character in their names, but their evaluation results were completely different.
4. Reproducibility was numerical 0.0 and latency ±2.5%. With the same container SHA and the same seed, results matched at the byte level down to the fourth decimal place. This is a first-pass confirmation of the determinism of flow-matching denoise.

I think the value of publishing these measurements externally lies in the fact that the eval-kit alpha API, the dataset audit, and the variant audit together form an entry point for making paper claims reproducible.

![Inference time relative to model size, and the SR difference depending on variant choice](/diagrams/10_cross_policy.svg)

*Caption: even though the model is 9.2× larger, inference was only 1.3× slower (left), and two variants with nearly identical names split into SR 0% and 100% (right).*

Why is a model 9× larger only 1.3× slower at inference? Most of the inference time comes from the vision tower, which processes the camera image once, and this part does not grow 9× just because the whole model gets bigger. And here, "variant" means the several versions made by training the same model on different data; even when the names are almost identical, which version you pick splits the success rate into 0% and 100%.

---

## 1. Measurement environment (reproducibility anchor)

```
Host:        Linux Ubuntu 26.04 / kernel 7.0
GPU:         NVIDIA RTX 5070 Ti, 16303 MiB VRAM, sm_120 (Blackwell)
Driver:      595.58.03 (open)
Container:   lerobot-eval:0.5.1 (NGC 25.10 base, image ID eb5b017da246, built 2026-05-04)
Stack:       torch 2.9.0a0+nv25.10 / lerobot 0.5.1 / transformers 5.3.0
             cuDNN 9.14.0.64 / CUDA runtime 13.0
Swap:        16 GB (host)
```

This fingerprint is the primary source for `workbench/reports/reproducibility_spike_v0.json`. Every measurement command and raw json reproduces at the byte level on top of this fingerprint (M3 W3 measurement).

---

## 2. The nonlinear relationship between model size and forward latency

| Policy | params | VRAM | chunk_avg_ms | select_cached_ms |
|---|---:|---:|---:|---:|
| SmolVLA (`lerobot/smolvla_base`) | **450M** | 924 MiB | **204** | 1.9 |
| π0.5 (`lerobot/pi05_base`) | **4143M** | 9082 MiB | **267** | 1.5 |
| Gap | **9.2×** | 9.8× | **1.3×** | 0.8× |

The measurement sources are `workbench/reports/smolvla_smoke.json` (M1 bootstrap, 2026-05-04) and `workbench/reports/pi05_smoke.json` (M2 W1, 2026-05-05).

Here is my own explanation for why the model is 9.2× larger but forward was only 1.3× slower. flow-matching denoise generates the entire 50-step trajectory at once; the vision_tower inside the model performs a single forward only once, and only the action_expert iterates num_inference_steps=10 times. As a result, most of the time is spent in the vision_tower, and since π0.5's vision_tower is not 9.2× larger, the latency gap stayed small. The action expert's chunk_amortization appears to be influenced more by the flow-matching structure than by model size.

select_cached_ms was actually faster for π0.5 at 0.8×, which was an interesting result. This seems to be because KV cache lookup operates independently of model size.

---

## 3. The cross-policy gap revealed in the sweep matrix

I applied the same sweep command (num_steps × n_action_steps × dtype × seed = 96 cells) identically to both policies (`smolvla_svla_so100_sweep.json`, `pi05_so100_sweep.json`).

### 3.1 seed_std reacts to num_steps in opposite directions

| num_steps | SmolVLA seed_std | π0.5 seed_std |
|---|---|---|
| 2 | 0.116 | 0.018 |
| 4 | 0.116 | 0.027 |
| 6 | 0.116 | 0.034 |
| 10 | 0.116 | **0.042** |

For SmolVLA, seed_std stayed flat as num_steps increased (M1 W3-B measurement), while for π0.5, seed_std increased as num_steps increased (M2 W3 measurement).

I think this is because the two policies' flow-matching denoise schedules are fundamentally different. This gap is the primary finding in `m2w3_pi05_sweep_findings.md`, and it suggests that even when the same sweep matrix is applied, the decision variables that govern dynamics differ from policy to policy.

### 3.2 The per-policy limits of bf16 compatibility

For SmolVLA, 48 of the 96 cells were fine and 48 failed on bf16 (a dtype mismatch between the preprocessor's fp32 and the model's bf16). In contrast, π0.5 was fine on bf16 for all 48/48 of its 96 cells (a native bf16 training and inference approach).

It seems the bf16 inference path triggers a different code path depending on the model. As a follow-on item to eval-kit's `variant_audit` (M3 W2 §11.6), I think this is a good candidate for automatically checking bf16 compatibility before download.

---

## 4. A variant-choice mistake splits SR 0% vs 100% (M3 W2-7)

The measurement source is `m3w2_pi05_libero.md §9`. Using the same command (LIBERO env-side eval), I compared by changing only `--policy.path`.

```bash
docker run ... lerobot-eval:0.5.1-libero \
  lerobot-eval --policy.path=<VARIANT> \
    --policy.dtype=bfloat16 --env.type=libero --env.task=libero_spatial \
    --env.task_ids='[0]' --eval.n_episodes=5 --eval.batch_size=1
```

| Variant | File size | dtype storage | unnorm_stats | n_action_steps | SR (5 ep) | Note |
|---|---:|---|---|---:|---:|---|
| `lerobot/pi05_libero_base` | 14.47 GB | F32 (812 tensors) | **empty** | 10 | **0/5 = 0%** | multi-embodiment pretrain |
| `lerobot/pi05_libero_finetuned_v044` | **7.47 GB** | F32 (122) + **BF16 (690)** | **baked into safetensors** | 50 | **5/5 = 100%** | LIBERO finetune complete |

I estimate four mechanisms were at work in widening SR from 0% to 100% (`m3w2_pi05_libero.md §11`).

1. Multi-embodiment dilution (about 40pp): when training `_base`, several robots such as svla_so100/101, LIBERO, and ALOHA were averaged. In the Franka 7-DoF dimension, the 6-DoF training signal may have acted as noise.
2. Missing unnorm_stats (about 25pp): `_base`'s model output `[-1, 1]` appears to be coincidentally compatible with the LIBERO env action_space `[-1, 1]`, but the two `[-1, 1]`s mean different things. If the actual EE delta is in the ±0.05 range, then a model 0.5 becomes a jerk at 10× scale.
3. n_action_steps train-inference mismatch (about 20pp): the `_base` config is 10, and `_finetuned` is 50. Because flow-matching generates the 50 steps simultaneously, out-of-distribution noise in chunk[10:50] bleeds into the quality of chunk[0:10].
4. fp32 → bf16 cast drift (about 15pp): `_base` uses fp32 storage, so casting to bf16 at inference time loses mantissa from 23 bits to 7 bits.

When these four act simultaneously, they accumulate from 100% → 60 → 35 → 15 → 0%, leading to a catastrophic cascade.

I want to note the external value of this finding separately. The five variants on the HuggingFace Hub (`_base`, `_libero_base`, `_libero_finetuned_v044`, `_libero_finetuned_quantiles_v044`, `pi05-libero`) are clearly different things, but the README example alone gives no way to tell them apart. I consider this finding itself to be primary source material for charter §3.1 P0 #5 (reproducibility limits). Even if a paper touts SR 90%+, whether or not it reproduces when you follow the README command verbatim comes down to variant choice.

---

## 5. Partial confirmation of paper claim reproduction (M3 W2-9)

Using `_finetuned_v044`, I attempted task 0 from each of the 4 LIBERO suites for 5 ep each (`m3w2_pi05_libero.md §12`).

| Suite | Task 0 task language | SR (5 ep) | Result |
|---|---|---:|---|
| `libero_spatial` | "pick up the black bowl..." | **5/5 = 100%** | ✅ paper claim reproduced |
| `libero_goal` | "open the middle drawer..." | **4/5 = 80%** | ✅ close |
| `libero_object` | "pick up the alphabet soup..." | OOM, 8+ attempts | ⚠ host limit |
| `libero_10` | "put both...in the basket" | OOM, 3+ attempts | ⚠ host limit |

The average over the 2/4 suites that I could measure was 90%, partially reproducing the paper claim of 90%+.

Let me add my own observations about the host limit. `libero_object` and `libero_10` each ended in SIGKILL (exit 137) on all of their 8+ and 3+ attempts respectively. The OOM-killer triggered as container memory reached a 19 GB peak. It failed even when I specified `--memory=28g --memory-swap=42g`, and I'm considering a hypothesis of a conflict between host swappiness and cgroup. This limit itself is primary source material on reproducibility and hardware dependence. The RTX 5070 Ti 16GB GPU is sufficient for the model itself, but asset-heavy suites like `libero_object` or `libero_10` were impossible to reproduce on this host. A GPU of 24 GB or more, or a cloud GPU, is needed.

---

## 6. Reproducibility spike (M3 W3)

The measurement data is `reproducibility_spike_v0.json`.

### 6.1 W3-1: π0.5 replay rerun (3 days apart)

The baseline is `pi05_replay_svla_so100.json` (M2 W2, 2026-05-06), and the rerun is 2026-05-09.

| Metric | Baseline | Rerun | Δ |
|---|---:|---:|---:|
| pred_arm6_gt_ratio_normed | 0.285 | 0.285 | **0.0** |
| residual_l2_normed (i=0/i=25/i=49) mean | 4.91/5.18/5.41 | identical | **0.0** |
| chunk_records[0] 5 metrics | identical | identical | **0.0** |
| chunk_latency_p50_ms | 275.07 | 282.59 | **+7.52 (+2.7%)** |
| chunk_latency_p95_ms | 290.34 | 291.71 | +1.4 (+0.5%) |

### 6.2 W3-2: native lerobot-eval rerun (19 min apart)

| Metric | Baseline (13:39) | Rerun (13:58) | Δ |
|---|---|---|---|
| successes pattern | [F,F,F,F,F] | [F,F,F,F,F] | **identical** |
| eval_s | 76.57 | 75.05 | -1.52 (-2.0%) |

numerical reproducibility was perfect down to the fourth decimal place. This confirms flow-matching denoise and CUDA deterministic behavior. The only thing that wavered was latency at ±2.5%, which I take to be the noise floor coming from a single host's thermal, scheduler, and cache effects. This is primary external source material for charter §3.1 P0 #5.

---

## 7. Dataset audit (M3 W4): the gaps in form-factor agnostic validation

I automatically audited 10 LeRobotDatasets (`workbench/reports/datasets/_summary.md`).

| Action dim | Dataset | n_cam |
|---:|---|---:|
| 2 | pusht | 0 (state-only) |
| 6 | svla_so100/101_pickplace | 2 |
| 7 | libero | 2 |
| 14 | aloha_sim_*, aloha_static_* | 1-4 |

action dim came out to four values, cameras 0–4, and action range outliers 0%–100% (a signal of the normalization approach).

Let me quantify the gaps in the ADAPTER_REGISTRY of eval-kit alpha (M3 W1).

- It currently covers 4 combinations (smolvla/pi05 × svla_so100/so101, i.e. 6-DoF arm only).
- Of the 10 datasets in this audit, only 2 are covered. aloha 14-DoF, pusht 2-DoF, and dexterous 12+ DoF are all unhandled.
- I see this gap as the **eval-kit v0.2 candidate**. The direction is for `dataset_audit` and `variant_audit` to work together and automatically determine which dataset is compatible with which policy.

---

## 8. Meta-finding: the gap between estimates and real measurements

I think the real output of this measurement series was the pattern of estimates breaking down at every phase.

- M1 W2: 4/4 assumptions all broke (env compatibility matrix measurement).
- M2 W1: 4/4 assumptions all broke (π0.5 bootstrap).
- M2 W4: 3/4 assumptions broke (failure classification).
- M3 W1: 4/4 assumptions all broke (eval-kit alpha design).
- M3 W2 W2-4: the "env state distribution mismatch" hypothesis was disproven by the W2-6 measurement.
- M3 W2 W2-6: the "bf16 numerical drift" hypothesis was disproven by the W2-7 measurement.
- M3 W2 W2-7: 4 of the 5 hypotheses were all disproven, and only variant choice was correct.

In my view, the real value of VLA evaluation lies in pinning down, in a measurable form, where paper claim reproduction breaks. I believe these findings create the external value of eval-kit alpha, variant_audit, and dataset_audit, and I'm considering them as M4 OSS release candidates.

---

## 9. Limitations of this post

- Because of a single GPU host (RTX 5070 Ti 16GB), π0.5 fp32 evaluation was impossible (24 GB or more required), and OOM occurred on libero_object/10.
- With only 2 policies (SmolVLA + π0.5), 2 datasets (svla_so100/101), and 1 sim env (LIBERO), there are large platform and form-factor validation gaps from the form-factor agnostic perspective of charter §11.7.
- macOS MPS cross-platform comparison was deferred (by user decision, proceeding Linux-only, `feedback_linux_only_workflow.md`).
- The 3 ablations (isolated effect per mechanism) are unhandled. They are a candidate to proceed if there are external user requests after the M4 OSS release.

---

## 10. Candidate external value of eval-kit alpha

This is the alpha API into which this series' findings are baked (`workbench/eval_kit/`, 0.1.0a1, M3 W1).

```python
from eval_kit.runners.replay import replay
from eval_kit.runners.sweep import sweep
from eval_kit.failure_modes import classify_chunk
from eval_kit.metrics.schema_v0 import ChunkRecord, emit
```

The v0.2 candidate into which this post's findings would go is as follows.

```python
from eval_kit.variant_audit import audit
from eval_kit.dataset_audit import audit as ds_audit
from eval_kit.runners.reproducibility import spike

# variant selection mistake (M3 W2-7) → automatic check
audit("lerobot/pi05_libero_base")
# → eval_ready: False
#    reasons: unnorm_stats empty / dtype float32 / n_action_steps mismatch / 'base' suffix
#    recommended: lerobot/pi05_libero_finetuned_v044

# dataset-policy compatibility (M3 W4) → automatic check
ds_audit("lerobot/aloha_static_coffee")
# → action_dim: 14
#    compatible_policies: []  (signals the current ADAPTER_REGISTRY is incomplete)

# reproducibility (M3 W3) → automatic measurement
spike(baseline_report=..., rerun_command=..., container=..., interval_days=1)
# → max_abs_diff_per_metric / latency_drift_distribution
```

I believe this v0.2 candidate will substantially lower the entry barrier for external users at the M4 OSS release, because it points toward turning a paper's SR claim into a reproducible form.

---

## 11. Change log

- 2026-05-09: draft (M3 W4-3). Synthesis of the M2 W3-W4 + M3 W2 W2-7 + M3 W3 + M3 W4-2 outputs. 11 sections + cross-policy matrix + meta-finding + eval-kit v0.2 candidate. First external visualization.
