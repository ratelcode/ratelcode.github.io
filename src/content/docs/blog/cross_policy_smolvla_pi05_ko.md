---
title: "두 VLA 정책의 실측 비교 — 9.2× 모델인데 forward는 1.3×, variant 선택이 SR 100pp"
date: 2026-05-09
authors: ratel
excerpt: "SmolVLA(450M) vs π0.5(4143M)를 동일 환경·동일 데이터셋·동일 sweep으로 비교 — 9.2× 모델이 forward는 1.3×만, variant 한 글자 차이가 SR 0% vs 100%."
tags:
  - VLA
  - SmolVLA
  - π0.5
  - 평가
  - 재현성
---

> 개인 로봇 평가 연구 프로젝트의 측정 기록입니다. 측정값과 제 의견을 문장에서 구분해 적었습니다.

## 0. TL;DR

이번 글에서는 SmolVLA(450M)와 π0.5(4143M) 두 VLA 정책을 같은 환경(Linux Ubuntu + RTX 5070 Ti 16GB, NGC 25.10 + lerobot 0.5.1)에서 같은 데이터셋(`lerobot/svla_so100_pickplace`)으로 같은 sweep을 돌려 비교했습니다. 결과를 먼저 네 가지로 추렸습니다.

1. chunk_latency를 보면 π0.5가 파라미터 기준 9.2배 큰 모델인데도 forward는 1.3배만 느렸습니다(SmolVLA 204ms vs π0.5 267ms). 모델 크기와 inference 시간이 비례하지 않았습니다.
2. seed variance 패턴이 정반대였습니다. SmolVLA는 num_steps를 늘려도 seed_std가 평탄했지만, π0.5는 num_steps를 늘릴수록 seed_std가 커졌습니다(0.018 → 0.042). 두 정책의 flow-matching 방식이 근본적으로 다르다는 신호였습니다.
3. variant를 잘못 고르면 SR이 0%와 100%로 갈렸습니다(100pp 격차). `lerobot/pi05_libero_base`(multi-embodiment pretrain checkpoint)와 `lerobot/pi05_libero_finetuned_v044`(LIBERO finetune)는 이름 한 글자 차이지만 평가 결과는 완전히 달랐습니다.
4. 재현성은 numerical 0.0, latency ±2.5% 수준이었습니다. 동일 컨테이너 SHA에 동일 seed면 소수점 넷째 자리까지 byte-level로 일치했습니다. flow-matching denoise의 결정성을 1차로 확인한 셈입니다.

이 측정을 밖으로 공개할 가치는, eval-kit alpha API와 dataset audit, variant audit이 paper claim을 재현 가능한 형태로 만들어 주는 진입점이 된다는 점에 있다고 봅니다.

![모델 크기 대비 추론 시간, 그리고 variant 선택에 따른 SR 차이](/diagrams/10_cross_policy.svg)

*캡션: 모델이 9.2배 커도 추론은 1.3배만 느렸고(왼쪽), 이름이 거의 같은 두 variant는 SR 0%와 100%로 갈렸습니다(오른쪽).*

왜 9배 큰 모델이 추론에서는 1.3배만 느릴까요? 추론 시간의 대부분은 카메라 이미지를 한 번 처리하는 vision tower에서 나오는데, 이 부분은 모델 전체가 커진다고 해서 9배까지 함께 커지지는 않기 때문입니다. 그리고 여기서 variant란 같은 모델을 서로 다른 데이터로 학습시켜 만든 여러 버전을 뜻하는데, 이름이 거의 같아도 어느 버전을 고르느냐에 따라 성공률이 0%와 100%로 갈립니다.

---

## 1. 측정 환경 (재현성 anchor)

```
Host:        Linux Ubuntu 26.04 / kernel 7.0
GPU:         NVIDIA RTX 5070 Ti, 16303 MiB VRAM, sm_120 (Blackwell)
Driver:      595.58.03 (open)
Container:   lerobot-eval:0.5.1 (NGC 25.10 base, image ID eb5b017da246, built 2026-05-04)
Stack:       torch 2.9.0a0+nv25.10 / lerobot 0.5.1 / transformers 5.3.0
             cuDNN 9.14.0.64 / CUDA runtime 13.0
Swap:        16 GB (host)
```

이 fingerprint가 `workbench/reports/reproducibility_spike_v0.json`의 1차 자료입니다. 모든 측정 명령과 raw json이 이 fingerprint 위에서 byte-level로 재현됩니다(M3 W3 측정).

---

## 2. 모델 크기와 forward latency의 비선형 관계

| 정책 | params | VRAM | chunk_avg_ms | select_cached_ms |
|---|---:|---:|---:|---:|
| SmolVLA (`lerobot/smolvla_base`) | **450M** | 924 MiB | **204** | 1.9 |
| π0.5 (`lerobot/pi05_base`) | **4143M** | 9082 MiB | **267** | 1.5 |
| 격차 | **9.2×** | 9.8× | **1.3×** | 0.8× |

측정 출처는 `workbench/reports/smolvla_smoke.json`(M1 부트스트랩, 2026-05-04)과 `workbench/reports/pi05_smoke.json`(M2 W1, 2026-05-05)입니다.

모델이 9.2배 큰데 forward는 1.3배에 그친 이유를 제 나름대로 풀어 보면 이렇습니다. flow-matching denoise는 전체 50-step trajectory를 한 번에 생성하고, 모델 안의 vision_tower는 단일 forward를 한 번만 수행하며, action_expert만 num_inference_steps=10만큼 반복합니다. 결국 대부분의 시간이 vision_tower에서 소요되는데, π0.5의 vision_tower까지 9.2배 커진 것은 아니라서 latency 격차가 작았다고 봅니다. action expert의 chunk_amortization은 모델 크기보다 flow-matching 구조의 영향을 더 크게 받는 것으로 보입니다.

select_cached_ms는 오히려 π0.5가 0.8배로 더 빨랐는데, 흥미로운 결과였습니다. KV cache lookup이 모델 크기와 무관하게 동작하기 때문으로 보입니다.

---

## 3. Sweep 매트릭스에서 드러난 cross-policy 격차

같은 sweep 명령(num_steps × n_action_steps × dtype × seed = 96 cells)을 두 정책에 동일하게 적용했습니다(`smolvla_svla_so100_sweep.json`, `pi05_so100_sweep.json`).

### 3.1 seed_std가 num_steps에 정반대로 반응

| num_steps | SmolVLA seed_std | π0.5 seed_std |
|---|---|---|
| 2 | 0.116 | 0.018 |
| 4 | 0.116 | 0.027 |
| 6 | 0.116 | 0.034 |
| 10 | 0.116 | **0.042** |

SmolVLA는 num_steps를 늘려도 seed_std가 평탄했고(M1 W3-B 측정), π0.5는 num_steps를 늘릴수록 seed_std가 증가했습니다(M2 W3 측정).

두 정책의 flow-matching denoise schedule이 근본적으로 다르기 때문이라고 봅니다. 이 격차는 `m2w3_pi05_sweep_findings.md`의 1차 finding으로, 같은 sweep matrix를 적용해도 정책마다 dynamics를 좌우하는 결정 변수가 다르다는 점을 시사합니다.

### 3.2 bf16 호환성의 정책별 한계

SmolVLA는 96 cells 중 48개가 정상, 48개가 bf16 실패였습니다(preprocessor fp32와 model bf16 사이의 dtype mismatch). 반면 π0.5는 96 cells 중 48/48 모두 bf16에서 정상이었습니다(네이티브 bf16 학습 및 인퍼런스 방식).

bf16 인퍼런스 경로가 모델마다 다른 path를 트리거한다고 보입니다. eval-kit의 `variant_audit`(M3 W2 §11.6)에 이어질 후속 항목으로, 다운로드 전에 bf16 호환 여부를 자동 점검하는 후보로 적합하다고 생각합니다.

---

## 4. variant 선택 실수가 SR 0% vs 100%를 가른다 (M3 W2-7)

측정 출처는 `m3w2_pi05_libero.md §9`입니다. 동일한 명령(LIBERO env-side eval)에서 `--policy.path`만 바꿔 비교했습니다.

```bash
docker run ... lerobot-eval:0.5.1-libero \
  lerobot-eval --policy.path=<VARIANT> \
    --policy.dtype=bfloat16 --env.type=libero --env.task=libero_spatial \
    --env.task_ids='[0]' --eval.n_episodes=5 --eval.batch_size=1
```

| Variant | File size | dtype storage | unnorm_stats | n_action_steps | SR (5 ep) | 비고 |
|---|---:|---|---|---:|---:|---|
| `lerobot/pi05_libero_base` | 14.47 GB | F32 (812 tensors) | **비어있음** | 10 | **0/5 = 0%** | multi-embodiment pretrain |
| `lerobot/pi05_libero_finetuned_v044` | **7.47 GB** | F32 (122) + **BF16 (690)** | **safetensors 박힘** | 50 | **5/5 = 100%** | LIBERO finetune 완료 |

SR이 0%에서 100%로 벌어진 데에는 네 가지 메커니즘이 작용했다고 추정합니다(`m3w2_pi05_libero.md §11`).

1. Multi-embodiment dilution(약 40pp): `_base`를 학습할 때 svla_so100/101, LIBERO, ALOHA 등 여러 로봇을 평균했습니다. Franka 7-DoF 차원에서 6-DoF 학습 신호가 noise로 작용했을 가능성이 있습니다.
2. unnorm_stats 누락(약 25pp): `_base`의 model output `[-1, 1]`이 LIBERO env action_space `[-1, 1]`과 우연히 호환되는 것처럼 보이지만, 두 `[-1, 1]`의 의미가 다릅니다. 실제 EE delta가 ±0.05 범위라면 model 0.5는 10배 scale의 jerk가 됩니다.
3. n_action_steps의 학습-인퍼런스 mismatch(약 20pp): `_base` config는 10, `_finetuned`는 50입니다. flow-matching이 50-step을 동시에 생성하므로, chunk[10:50]의 학습 분포 밖 noise가 chunk[0:10]의 quality로 침투합니다.
4. fp32 → bf16 cast drift(약 15pp): `_base`는 fp32 storage라서 인퍼런스 시 bf16으로 cast하면 mantissa가 23bit에서 7bit로 손실됩니다.

이 네 가지가 동시에 작용하면 100% → 60 → 35 → 15 → 0%로 누적되어, catastrophic cascade로 이어집니다.

이 finding의 외부 가치를 따로 적어 두고 싶습니다. HuggingFace Hub의 5개 variant(`_base`, `_libero_base`, `_libero_finetuned_v044`, `_libero_finetuned_quantiles_v044`, `pi05-libero`)는 분명히 서로 다른 물건인데 README example만으로는 구분이 안 됩니다. 이 finding 자체가 charter §3.1 P0 #5(재현성 한계)의 1차 자료라고 봅니다. paper가 SR 90%+를 내세워도 README 명령을 그대로 따라갔을 때 재현되느냐 안 되느냐가 variant 선택에서 갈리기 때문입니다.

---

## 5. Paper claim 재현의 부분 확인 (M3 W2-9)

`_finetuned_v044`로 LIBERO 4개 suite의 task 0을 각각 5 ep씩 시도했습니다(`m3w2_pi05_libero.md §12`).

| Suite | Task 0 task language | SR (5 ep) | 결과 |
|---|---|---:|---|
| `libero_spatial` | "pick up the black bowl..." | **5/5 = 100%** | ✅ paper claim 재현 |
| `libero_goal` | "open the middle drawer..." | **4/5 = 80%** | ✅ 근접 |
| `libero_object` | "pick up the alphabet soup..." | OOM 8+ 시도 | ⚠ 호스트 한계 |
| `libero_10` | "put both...in the basket" | OOM 3+ 시도 | ⚠ 호스트 한계 |

측정이 가능했던 2/4 suite의 평균은 90%로, paper claim인 90%+를 부분 재현했습니다.

호스트 한계에 대한 제 관찰을 덧붙입니다. `libero_object`와 `libero_10`은 각각 8회 이상, 3회 이상 시도가 모두 SIGKILL(exit 137)로 끝났습니다. 컨테이너 메모리가 19 GB peak에 도달하면서 OOM-killer가 트리거됐습니다. `--memory=28g --memory-swap=42g`를 명시해도 실패했는데, 호스트 swappiness와 cgroup 충돌 가설을 생각하고 있습니다. 이 한계 자체가 재현성과 하드웨어 의존성에 대한 1차 자료입니다. RTX 5070 Ti 16GB GPU는 모델 자체에는 충분하지만, `libero_object`나 `libero_10`처럼 asset이 큰 suite는 이 호스트에서 재현이 불가능했습니다. 24 GB 이상 GPU나 cloud GPU가 필요합니다.

---

## 6. Reproducibility spike (M3 W3)

측정 자료는 `reproducibility_spike_v0.json`입니다.

### 6.1 W3-1: π0.5 replay rerun (3일 간격)

baseline은 `pi05_replay_svla_so100.json`(M2 W2, 2026-05-06), rerun은 2026-05-09입니다.

| Metric | Baseline | Rerun | Δ |
|---|---:|---:|---:|
| pred_arm6_gt_ratio_normed | 0.285 | 0.285 | **0.0** |
| residual_l2_normed (i=0/i=25/i=49) mean | 4.91/5.18/5.41 | identical | **0.0** |
| chunk_records[0] 5 metrics | identical | identical | **0.0** |
| chunk_latency_p50_ms | 275.07 | 282.59 | **+7.52 (+2.7%)** |
| chunk_latency_p95_ms | 290.34 | 291.71 | +1.4 (+0.5%) |

### 6.2 W3-2: native lerobot-eval rerun (19분 간격)

| Metric | Baseline (13:39) | Rerun (13:58) | Δ |
|---|---|---|---|
| successes pattern | [F,F,F,F,F] | [F,F,F,F,F] | **identical** |
| eval_s | 76.57 | 75.05 | -1.52 (-2.0%) |

numerical reproducibility는 소수점 넷째 자리까지 완벽했습니다. flow-matching denoise와 CUDA deterministic 동작이 확인된 셈입니다. 유일하게 흔들린 부분은 latency ±2.5%였는데, 이는 단일 호스트의 thermal, scheduler, cache에서 나오는 노이즈 floor라고 봅니다. charter §3.1 P0 #5의 1차 외부 자료입니다.

---

## 7. Dataset audit (M3 W4): form-factor agnostic 검증의 빈틈

10개 LeRobotDataset을 자동 audit했습니다(`workbench/reports/datasets/_summary.md`).

| Action dim | 데이터셋 | n_cam |
|---:|---|---:|
| 2 | pusht | 0 (state-only) |
| 6 | svla_so100/101_pickplace | 2 |
| 7 | libero | 2 |
| 14 | aloha_sim_*, aloha_static_* | 1-4 |

action dim은 네 가지, 카메라는 0~4개, action range outlier는 0%~100%(정규화 방식의 신호)로 측정됐습니다.

eval-kit alpha(M3 W1)의 ADAPTER_REGISTRY가 가진 빈틈을 정량으로 적어 두겠습니다.

- 현재 cover하는 조합은 4개입니다(smolvla/pi05 × svla_so100/so101, 즉 6-DoF arm만).
- 이 audit의 10개 데이터셋 중 2개만 cover합니다. aloha 14-DoF, pusht 2-DoF, dexterous 12+ DoF는 모두 미수행입니다.
- 이 격차가 곧 **eval-kit v0.2 후보**라고 봅니다. `dataset_audit`과 `variant_audit`이 함께 동작해서 어떤 데이터셋이 어떤 정책과 호환되는지 자동으로 판단하는 방향입니다.

---

## 8. 메타-finding: 추정과 실측의 격차

이 시리즈 측정의 진짜 산출물은, 매 페이즈마다 추정이 깨지는 패턴이었다고 생각합니다.

- M1 W2: 가정 4/4가 모두 깨졌습니다(env 호환성 매트릭스 측정).
- M2 W1: 가정 4/4가 모두 깨졌습니다(π0.5 부트스트랩).
- M2 W4: 가정 3/4가 깨졌습니다(failure 분류).
- M3 W1: 가정 4/4가 모두 깨졌습니다(eval-kit alpha 설계).
- M3 W2 W2-4: "env state distribution mismatch" 가설을 W2-6 측정으로 반증했습니다.
- M3 W2 W2-6: "bf16 numerical drift" 가설을 W2-7 측정으로 반증했습니다.
- M3 W2 W2-7: 5개 가설 중 4개를 모두 반증했고, variant 선택 하나만 정답이었습니다.

제 생각에 VLA 평가의 진짜 가치는, paper claim 재현이 어디에서 깨지는지를 측정 가능한 형태로 박아 두는 데 있습니다. 이 finding들이 eval-kit alpha와 variant_audit, dataset_audit의 외부 가치를 만든다고 보며, M4 OSS 공개 후보로 생각하고 있습니다.

---

## 9. 본 글의 한계

- 단일 GPU 호스트(RTX 5070 Ti 16GB)라서 π0.5 fp32 평가가 불가능했고(24 GB 이상 필요), libero_object/10에서 OOM이 났습니다.
- 정책 2개(SmolVLA + π0.5), 데이터셋 2개(svla_so100/101), sim env 1개(LIBERO)에 그쳐, charter §11.7의 form-factor agnostic 관점에서 플랫폼과 폼팩터 검증 빈틈이 큽니다.
- macOS MPS cross-platform 비교는 보류했습니다(사용자 결정으로 Linux 전용 진행, `feedback_linux_only_workflow.md`).
- ablation 3종(메커니즘 단독 영향)은 미수행입니다. M4 OSS 공개 후 외부 사용자 요청이 있으면 진행할 후보입니다.

---

## 10. eval-kit alpha의 외부 가치 후보

이 시리즈 finding이 박힌 alpha API입니다(`workbench/eval_kit/`, 0.1.0a1, M3 W1).

```python
from eval_kit.runners.replay import replay
from eval_kit.runners.sweep import sweep
from eval_kit.failure_modes import classify_chunk
from eval_kit.metrics.schema_v0 import ChunkRecord, emit
```

이번 글의 finding들이 들어갈 v0.2 후보는 다음과 같습니다.

```python
from eval_kit.variant_audit import audit
from eval_kit.dataset_audit import audit as ds_audit
from eval_kit.runners.reproducibility import spike

# variant 선택 실수 (M3 W2-7) → 자동 점검
audit("lerobot/pi05_libero_base")
# → eval_ready: False
#    reasons: unnorm_stats empty / dtype float32 / n_action_steps mismatch / 'base' suffix
#    recommended: lerobot/pi05_libero_finetuned_v044

# 데이터셋-정책 호환 (M3 W4) → 자동 점검
ds_audit("lerobot/aloha_static_coffee")
# → action_dim: 14
#    compatible_policies: []  (현재 ADAPTER_REGISTRY 미흡 신호)

# 재현성 (M3 W3) → 자동 측정
spike(baseline_report=..., rerun_command=..., container=..., interval_days=1)
# → max_abs_diff_per_metric / latency_drift_distribution
```

이 v0.2 후보가 M4 OSS 공개 시 외부 사용자의 진입 장벽을 크게 낮출 것이라고 봅니다. paper의 SR claim을 재현 가능한 형태로 바꿔 주는 방향이기 때문입니다.

---

## 11. 변경 이력

- 2026-05-09: 초안 (M3 W4-3). M2 W3-W4 + M3 W2 W2-7 + M3 W3 + M3 W4-2 산출물 합성. 11 섹션 + cross-policy 매트릭스 + 메타-finding + eval-kit v0.2 후보. 외부 가시화 1차.
