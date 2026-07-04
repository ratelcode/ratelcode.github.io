---
title: "LeRobotDataset의 silent failure 카탈로그 — 정책 평가가 조용히 깨지는 7가지"
date: 2026-05-09
authors: ratel
excerpt: "평가가 오류 없이 끝나는데 결과가 의도한 metric과 무관한 7가지 silent failure — variant 선택부터 form-factor mismatch, language coverage 0%, OOM까지."
homeCover: "/diagrams/11_silent_failure.svg"
metric: "7가지 실패"
tags:
  - LeRobot
  - 데이터셋
  - 평가
  - 재현성
---

> 개인 로봇 평가 연구 프로젝트의 측정 기록입니다. 측정값과 제 의견을 문장에서 구분해 적었습니다.

## 0. TL;DR

여기서 말하는 silent failure는 평가가 오류 없이 끝까지 돌아가는데, 정작 나온 결과가 우리가 재고 싶었던 metric과 아무 상관이 없는 상황을 말합니다. 이 글에서는 LeRobotDataset 양식과 정책 평가 pipeline에서 이렇게 조용히 깨지는 경우 7가지를 정리했습니다. 모두 이 프로젝트의 M1-M5 측정 과정에서 직접 마주친 것들입니다.

![오류 없이 종료되지만 결과만 조용히 깨지는 silent failure 7가지](/diagrams/11_silent_failure.svg)

*캡션: 데이터셋 로드부터 정상 종료(exit 0)까지 전 구간이 초록(오류 없음)으로 통과하는데, 정작 결과만 빨강(SR 0%)으로 깨지고 어디서 어긋났는지 알려 주는 신호는 없습니다. 위쪽 주황색 7곳이 조용히 묻힌 균열 지점입니다.*

silent failure가 무서운 이유는 에러가 한 번도 뜨지 않기 때문입니다. 프로그램이 끝까지 멀쩡하게 돌고 정상 종료(exit 0)까지 찍으니 화면만 봐서는 다 잘된 것처럼 보입니다. 그래서 결과 숫자(SR 0%)를 확인하기 전까지는 위 7곳 중 어디가 조용히 어긋났는지 알아챌 방법이 없습니다.

| # | Silent failure | 측정 출처 | 격차 |
|---|---|---|---|
| 1 | variant 선택 실수 (`_base` vs `_finetuned`) | M3 W2-7 | **SR 0% vs 100% (100pp)** |
| 2 | form-factor mismatch (action_dim 6→7 슬라이싱) | M1 W3-A | SR=0 *조용히* (어댑터 통과) |
| 3 | language coverage 0% (task 컬럼 비어있음) | M3 W4-2 dataset audit | 정책이 *random task* 로 forward |
| 4 | image orientation flip 누락 | M3 W2-3 vs native LiberoEnv | perception 분포 깨짐 |
| 5 | outlier % 분포 dataset 별 다름 (정규화 양식) | M3 W4-2 audit (pusht 100% / svla 90% / libero 0%) | 단일 stats 적용 시 *물리 의미* 깨짐 |
| 6 | bf16 cast vs native bf16 storage | M3 W2-7 메커니즘 4 | mantissa 23→7bit 손실 |
| 7 | host swappiness vs docker cgroup OOM | M3 W2-9 (libero_object/10) | SIGKILL 후 *조용한* exit 137 |

이 7가지를 관통하는 한 가지는, 평가 도구가 명시적인 error를 던지지 않고 무엇이 잘못됐는지 측정으로 드러낼 양식도 없다는 점입니다. 그래서 외부 사용자가 paper claim을 재현하려고 할 때 정확히 어디에서 깨지는지를 알 길이 없습니다. 제 생각에 이 글의 외부 가치는 7가지 silent failure를 진단 가능한 형태로 박아 두는 데 있고, 그 진단 entry로 eval-kit alpha v0.2의 `variant_audit / dataset_audit / spike` API를 함께 제시합니다.

---

## 1. 측정 환경

```
Host:        Linux Ubuntu 26.04 / kernel 7.0
GPU:         NVIDIA RTX 5070 Ti, 16303 MiB, sm_120 (Blackwell)
Container:   lerobot-eval:0.5.1 (NGC 25.10, image ID eb5b017da246)
Stack:       torch 2.9.0a0+nv25.10 / lerobot 0.5.1 / transformers 5.3.0
```

이 fingerprint가 아래 7가지 finding을 byte-level로 재현하기 위한 기준점입니다.

---

## 2. Silent failure 1: variant 선택 실수 — `_base` 가 paper claim 광고하는데 SR=0%

먼저 측정값입니다 (`m3w2_pi05_libero.md §9`).

| Variant | SR (5 ep) | File | dtype storage | unnorm_stats |
|---|---:|---:|---|---|
| `lerobot/pi05_libero_base` | **0/5 = 0%** | 14.47 GB | F32 (812 tensors) | 비어있음 |
| `lerobot/pi05_libero_finetuned_v044` | **5/5 = 100%** | 7.47 GB | F32 (122) + BF16 (690) | safetensors 박힘 |

두 줄 모두 같은 명령(`lerobot-eval --policy.path=<VARIANT> --env.task=libero_spatial --task_ids=[0] --n_episodes=5`)으로 돌렸고, `--policy.path`만 바꿨습니다. README example에 적힌 `_base` 호출을 그대로 실행하면 실패하는데, 그 어떤 명시적 error도 뜨지 않습니다.

이게 왜 조용히 넘어가는지에 대한 제 해석입니다. `_base` 모델은 정상적으로 로드되고 forward도 정상이며 episode가 200 step까지 도달한 다음 조용히 `success=False`를 반환합니다. paper는 SR 90%+를 광고하는데 README example 명령 그대로 돌리면 SR=0%가 나오고, 어디가 잘못됐는지에 대한 설명은 없습니다. multi-embodiment dilution, unnorm_stats 누락, n_action_steps, fp32→bf16 cast라는 4가지 메커니즘이 동시에 작용하는데, 어느 것도 runtime error를 던지지 않습니다.

이걸 잡으려고 만든 게 eval-kit v0.2의 `variant_audit`입니다 (`workbench/eval_kit/eval_kit/variant_audit.py`).
```python
audit("lerobot/pi05_libero_base")
# → eval_ready: False
#   reasons:
#     - unnorm_stats 박힘 X (~25pp)
#     - safetensors all-fp32 storage → bf16 cast drift (~15pp)
#     - n_action_steps=10 vs chunk_size=50 → 학습-인퍼런스 mismatch (~20pp)
#     - 'base' suffix → finetune 시작점, 평가용 X
#   recommended: lerobot/pi05_libero_finetuned_v044
```

이 audit은 다운로드 전에 자동으로 점검해 주기 때문에, 14.47 GB짜리 헛된 다운로드와 200 step짜리 헛된 compute를 미리 피할 수 있습니다.

---

## 3. Silent failure 2: form-factor mismatch — 어댑터가 *조용히* 통과

측정 근거는 `docs/learning/w2_env_compat_metric_schema.md`의 M1 W3-A 측정입니다.

LIBERO env는 7-DoF Franka EE delta를 출력하는데, SmolVLA 정책은 6-DoF SO-ARM100으로 학습돼 있습니다. 둘을 붙여 보려고 만든 어댑터(`workbench/scripts/libero_adapter.py`)는 shape만 맞추는 수준이었습니다. state는 8→6으로(Franka joint의 첫 6 dim만 슬라이싱), action은 6→7로(SmolVLA의 6-DoF를 OSC_POSE 첫 6 자리에 그대로 넣고 gripper 1 dim 추가) 맞췄습니다.

결과는 SmolVLA × libero_spatial/0 × 5 ep × 3 seed로 15 trials를 돌려 **SR=0/15 = 0%**였습니다.

이것도 조용히 넘어가는데, 제가 보기엔 이유가 이렇습니다. 어댑터가 runtime error를 던지지 않습니다. shape만 맞으면 SmolVLA forward는 정상적으로 돕니다. env도 7-DoF action을 정상적으로 받아([-1, 1] clip) 200 step을 끝까지 실행합니다. 문제는 의미가 어긋난다는 데 있습니다. SmolVLA의 normalized action space는 Franka EE delta가 아니라서, 로봇이 사실상 임의의 행동을 하고 임무를 달성하지 못합니다. 외부 사용자 입장에서는 왜 실패했는지 진단할 entry 자체가 없습니다.

진단 방향은 action_dim 호환을 점검하고 ADAPTER_REGISTRY 등록 양식을 명시하는 것입니다.
```python
from eval_kit.dataset_audit import audit
audit("lerobot/libero")
# → schema: action_dim=7, n_cameras=2, state_dim=8
#   compatibility:
#     compatible_policies: [pi05_libero_finetuned_v044]
#     notes: "7-DoF Franka + 2 camera = LIBERO 양식.
#            SmolVLA (6-DoF) 직접 호환 X — 어댑터의 shape-only 매핑은
#            의미 mismatch 야기 (M1 W3-A SR=0 finding)."
```

---

## 4. Silent failure 3: language coverage 0% — task 컬럼이 *비어있는데* forward 통과

측정값은 `workbench/reports/datasets/svla_so100_pickplace.json`의 quality 항목입니다.

```json
"language_coverage": {
  "sampled_frames": 100,
  "coverage_pct": 0.0,
  "sample_unique_tasks": []
}
```

데이터셋 10개를 audit한 결과 다수가 `coverage_pct: 0.0`으로 나왔습니다. 즉 `task` 컬럼이 비어 있거나, 아니면 우리 audit 코드의 접근 양식이 어긋난 것입니다.

조용히 넘어가는 이유는 이렇게 봅니다. `task` 컬럼이 비어 있어도 정책 forward는 empty string으로 그냥 진행됩니다. PaliGemma tokenizer가 빈 입력을 받으면 `[BOS]`만 남고 나머지는 padding으로 채워집니다. 그러면 정책이 task와 무관한 행동을 만들어 내고(학습 때 본 평균적인 task instruction 분포 쪽으로 쏠려서), 동작에 일관성이 없어집니다. 그런데도 runtime error는 없고 SR만 조용히 떨어집니다.

여기서부터는 제 가설입니다. `task_index`(int)를 `meta.tasks` lookup으로 풀어서 진짜 task description을 회수해야 합니다. 그런데 LeRobotDataset.meta.tasks의 양식이 데이터셋마다 달라서, v0.1 audit에서는 `meta.tasks` lookup을 후보로만 두고 있습니다.

이 finding 자체가 우리 audit script의 한계를 보여 주는 신호이자, LeRobotDataset 양식이 데이터셋별로 다르다는 점을 보여 주는 1차 자료이기도 합니다.

---

## 5. Silent failure 4: image orientation flip — perception 분포 깨짐

측정 근거는 `m3w2_pi05_libero.md §1.4`의 M3 W2-4 측정입니다.

LIBERO env(`OffScreenRenderEnv`)는 mujoco viewport 관습 때문에 상하 반전된 image를 출력합니다. 그래서 우리가 직접 만든 어댑터(`libero_pi05_adapter.py`)는 `arr[::-1, :, :]`로 vertical flip을 적용했습니다. 그런데 lerobot의 native `lerobot.envs.libero.LiberoEnv._format_raw_obs`는 flip 없이 image를 그대로 정책에 전달합니다 ([source](https://github.com/huggingface/lerobot/blob/main/lerobot/envs/libero.py)).

그렇다면 둘 중 어느 쪽이 학습 분포일까요. 학습 데이터셋(`HuggingFaceVLA/libero` 또는 `lerobot/libero`)의 image가 flip돼 있는지는 직접 측정하지 못했습니다. 다만 native LiberoEnv가 SR>0으로 작동한다는 점(M3 W2-7 100%)을 보면, native 양식이 학습 분포일 것이라고 봅니다. 우리 어댑터(flip 적용)가 SR=0이었던 걸 보면, vertical flip이 오히려 과잉 처리였고 학습 분포에서 벗어난 것이라고 추정합니다.

이게 조용히 넘어가는 이유는, flip이 빠졌든 과했든 runtime error가 없기 때문입니다. image shape (3, H, W)는 멀쩡하고, vision_tower는 어떤 image든 forward를 정상적으로 합니다. 단지 학습 분포 밖의 입력에 대해 임의의 feature를 뱉을 뿐입니다. perception이 깨진 건 비디오를 보면 로봇이 엉뚱한 위치를 잡는 식으로 눈에 명백히 드러나지만, runtime metric으로는 그저 조용한 SR 0%로만 보입니다.

진단은 image hashing이나 pixel statistics를 비교하고, native env와의 image space를 byte-level로 diff하는 방향입니다.

---

## 6. Silent failure 5: outlier % 분포 dataset 별 다름 — 정규화 양식 결정 변수

측정값은 `workbench/reports/datasets/_summary.md`의 M3 W4-2 audit입니다.

| Dataset | action_dim | outlier % (`abs(action) > 1.0`) | 정규화 양식 |
|---|---:|---:|---|
| `lerobot/pusht` | 2 | **100.0%** | raw env coords (정규화 X) |
| `lerobot/svla_so100_pickplace` | 6 | **89.7%** | raw deg (정규화 X) |
| `lerobot/svla_so101_pickplace` | 6 | **97.8%** | raw deg |
| `lerobot/libero` | 7 | **0.0%** | normalized [-1, 1] |
| `lerobot/aloha_sim_insertion_human` | 14 | **2.4%** | normalized (소수 outlier) |
| `lerobot/aloha_static_coffee` | 14 | **4.4%** | normalized |

같은 lerobot toolkit 안인데도 데이터셋마다 정규화 양식이 대조적입니다. `pusht`의 100%와 `libero`의 0%가 둘 다 정상이라는 게 핵심적인 함정입니다. 외부 사용자가 어떤 stats를 어떻게 적용할지 명확히 모른 채 그냥 덮어쓰면 물리적 의미가 깨집니다. 앞서 M3 W2-7 메커니즘 2에서 본 `unnorm_stats=={}` 양식도 같은 맥락인데, `_base`의 model output `[-1, 1]`이 LIBERO env의 `[-1, 1]`과 우연히 호환되는 것처럼 보일 뿐 두 `[-1, 1]`의 물리적 의미는 다릅니다.

진단은 eval-kit v0.2의 `dataset_audit`으로 합니다.
```python
audit("lerobot/pusht")
# → outlier_pct: 100.0 → "raw env coords, 정규화 X. 정책 학습 시 stats 박혀있어야"
```

---

## 7. Silent failure 6: bf16 cast vs native bf16 — mantissa 23→7bit 손실

측정 근거는 `m3w2_pi05_libero.md §11.4`입니다.

| Variant | safetensors dtype | config.dtype | 인퍼런스 dtype (CLI override) |
|---|---|---|---|
| `pi05_libero_base` | F32 (812 tensors all) | "float32" | `bfloat16` (CLI cast) |
| `pi05_libero_finetuned_v044` | F32 (122) + BF16 (690) | "bfloat16" | `bfloat16` (학습-인퍼런스 일치) |

CLI에서 주는 `--policy.dtype=bfloat16` flag는 두 variant가 동일해서, 외부 사용자가 보기에는 dtype 처리가 똑같아 보입니다. 그런데 weights 자체의 storage가 다릅니다. fp32에서 bf16으로 cast할 때 mantissa가 23bit에서 7bit로 줄면서 weight 분포의 세밀한 정보가 손실됩니다. 단독 효과만 따지면 대략 15pp 수준이라고 보는데(m3w2 §11.4 가설), 나머지 3가지 메커니즘과 동시에 작용하면 catastrophic cascade로 번집니다. 이 과정에서 runtime warning은 하나도 뜨지 않습니다. 같은 명령으로 돌렸는데 근본적으로 다른 numerical 분포가 나오는 셈입니다.

진단은 eval-kit v0.2의 `variant_audit`으로 합니다.
```python
audit("lerobot/pi05_libero_base")
# → reasons:
#     - safetensors all-fp32 storage → bf16 cast drift 위험 (~15pp)
```

---

## 8. Silent failure 7: host swappiness vs docker cgroup OOM — *조용한* exit 137

측정 근거는 `m3w2_pi05_libero.md §12.2`의 M3 W2-9 측정입니다. `_finetuned_v044 × libero_object/0 × 5ep`은 8번 넘게 시도했는데 모두 SIGKILL(exit 137)이었고, `_finetuned_v044 × libero_10/0 × 5ep`도 3번 넘게 시도해 모두 SIGKILL이었습니다. `--memory=28g --memory-swap=42g`를 명시해도 실패했고, 컨테이너 메모리 monitor로는 19 GB peak 직후 SIGKILL이 떨어졌습니다.

이게 조용한 이유를 정리해 보면, exit 137은 SIGKILL이고 보통은 OOM-killer가 원인입니다. 그런데 호스트에는 28 GB가 available하고 swap도 12 GB가 있어서 memory cgroup 한계는 아니었습니다. 제 가설은 object와 libero_10의 큰 BDDL/scene asset이 모델 로드와 env render 시점에 동시에 spike를 일으키고, host swappiness 설정 때문에 swap을 쓰기도 전에 커널 OOM-killer가 trigger된다는 것입니다. exit 137 말고는 traceback도 log도 남지 않습니다. python stderr의 마지막 줄이 "Making policy."이고 그 직후에 죽습니다. 외부 사용자 입장에서는 어디서 메모리 spike가 났는지 진단할 수 없고, 그냥 "이 호스트에서는 안 된다" 정도로만 인식하게 됩니다.

진단 방향은 다음과 같습니다. `docker stats` 모니터로 메모리 trajectory를 추적하고(M3 W2-9 추가 측정), 호스트별 reproducibility class를 박아 둡니다. 예를 들면 "host_compatible_suites": [spatial, goal], "host_incompatible_suites": [object, 10] 식입니다. eval-kit v0.3 후보로는 host RAM과 GPU VRAM, swap을 측정해 호환 가능한 suite list를 뽑아 주는 `host_compatibility` 자동 점검을 생각하고 있습니다.

---

## 9. silent failure 의 공통 패턴 + eval-kit 의 진단 entry

7가지 silent failure를 묶어 보면 공통점이 이렇습니다. 첫째, runtime error가 없습니다. pipeline은 정상적으로 끝까지 돌고 exit 0으로 종료됩니다. 둘째, 표면에 드러나는 신호는 SR이나 accuracy 같은 명시적 metric이 깨졌다는 것뿐인데, 정작 왜 깨졌는지에 대한 진단 entry가 없습니다. 셋째, 그래서 외부 사용자가 paper claim을 재현할 때 어디에서 깨지는지를 알 수 없고, 이게 toolkit의 결정적인 외부 가시화 한계라고 봅니다.

eval-kit alpha v0.2의 진입점은 다음과 같습니다 (`docs/ko/blog/cross_policy_smolvla_pi05.md §11`). `variant_audit(repo_id)`는 silent failure 1과 6을 직접 진단하고, `dataset_audit(repo_id)`는 silent failure 2, 3, 4, 5를 자동으로 점검하며, `runners.reproducibility.spike(baseline, rerun)`은 silent failure 7의 host 의존성에 대한 1차 자료를 만듭니다.

이번 글의 finding에서 끌어낸 v0.3 후보는 이렇습니다. silent failure 7을 위한 `host_compatibility(suite_id, host_ram, host_swap, gpu_vram)`, silent failure 4를 위한 `image_orientation_check(env, dataset)`, 그리고 silent failure 3을 위한 `task_metadata_check(dataset)`(`meta.tasks` lookup)입니다.

---

## 10. 본 글의 외부 가시화 가치

7가지 silent failure를 측정 가능한 형태로 박아 두면, 외부 사용자가 paper claim을 재현할 때 어디에서 깨지는지를 자동으로 진단할 entry가 생깁니다. lerobot OSS의 README example이 여러 silent failure를 trigger할 수 있는 만큼, 이 글의 카탈로그가 1차 외부 reference가 될 수 있다고 봅니다.

외부 인용 시도로는 다음을 생각하고 있습니다 (charter §7 deliverable #3). huggingface/lerobot의 GitHub issue 초고(M4 W4) "README example reproduces SR ≈ 0%"는 silent failure 1을 외부로 처음 드러내는 글입니다. 그 외에 HuggingFace Hub의 각 정책 README를 갱신하는 후보(silent failure 6의 dtype storage 명시), LeRobotDataset 양식 가이드(silent failure 3, 5의 metadata 명시 양식)도 있습니다.

한계도 적어 둡니다. 이 카탈로그는 measurement 자료를 narrative로 합성한 것이지 paper-style 학술 보고서는 아니고, arXiv 양식은 지원하지 않습니다. 또 silent failure가 8번째 이후로도 더 있을 수 있습니다. 개인적으로는 M5 W1에서 본 SmolVLA non-determinism(vision encoder dropout 때문이 아닐까 짐작합니다)이 이 카탈로그에 추가될 후보라고 봅니다.

---

## 11. 본 글의 근거 자료

| Silent failure | 근거 자료 |
|---|---|
| 1. variant 선택 | `m3w2_pi05_libero.md §9, §11`, `workbench/reports/lerobot_eval_libero_spatial_*.json` |
| 2. form-factor mismatch | `m3w2_pi05_libero.md §1`, `libero_smoke.py` SR=0 측정 |
| 3. language coverage | `workbench/reports/datasets/_summary.md`, 10 audit JSON |
| 4. image orientation | `m3w2_pi05_libero.md §4`, `lerobot/envs/libero.py` source |
| 5. outlier % | M3 W4-2 audit 10 데이터셋 |
| 6. dtype cast | `m3w2_pi05_libero.md §11.4` |
| 7. cgroup OOM | `m3w2_pi05_libero.md §12.2`, `docker stats` monitor |

이 자료는 모두 raw json과 학습 노트 양식으로 박혀 있어서 byte-level로 재현할 수 있습니다.

---

## 12. 변경 이력

- 2026-05-09: 초안 (M5 W4). 7 silent failure 카탈로그 + eval-kit v0.2/v0.3 진단 entry 박힘. M3 W4-2 audit + W2-7 variant + W3 reproducibility + W2-9 host 한계 자료 합성.
