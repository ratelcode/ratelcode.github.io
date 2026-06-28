---
title: "파인튜닝으로 프레임·스케일 격차 닫기 — 무엇이 풀리고 무엇이 안 풀리나"
date: 2026-06-21
authors: ratel
excerpt: "30ep FT로 elbow out-of-support 74%→0.1%. 단 평가=학습 데이터라 낮은 잔차는 일반화가 아닌 in-distribution 충실도라는 경고."
tags:
  - SmolVLA 튜닝
  - SmolVLA
  - 파인튜닝
  - SO-101
---

> SmolVLA를 SO-101 실물 로봇에 올리면서 겪은 튜닝 과정을 정리한 시리즈입니다 (4/9). 측정·커밋으로 확인한 내용과 제 의견을 문장에서 구분해 적었습니다.

## TL;DR

- 사전학습된 `smolvla_base`를 우리 SO-101 로봇에 그대로 돌렸더니, elbow(팔꿈치) 관절의 예측이 사람이 시연한 범위를 한참 벗어났습니다(out-of-support 74%). 동작 폭이 약 3배로 부풀었고, 실기에서는 팔이 테이블을 파고들었습니다.
- 이유는 `smolvla_base`가 다른 분포(다른 로봇 프레임 + 다른 카메라 시점)로 학습된 정책이기 때문입니다. 우리 데이터의 정규화·관절 스케일·관절 조합·시점과 어긋났는데, 관절별 선형 보정으로는 "관절 조합"까지는 제약할 수 없었습니다.
- 그래서 우리 30개 에피소드(ep)로 `smolvla_base`를 파인튜닝(FT)했습니다. 한 번의 재학습으로 정규화·스케일·관절결합·시점이 모두 우리 분포로 맞춰졌습니다.
- 숫자로 보면 elbow out-of-support 0.738 → **0.001**, elbow MAE 99.55° → **4.71°**, 전체 잔차(residual_l2) i0 99.17 → **3.27**까지 떨어졌습니다. 학습 loss는 0.177 → 0.037이었고, RTX 5070 Ti에서 약 1시간 6분 걸렸습니다.
- 다만 안 풀린 부분이 있습니다. 평가에 쓴 30 ep가 학습에 쓴 바로 그 데이터라, 낮은 잔차는 "일반화"가 아니라 in-distribution 충실도(부분 암기 포함)에 가깝습니다. 데이터 다양성 부족(grasp이 물체 오른쪽으로 치우치는 편향)은 FT로 못 닫았습니다.

---

## 0. 배경 — 이 글만 읽어도 되게

SmolVLA는 카메라 이미지와 로봇 관절 상태(obs)를 받아, 앞으로 50스텝의 관절 명령(action chunk)을 한 번에 예측하는 정책(policy)입니다. 저는 이 정책의 사전학습 가중치 `smolvla_base`를 SO-101이라는 6관절 로봇팔에 적용하려 했습니다.

문제는 사전학습 정책이 우리 로봇, 우리 카메라로 찍은 데이터를 본 적이 없다는 점입니다. 그래서 zero-shot(추가 학습 없이 바로 적용)으로는 예측이 엉뚱하게 나왔습니다. 이 글은 그 격차를 파인튜닝으로 닫는 이야기입니다. 앞 글들에서 정규화 프레임·카메라 슬롯 문제를 부분적으로 손봤지만, 그래도 남은 격차가 있었고 그게 이 글의 출발점입니다.

> 데이터셋 기준을 먼저 적어 둡니다. 이 글의 FT와 측정은 첫 데이터셋(`so101_pnp_tray_v0`, 트레이 태스크 **30ep**)으로 진행했습니다. 시리즈 3·7·8·9편에서 쓰는 큐브 벤치마크(`so101_cube_v0`, **51ep**)는 같은 FT 기법을 더 큰 데이터로 반복한 다음 라운드입니다(이 글 §5의 "30→55~60ep" 계획이 실현된 것입니다).

---

## 1. 증상 — 무엇이 안 됐나

사전학습 `smolvla_base`를 SO-101 실기에 zero-shot으로 돌리니 elbow_flex(팔꿈치 굽힘) 예측이 사람 시연 범위 밖으로 외삽됐습니다. 정량적으로는, 오프라인 chunk 비교에서 예측 중 73.8%가 시연 envelope(시연에서 관측된 최소~최대 범위) 밖이었습니다(out-of-support fraction = 0.738). 그 결과 elbow 동작 폭이 부풀어, 실기에서 팔이 테이블을 파고드는 충돌이 발생했습니다.

여기서 "out-of-support fraction"이라는 지표가 중요합니다. 말 그대로 예측 액션이 사람이 시연한 적 없는 영역으로 얼마나 벗어나는지를 0~1로 잰 값입니다. 0이면 예측이 항상 시연 범위 안, 1이면 항상 밖입니다. 실기에서 본 적 없는 영역으로 명령을 내린다는 건 곧 안전하지 않다는 뜻이라, 이 지표는 충돌 위험의 직접 신호가 됩니다.

이 격차는 한 층이 아니라 3층이었습니다.

1. 좌표 프레임 오프셋 — 다른 로봇 프레임(SO-100 vs SO-101)으로 학습된 기준점 차이 (통계 remap으로 부분 완화)
2. z-스케일 증폭 — 동작 폭이 약 3배로 부풀음 (관절별 envelope 클립으로 부분 완화)
3. 관절 상관 + 시각 도메인 갭 — 관절들이 함께 움직이는 조합 자체가 어긋남 + 카메라 시점 차이 (선형/관절별 보정으로는 못 고침)

①②는 관절별로 더하기·자르기(오프셋·클립)로 손댈 수 있었지만, ③은 그게 안 됐습니다. 관절을 하나씩 따로 자르면 각 관절은 범위 안에 들어가더라도, 여러 관절이 같이 만드는 자세는 여전히 엉뚱할 수 있기 때문입니다.

---

## 2. 원인 — 왜 그랬나 (메커니즘)

`smolvla_base`는 우리 것과 다른 분포(다른 로봇 프레임 + 사전학습 카메라 시점)에서 학습된 정책입니다. 그래서 우리 SO-101 데모 분포와 네 가지가 어긋납니다. ① 정규화 ② 관절 스케일 ③ 관절 조합(상관) ④ 카메라 시점입니다.

이 중 ①②는 관절마다 따로 손볼 수 있는 격차라고 봅니다. shoulder는 +5°, elbow는 ÷3 같은 식이죠. 하지만 ③ 관절 조합은 다릅니다. "팔꿈치를 굽힐 때 어깨가 얼마나 따라 내려가는가" 같은 관절 간 결합은 관절을 하나씩 자르는 per-joint 클립으로는 표현할 수 없습니다. 그래서 클립을 걸어도 실기에서 여전히 땅을 파는 자세가 나왔습니다.

파인튜닝은 이 문제를 한 번에 흡수합니다. FT는 정책 가중치를 우리 데모 분포로 다시 최적화하므로, base가 외삽하던 관절 조합·스케일을 시연 envelope 안으로 끌어들입니다. ①②③④가 따로따로가 아니라 동시에 우리 데이터에 맞춰집니다.

그 결과 예측 액션이 시연 min~max 안에 머물러 out-of-support fraction이 0에 수렴했고, "예측이 시연 범위를 안 벗어난다"는 안전성 조건이 충족됐습니다. 실기에서 땅을 파는 충돌 위험이 사라진 것입니다.

> 메커니즘에는 함정이 하나 있습니다. 평가에 쓴 데이터가 학습에 쓴 데이터와 같다는 점입니다. 그래서 낮아진 잔차는 "새 상황에서도 잘한다(일반화)"가 아니라 "학습한 분포를 잘 맞춘다(in-distribution 충실도, 부분 암기 포함)"를 반영합니다. base는 이 데이터를 본 적이 없으니, 이 비교는 공정한 일반화 비교가 아니라 "FT는 학습분포를 맞추고 base는 못 맞춘다"만 보여줍니다. 자세한 함의는 §4·§5에서 다룹니다.

---

## 3. 무엇을 바꿨나 (파일·플래그·값)

`workbench/so101/scripts/train_smolvla.py`로 `smolvla_base`를 FT했습니다 (`lerobot-train` 래퍼). 커밋은 `e39ac52`(카메라 매핑 해결) + `d174214`(FT 성공 + 비교 측정)입니다.

### 3-1. 카메라 슬롯 매핑이 관건

`smolvla_base`의 config는 입력에 `observation.images.camera1/2/3`을 명시하고 `empty_cameras=0`으로 둡니다. 우리 데이터는 카메라가 `top`/`wrist` 2개뿐이라, 그냥 FT하면 내부 `prepare_images`에서 일치하는 카메라 키가 0개가 되어 "All image features missing" 크래시가 납니다. 무음 실패가 아니라 즉시 에러가 떨어집니다.

이건 lerobot의 `--rename_map`으로 런타임에 키 이름을 슬롯에 맞춰서 해결했습니다.

```text
--rename_map='{"observation.images.top":"observation.images.camera1",
               "observation.images.wrist":"observation.images.camera2"}'
```

camera3은 매핑에서 제외했으니 실카메라 2개로 학습한 셈입니다. 이미지 정규화가 `VISUAL=IDENTITY`라 카메라 키-통계 불일치는 무관하고, state/action만 `MEAN_STD`로 정규화돼 키가 일치합니다.

### 3-2. 학습 설정

학습 설정은 다음과 같습니다 (`m7w3_smolvla_finetune.md` §1).

| 항목 | 값 |
|---|---|
| trainable / 전체 파라미터 | 99.9M / 450M (`freeze_vision_encoder=True` + `train_expert_only=True`, SmolVLA 기본) |
| batch_size | 8 (약 11.7GB / 16GB VRAM) |
| steps | 20,000 (약 5.9 epoch) |
| save_freq | 2,000 |
| seed | 1000 |
| 기타 필수 | `--policy.device=cuda`, `--policy.push_to_hub=false`, `--wandb.enable=false` |
| 학습 시간 | 약 1시간 6분 (RTX 5070 Ti) |
| loss | 0.177 → 0.037 (단조 수렴, NaN 없음) |

비교는 `workbench/so101/scripts/svla_compare_ft.py`로 했습니다. 로봇 없이 오프라인으로 chunk fidelity를 측정합니다. base·FT 둘 다 unnormalize는 데이터셋 통계로 통일하고, 둘 다 모델 config가 기대하는 이미지 슬롯(`camera1/2/3`)을 자동으로 채워(누락 슬롯 = zeros) 입력 텐서 계약을 맞춥니다.

---

## 4. 전후 비교

![파인튜닝 전후 예측이 시연 범위를 벗어나는 정도](/diagrams/04_finetune_oos.svg)

*캡션: 회색 띠는 사람이 시연으로 보여준 동작 범위입니다. 파인튜닝 전(위)에는 예측(빨강 점)이 띠 밖으로 많이 삐져나갔지만, 30ep 파인튜닝 후(아래)에는 예측(초록 점)이 대부분 띠 안으로 모였습니다.*

여기서 "out-of-support(시연 범위 밖)"란, 정책이 내놓은 예측이 사람이 한 번도 시연하지 않은 동작 영역으로 벗어났다는 뜻입니다. 로봇이 가 본 적 없는 자세로 명령이 나가는 것이라, 띠 밖의 점이 많을수록 테이블을 파고드는 것 같은 충돌 위험이 커집니다. 파인튜닝은 바로 이 점들을 띠 안쪽으로 끌어들이는 작업입니다.

아래 수치는 전부 `workbench/reports/svla_compare_base_vs_ft.json`에서 측정했습니다 (30 ep × 5개 chunk 시작점, chunk_size 50, 오프라인).

| 지표 | base (zero-shot) | FT | 출처 키 |
|---|---|---|---|
| **elbow_flex out-of-support fraction** | 0.738 | **0.001** | `out_of_support_fraction.elbow_flex` |
| out-of-support fraction (전체 평균) | 0.217 | **0.013** | `oos_overall` |
| shoulder_lift OOS fraction | 0.27 | **0.034** | `out_of_support_fraction.shoulder_lift` |
| wrist_roll OOS fraction | 0.129 | **0.001** | `out_of_support_fraction.wrist_roll` |
| elbow_flex MAE (deg) | 99.55 | **4.71** | `per_joint_mae.elbow_flex` |
| shoulder_lift MAE (deg) | 78.09 | **3.61** | `per_joint_mae.shoulder_lift` |
| shoulder_pan MAE (deg) | 19.62 | **1.19** | `per_joint_mae.shoulder_pan` |
| residual_l2 i0 (chunk 첫 스텝) | 99.17 | **3.27** | `residual_l2.i0` |
| residual_l2 i49 (chunk 끝 스텝) | 173.68 | **12.11** | `residual_l2.i49` |
| elbow_flex 예측 motion range (deg) | 129.37 | **21.92** | `pred_motion_range_mean.elbow_flex` |
| training loss | 0.177 | **0.037** | `m7w3_smolvla_finetune.md §1` |
| 추론 지연 평균 (ms, chunk predict) | 239.4 (base 3cam) | 284.4 (ft 2cam) | `latency_ms_mean` |

표를 읽는 법은 이렇습니다.

- OOS가 0으로 수렴했다는 건 예측이 시연 범위 밖으로 거의 안 나간다는 뜻이고, 안전성 조건을 충족했다는 의미입니다. elbow가 74%에서 0.1%로 떨어진 게 가장 큰 변화입니다.
- MAE와 residual_l2가 급감한 건 예측이 사람 시연 궤적을 정밀하게 따라간다는 뜻입니다. 단 §5의 함정 때문에 이건 "일반화"가 아니라 "학습분포 적합"으로 읽어야 합니다.
- elbow 예측 motion range가 129°에서 22°로 줄어든 건, 부풀었던 z-스케일이 시연 스케일로 정상화됐다는 신호입니다.
- 지연은 오히려 늘었습니다(239 → 284ms). base는 3캠(빈 슬롯 포함), FT는 2캠 구성이라 단순 비교는 아니지만, FT가 추론을 더 빠르게 만들지는 않았다는 점은 기록해 둘 가치가 있다고 봅니다.

---

## 5. 무엇이 안 풀렸나 — 방법론적 경고

이 글에서 제일 강조하고 싶은 부분입니다.

비교 평가에 쓴 30 ep는 FT 학습에 쓴 바로 그 30 ep입니다. 이번엔 별도 held-out split 없이 전부 학습에 썼습니다. 따라서 FT의 낮은 residual은 일반화가 아니라 in-distribution 충실도(부분 암기 포함)이며, base와의 공정한 일반화 비교가 아닙니다.

그럼에도 이 측정에 의미가 있다고 보는 이유가 있습니다. out-of-support fraction이 0에 수렴한 건 암기와 무관한 안전성 지표이기 때문입니다. "예측이 시연 범위를 안 벗어난다"는 건 학습 데이터를 외웠는지와 별개로, 실기 충돌 위험이 사라졌다는 직접 신호입니다. 그래서 이 한 가지는 신뢰할 수 있다고 봅니다.

FT가 닫은 것은 분포 안의 격차(정규화·스케일·관절결합·시점)뿐입니다. 데이터 다양성 부족, 즉 분포 밖의 격차는 못 닫았습니다. 실제 후속 실기 rollout에서 grasp 타겟이 오른쪽으로 체계적으로 치우치는 편향이 남았고, 제어 rate를 풀어도 동일했습니다. 이는 제어가 아니라 학습된 grasp 타겟 자체의 편향, 즉 BC(behavior cloning) 데이터 부족의 전형적 신호입니다.

제 계획으로는 이 편향을 다음 데이터 라운드(30 → 55~60 ep)와 train/val 분리로 보완하려 합니다 (시리즈 9편 주제). "FT로 풀리는 격차"와 "데이터로만 풀리는 격차"는 종류가 다르다는 점이 요점입니다.

> 진짜 일반화는 새 물체 위치에서의 실기 rollout으로만 판정됩니다. 오프라인 잔차가 아무리 낮아도, held-out이 아니면 그건 일반화의 증거가 아닙니다.

---

## 6. 회사에 적용한다면 (transferable 체크포인트)

ML/로봇 파이프라인에 이 기법을 옮길 때 점검할 항목을 정리했습니다.

1. 평가 데이터와 학습 데이터를 절대 섞지 마세요. 같은 데이터로 평가하면 낮은 loss/residual은 일반화가 아니라 in-distribution 충실도(부분 암기)일 뿐입니다. FT 전부터 train/val을 분리하고, 진짜 일반화는 held-out(여기선 새 물체 위치 실기 rollout)으로만 판정하세요.

2. "안전성"과 "일반화"를 다른 지표로 분리해 측정하세요. 예측이 학습/시연 범위 밖으로 안 나가는지(out-of-support fraction 같은 분포-안 지표)와, 새 상황에서 성공하는지(held-out 성능)는 별개입니다. 전자는 암기 여부와 무관하게 신뢰할 수 있고, 후자는 held-out 없이는 알 수 없습니다.

3. 사전학습 정책의 입력 텐서 계약을 FT 전에 config에서 직접 확인하세요. `smolvla_base`처럼 `camera1/2/3` 슬롯 + `empty_cameras=0`을 강제하는 경우, 카메라 수가 달라도 `rename_map`으로 슬롯에 맞추고 누락 슬롯은 zeros로 채워야 합니다. 안 그러면 "All image features missing" 크래시나 추론 freeze가 납니다.

4. "FT로 풀리는 격차"와 "데이터로만 풀리는 격차"를 구분하세요. 분포-안 격차(정규화·스케일·관절결합·시점)는 FT가 한꺼번에 닫습니다. 하지만 데이터 다양성 부족(체계적 편향 등 분포-밖)은 FT로 안 풀리고 데이터 수집 라운드로만 해결됩니다. 잔차가 안 떨어지는 게 아니라 특정 방향으로 일관되게 틀린다면, 그건 학습 문제가 아니라 데이터 문제입니다.

---

## 시리즈 내비

- 이전 글: [2개 카메라로 학습했는데 정책이 멈춘 이유 — 카메라 슬롯 함정](./smolvla_tuning_03_camera_slot.md)
- 다음 글: [팔이 reach만 반복하고 안 들렸던 이유 — 액션 청크 실행 구간](./smolvla_tuning_05_exec_horizon.md)

> 원본 학습 노트: `docs/learning/m7w3_smolvla_finetune.md` · 측정 raw: `workbench/reports/svla_compare_base_vs_ft.json`
