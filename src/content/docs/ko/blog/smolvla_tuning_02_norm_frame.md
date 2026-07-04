---
title: "왜 사전학습 SmolVLA가 팔도 못 들었나 — 정규화 프레임 불일치"
date: 2026-06-21
authors: ratel
excerpt: "정규화 통계가 다른 로봇(SO-100) 좌표 프레임에 박혀 좌표가 조용히 점프했다. 통계 한 세트 교체로 chunk 시작점 잔차 232.7→105.8."
homeCover: "/diagrams/02_norm_frame.svg"
metric: "232.7 → 105.8"
tags:
  - SmolVLA 튜닝
  - SmolVLA
  - 정규화
  - SO-101
---

> SmolVLA를 SO-101 실물 로봇에 올리면서 겪은 튜닝 과정을 정리한 시리즈입니다 (2/9). 측정·커밋으로 확인한 내용과 제 의견을 문장에서 구분해 적었습니다.

## TL;DR

공개 사전학습 정책 `lerobot/smolvla_base`를 우리가 직접 수집한 SO-101 실기 데이터셋에 zero-shot으로 돌렸더니 팔이 거의 못 움직였습니다. 오프라인 chunk 비교에서 오차가 두 관절(shoulder_lift, elbow_flex)에만 몰렸고(각각 MAE 203°·127°), 온라인 rollout에서는 팔꿈치가 시연에 없던 영역까지 과신전해 전완이 테이블 평면 아래로 꽂혔습니다(SR 0/1, 사용자 중단).

원인은 action 정규화 통계(mean/std)가 다른 로봇(SO-100) 좌표 프레임에 박혀 있던 것이었습니다. 정책 출력은 정규화 공간 값이라, 복원에 쓰는 통계가 곧 로봇 좌표계를 정의합니다. 통계가 다른 rig 것이면 같은 출력이 우리 좌표로 디코딩될 때 좌표가 조용히 점프합니다.

해결은 단순했습니다. unnormalize 통계를 사전학습(so100-blue) 대신 우리 데이터셋 통계로 교체했습니다(플래그 `UNNORM=ours`). 가중치와 입력은 그대로 뒀습니다.

수치로 보면 chunk 시작점 잔차 L2가 **232.7 → 105.8**, shoulder_lift MAE가 **203° → 76°**, wrist_roll MAE가 21.6° → 4.9°로 떨어졌습니다. 다만 elbow는 std 비율(우리가 약 3배 넓음) 때문에 평균 보정만으로는 127° → 105.6°에 그쳐 잔차가 남았습니다.

---

## 1. 증상 — 무엇이 안 됐나

배경을 한 줄로 정리하면, SmolVLA는 카메라 영상과 로봇 관절 상태를 받아 앞으로 50스텝의 action을 한 번에 내놓는 정책입니다. "zero-shot"은 우리 로봇 데이터로 추가 학습 없이 공개 가중치를 그대로 쓰는 것을 말합니다.

우리는 SO-101 팔로 픽앤플레이스 5에피소드(`so101_pnp_tray_v0`)를 사람 손(teleop)으로 직접 수집했습니다. 그걸 정답(teleop GT)으로 두고, 같은 입력을 사전학습 SmolVLA에 넣어 출력 chunk를 비교했습니다(offline replay, 로봇 비구동). 그리고 안전 가드를 달고 실제 팔에서도 돌려봤습니다(online rollout).

오프라인 비교 결과(`smolvla_real_so101_pnp_replay.json`)는 이렇습니다.

- chunk 시작점(i0) 잔차 L2가 **232.65**였습니다. M2에서 SO-100 학습분포에 가까운 공개셋을 같은 방식으로 잰 비교값(107.5, 노트 §1)의 약 2.2배입니다.
- 실패가 두 관절에 집중된 비대칭이었습니다. per-joint MAE가 shoulder_lift 203.2°, elbow_flex 127.1°였는데, 나머지 팔 관절(shoulder_pan 14.4°, wrist_flex 14.1°, wrist_roll 21.6°)은 14~21° 수준에 머물렀습니다.

온라인 rollout(`smolvla_rollout_so101_v0.json` + 노트 §5·§6)에서 일어난 일은 다음과 같습니다.

우리-통계 remap에 상대목표 클램프(5°/step), 15Hz, chunk 50 중 앞 10스텝만 실행 후 재예측 조건으로 45 chunk / 450 step을 완주했고, 처음엔 인코더 지표상 충돌 없이 "hover"로 보였습니다(`v0` JSON). 그런데 다음 날(2026-06-12) `.rrd` 명령 시계열을 GT envelope와 대조하니, elbow가 시연에 없던 139°까지 과신전(GT 최대 83°+56°)해 전완이 테이블 평면 아래로 꽂히는 "땅 파기"였음이 드러났습니다(노트 §6). SR은 0/1이었고, 모터가 테이블을 상대로 반복 stall하면서 파손 위험이 생겨 사용자가 실험을 중단했습니다.

> 139° 명령 최대값과 관절별 명령 범위는 `.rrd`(미추적 영상 로그)에서 추출한 노트 §6 서술값입니다. 추적되는 rollout JSON에는 명령 시계열이 저장돼 있지 않아 JSON으로 직접 재확인하진 못했습니다.

화면으로 보기 전, 그러니까 인코더 지표만 볼 때는 이 거동을 "hover(머뭇거림)"로 오독했습니다. 실제로는 "땅 파기"였습니다. 개인적으로는 인코더 지표보다 육안 관찰이 먼저였어야 했다고 봅니다.

---

## 2. 원인 — 왜 그랬나

### 정규화 공간과 복원식

SmolVLA의 `predict_action_chunk`는 정규화 공간의 텐서 `(B, 50, 6)`를 내놓습니다. 이걸 실제 로봇 각도로 되돌리는 식은 간단합니다.

```
로봇 좌표값 = 정규화_출력 * std + mean
```

여기서 `mean`/`std`는 학습할 때 쓴 데이터의 통계입니다. 즉 이 mean/std가 어떤 로봇 좌표계인지를 정의합니다. 정규화 출력 자체는 단위 없는 숫자일 뿐이고, 어떤 좌표로 풀리느냐는 전적으로 복원에 쓰는 통계에 달려 있습니다.

### 통계가 다른 로봇 것이었다

노트 §2에서 세 소스의 action mean을 직접 대조해 보니, `smolvla_base`는 SO-100 rig 분포로 action을 정규화해 학습했고 복원 통계가 `so100-blue.buffer.action`에 박혀 있었습니다(스크립트 출력 `unnorm_stats_key="so100-blue.buffer.action"`로 확인). 그 통계의 lift mean은 +125.7°, elbow mean은 +125.4°, roll mean은 -106.5°였습니다.

그런데 우리 SO-101 시연에서 그 관절들의 실제 작동 중앙은 lift -27.6°, elbow +22.4°, roll -4.6°였습니다. 복원식이 SO-100 mean을 강제하니 lift/elbow 결과가 GT 대비 100°가 넘게 어긋납니다. 다른 관절은 두 rig의 프레임이 우연히 비슷해서 14~21°로 멀쩡했던 것이고, 그래서 실패가 두 관절에만 몰린 비대칭이 나왔습니다.

여기서 짚고 싶은 건 "우리 calibration이 틀렸나?"가 아니라는 점입니다. 우리 rig 작동 범위(lift -27.6°, elbow +22.4°)는 SO-101 커뮤니티 데이터셋과 같은 영역대였습니다(노트 §2 비교). 제 생각에는 비표준인 쪽은 우리 셋업이 아니라 모델이 들고 있는 통계의 프레임입니다.

정책이 내놓는 출력은 그 자체로는 단위가 없는 숫자라, 어떤 mean/std를 곱해 되돌리느냐가 곧 로봇 좌표계를 정합니다. 그래서 같은 출력 하나라도 SO-100 통계로 풀면 엉뚱한 관절 각도로, 우리 데이터 통계로 풀면 제자리로 디코딩됩니다. 아래 그림이 이 갈림을 한눈에 보여줍니다.

![정규화 복원 통계 차이로 생기는 좌표 점프](/diagrams/02_norm_frame.svg)

*같은 정규화 출력이라도 SO-100 통계로 복원하면 좌표가 점프해 시작점 잔차가 232.7이 되고, 우리 데이터 통계로 복원하면 제자리로 돌아와 잔차가 105.8로 줄어듭니다.*

### 평균만 옮긴다고 다 풀리진 않는다 (2차 원인)

우리 elbow std는 53.6°로, 사전학습 rig elbow std(노트 §6 서술값 18.2°)의 약 3배입니다. 선형 mean/std remap은 평균(1차 모멘트)은 맞춰주지만 분산 비율은 못 고칩니다. 모델이 "z ±2 정도"의 의도로 낸 출력이 우리 std로 복원되면 ±107° 영역으로 디코딩되니, z-스케일이 과증폭됩니다.

> 사전학습 rig elbow std 18.2°는 노트 본문 서술값입니다. repo 내 별도 통계 파일로 직접 재확인하진 못했습니다. 우리 53.6°와의 약 3배 비율은 노트와 일관하며, 이 글의 결론(분산 비율이 2차 원인)은 그 비율에만 의존합니다.

3차 원인으로 관절 상관과 시각 도메인 갭(시연 manifold 자체가 다름, 노트 §9)도 있지만, 이 글의 본론은 1·2차입니다.

---

## 3. 무엇을 바꿨나

가중치도, 입력도 건드리지 않았습니다. 복원에 쓰는 통계 한 세트만 바꿨습니다.

파일은 `workbench/so101/scripts/svla_real_replay.py`이고, 플래그는 환경변수 `UNNORM=ours`입니다.

`postprocess.steps`에 박힌 `so100-blue.buffer.action` 통계 대신, 우리 데이터셋의 `meta/stats.json["action"]`에서 mean/std를 읽어 복원합니다(출력 키가 `OURS:so101_pnp_tray_v0`로 바뀝니다).

```python
# UNNORM=ours 일 때: 우리 데이터셋 통계로 복원
stats = json.load(open(DS_ROOT / "meta/stats.json"))["action"]
action_mean = torch.tensor(stats["mean"], ...)
action_std  = torch.tensor(stats["std"],  ...)
pred = pred * action_std + action_mean   # SO-100 통계 대신 우리 프레임
```

추가로 둔 안전 가드는 두 가지입니다.

- `ENV_CLIP=1`은 예측을 시연 min~max(stats.json의 action min/max)로 클립하고, 관절별 클립 발동률을 기록합니다(`clip_fraction_per_joint`). 이게 다음 절의 "out-of-support fraction" 지표가 됩니다.
- 온라인 rollout(`svla_rollout.py`)에는 `unnorm=OURS`에 더해 상대목표 클램프(`max_relative_target=5°/step`, v0 기준), 15Hz, chunk 50 중 앞 10스텝만 실행 후 재예측, 40초 자동 종료를 걸었습니다.

커밋 `bf9ca5e`(2026-06-12)가 `svla_rollout.py`에 안전장치 3종을 넣었습니다. (1) GT envelope 클립, (2) `Present_Load` 과부하 자동중단, (3) 종료 시 torque off 검증입니다. 이 가드들은 실기 출력을 신뢰하기 전의 전제라 이번 정규화 검증부터 끝까지 깔고 갔습니다. 동작 원리나 deg/s 불변 같은 안전 가드의 자세한 설계는 [6편](./smolvla_tuning_06_safety_clamp.md)에서 전담합니다.

> 맥락을 분리해 두면, teleop pan ~25° 틀어짐은 최초 calibration의 homing 자세가 한쪽 팔만 ~25° 돌아간 채 기록된 별개 문제였습니다(노트/셋업 §9). 재캘리브레이션으로 완치됐고 이번 정규화 이슈와는 무관합니다.

---

## 4. 전후 비교

통계 교체(remap)만으로 평균이 맞는 관절은 급회복했습니다. elbow는 std 비율 왜곡이 남아 envelope 클립을 추가로 걸어야 절반으로 줄었습니다. 아래 수치는 세 리포트 JSON(`_replay` / `_unnorm_ours` / `_envclip`)에서 직접 대조한 값입니다.

| 지표 | 전 | 후 | 출처 |
|---|---:|---:|---|
| 잔차 L2 i0 (chunk 시작점) | 232.7 | **105.8** | `_replay.json` → `_unnorm_ours.json` |
| 잔차 L2 i49 (chunk 끝) | 266.0 | 173.3 | 동일 |
| shoulder_lift MAE (°) | 203.2 | **76.1** | 동일 (per-joint) |
| wrist_roll MAE (°) | 21.6 | **4.9** | 동일 (-77%) |
| elbow_flex MAE (°) — remap만 | 127.1 | 105.6 | 동일 (-17%, 잔존) |
| elbow_flex MAE (°) — remap + envelope 클립 | 105.6 | **51.3** | `_unnorm_ours.json` → `_envclip.json` (-51%) |
| 잔차 L2 i49 — remap → remap+클립 | 173.3 | 127.9 | 동일 (-26%) |

위험을 정량화하면, envelope 클립 리포트(`_envclip.json`)에서 elbow_flex의 클립 발동률(out-of-support fraction)이 0.834였습니다. 즉 elbow 예측의 83.4%가 시연 범위 밖이었습니다(`clip_fraction_per_joint`). 이게 std 약 3배 증폭이 만든 외삽 행동의 정량화입니다.

온라인 rollout(노트 §6, `.rrd` 명령 시계열 추출)에서도 같은 그림이 나왔습니다. GT elbow 범위는 -51~83°였는데 rollout 명령은 -46~139°(GT 최대를 +56° 초과)까지 휘둘렸습니다.

i0(시작점)은 잘 회복했지만 i49(끝, open-loop 후반)와 실기 충돌은 남았습니다. offline 지표가 좋다고 online이 안전하다는 보장은 없다고 봅니다.

> §4 표의 per-joint MAE·clip_fraction·잔차 L2·unnorm_stats_key 값은 위 세 리포트 JSON에서 직접 대조했습니다(잔차 L2 집계 평균은 각 리포트의 chunk_records 평균과 일치 확인). 온라인 rollout의 latency나 클램프 발동 횟수 등은 노트 본문값이라 이 글에선 단언하지 않습니다(원노트 §5 참조).

---

## 5. 회사에 적용한다면

다른 form factor/rig에서 학습된 사전학습 정책을 평가·배포할 때의 체크포인트입니다.

1. action 통계의 출처 프레임을 먼저 확인합니다. 제 생각에는 mean/std는 단순 정규화 메타가 아니라 로봇 좌표 프레임을 정의하는 1급 변수입니다. 같은 체크포인트가 통계 한 줄(SO-100 vs 우리 데이터셋) 차이로 SR 0과 의미 있는 거동을 오갑니다. SR만 보고하면 프레임 불일치가 '모델 실패'로 오기록됩니다. 타깃 rig 통계로 remap하는 것이 1순위입니다.

2. 평균(remap)으로 안 풀리는 잔차는 분산 비율을 의심합니다. 선형 mean/std 보정은 평균을 옮길 뿐 스케일 왜곡은 못 고칩니다. 우리 elbow는 std가 약 3배라 remap 후에도 105.6°가 잔존했습니다. std ratio(z-스케일)를 같이 봅니다.

3. 'out-of-support action fraction'을 표준 진단 지표로 채택합니다. 예측이 시연 min~max 밖인 비율을 말합니다. SR 0인 두 정책도 이 지표로 '위험한 외삽형(elbow 83% OOS)'과 '범위 내 미달형'으로 구분된다고 봅니다. 어느 쪽인지에 따라 대응이 완전히 달라집니다.

4. 실기 전에 안전 가드를 선장착합니다. envelope 클립, 상대목표 클램프(5°/step), 부하(`Present_Load`) 자동중단을 기본값으로 둡니다. 그리고 인코더 지표보다 육안 관찰을 우선합니다. 우리의 'hover' 오독이 실제로는 '땅 파기'였으니까요.

cross-embodiment zero-shot 실패는 우리 셋업 결함이 아니라 분야 전체의 구조적 한계라고 봅니다. 표준 해법은 타깃 rig에서 소규모(예: 50ep) 파인튜닝이라, remap이나 클립으로 못 푼 elbow z-스케일 잔존은 다음(4편) 파인튜닝 경로로 넘깁니다.

> cross-embodiment zero-shot ≈0%가 분야 합의라는 서술은 외부 문헌(SmolVLA paper, VLA survey 2508.13073, VLA-Pilot 2511.14178 등) 인용으로, repo 내부 측정이 아닙니다(노트 §7.1, 2026-06-20 웹 검색 기반).

---

## 시리즈 내비

- 이전: [SmolVLA 실기 튜닝 여정 — 팔도 못 들던 정책을 부드러운 픽앤플레이스로 (시리즈 지도)](./smolvla_tuning_01_intro.md)
- 다음: [2개 카메라로 학습했는데 정책이 멈춘 이유 — 카메라 슬롯 함정](./smolvla_tuning_03_camera_slot.md)
