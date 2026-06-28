---
title: "실물 로봇 안전 장치 — 상대목표 클램프 + 과부하 자동중단"
date: 2026-06-21
authors: ratel
excerpt: "정책 raw 출력을 모터로 직송하면 위험하다. envelope 클립 + per-step 클램프 + Present_Load 과부하 중단 3층 가드, 그리고 deg/s 불변 설계."
tags:
  - SmolVLA 튜닝
  - SmolVLA
  - 안전
  - SO-101
---

> SmolVLA를 SO-101 실물 로봇에 올리면서 겪은 튜닝 과정을 정리한 시리즈입니다 (6/9). 측정·커밋으로 확인한 내용과 제 의견을 문장에서 구분해 적었습니다.

## TL;DR

- 실물 SO-101에서 SmolVLA가 내놓은 명령이 시연했던 동작 범위를 크게 벗어났습니다. 팔이 들리는 대신 테이블 평면 아래로 꽂혀 모터가 stall에 빠졌고, 파손 위험이 보여 사용자가 온라인 실험을 즉시 중단했습니다(2026-06-12).
- 원인은 정책의 raw 출력을 모터로 보내기 전에 (a) 분포 밖 외삽 차단, (b) 한 스텝 이동량 제한, (c) 과부하 자동중단, 이 세 가드가 전혀 없었기 때문입니다.
- 그래서 커밋 `bf9ca5e`에서 3종 가드를 추가했습니다. GT envelope 클립, per-step 상대목표 클램프, Present_Load 과부하 자동중단, 그리고 종료 시 토크 해제 재검증까지 붙였습니다.
- 오프라인 선검증에서는 elbow 명령의 **83.4%**가 시연 범위 밖이라 클립이 가장 강하게 발동했고, elbow MAE는 105.6°에서 51.3°로 떨어졌습니다. 이후 FT 체크포인트 실기 rollout(2026-06-21)은 `aborted=null`, `torque_off_verified=true`로 안전하게 종료됐습니다.

---

## 1. 증상 — 무엇이 안 됐나

먼저 배경부터 짚겠습니다. SmolVLA는 카메라 이미지와 관절 상태(obs)를 받아 다음 50스텝의 관절 명령(action chunk)을 한 번에 내놓는 정책입니다. 이 명령을 실물 6관절 팔(SO-101)의 모터로 그대로 흘려보내는 것이 "온라인 rollout"입니다.

문제는 정책이 내놓은 명령이 사람이 직접 시연하며 모은 데이터의 동작 범위를 한참 벗어났다는 데 있었습니다.

시연 데이터에서 elbow_flex 관절이 실제로 도달한 범위는 -51~83°였습니다. 그런데 클립을 걸기 전 zero-shot rollout에서 정책 명령은 **-46~139°**까지 나갔습니다. 시연 최대치 83°를 56°나 초과한 과신전입니다(`m7w2_smolvla_real_zeroshot.md` §6 표).

그 결과 전완이 테이블 평면 아래로 꽂혔습니다. 팔을 들지 않고 땅을 파는 모양새였고, 모터는 테이블을 상대로 stall에 빠졌습니다. 파손 위험이 분명해서 사용자가 온라인 실험을 즉시 중단했습니다(2026-06-12).

요약하면 정책 출력을 그대로 믿어도 되는지 검증하는 가드가 한 겹도 없던 상태였습니다.

---

## 2. 원인 — 왜 그랬나

명령이 범위를 벗어난 1차 원인은 이 시리즈 2편에서 다룬 rig 간 통계 불일치입니다.

우리 데이터의 elbow 표준편차(std)는 53.6°인데, SmolVLA가 사전학습된 rig의 elbow std는 18.2°였습니다. 약 3배 차이입니다. 모델이 "평균에서 z=±2만큼 움직여라"라고 의도하면, std가 3배 큰 우리 좌표계에서는 그게 ±107°짜리 휘두르기로 디코딩됩니다. 그래서 시연에 없는 139°까지 외삽이 일어난 것입니다(`m7w2` §6).

선형 mean/std remap, 즉 평균·분산으로 좌표를 맞추는 보정은 1차 모멘트(평균)만 맞춰 줄 뿐, 분산비(z-스케일) 왜곡까지는 보정하지 못합니다.

여기에 더해 가드 자체가 없었기 때문에 다음 세 가지가 모두 무방비였습니다.

1. 분포 밖 외삽을 차단할 장치가 없었습니다. 시연이 한 번도 가본 적 없는 좌표로 명령이 나가도 막을 게 없었습니다.
2. 스텝 목표의 점프를 제한할 장치가 없었습니다. 한 번에 큰 각도를 점프하면 모터에 토크 스파이크가 걸립니다.
3. 과부하 중단과 종료 검증이 없었습니다. 모터가 무언가에 막혀 부하가 치솟아도 멈출 장치가 없었고, 끝낼 때 토크가 실제로 풀렸는지도 확인하지 않았습니다.

여기에 더해, 이건 나중에 M7W4에서 보강하며 깨달은 부분인데, per-step 이동 제한을 "스텝당 몇 도(°/step)"라는 상수로 박으면 제어 주파수(Hz)에 묶이는 구조적 결함이 생깁니다. Hz를 바꾸면 실효 속도 천장(°/s)이 조용히 달라집니다. 예컨대 8°/step을 둔 채 15Hz에서 30Hz로 올리면 안전 천장이 120°/s에서 240°/s로 2배 완화되어 버립니다.

---

## 3. 무엇을 바꿨나

커밋 `bf9ca5e`가 `workbench/so101/scripts/svla_rollout.py`에 send 직전 3층 가드를 추가했습니다(이 커밋은 해당 파일 한 개만 수정합니다). 순서가 중요합니다. 역정규화 → envelope 클립 → per-step 클램프 → 과부하 감시 순서로 동작합니다.

![정책 출력에서 모터로 가기 전 3층 안전 가드](/diagrams/06_safety_guards.svg)

*캡션: 정책이 내놓은 raw 명령은 envelope 클립 → per-step 클램프 → 과부하 감시를 차례로 통과한 뒤에야 모터로 전달됩니다.*

세 가드는 막는 대상이 서로 다릅니다. envelope 클립은 시연이 한 번도 가본 적 없는 좌표를, per-step 클램프는 한 스텝에 너무 큰 각도로 점프하는 명령을, 과부하 감시는 모터가 무언가에 막혀 부하가 치솟는 상황을 각각 끊어 냅니다. 이 세 단계는 한 번만 거치는 관문이 아니라 매 제어 스텝마다 같은 순서로 다시 실행됩니다.

### (a) GT envelope 클립 — 분포 밖 외삽 차단

역정규화 직후, 시연 데이터셋 `stats.json`의 관절별 min~max로 명령을 독립적으로 잘라냅니다. 사전학습 분포에서 외삽된 명령이 "시연이 실제로 가본 좌표"로 잘리므로, 테이블 평면 침투 같은 물리 충돌을 막을 수 있습니다.

```python
# stats.json의 관절별 min/max로 텐서 구성
a_min = torch.tensor(STATS["min"], dtype=torch.float32, device="cuda")
a_max = torch.tensor(STATS["max"], dtype=torch.float32, device="cuda")
...
chunk = chunk * a_std + a_mean             # 역정규화
if env_clip:
    chunk = torch.clamp(chunk, a_min, a_max)   # ENV_CLIP=0 으로 끔
```

### (b) per-step 상대목표 클램프 — 한 스텝 이동량 제한

lerobot의 `ensure_safe_goal_position`(`lerobot/src/src/lerobot/robots/utils.py:91`)이 관절별로 `|goal - present|`를 cap으로 제한합니다. 한 스텝 목표가 크게 점프하면 cap만큼만 움직이고 나머지는 다음 스텝으로 미뤄, 토크 스파이크를 막습니다.

```python
# 동기 스크립트: 상수 (sweep 변수)
MAX_REL = float(os.environ.get("MAX_REL", "5.0"))   # °/step
SOFollowerRobotConfig(max_relative_target=MAX_REL, ...)
```

이후 M7W4의 RTC 스크립트(`svla_rollout_rtc.py:74-75`)에서는 이 cap을 각속도에서 파생하도록 바꿨습니다. `MAX_REL = MAX_VEL_DPS / CONTROL_HZ`(기본 `MAX_VEL_DPS=120`°/s)로 두면, Hz를 올려도 속도 천장이 120°/s로 고정됩니다.

### (c) Present_Load 과부하 자동중단 + 종료 검증

매 스텝 `sync_read`로 최대 부하를 읽어, raw 값 500(약 50%)을 연속 5스텝(15Hz 동기 스크립트 기준 약 0.33s) 초과하면 abort하고 루프를 빠져나옵니다. 종료할 때는 토크를 끄고, 실제로 꺼졌는지 재read로 검증합니다.

```python
OVERLOAD_RAW = 500    # Present_Load raw (~50%)
OVERLOAD_STEPS = 5    # 연속 5스텝 초과 시 abort
overload_streak = overload_streak + 1 if max_load > OVERLOAD_RAW else 0
if overload_streak >= OVERLOAD_STEPS:
    ...  # abort + break (reason/load/t_s 메타 기록)
...
robot.bus.disable_torque()
tq = robot.bus.sync_read("Torque_Enable", normalize=False)
out["torque_off_verified"] = not any(tq.values())
```

M7W4 효율화 리뷰에서 "Present_Load read를 N스텝마다로 줄여 critical-path를 단축하자"는 제안이 나왔지만, 저는 이걸 의도적으로 기각했습니다. 여유는 안전 비용이 0인 곳(jpeg 압축·CPU 역정규화)에서만 회수하고, 부하나 이상 감시 read는 매 제어 스텝 유지하는 게 맞다고 봅니다.

---

## 4. 전후 비교

아래 수치는 오프라인 replay 선검증(2026-06-12, smolvla_base zero-shot, `so101_pnp_tray_v0` 5ep)에서 측정한 값입니다. 실기 구동이 아니라 정책 출력을 시연 데이터에 대고 비교한 결과입니다.

| 지표 | before (remap만) | after (remap+클립) |
|---|---|---|
| elbow_flex 클립 발동률(예측 스텝 중 시연 envelope 밖 비율) | 83.4% | 시연 min~max로 절단 |
| elbow MAE | 105.6° | 51.3° |
| residual L2 i25 | 165.0 | 125.9 |
| residual L2 i49 | 173.3 | 127.9 |
| elbow_flex 명령 범위 vs 시연 envelope(zero-shot, 클립 전) | -46~139° (GT -51~83° 대비 56° 초과) | envelope 클립으로 시연 min~max 내 절단 |
| per-step 클램프 파생 방식(M7W4 deg/s 불변) | MAX_REL 상수 5.0°/step (Hz 의존) | MAX_REL = 120°/s ÷ CONTROL_HZ (Hz 독립) |

실기에서 가드 3종이 실제 안전 성공으로 확증된 실행은 이후 FT 체크포인트(`smolvla_cube_frugal`) 기반 RTC rollout입니다.

| 지표(RTC 24Hz 실기, 2026-06-21) | before(가드 없을 때) | after |
|---|---|---|
| 안전 종료 결과 | stall·파손 위험으로 사용자 중단 | `aborted=null`, `torque_off_verified=true`, `queue_starved_steps=0` |

출처: `smolvla_real_so101_pnp_replay_envclip.json`, `smolvla_rollout_rtc_so101_ft_v0.json`, `m7w2_smolvla_real_zeroshot.md` §6·§8.

### 주의(caveat)

- elbow MAE 105.6→51.3 같은 수치는 실기가 아니라 offline replay 선검증 값입니다.
- 클립 발동률 83.4%와 MAE 개선폭은 base 모델 + tray 데이터셋 기준입니다. FT 후에는 분포 밖(OOS) 명령이 거의 0이라 클립이 사실상 무발동하는 2차 안전망으로 남습니다(`svla_rollout.py` docstring).
- `OVERLOAD_RAW=500`(~50%)과 `OVERLOAD_STEPS=5`는 실측 튜닝이 아니라 제가 설정해 둔 임계값입니다. 확인된 리포트 중 실제 과부하 abort가 트리거된 기록은 없습니다(`aborted=null`).

---

## 5. 회사에 적용한다면

정책이나 어떤 학습된 모델이든, 그 출력을 실물 액추에이터로 보내는 모든 파이프라인에 그대로 옮길 수 있는 체크포인트입니다.

1. 가드는 한 겹이 아니라 다층으로 둡니다.
   - 분포-기반 클립으로 시연·학습 데이터의 min~max 밖 출력을 잘라냅니다. 통계 remap이 못 잡는 분산비 왜곡을 후단에서 부분 보정하는 값싼 2차 안전망입니다.
   - per-step rate 제한으로 이상치나 폭주를 단일 스텝 cap으로 흡수합니다.
   - 부하 기반 자동중단은 알 수 없는 실패에 대한 최종 방어선입니다.

2. 안전 불변량은 per-step이 아니라 per-second로 묶고, 제어 주기에서 파생합니다.
   - 제 생각에는, 속도·가속·토크 같은 물리 단위(°/s 등)로 천장을 정의하고 `cap = 천장 / Hz`로 파생하면, 제어 주파수를 바꿔도 안전 천장이 자동으로 고정됩니다. 상수(°/step)로 박으면 주기를 올릴 때 안전이 조용히 완화됩니다.

3. 안전 감시 주파수는 성능 최적화와 거래하지 않습니다.
   - 개인적으로는, critical-path 여유는 안전 비용이 0인 곳(로깅·압축·역정규화)에서만 회수해야 한다고 봅니다. 부하나 이상 감시 read는 매 제어 스텝 유지합니다.

4. 종료 경로는 "했겠지"에 의존하지 말고 실제로 재read해 검증합니다.
   - 토크 해제 후 `Torque_Enable`을 다시 읽어 0인지 확인했습니다(`torque_off_verified`). disconnect가 알아서 했을 거라고 가정하지 않습니다.

---

## 시리즈 내비

- 이전 글: [팔이 reach만 반복하고 안 들렸던 이유 — 액션 청크 실행 구간](./smolvla_tuning_05_exec_horizon_ko.md)
- 다음 글: [청크 경계 멈칫 없애기 — RTC 비동기 추론 구조](./smolvla_tuning_07_rtc_async_ko.md)
