---
title: "2개 카메라로 학습했는데 정책이 멈춘 이유 — 카메라 슬롯 함정"
date: 2026-06-21
authors: ratel
excerpt: "2캠으로 학습해도 config는 camera1/2/3 세 슬롯을 기대한다. camera3 누락이 freeze의 진짜 원인 — 채우니 shoulder_lift +4°→+87°."
tags:
  - SmolVLA 튜닝
  - SmolVLA
  - 카메라
  - SO-101
---

> SmolVLA를 SO-101 실물 로봇에 올리면서 겪은 튜닝 과정을 정리한 시리즈입니다 (3/9). 측정·커밋으로 확인한 내용과 제 의견을 문장에서 구분해 적었습니다.

## TL;DR

- 큐브 데이터셋으로 다시 파인튜닝(FT)한 SmolVLA로 실기 rollout을 돌렸더니 팔을 거의 못 들고 제자리에서 꿈틀거리다 끝났습니다(freeze). 그런데 오프라인 충실도 지표는 멀쩡했습니다.
- 카메라는 2대(top/wrist)만 연결했는데, 모델 config는 camera1/2/3 세 개의 이미지 슬롯을 기대하고 있었습니다. 추론할 때 camera3 슬롯을 안 넣어서 입력 스키마가 깨진 게 원인이었습니다. 모델 내부 문제가 아니라 입력 텐서 한 칸이 빠진 것이었습니다.
- `policy.config.input_features`를 읽어 camera1/2 외의 슬롯을 자동 감지하고 `torch.zeros`로 채우도록 고쳤습니다. 수동 플래그는 없앴습니다.
- camera3을 빼면 shoulder_lift가 **+4°**에 그쳐 freeze가 재현됐고, 채우면 **+87°**로 정상 lift가 나왔습니다. 수정 후에는 실기에서 reach→grasp→lift→place 전 구간을 완주했습니다.

---

## 1. 증상 — 무엇이 안 됐나

배경부터 짧게 적겠습니다. SmolVLA는 카메라 이미지 여러 장과 로봇 관절 상태(state)를 입력(obs)으로 받아, 다음 50스텝의 동작 묶음(action chunk)을 한 번에 예측하는 정책(policy)입니다. 저는 SO-101 팔 로봇으로 큐브를 집어 옮기는 데이터(`so101_cube_v0`, 51 ep)를 모아 이 정책을 다시 파인튜닝했습니다.

그 체크포인트로 실기 온라인 rollout을 돌렸더니 결과가 이상했습니다.

팔이 거의 안 움직이고 제자리에서 꿈틀거리다 끝났습니다. 특히 어깨를 들어 올리는 관절 `shoulder_lift`가 시작 자세에서 사실상 변하지 않았습니다. 동작이 살짝 어긋나는 정도가 아니라 모션 자체가 없는 freeze였습니다.

결정적으로 모순되는 신호가 하나 있었습니다. 같은 체크포인트의 오프라인 chunk 충실도(`svla_compare_ft`)는 멀쩡했습니다. 예측 동작과 정답 동작의 잔차(residual, 두 동작의 차이)가 낮았고, 시연 범위 밖 비율(OOS, out-of-support — 학습 데이터가 보여준 적 없는 동작의 비율)도 거의 0이었습니다.

오프라인 지표는 정상인데 online rollout만 freeze였습니다. 이 괴리는 모델 자체가 망가진 게 아니라 online과 offline의 입력 경로 차이를 봐야 한다는 전형적인 신호였습니다(`m7w4_camera3_rollout_fix.md` §1).

## 2. 원인 — 왜 그랬나

처음엔 모델 내부를 의심했습니다. freeze를 보고 정지 프레임이 많은 데이터로 학습돼 평균 동작이 0 근처로 눌린 것(idle-bias)이라 보고, 정지 프레임 비율을 줄인 트림본(`so101_cube_v0_trim`, frozen 37% → 7.5%)까지 만들어 다시 FT했습니다. 그래도 똑같이 freeze였습니다. 출력 분포를 직접 프로브해 봐도 무동작이 아니었으니, idle-bias 가설은 반증됐습니다. 역정규화 버그, 카메라 물리 이동, 시작 state OOD(out-of-distribution, 학습 분포 밖) 같은 후보도 차례로 배제했습니다.

남은 건 추론에 들어가는 입력 텐서 구성 그 자체였습니다.

여기가 함정이었습니다. SmolVLA는 config의 `input_features`에 정의된 모든 이미지 슬롯을 입력으로 받도록 학습됩니다. 학습 시 실카메라 2대만 매핑했더라도, 모델 config는 base를 상속해 `camera1 / camera2 / camera3` 세 슬롯을 그대로 유지합니다. 즉 config가 진실의 원천(source of truth)이고, 거기엔 camera3이 여전히 살아 있었습니다.

- 학습 경로에선 2캠만 매핑해도 동작했습니다(rename_map으로 처리). 그래서 "camera3은 안 써도 되는구나"라고 착각하기 쉬웠습니다.
- 하지만 추론 경로에서 모델은 여전히 camera3 슬롯을 기대합니다. obs 배치에 `observation.images.camera3` 키를 안 넣으면 기대 입력 스키마가 깨지고, 모델은 거의 무동작 출력을 냅니다. 이게 freeze의 직접 원인이었습니다.

![카메라 3슬롯 중 camera3 누락 vs zeros 채움](/diagrams/03_camera_slot.svg)

*캡션: 모델은 이미지 슬롯 3칸을 기대하는데, camera3을 비우면 멈추고(+4°) 그 칸을 0으로 채우면 정상으로 팔을 듭니다(+87°).*

그림으로 보면 함정이 단순합니다. 모델은 입력으로 이미지 칸 3개를 늘 기대하는데, 카메라를 2대만 연결하면 세 번째 칸이 비어 입력의 모양 자체가 어긋납니다. 비어 있는 그 칸을 같은 크기의 0으로 채워 주기만 하면 모양이 맞아 정책이 다시 움직입니다.

그러면 왜 오프라인 충실도는 멀쩡해 보였을까요. 정상으로 나온 오프라인 진단(probe·모델 비교 도구)은 camera3 슬롯을 자동으로 zeros로 채워 입력을 구성하고 있었기 때문입니다. 그 경로는 세 슬롯을 다 채워 정상이었고, online rollout만 한 칸을 빠뜨려 실패하는 괴리가 생긴 것입니다. 다만 같은 버그가 비교 스크립트 `svla_compare_ft`의 FT 경로에도 숨어 있었다는 사실은 §3의 감사에서 뒤늦게 드러났습니다.

원인은 idle-bias나 역정규화, state OOD 같은 모델 내부 가설이 아니라 학습 스키마와 추론 스키마의 불일치, 곧 이미지 슬롯 한 칸 누락이었습니다(`m7w4_camera3_rollout_fix.md` §2·§3).

### online 실패를 offline로 재현 — 인과를 수치로 확정

"camera3이 원인"임을 추측이 아니라 통제 실험으로 못 박은 방법이 이번 디버깅의 핵심이었습니다. 실기 rollout 중 저장해 둔 `.rrd` 기록(rerun이 남기는 입출력 로그 파일)에서 로봇이 실제로 본 라이브 top/wrist 프레임을 추출해, 그 입력을 오프라인에서 똑같이 재생했습니다. 그리고 camera3 슬롯만 켜고 끄며 비교했습니다.

로봇을 다시 구동할 필요가 없으니 마모나 안전 부담 없이 가설 하나당 한 번의 재생으로 검증됩니다.

## 3. 무엇을 바꿨나

수정의 핵심은 사람이 카메라 슬롯 개수를 기억해 플래그로 켜는 구조를 없앤 것입니다. 그 대신 모델 config에서 기대 슬롯을 직접 읽어 자동으로 채웁니다.

대상 파일은 `workbench/so101/scripts/svla_rollout.py`입니다.

- 기존의 수동 `ADD_CAM3` 환경변수 플래그를 폐지했습니다.
- `main()`에서 `policy.config.input_features`를 읽어, `observation.images`로 시작하면서 camera1/camera2가 아닌 슬롯을 `EXTRA_CAMS`로 자동 감지하게 했습니다.
- `obs_to_batch()`에서 `EXTRA_CAMS`의 각 키를 camera1과 같은 shape의 `torch.zeros_like`로 채웁니다.

핵심 두 조각만 보면 이렇습니다.

```python
# main(): 모델 config에서 기대 이미지 슬롯을 자동 감지 (camera1/2 제외분 = EXTRA_CAMS)
EXTRA_CAMS = [k for k in getattr(policy.config, "input_features", {})
              if k.startswith("observation.images")
              and k not in ("observation.images.camera1", "observation.images.camera2")]

# obs_to_batch(): 감지된 추가 슬롯(camera3 등)을 zeros로 자동 충족
for k in EXTRA_CAMS:
    out[k] = torch.zeros_like(out["observation.images.camera1"])
```

같은 버그 클래스를 다른 진입점에도 전파했습니다.

- `svla_compare_ft.py`에서는 `add_cam3` 플래그를 `expected_image_keys()` 헬퍼로 대체했습니다. 알고 보니 이 스크립트의 FT 경로도 camera3을 누락하고 있어서, 그 이전 리포트의 FT 수치는 무효 처리했습니다.
- 신설한 `svla_rollout_rtc.py`(RTC 비동기 추론)에는 동일한 `EXTRA_CAMS` 자동충족을 처음부터 적용했습니다.

관련 커밋은 `91f1ec8`(svla_rollout: ADD_CAM3 폐지 → 자동 감지·zeros 채움)과 `3ea5609`(compare_ft 동일 버그 수정 + rollout_rtc 신설 + 학습노트)입니다(커밋 메시지, `m7w4_camera3_rollout_fix.md` §5).

## 4. 전후 비교

`.rrd`에 잡아 둔 라이브 입력 프레임으로 camera3 슬롯만 토글한 통제 실험 결과와, 수정 후 실기 결과입니다.

| 지표 | camera3 빠짐 (before) | camera3 채움 / 수정 후 (after) | 출처 |
|---|---|---|---|
| shoulder_lift 변화 (.rrd 오프라인 재현) | **+4°** (실기 freeze 정확히 재현) | **+87°** (정상 lift) | 노트 §4 + 커밋 91f1ec8 |
| 실기 pick-and-place 완주 | freeze — 모션 없음 (shoulder_lift 시작 자세서 불변) | reach→grasp→lift→place 전 구간 완주 (shoulder_lift −68→+41→−73), abort 없이 안전 | 노트 §1·§6 |
| 트림본 frozen 프레임 비율 (idle-bias 검증) | 37% | 7.5% (트림 재-FT해도 동일 freeze → idle-bias 반증) | 노트 §1·§2 |

camera3을 빼면 실기가 보였던 무동작이 오프라인에서 그대로 재현되고, 채우면 lift가 복구됩니다. 이것으로 camera3 누락이 원인임을 인과적으로 확정했습니다.

곁가지 발견이 두 가지 남아 있습니다. 첫째, 조명을 끄면 grasp이 큐브 오른쪽으로 빗나갑니다. 학습 데이터를 라이트 ON으로 수집해서 조명이 분포 밖이면 서보잉이 흔들립니다. 다만 이건 조명 OOD 사례이고, grasp의 체계적 오른쪽 편향 자체의 주원인은 데이터 위치 다양성 부족인데 이건 9편에서 정량 분석하겠습니다. 둘째, 스텝당 이동 클램프 `MAX_REL=5`가 거의 매 스텝 발동해 실행 랙이 생겨서 `MAX_REL=8` 상향을 검토 중입니다. 안전 전제(envelope 클립 + 상대목표 rate limit + Present_Load 과부하 abort + .rrd)는 전 과정 유지했습니다(노트 §6).

한 가지 주의할 점은, 신설한 `svla_rollout_rtc.py`의 자동충족이 lerobot 코드 기준 정적 검증만 거쳤다는 것입니다. 실기 closed-loop 첫 실행 검증은 아직 안 했습니다(커밋 3ea5609 ⚠ 표기).

## 5. 회사에 적용한다면

이 사례는 SO-101에 한정된 이야기가 아닙니다. 입력이 여러 슬롯(카메라/센서)으로 구성되는 멀티모달 정책을 실서비스에 올릴 때 누구나 밟는 함정이라고 봅니다. 옮길 수 있는 체크포인트를 정리해 보겠습니다.

- 제 생각에는, offline 지표가 좋아도 online 실패는 입력 텐서 구성부터 의심하는 게 맞습니다. loss·residual·OOS는 올바른 입력을 줬을 때의 충실도일 뿐, online 경로가 같은 입력을 구성한다는 보장이 아닙니다. 모델 내부(역정규화·분포 편향·OOD)를 파기 전에 학습 스키마와 추론 스키마가 일치하는지를 먼저 확인하는 편이 낫습니다.
- 개인적으로는 사람이 슬롯 개수를 기억해 플래그로 켜는 구조 자체가 버그의 온상이라고 봅니다. 모델 config가 진실의 원천이니, 추론·평가 진입점에서 config가 요구하는 모든 입력 슬롯이 실제 obs에 채워졌는지 검증하고, 빠진 슬롯은 zeros로 자동 충족하거나 최소한 경고하는 가드를 넣는 게 좋습니다. 이렇게 하면 카메라 수가 바뀌어도 코드가 따라가므로 form-factor / N-camera에 무관해집니다.
- "online 실패를 offline로 재현"하는 흐름은 진단 표준으로 삼을 만하다고 봅니다. 실기·실서비스 rollout 중 라이브 입력을 그대로 기록(여기선 `.rrd`)해 두면, 시스템을 재구동하지 않고 입력 구성만 토글해 원인을 통제 실험으로 분리할 수 있습니다. 가설당 한 번의 오프라인 재생이면 충분하고, 로봇 마모나 안전 부담도 없습니다.

---

### 시리즈 내비

- 이전 글: [왜 사전학습 SmolVLA가 팔도 못 들었나 — 정규화 프레임 불일치](./smolvla_tuning_02_norm_frame_ko.md)
- 다음 글: [파인튜닝으로 프레임·스케일 격차 닫기 — 무엇이 풀리고 무엇이 안 풀리나](./smolvla_tuning_04_finetune_ko.md)
- 시리즈 지도: [SmolVLA 실기 튜닝 여정 (시리즈 지도)](./smolvla_tuning_01_intro_ko.md)
