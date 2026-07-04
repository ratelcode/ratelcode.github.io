---
title: "LeRobot 파이프라인을 에이전트 루프로 — LeAgent 개발기 (M0)"
date: 2026-07-04
authors: ratel
excerpt: "collect→train→eval을 손으로 돌리는 대신 결정론적 루프 + 에이전트로 감쌌다. 딥리서치로 설계 근거를 검증(24건 확인, 1건 반박)하고, 하루 만에 실 GPU에서 3사이클 자동 루프를 완주하기까지의 실환경 디버깅 카탈로그."
tags:
  - LeAgent
  - LeRobot
  - 에이전트
  - 자동화
  - SmolVLA
---

> 개인 프로젝트 [LeAgent](https://github.com/ratelcode/LeAgent)의 개발 기록입니다. 측정·커밋으로 확인한 내용과 제 의견을 문장에서 구분해 적었습니다. English version: [LeAgent devlog (M0)](/blog/leagent_m0_devlog/)

> **용어 1줄 정리**
> - **LeRobot**: Hugging Face의 로보틱스 라이브러리입니다. `lerobot-train` / `lerobot-eval` CLI와 LeRobotDataset v3.0 포맷을 제공합니다.
> - **SmolVLA**: 450M 파라미터 VLA(vision-language-action) 베이스 모델입니다. 파인튜닝 전제로 설계됐습니다.
> - **LIBERO**: 130개 조작 태스크로 구성된 시뮬레이션 벤치마크입니다. LeRobot v0.4.0부터 공식 지원됩니다.
> - **사이클(cycle)**: LeAgent에서 collect→train→eval→decide 한 바퀴를 말합니다.

---

## TL;DR

- SmolVLA 튜닝 시리즈를 하면서 매번 손으로 돌리던 "데이터 수집 → 학습 → 평가 → 판단" 루프를 통째로 자동화하는 프로젝트(LeAgent)를 시작했습니다.
- 설계 전에 딥리서치를 돌렸고(108개 서브에이전트, 소스 26개, 주장 130건 추출 → 상위 25건 3표 적대적 검증), **24건 확인 / 1건 반박**이 나왔습니다. 반박된 1건("RoboGen은 무인 무한 데이터 플라이휠로 쓸 수 있다")이 그대로 설계 규칙이 됐습니다: 루프 경계마다 검증 게이트, 무한 자동화 가정 금지.
- 핵심 설계 원칙: **제어 흐름은 LLM이 아니라 순수 Python 상태머신 + SQLite.** LLM은 제안(태스크 큐레이션, 지식 증류)만 하고, promote/iterate/escalate/rollback 결정은 eval 델타의 순수 함수입니다.
- 유닛 테스트 55개가 전부 통과해도 실제 CLI를 돌리면 깨집니다. lerobot 0.5.1 실환경에서만 드러난 이슈가 **10건** 나왔고 아래에 카탈로그로 정리했습니다.
- 첫 실 GPU 자동 루프(미니 M0: 3사이클, 데이터 8→16→32 에피소드, RTX 5070 Ti)는 22분에 완주했지만 성공률이 0%로 평평했고 — 루프가 이걸 "정책 한계"로 오판해 더 큰 모델로 escalate했습니다. 이 관측이 `escalate_floor` 가드(0% 플래토 = 학습량 부족 → iterate)로 이어졌습니다. **측정이 설계를 고친 사례**입니다.

---

## 왜 만들었나

SmolVLA 튜닝 시리즈(1~9편)를 진행하는 동안 제 워크플로는 이랬습니다: 데이터를 좀 모으고, `lerobot-train`을 돌리고, 끝나면 `lerobot-eval`을 돌리고, 결과를 보고 "데이터를 더 모을까, 하이퍼파라미터를 바꿀까, 모델을 바꿀까"를 매번 손으로 판단했습니다. 판단 기준은 머릿속에 있었고, 판단 이력은 어디에도 없었습니다.

이 루프 자체를 코드로 만들면 판단 기준이 명시되고, 이력이 남고, 밤새 돌릴 수 있습니다. 그게 LeAgent입니다: 오케스트레이터가 데이터/학습/평가/개선 에이전트를 조율하고, 대시보드로 플로우를 보는 공개 프로젝트.

## 설계를 리서치로 검증하고 시작했다

바로 코드부터 짜는 대신 2024–2026 논문과 LeRobot 코드베이스에 대한 딥리서치를 먼저 돌렸습니다. 108개 서브에이전트가 소스 26개에서 주장 130건을 뽑고, 상위 25건을 각각 3표 적대적 검증(반박 시도)에 부쳤습니다. 결과는 24건 확인, 1건 반박.

확인된 것들이 설계의 뼈대가 됐습니다:

| 검증된 사실 | 설계 반영 |
|---|---|
| `lerobot-train`/`lerobot-eval`은 스크립트 가능한 CLI 진입점 | 에이전트 = 서브프로세스 래퍼. LeRobot 내부 API에 결합하지 않음 |
| LeRobotDataset v3.0이 모든 스테이지의 단일 데이터 계약 | 에이전트 간 통신은 데이터셋/체크포인트 참조로만 |
| AutoRT: LLM 오케스트레이션이 실 로봇 52대·7개월 규모에서 작동, 단 **constitution 안전 필터 + 인간 감독 유지** | constitution.yaml 게이트 (실하드웨어 경로·CVE 경로 차단) |
| DexFlyWheel: IL→residual RL→성공 필터링→증강 사이클로 데모 1개→2,000개 | M1 자기개선 루프의 템플릿 |
| SmolVLA: 20k스텝 ≈ A100 4시간, 변형당 ~50 에피소드 필요 | 루프 케이던스와 데이터 수집 목표치의 근거 |

그리고 반박된 1건 — "RoboGen을 무한 무인 데이터 플라이휠로 쓸 수 있다"(1-2로 기각: 시뮬 한정, 감독 필요, 스킬 검증이 미해결 병목) — 이 가장 중요한 설계 규칙이 됐습니다. **무한 자동화를 가정하지 않는다.** 예산(사이클 수, GPU 시간)이 하드 리밋이고, blessed 체크포인트 승격은 사람이 뒤집을 수 있어야 합니다.

제 의견: 설계 문서에 "verified"와 "recommendation"을 구분해 적는 것만으로도 이후 디버깅에서 뭘 의심해야 할지가 명확해졌습니다. 실제로 이날 터진 문제들은 전부 "recommendation" 영역(스트리밍, 프레임워크 선택)이거나 문서 밖(CLI 세부 동작)이었습니다.

## 아키텍처: LLM은 제안만, 결정은 순수 함수

```
Orchestrator (Python 상태머신 + SQLite)
 ├─ Proposer      : 다음에 뭘 수집할지 제안 (LLM 또는 결정론적 — 교체 가능한 Protocol)
 ├─ Constitution  : sim-only 게이트, 실하드웨어/CVE 경로 차단
 ├─ Data Agent    : 시드 데이터셋 해석, 에피소드 스케줄 (M1: 큐레이션/증폭/증강)
 ├─ Train Agent   : lerobot-train 래퍼 (blessed 체크포인트에서 이어서 학습)
 ├─ Eval Agent    : lerobot-eval 래퍼, LIBERO 게이트
 ├─ Knowledge Agent: 사이클 결과를 OKF 마크다운 위키로 증류 (Karpathy LLM Wiki 3-레이어)
 └─ decide()      : promote / iterate / escalate / rollback — eval 델타의 순수 함수
```

결정 함수가 LLM이 아닌 이유: 재현 가능해야 하고, 테이블 기반 유닛 테스트가 가능해야 하고, 새벽 3시에 GPU 예산을 태우는 판단을 언어 모델의 기분에 맡길 수 없기 때문입니다. LLM 어댑터는 프로바이더 독립(`anthropic:* | openai:*[@base_url]`, Ollama/vLLM 포함)이고, **LLM이 없어도 모든 플로우가 결정론적 폴백으로 동작**합니다.

## 실환경 디버깅 카탈로그 — lerobot 0.5.1

유닛 테스트 55개(가짜 러너)가 전부 초록이어도, 진짜 `lerobot-*`을 서브프로세스로 돌리는 순간부터가 본편이었습니다. 하루 동안 만난 이슈 전부:

| # | 증상 | 원인/해결 |
|---|---|---|
| 1 | `--policy.path`가 `--help`에 없음 | 실제로는 파서가 특수 처리해서 동작함. help를 믿지 말고 실행으로 확인 |
| 2 | 학습 시작 거부: `'policy.repo_id' argument missing` | v0.5는 Hub 푸시가 기본. `--policy.push_to_hub=false` 필수 |
| 3 | `FileExistsError: Output directory ... already exists` | 로그 파일을 output_dir 안에 먼저 만들던 게 원인. 로그를 한 단계 위로 |
| 4 | eval 즉사: `batch size is greater than the number of eval episodes (50 > 2)` | eval 기본 batch 50. `--eval.batch_size=min(batch, episodes)` |
| 5 | eval_info.json 파싱 실패 | 실제 스키마는 `overall/per_group/per_task`, `pc_success`는 0–100 퍼센트 |
| 6 | 70GB 데이터셋(HuggingFaceVLA/libero)을 스모크에 통째로 못 받음 | `--dataset.episodes=[0,1,...]` 부분 샤드 다운로드 |
| 7 | `--dataset.streaming=true` + SmolVLA → 빈 배치로 즉사 | **스트리밍은 액션 청킹 정책(SmolVLA/ACT/π0)과 비호환** (delta-timestamp 액션 윈도우 미지원). 리서치가 "rough edges"로 경고했던 지점의 정확한 실패 모드 |
| 8 | 피처 미스매치: 정책은 `camera1/2/3`, LIBERO는 `image/image2` | train·eval 양쪽에 `--rename_map`. 카메라 2/3개는 부분집합 검증으로 통과 |
| 9 | eval에서 CUDA OOM (16GB) | 병렬 LIBERO 환경 5개(MuJoCo EGL 렌더링) + 정책 추론. batch 2로 캡 |
| 10 | `egl-probe` 빌드 실패 → LIBERO 설치 불가 | ① 시스템 EGL 헤더 필요 ② CMake 4.x가 2018년산 CMakeLists 거부 → `CMAKE_POLICY_VERSION_MINIMUM=3.5` ③ `libero` 패키지는 첫 임포트에 대화형 프롬프트 → `echo N \| python -c "import libero.libero"` 1회 초기화 |

측정값 몇 개: RTX 5070 Ti(16GB)에서 SmolVLA 파인튜닝 ~0.4s/step(batch 8), 300스텝 학습 ~130초, LIBERO 실평가 4에피소드 ~308초. PushT 스모크(2D, EGL 불필요)는 풀사이클 ~1분이라 파이프라인 검증용으로 유용했습니다.

## 첫 자동 루프가 가르쳐준 것: 0% 플래토는 정책 한계가 아니다

미니 M0(사이클당 300스텝, 데이터 8→16→32 에피소드, 가중치는 blessed 체크포인트에서 승계)는 22분에 3사이클을 완주했습니다. 결정 캐스케이드는 설계대로였습니다: promote(첫 사이클=베이스라인) → iterate → 플래토 감지 → **π0.5로 escalate**.

문제는 마지막 결정입니다. 성공률이 3사이클 내내 0.0%였는데 — 300스텝×32에피소드로는 당연한 결과 — 루프는 이 평평한 신호를 "정책이 한계에 달했다"로 읽고 7GB짜리 더 큰 모델로 올라가려 했습니다. 0% 플래토의 원인은 거의 항상 학습량/데이터 부족이지 모델 용량이 아닙니다.

그래서 결정 함수에 가드를 넣었습니다: **베이스라인이 `escalate_floor`(기본 5%)를 넘을 때만 플래토가 escalate로 이어지고, 그 아래면 iterate.** 이 관측은 그대로 유닛 테스트(`test_zero_plateau_iterates_instead_of_escalating`)로 박제했습니다.

제 의견: 이런 게 루프를 코드로 만드는 이유라고 생각합니다. 손으로 돌릴 때는 제가 무의식적으로 하던 판단("아직 언더트레이닝이지")이 코드에는 없었고, 자동 루프가 그 공백을 하루 만에 드러냈습니다.

## 지식 레이어: 런 이력이 아니라 교훈을 쌓기

루프는 데이터(이벤트, 체크포인트)를 쌓지만 교훈은 쌓지 않습니다. 그래서 Karpathy의 LLM Wiki 패턴(2026-04)과 Google OKF 스펙(2026-06)을 따라 지식 레이어를 넣었습니다: 불변 원천(events.jsonl, SQLite) → 에이전트가 관리하는 마크다운 위키(`knowledge/` — 태스크/정책별 페이지, YAML frontmatter에 run/cycle provenance) → 스키마 파일(KNOWLEDGE.md). 관측이 2회 이상 재현되면 `observed-once → replicated`로 승격되고, `human-confirmed`는 에이전트가 절대 강등하지 못합니다. 지식 페이지는 proposer의 컨텍스트로만 쓰이고 — **제어 흐름에는 절대 개입하지 않습니다.**

## 현재 상태와 다음

- 이 글을 쓰는 시점에 풀스케일 M0(사이클당 20k스텝, 데이터 40→80→160 에피소드, ~7시간)가 돌고 있고, 대시보드(`leagent dash` — 사이클 파이프라인, eval 차트, 롤아웃 영상, 이벤트, 지식 브라우저)로 관전 중입니다.
- 다음: M1 잔여(DexFlyWheel residual RL, RoboGene식 LLM 큐레이션), 실로봇(M3)은 CVE-2026-25874(async-inference gRPC의 pickle RCE)가 수정되는 lerobot 0.6.0 이후로.
- 코드: [github.com/ratelcode/LeAgent](https://github.com/ratelcode/LeAgent) — 이 글이 올라갈 때쯤 public 전환 예정입니다.
