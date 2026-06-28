# Analytics (Tinybird)

사이트 비콘(`src/components/Head.astro`)이 보내는 이벤트를 저장·질의하는 데이터 프로젝트.
대시보드 없음 — 분석은 Claude Code가 CLI/API로 수행.

- 리전: AWS `us-west-2` (API 호스트 `https://api.us-west-2.aws.tinybird.co`)
- 수집 데이터: pageview / contact_click + path·referrer·lang·UA. **쿠키·IP·식별자 없음.**

## 최초 배포 (1회)

```bash
curl https://tinybird.co | sh   # tb CLI 설치
tb login                        # 브라우저 인증
cd tinybird
tb --cloud deploy               # datasource + endpoint 배포
```

배포 후 **append 전용 토큰** 생성 → `Head.astro`의 `TB_TOKEN`에 입력:

```bash
tb --cloud token create static events_append --scope DATASOURCES:APPEND:analytics_events
tb --cloud token ls
```

(append 전용이라 클라이언트 노출 안전 — 읽기/관리 권한 없음)

## 질의 치트시트 (Claude Code용)

```bash
# 자유 SQL (관리 토큰으로 CLI에서)
tb --cloud sql "SELECT toDate(timestamp) d, count() FROM analytics_events GROUP BY d ORDER BY d DESC LIMIT 14"

# 배포된 엔드포인트 (read 토큰으로 curl)
curl "https://api.us-west-2.aws.tinybird.co/v0/pipes/daily_hits.json?days=30&token=$TB_READ_TOKEN"
curl "https://api.us-west-2.aws.tinybird.co/v0/pipes/top_pages.json?days=30&token=$TB_READ_TOKEN"
curl "https://api.us-west-2.aws.tinybird.co/v0/pipes/referrers.json?days=90&token=$TB_READ_TOKEN"
```

| 엔드포인트 | 내용 | 파라미터 |
|---|---|---|
| `daily_hits` | 일별 PV + 문의 클릭 (봇 제외) | `days=30` |
| `top_pages` | 경로별 조회수 | `days`, `limit` |
| `referrers` | 유입 도메인 (GTM 선행지표) | `days`, `limit` |

## KPI 정의 (GTM 09 §6)

- **인바운드 문의 수** ≈ `contact_click` (mailto 클릭) — 발행 리포트와 시계열 대조
- **리포트 도달** ≈ `/blog/*` pageview, 유입원 = `referrers`

## 스코프 고정

Worker·대시보드·세션·리플레이 등 추가 금지. 이 프로젝트는
"테이블 1개 + 엔드포인트 3개 + 비콘 30줄"이 상한이다 (02 자본 원칙).
