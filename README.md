# ratelcode.github.io

공개 VLA 정책의 실로봇 재현성·silent failure를 측정하는 **독립 평가 리포트** 사이트.
Astro + Starlight 기반.

- 프레임워크: [Astro](https://astro.build) + [Starlight](https://starlight.astro.build)
- 블로그 플러그인: [`starlight-blog`](https://github.com/HiDeoo/starlight-blog)
- 호스팅: GitHub Pages (`main` 브랜치 푸시 시 자동 배포)

## 개발

```bash
npm install   # 최초 1회
npm run dev   # 개발 서버 (http://localhost:4321)
npm run build # 정적 사이트 빌드 → ./dist
npm run preview # 빌드 결과 로컬 미리보기
```

## 사이트 언어 구조

**영어(루트) + 한국어(`/ko/`)** — 2026-07에 기본 로케일을 ko→en으로 전환.

- 영어 원문: `src/content/docs/` (루트)
- 한국어: `src/content/docs/ko/` — 같은 파일명(slug)으로 두면 언어 스위처가 자동 연결
- 한국어 번역이 없는 페이지는 영어로 폴백
- 전환 전에 발행된 옛 URL(`/blog/*_ko/`, `/en/*`)은 `astro.config.mjs`의
  `redirects`가 새 주소로 넘김 — 링크가 이미 퍼진 글의 slug는 바꾸지 말 것

## 글 작성

- 평가 리포트(블로그): `src/content/docs/blog/`에 영어 원문 추가, frontmatter에 `date`, `authors` 지정
  - 발행 전 [측정 프로토콜의 발행 게이트 5항](src/content/docs/methodology/protocol.md) 충족 여부 확인
  - 한국어 병행 발행: 같은 파일명으로 `src/content/docs/ko/blog/`에 추가
- 방법론 문서: `src/content/docs/methodology/` (+ `ko/methodology/`)
- 설명 다이어그램(SVG): `public/diagrams/`

## 디자인

- 테마: `src/styles/theme.css` — "Phosphor Terminal" (Geist 모노크롬 + 형광 그린 포인트)
- 폰트: Geist/Geist Mono는 Astro Fonts API, Noto Sans KR은 `@fontsource` customCss
  (한글 슬라이스를 캐시 가능한 외부 CSS로 유지하기 위함 — `astro.config.mjs` 주석 참조)
- 커스텀 컴포넌트: `src/components/` — Head(OG/비콘), SiteTitle(브랜드 마크),
  Footer(면책·상태바), HomeHero(스플래시 포스터), RecentPosts(홈 최근 글)
- 스플래시(홈)는 사이드바·햄버거가 없어 **HomeHero의 CTA 버튼이 모바일 유일 내비게이션**

## 애널리틱스

`tinybird/` 데이터 프로젝트로 쿠키 없는 자체 수집(pageview/contact_click).
`src/components/Head.astro`의 `TB_TOKEN`이 비어 있으면 수집 꺼짐 —
활성화 절차는 `tinybird/README.md` 참조.

## 배포

`main` 브랜치에 푸시하면 `.github/workflows/deploy.yml`이 빌드 후 GitHub Pages로 배포합니다.
저장소 설정 → **Settings → Pages → Source**를 `GitHub Actions`로 한 번 설정해 주세요.
