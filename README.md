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

## 글 작성

- 평가 리포트(블로그): `src/content/docs/blog/` 아래에 추가하고 frontmatter에 `date`, `authors` 지정
  - 발행 전 [측정 프로토콜의 발행 게이트 5항](src/content/docs/methodology/protocol.md) 충족 여부 확인
  - 영어 병행 발행: 같은 파일명으로 `src/content/docs/en/blog/`에 번역본 추가
- 방법론 문서: `src/content/docs/methodology/`
- 사이트 언어: 한국어(루트) + 영어(`/en/`) — 영어 번역이 없는 페이지는 한국어로 폴백

## 배포

`main` 브랜치에 푸시하면 `.github/workflows/deploy.yml`이 빌드 후 GitHub Pages로 배포합니다.
저장소 설정 → **Settings → Pages → Source**를 `GitHub Actions`로 한 번 설정해 주세요.
