# ratelcode.github.io

**Independent evaluation reports** measuring the real-robot reproducibility and
silent failures of open VLA policies. Built with Astro + Starlight.

- Framework: [Astro](https://astro.build) + [Starlight](https://starlight.astro.build)
- Blog plugin: [`starlight-blog`](https://github.com/HiDeoo/starlight-blog)
- Hosting: GitHub Pages (auto-deploys on push to `main`)

## Development

```bash
npm install   # once
npm run dev   # dev server (http://localhost:4321)
npm run build # static build → ./dist
npm run preview # preview the build locally
```

## Locale structure

**English (root) + Korean (`/ko/`)** — the default locale flipped ko→en in 2026-07.

- English originals: `src/content/docs/` (root)
- Korean: `src/content/docs/ko/` — same filename (slug) links the two via the
  language switcher
- Pages without a Korean translation fall back to English
- Old URLs published before the flip (`/blog/*_ko/`, `/en/*`) are forwarded by
  the `redirects` map in `astro.config.mjs` — never change the slug of a post
  whose link is already out in the wild

## Writing

- Evaluation reports (blog): add the English original under
  `src/content/docs/blog/` with `date` and `authors` in the frontmatter
  - Before publishing, check the five
    [publication gates](src/content/docs/methodology/protocol.md)
  - Korean edition: add the same filename under `src/content/docs/ko/blog/`
- Methodology docs: `src/content/docs/methodology/` (+ `ko/methodology/`)
- Explanatory diagrams (SVG): `public/diagrams/`

## Design

- Theme: `src/styles/theme.css` — "Phosphor Terminal" (Geist monochrome with a
  phosphor-green accent)
- Fonts: Geist / Geist Mono via the Astro Fonts API; Noto Sans KR via
  `@fontsource` customCss (keeps the sliced hangul chunks in one cacheable
  external stylesheet — see the note in `astro.config.mjs`)
- Custom components: `src/components/` — Head (OG / analytics beacon),
  SiteTitle (brand mark), Footer (disclaimer + status bar), HomeHero (splash
  poster), RecentPosts (home post list)
- The splash (home) page has no sidebar or hamburger menu, so **the HomeHero
  CTA buttons are the only navigation on mobile**

## Analytics

Cookie-less self-hosted collection (pageview / contact_click) via the
`tinybird/` data project. Collection is off while `TB_TOKEN` in
`src/components/Head.astro` is empty — see `tinybird/README.md` to enable.

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which builds and
deploys to GitHub Pages. One-time setup: repository **Settings → Pages →
Source** must be set to `GitHub Actions`.
