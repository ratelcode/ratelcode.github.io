// @ts-check
import { defineConfig, fontProviders } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightBlog from 'starlight-blog';

// https://astro.build/config
export default defineConfig({
	site: 'https://ratelcode.github.io',

	// 2026-07: default locale flipped ko -> en. Old published Korean URLs
	// (root locale) redirect to their new /ko/ homes; the _ko filename
	// suffix was dropped so slugs pair across locales for the switcher.
	redirects: {
		// Note: /blog/2026-05-09/ is NOT redirected — like /about/ and the
		// /methodology/* pages, that URL had unsuffixed Korean content
		// before the flip and now serves the new English translation in
		// place (a content swap, not a move). Only the _ko-suffixed old
		// URLs below actually moved.
		'/blog/cross_policy_smolvla_pi05_ko/': '/ko/blog/cross_policy_smolvla_pi05/',
		'/blog/lerobot_dataset_silent_failure_ko/': '/ko/blog/lerobot_dataset_silent_failure/',
		'/blog/smolvla_tuning_01_intro_ko/': '/ko/blog/smolvla_tuning_01_intro/',
		'/blog/smolvla_tuning_02_norm_frame_ko/': '/ko/blog/smolvla_tuning_02_norm_frame/',
		'/blog/smolvla_tuning_03_camera_slot_ko/': '/ko/blog/smolvla_tuning_03_camera_slot/',
		'/blog/smolvla_tuning_04_finetune_ko/': '/ko/blog/smolvla_tuning_04_finetune/',
		'/blog/smolvla_tuning_05_exec_horizon_ko/': '/ko/blog/smolvla_tuning_05_exec_horizon/',
		'/blog/smolvla_tuning_06_safety_clamp_ko/': '/ko/blog/smolvla_tuning_06_safety_clamp/',
		'/blog/smolvla_tuning_07_rtc_async_ko/': '/ko/blog/smolvla_tuning_07_rtc_async/',
		'/blog/smolvla_tuning_08_rtc_speed_ko/': '/ko/blog/smolvla_tuning_08_rtc_speed/',
		'/blog/smolvla_tuning_09_data_diversity_ko/': '/ko/blog/smolvla_tuning_09_data_diversity/',
		// Old /en/ URLs (English was a prefixed locale before the flip) —
		// listed individually; a `/en/[...slug]` wildcard reuses the shared
		// catch-all route's static paths (including /ko/ ones), which
		// breaks the build.
		'/en/': '/',
		'/en/about/': '/about/',
		'/en/methodology/protocol/': '/methodology/protocol/',
		'/en/methodology/test-rig/': '/methodology/test-rig/',
		'/en/blog/2026-05-09/': '/blog/2026-05-09/',
		'/en/blog/cross_policy_smolvla_pi05/': '/blog/cross_policy_smolvla_pi05/',
		'/en/blog/lerobot_dataset_silent_failure/': '/blog/lerobot_dataset_silent_failure/',
		'/en/blog/smolvla_tuning_01_intro/': '/blog/smolvla_tuning_01_intro/',
		'/en/blog/smolvla_tuning_02_norm_frame/': '/blog/smolvla_tuning_02_norm_frame/',
		'/en/blog/smolvla_tuning_03_camera_slot/': '/blog/smolvla_tuning_03_camera_slot/',
		'/en/blog/smolvla_tuning_04_finetune/': '/blog/smolvla_tuning_04_finetune/',
		'/en/blog/smolvla_tuning_05_exec_horizon/': '/blog/smolvla_tuning_05_exec_horizon/',
		'/en/blog/smolvla_tuning_06_safety_clamp/': '/blog/smolvla_tuning_06_safety_clamp/',
		'/en/blog/smolvla_tuning_07_rtc_async/': '/blog/smolvla_tuning_07_rtc_async/',
		'/en/blog/smolvla_tuning_08_rtc_speed/': '/blog/smolvla_tuning_08_rtc_speed/',
		'/en/blog/smolvla_tuning_09_data_diversity/': '/blog/smolvla_tuning_09_data_diversity/',
	},

	prefetch: {
		prefetchAll: true,
		defaultStrategy: 'hover',
	},

	fonts: [
		{
			provider: fontProviders.fontsource(),
			name: 'Geist',
			cssVariable: '--font-geist',
			// 600 is load-bearing: Starlight styles every content heading
			// (h1–h6) at font-weight 600 — without the face browsers snap
			// those to 700.
			weights: [400, 600, 700],
			fallbacks: ['ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'],
		},
		{
			provider: fontProviders.fontsource(),
			name: 'Geist Mono',
			cssVariable: '--font-geist-mono',
			// 600 for the brand mark, card titles, and post-row titles
			// (--font-display resolves to this mono chain).
			weights: [400, 600, 700],
			fallbacks: ['ui-monospace', 'SF Mono', 'Menlo', 'Consolas', 'monospace'],
		},
	],
	// Noto Sans KR deliberately does NOT go through the Fonts API: its hangul
	// slices (~120 unicode-range chunks) would be inlined as ~25KB gzip of
	// <style> into every page's HTML, re-fetched on each navigation. Importing
	// the @fontsource CSS via Starlight customCss (see below) keeps the same
	// self-hosted sliced files in one cacheable external stylesheet instead.

	integrations: [
		starlight({
			title: 'ratelcode',
			defaultLocale: 'root',
			// en is the default locale (served at the root) for the global LeRobot/VLA
			// community; Korean is published under /ko/ (GTM 09 §3 — inbound credibility).
			locales: {
				root: { label: 'English', lang: 'en' },
				ko: { label: '한국어', lang: 'ko' },
			},
			// Show git-based "last updated" on every page — reports get corrected,
			// and visible revision dates are part of the reproducibility promise.
			lastUpdated: true,
			social: [
				{
					icon: 'github',
					label: 'GitHub',
					href: 'https://github.com/ratelcode/ratelcode.github.io',
				},
			],
			customCss: [
				'@fontsource/noto-sans-kr/400.css',
				'@fontsource/noto-sans-kr/700.css',
				'./src/styles/theme.css',
			],
			components: {
				Head: './src/components/Head.astro',
				SiteTitle: './src/components/SiteTitle.astro',
				Footer: './src/components/Footer.astro',
			},
			plugins: [
				starlightBlog({
					title: 'Blog',
					authors: {
						ratel: {
							name: 'ratelcode',
							url: 'https://github.com/ratelcode',
						},
					},
				}),
			],
			sidebar: [
				{
					label: 'Methodology',
					translations: { ko: '방법론' },
					items: [
						{ label: 'Evaluation protocol', translations: { ko: '측정 프로토콜' }, slug: 'methodology/protocol' },
						{ label: 'Test rig (SO-101)', translations: { ko: '평가 리그 (SO-101)' }, slug: 'methodology/test-rig' },
					],
				},
				{ label: 'About', translations: { ko: '소개' }, slug: 'about' },
			],
		}),
	],
});
