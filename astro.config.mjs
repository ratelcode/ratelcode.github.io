// @ts-check
import { defineConfig, fontProviders } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightBlog from 'starlight-blog';

// https://astro.build/config
export default defineConfig({
	site: 'https://ratelcode.github.io',

	prefetch: {
		prefetchAll: true,
		defaultStrategy: 'hover',
	},

	fonts: [
		{
			provider: fontProviders.fontsource(),
			name: 'Geist',
			cssVariable: '--font-geist',
			weights: [400, 700],
			fallbacks: ['ui-sans-serif', 'system-ui', '-apple-system', 'sans-serif'],
		},
		{
			provider: fontProviders.fontsource(),
			name: 'Geist Mono',
			cssVariable: '--font-geist-mono',
			weights: [400, 700],
			fallbacks: ['ui-monospace', 'SF Mono', 'Menlo', 'Consolas', 'monospace'],
		},
		{
			provider: fontProviders.fontsource(),
			name: 'Noto Sans KR',
			cssVariable: '--font-noto-kr',
			weights: [400, 700],
			fallbacks: ['Apple SD Gothic Neo', 'sans-serif'],
		},
	],

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
			customCss: ['./src/styles/theme.css'],
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
