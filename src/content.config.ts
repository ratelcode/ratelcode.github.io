import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';
import { blogSchema } from 'starlight-blog/schema';

export const collections = {
	docs: defineCollection({
		loader: docsLoader(),
		schema: docsSchema({
			extend: (context) =>
				blogSchema(context).extend({
					// Home-page ledger extras, both optional per post:
					// one-line headline measurement ("0.5x → 1.6x") and a
					// diagram path under public/ for the featured card.
					// Named homeCover because starlight-blog owns `cover`
					// (an {alt, image|dark/light} object rendered atop the
					// post page) — overriding it breaks post rendering.
					metric: z.string().optional(),
					homeCover: z.string().optional(),
				}),
		}),
	}),
	// UI string overrides — starlight-blog ships no Korean strings, so we
	// provide them ourselves (src/content/i18n/ko.json). catchall allows
	// plugin-namespaced keys (starlightBlog.*) alongside Starlight's own.
	i18n: defineCollection({
		loader: i18nLoader(),
		schema: i18nSchema({ extend: z.object({}).catchall(z.string()) }),
	}),
};
