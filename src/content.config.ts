import { defineCollection } from 'astro:content';
import { z } from 'astro/zod';
import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';
import { blogSchema } from 'starlight-blog/schema';

export const collections = {
	docs: defineCollection({
		loader: docsLoader(),
		schema: docsSchema({
			extend: (context) => blogSchema(context),
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
