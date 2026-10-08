<!-- Site footer: privacy promise, links, licences and credits. The counts come from the root layout's server load
     (catalog/slices.ts), so no page loads the catalog modules for them. -->
<script lang="ts">
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import { ui, t } from '$lib/i18n/i18n.svelte';
	import { REPO_URL } from '$lib/project';
	import type { SiteCounts } from '$lib/catalog/types';

	// Every prerendered page has them. Only the 404 fallback shell (an unknown URL opened directly) has no server data, and
	// leaves the line out rather than load the catalog for it (a lazy import re-split the shared chunks of every page).
	const counts = $derived((page.data as { site?: SiteCounts }).site);
</script>

<footer class="ftr">
	<div class="container inner">
		<div class="col">
			<p class="brand">Model Atlas · <span lang="zh-CN">模型图鉴</span></p>
			<p class="privacy">
				<svg viewBox="0 0 12 12" width="12" height="12" aria-hidden="true"><rect x="2" y="5.2" width="8" height="5.6" rx="1.2" fill="currentColor" /><path d="M3.8 5.2V3.8a2.2 2.2 0 0 1 4.4 0v1.4" fill="none" stroke="currentColor" stroke-width="1.4" /></svg>
				{ui('footerPrivacy')}
			</p>
			{#if counts}
				<p class="muted small">
					{ui('countModels', { n: counts.models })} · {ui('countConcepts', { n: counts.concepts })} ·
					{ui('countLive', { n: counts.live })}
				</p>
			{/if}
		</div>
		<nav class="col links" aria-label="Footer">
			<a href="{base}/atlas/">{ui('navAtlas')}</a>
			<a href="{base}/train/">{ui('navTrain')}</a>
			<a href="{base}/glossary/">{ui('navGlossary')}</a>
			<a href="{base}/about/">{ui('footerHowNumbers')}</a>
			<a href="{base}/about/#licenses">{ui('footerLicenses')}</a>
			<a href="{base}/devkit/">{ui('navDevkit')}</a>
			<a href="{base}/dev/kit/">{ui('navKit')}</a>
		</nav>
		<div class="col small muted legal">
			<p>
				{t({
					en: 'Code: MIT licence. Lesson text, articles and figures: CC BY 4.0. Datasets, vocabularies and fonts keep their own licences.',
					zh: '代码采用 MIT 许可证；课程文字、文章和插图采用 CC BY 4.0；数据集、词表和字体保留各自的许可证。'
				})}
				<a href={REPO_URL} rel="noopener">{t({ en: 'Source on GitHub', zh: 'GitHub 上的源代码' })}</a>
			</p>
			<p>
				{t({
					en: 'A static site: every model runs or is replayed in your browser. Fonts: Fraunces, Atkinson Hyperlegible Next, JetBrains Mono (SIL OFL). Math: KaTeX (MIT).',
					zh: '纯静态网站：所有模型都在你的浏览器中运行或回放。字体：Fraunces、Atkinson Hyperlegible Next、JetBrains Mono（SIL OFL）。公式：KaTeX（MIT）。'
				})}
			</p>
		</div>
	</div>
</footer>

<style>
	.ftr {
		margin-top: var(--s8);
		border-top: 1px solid var(--rule);
		background: var(--surface-2);
		padding: var(--s6) 0 var(--s7);
		font-size: var(--fs-sm);
	}
	.inner {
		display: grid;
		grid-template-columns: 1.4fr 1fr 1.2fr;
		gap: var(--s5);
	}
	@media (max-width: 760px) {
		.inner {
			grid-template-columns: 1fr;
		}
	}
	.brand {
		font-family: var(--font-display);
		font-weight: 650;
		font-size: 1.05rem;
		margin-bottom: 6px;
	}
	.privacy {
		display: flex;
		gap: 6px;
		align-items: baseline;
		color: var(--ink-2);
	}
	.links {
		display: grid;
		gap: 4px;
		align-content: start;
	}
	.small {
		font-size: var(--fs-xs);
	}
	.legal p {
		margin: 0 0 8px;
	}
</style>
