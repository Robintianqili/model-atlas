<!-- Site footer (redesign spec §4.5): the brand, the privacy sentence and the provenance legend (with About, the only place
     the three kinds are explained in full) · Learn / Build / About columns · the licence line. The counts come from the
     root layout's server load (catalog/slices.ts), so no page loads the catalog modules for them. -->
<script lang="ts">
	import { base } from '$app/paths';
	import { page } from '$app/state';
	import Mark from './Mark.svelte';
	import { ui } from '$lib/i18n/i18n.svelte';
	import { REPO_URL } from '$lib/project';
	import type { SiteCounts } from '$lib/catalog/types';

	// Every prerendered page has them. Only the 404 fallback shell (an unknown URL opened directly) has no server data, and
	// leaves the line out rather than load the catalog for it (a lazy import re-split the shared chunks of every page).
	const counts = $derived((page.data as { site?: SiteCounts }).site);
</script>

<footer class="ftr">
	<div class="inner">
		<div class="about">
			<a class="brand" href="{base}/"><Mark size={26} /><span>Model Atlas</span><span class="zh" lang="zh-CN">模型图鉴</span></a>
			<p class="numbers">{ui('footerNumbers')}</p>
			<h2 class="visually-hidden">{ui('provLegend')}</h2>
			<ul class="provs">
				<li><span class="prov live"><span class="dot" aria-hidden="true"></span><b>{ui('provLive')}</b></span><span>{ui('provLiveNote')}</span></li>
				<li><span class="prov pre"><span class="dot" aria-hidden="true"></span><b>{ui('provPrecomputed')}</b></span><span>{ui('provPrecomputedNote')}</span></li>
				<li><span class="prov illus"><span class="dot" aria-hidden="true"></span><b>{ui('provIllustrative')}</b></span><span>{ui('provIllustrativeNote')}</span></li>
			</ul>
		</div>
		<nav aria-labelledby="ftr-learn">
			<h2 id="ftr-learn">{ui('footerLearn')}</h2>
			<a href="{base}/atlas/">{ui('footerAllLessons')}</a>
			<a href="{base}/#paths">{ui('footerPaths')}</a>
			<a href="{base}/glossary/">{ui('navGlossary')}</a>
		</nav>
		<nav aria-labelledby="ftr-build">
			<h2 id="ftr-build">{ui('footerBuild')}</h2>
			<a href="{base}/devkit/">{ui('footerDevkit')}</a>
			<a href="{base}/dev/kit/">{ui('navKit')}</a>
			<a href={REPO_URL} rel="noopener">{ui('footerSource')}</a>
		</nav>
		<nav aria-labelledby="ftr-about">
			<h2 id="ftr-about">{ui('navAbout')}</h2>
			<a href="{base}/about/">{ui('footerHowNumbers')}</a>
			<a href="{base}/about/#licenses">{ui('footerLicenses')}</a>
			<a href="{base}/about/#privacy">{ui('footerPrivacyLink')}</a>
		</nav>
		<div class="legal">
			<p>{ui('footerLegal')} {ui('footerTech')}</p>
			{#if counts}
				<p class="counts">
					{ui('countModels', { n: counts.models })} · {ui('countConcepts', { n: counts.concepts })} · {ui('countLive', { n: counts.live })}
				</p>
			{/if}
		</div>
	</div>
</footer>

<style>
	.ftr {
		margin-top: var(--s8);
		border-top: 1px solid var(--rule);
		background: var(--panel);
	}
	.inner {
		max-width: var(--page-max);
		margin: 0 auto;
		padding: 40px var(--gutter) 32px;
		display: grid;
		grid-template-columns: minmax(0, 2fr) repeat(3, minmax(0, 1fr));
		gap: var(--s6);
	}
	.brand {
		display: flex;
		align-items: center;
		gap: 10px;
		color: var(--ink);
		text-decoration: none;
		font-weight: var(--w-title);
	}
	.brand .zh {
		font-family: var(--font-cjk);
		font-weight: var(--w-body);
		color: var(--muted);
	}
	.numbers {
		max-width: 44ch;
		margin: 14px 0;
		color: var(--ink-2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.provs {
		display: grid;
		gap: 6px;
		margin: 0;
		padding: 0;
		list-style: none;
		color: var(--muted);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	.provs li {
		display: flex;
		align-items: center;
		gap: 10px;
	}
	.prov {
		display: inline-flex;
		align-items: center;
		gap: 8px;
		min-width: 126px;
	}
	.prov b {
		color: var(--ink-2);
		font-weight: var(--w-ui);
	}
	.dot {
		width: 8px;
		height: 8px;
		border-radius: 50%;
		background: var(--live);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--live) 22%, transparent);
	}
	.pre .dot {
		background: var(--pre);
		box-shadow: 0 0 0 3px color-mix(in srgb, var(--pre) 22%, transparent);
	}
	.illus .dot {
		background: transparent;
		border: 1.5px dashed var(--illus);
		box-shadow: none;
		width: 10px;
		height: 10px;
	}
	nav {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 8px;
	}
	nav h2 {
		margin: 0 0 2px;
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
	}
	nav a {
		color: var(--ink-2);
		font-size: var(--t-sm);
		line-height: var(--lh-sm);
		text-decoration: none;
		padding: 2px 0;
	}
	nav a:hover {
		color: var(--ink);
		text-decoration: underline;
	}
	.legal {
		grid-column: 1 / -1;
		border-top: 1px solid var(--rule);
		padding-top: var(--s4);
		color: var(--muted);
		font-size: var(--t-xs);
		line-height: var(--lh-xs);
	}
	.legal p {
		margin: 0 0 6px;
		max-width: 110ch;
	}
	@media (max-width: 759px) {
		.inner {
			grid-template-columns: 1fr 1fr;
			gap: var(--s5);
			padding: 32px var(--gutter) 24px;
		}
		.about {
			grid-column: 1 / -1;
		}
		nav a {
			padding: 6px 0;
		}
	}
</style>
