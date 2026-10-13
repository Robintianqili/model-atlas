<!-- vq-vae/codebook (see meta.ts): every code as the block it draws; the codes of your drawing; test-set usage. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmtPct } from '../../../../atlas/core/format.ts';
	import Notice from '../../../../atlas/ui/views/kit.parts/Notice.svelte';
	import Pic from '../../parts/Pic.svelte';
	import { blankOf, codesOf, tilesOf, usageOf } from '../../vq.ts';
	import { countsOf } from './meta.ts';

	let P: StageProps = $props();
	const usage = $derived(P.props.mode === 'usage');
	const u = $derived(usageOf(P.rt));
	const tiles = $derived(P.rt.model ? tilesOf(P.rt.model) : []);
	const K = $derived(tiles.length || u?.counts.length || 0);
	const blank = $derived(P.rt.model ? blankOf(P.rt.model) : (u?.blank ?? -1));
	const mine = $derived(countsOf(codesOf(P.rt), K));
	const rows = $derived(u ? u.counts.map((n, k) => ({ k, s: n / u.patches })).sort((a, b) => b.s - a.s || a.k - b.k) : []);
	const top = $derived(rows[0]?.s || 1);
	const sc = $derived(makeScale('sequential', [0, top], themeState.resolved));
	$effect(() => {
		if (P.autoplay && K) P.onend();
	});
</script>

{#if !K}
	<Notice text={{ en: 'The codebook is loading.', zh: '码本正在加载。' }} />
{:else if !usage}
	<div class="gallery" role="list" aria-label={t({ en: 'The codebook', zh: '码本' })}>
		{#each Array.from({ length: K }, (_, k) => k) as k (k)}
			<figure role="listitem" class:used={mine[k] > 0}>
				{#if tiles[k]}<Pic data={tiles[k]} size={44} label={t({ en: `Code ${k}`, zh: `码字 ${k}` })} />{/if}
				<figcaption>{k}{k === blank ? t({ en: ' (blank)', zh: '（空白）' }) : ''}{mine[k] ? ` ×${mine[k]}` : ''}</figcaption>
			</figure>
		{/each}
	</div>
	<p class="small muted">{t({ en: 'Each code drawn among empty patches. Outlined: the codes your drawing uses, ×n = how many of its patches.', zh: '每个码字放在空白图块中间画出的样子。加框的是你的画用到的码字，×n 是它的几个图块用了它。' })}</p>
{:else}
	<div class="usage" use:keyScroll>
		<p class="small muted">{t({ en: 'Share of the test patches that snapped to each code, most used first (bar length = share)', zh: '测试图块吸附到每个码字的比例，从多到少（条长 = 比例）' })}</p>
		<ol>
			{#each rows as r (r.k)}
				<li class:used={mine[r.k] > 0}>
					{#if tiles[r.k]}<Pic data={tiles[r.k]} size={22} label={t({ en: `Code ${r.k}`, zh: `码字 ${r.k}` })} />{:else}<span></span>{/if}
					<span>{t({ en: `code ${r.k}`, zh: `码字 ${r.k}` })}{r.k === blank ? t({ en: ' (blank)', zh: '（空白）' }) : ''}</span>
					<span class="bar"><span style:width="{(100 * r.s) / top}%" style:background={sc.css(r.s)}></span></span>
					<span class="v">{fmtPct(r.s)}</span>
					<span class="v">{mine[r.k] ? t({ en: `yours ×${mine[r.k]}`, zh: `你的 ×${mine[r.k]}` }) : ''}</span>
				</li>
			{/each}
		</ol>
	</div>
{/if}

<style>
	.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(58px, 1fr)); gap: 6px; min-width: 0; }
	figure { margin: 0; display: grid; justify-items: center; gap: 2px; padding: 3px; border: 1px dashed var(--rule); border-radius: 4px; }
	figure.used { border: 2px solid var(--accent); }
	figcaption, .small { font-size: var(--fs-sm); }
	figcaption { font-size: 11px; text-align: center; }
	.muted { color: var(--muted); }
	p { margin: 0; }
	.usage { max-height: 300px; overflow: auto; min-width: 0; }
	ol { list-style: none; margin: 4px 0 0; padding: 0; display: grid; gap: 2px; }
	li { display: grid; grid-template-columns: 24px 92px minmax(40px, 1fr) 48px 64px; align-items: center; gap: 6px; font-size: var(--fs-sm); }
	li.used { font-weight: 700; }
	.bar { height: 10px; background: var(--surface-2); border-radius: 2px; overflow: hidden; }
	.bar span { display: block; height: 100%; }
	.v { text-align: right; font-variant-numeric: tabular-nums; }
</style>
