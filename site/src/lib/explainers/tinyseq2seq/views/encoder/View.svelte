<!-- tinyseq2seq 'encoder' (meta.ts): one column per character, its numbers drawn as cells (blue < 0 < orange). In mode
     'read' a player reveals the states left to right; columns not read yet are left empty. The drawing scales to the
     stage and scrolls sideways when a long date does not fit. -->
<script lang="ts">
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import { keyScroll } from '$lib/kit/keyscroll';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtInt, fmtPct } from '../../../../atlas/core/format.ts';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { encoderOf } from './meta.ts';

	let P: StageProps = $props();
	const E = $derived(encoderOf(P));
	const sc = $derived(makeScale('diverging', [-E.m, E.m], themeState.resolved));
	const cw = 36;
	const ch = $derived(E.D > 20 ? 9 : 15);
	const top = $derived(E.mode === 'chars' ? 46 : 30);
	let step = $state(0);
	let hover = $state<{ j: number; d: number } | null>(null);
	$effect(() => void (step = P.autoplay ? 0 : Math.max(0, E.T - 1)));
	const shown = (j: number) => E.mode !== 'read' || j <= step;
</script>

{#snippet grid()}
	<div class="box" use:keyScroll>
		<svg viewBox="0 0 {E.T * cw + 4} {top + E.D * ch + 6}" style:width="min(100%, {E.T * cw + 4}px)" style:min-width="{E.T * 17}px" role="img" aria-label={t({ en: 'Numbers under each character', zh: '每个字符下面的数' })} onpointerleave={() => (hover = null)}>
			{#each E.chars as c, j (j)}
				<text x={j * cw + cw / 2 + 2} y="20" class="ch" class:cur={E.mode === 'read' && j === step}>{c}</text>
				{#if E.mode === 'chars'}<text x={j * cw + cw / 2 + 2} y="38" class="id">{E.ids[j]}</text>{/if}
				{#if shown(j) && E.x}
					{#each { length: E.D } as _, d (d)}
						<rect x={j * cw + 3} y={top + d * ch} width={cw - 4} height={ch - 1} fill={sc.css(E.x.data[j * E.D + d])} role="presentation" onpointerenter={() => (hover = { j, d })} />
					{/each}
				{/if}
			{/each}
			{#if E.T && E.mode !== 'chars'}<rect class="ring" x={(E.mode === 'read' ? step : E.T - 1) * cw + 1} y={top - 4} width={cw} height={E.D * ch + 7} rx="4" />{/if}
		</svg>
	</div>
	<p class="small num">
		{#if hover && E.x}{t({ en: `“${E.chars[hover.j]}” (position ${hover.j}), number ${hover.d}: ${fmt3(E.x.data[hover.j * E.D + hover.d])}`, zh: `“${E.chars[hover.j]}”（位置 ${hover.j}），第 ${hover.d} 个数：${fmt3(E.x.data[hover.j * E.D + hover.d])}` })}
		{:else}{t({ en: `${E.D} numbers per column (blue below 0, orange above) · hover or tap a cell`, zh: `每列 ${E.D} 个数（蓝色小于 0，橙色大于 0）· 悬停或点按一个格子` })}{/if}
	</p>
	{#if E.mode === 'last'}
		<table class="num">
			<caption>{t({ en: `Dates written exactly right, out of ${fmtInt(P.rt.num('fact:testN'))} test dates (measured offline)`, zh: `完全写对的日期，共 ${fmtInt(P.rt.num('fact:testN'))} 个测试日期（离线测得）` })}</caption>
			<thead><tr><th>{t({ en: 'state size', zh: '状态大小' })}</th><th>{t({ en: 'with attention', zh: '有注意力' })}</th><th>{t({ en: 'without', zh: '没有注意力' })}</th></tr></thead>
			<tbody>{#each [[E.D, 'testExact', 'noAttnExact'], [P.rt.num('fact:h8'), 'h8AttnExact', 'h8NoAttnExact']] as r (r[1])}<tr><td>{r[0]}</td><td>{fmtPct(P.rt.num(`fact:${r[1]}`))}</td><td>{fmtPct(P.rt.num(`fact:${r[2]}`))}</td></tr>{/each}</tbody>
		</table>
	{/if}
{/snippet}

<div class="enc">
	{#if E.mode === 'read' && E.T > 1}
		<StepPlayer count={E.T} bind:step interval={420} autoplay={P.autoplay} onend={P.onend} counter={{ en: 'Character {i} of {n}', zh: '第 {i} / {n} 个字符' }}>
			{#snippet children()}{@render grid()}{/snippet}
		</StepPlayer>
	{:else}
		{@render grid()}
	{/if}
</div>

<style>
	.enc { display: grid; gap: 6px; min-width: 0; }
	.box { overflow-x: auto; max-width: 100%; }
	svg { display: block; }
	.ch { font: 600 17px var(--font-mono); text-anchor: middle; fill: var(--ink); }
	.ch.cur { fill: var(--accent-strong); }
	.id { font: 12px var(--font-mono); text-anchor: middle; fill: var(--muted); }
	.ring { fill: none; stroke: var(--accent); stroke-width: 2.5; }
	.small { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); }
	table { border-collapse: collapse; font-size: var(--fs-sm); }
	caption { text-align: left; color: var(--ink-2); padding-bottom: 4px; }
	th, td { padding: 3px 12px 3px 0; text-align: left; border-bottom: 1px solid var(--rule); }
</style>
