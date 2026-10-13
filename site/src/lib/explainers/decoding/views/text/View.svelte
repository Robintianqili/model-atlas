<!--
  'decoding/text' (meta.ts has the props): the prompt (plain chips), then every new token as a chip shaded by the chance
  TinyGPT gave it at temperature 1 (one purple ramp, 0 to 100%), the newest outlined; with `loop`, the copies of the
  phrase that comes back are underlined. Under it: the candidates the newest token was drawn from, and u.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmtPct, visibleToken } from '$lib/kit/format';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { textPicOf } from './meta.ts';

	let { rt, at, props }: StageProps = $props();
	const pic = $derived(textPicOf({ rt, at, props }));
	const sc = $derived(makeScale('sequential', [0, 1], themeState.resolved));
	const inLoop = $derived.by(() => {
		const s = new Set<number>();
		const l = pic?.loop;
		if (l) for (const a of l.at) for (let q = 0; q < l.len; q++) s.add(a + q);
		return s;
	});
	const show = (s: string) => (s.trim() === '' ? visibleToken(s) : s.replace(/\n/g, '↵'));
	const rare = $derived(typeof props.rare === 'number' ? props.rare : -1);
	const nRare = $derived(pic ? pic.made.filter((c) => c.pModel < rare).length : 0);
	const maxTop = $derived(Math.max(...(pic?.top.map((e) => e.p) ?? [0]), 1e-9));
</script>

{#if pic}
	<div class="txt">
		<p class="chips">
			{#each pic.prompt as c, i (i)}<span class="chip pr">{show(c.text)}</span>{/each}{#each pic.made as c, i (i)}<span
					class="chip"
					class:now={i === pic.made.length - 1}
					class:loop={inLoop.has(i)}
					class:rare={c.pModel < rare}
					style="background: {sc.css(c.pModel)}; color: {sc.ink(c.pModel)}"
					title={fmtPct(c.pModel)}>{show(c.text)}</span
				>{/each}
		</p>
		<p class="legend">
			<span class="chip pr sw">{t({ en: 'prompt', zh: '提示' })}</span>
			<span class="ramp" style="background: linear-gradient(90deg, {sc.css(0)}, {sc.css(0.5)}, {sc.css(1)})"></span>
			<span>{t({ en: 'new token: the chance TinyGPT gave it (temperature 1), 0% to 100%', zh: '新词元：TinyGPT 给它的概率（温度 1），从 0% 到 100%' })}</span>
			{#if nRare}<span class="chip rare sw">{t({ en: `below ${fmtPct(rare, 0)}: ${nRare}`, zh: `低于 ${fmtPct(rare, 0)}：${nRare} 个` })}</span>{/if}
			{#if pic.loop}<span class="chip loop sw">{t({ en: 'a phrase that comes back', zh: '重复出现的短语' })}</span>{/if}
		</p>
		{#if pic.top.length}
			<div class="cands">
				<p class="ch">
					{t({ en: `Token ${pic.made.length}: drawn from`, zh: `第 ${pic.made.length} 个词元的候选` })}
					{#if pic.greedy}({t({ en: 'greedy: the top one, no random number', zh: '贪心：取第一名，不抽随机数' })}){:else}(u = <span class="num">{pic.u.toFixed(3)}</span>){/if}
				</p>
				<ol>
					{#each pic.top as e (e.id)}
						<li class:on={e.id === pic.token}>
							<code>{e.text}</code>
							<span class="b"><span style="width: {(e.p / maxTop) * 100}%; background: {sc.css(0.75)}"></span></span>
							<span class="num">{fmtPct(e.p)}</span>
						</li>
					{/each}
				</ol>
			</div>
		{/if}
	</div>
{/if}

<style>
	.txt {
		display: grid;
		gap: 10px;
		min-width: 0;
	}
	.chips {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		gap: 3px 2px;
		line-height: 1.5;
		font-size: var(--fs-sm);
	}
	.chip {
		padding: 0 3px;
		border-radius: 3px;
		white-space: pre;
		border: 1px solid transparent;
	}
	.chip.pr {
		border: 1px solid var(--rule-strong);
		color: var(--ink);
		background: var(--surface-2);
	}
	.chip.now {
		outline: 2px solid var(--highlight);
		outline-offset: 1px;
	}
	.chip.loop {
		text-decoration: underline;
		text-decoration-thickness: 2px;
		text-underline-offset: 3px;
		border-bottom: 2px solid var(--generated);
	}
	.chip.rare {
		border: 1.5px dashed var(--generated);
	}
	.chip.rare.sw {
		color: var(--ink-2);
	}
	.legend {
		margin: 0;
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 6px 10px;
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.sw {
		font-size: var(--fs-xs);
	}
	.chip.loop.sw {
		color: var(--ink-2);
	}
	.ramp {
		display: inline-block;
		width: 64px;
		height: 10px;
		border-radius: 2px;
		border: 1px solid var(--rule);
	}
	.cands {
		border-top: 1px solid var(--rule);
		padding-top: 6px;
		max-width: 520px;
	}
	.ch {
		margin: 0 0 4px;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	ol {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 2px;
	}
	li {
		display: grid;
		grid-template-columns: minmax(3rem, 7rem) minmax(0, 1fr) 4rem;
		gap: 8px;
		align-items: center;
		font-size: var(--fs-xs);
	}
	li.on {
		outline: 2px solid var(--highlight);
		outline-offset: 1px;
		border-radius: 3px;
	}
	li code {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.b {
		height: 8px;
		position: relative;
	}
	.b span {
		position: absolute;
		left: 0;
		top: 0;
		bottom: 0;
		border-radius: 2px;
	}
	li .num {
		text-align: right;
	}
</style>
