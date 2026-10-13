<!--
  'decoding/dist' (meta.ts has the props): the ten most likely next tokens as rows. The bar is the chance after the
  temperature (one purple), the outline behind it the chance at temperature 1; the right column the chance after the
  cutoffs, renormalized, or "cut". Dashed lines between rows mark where top-k and top-p cut. With a draw, a strip lays
  the kept chances end to end from 0 to 1 and a marker shows where the random number u falls.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmtInt, fmtPct } from '$lib/kit/format';
	import { fmtSetting } from '../../../../atlas/core/format.ts';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { V } from '../../decode.ts';
	import { distOf, settingsText } from './meta.ts';

	let { rt, at, props }: StageProps = $props();
	const pic = $derived(distOf({ rt, at, props }));
	const sc = $derived(makeScale('sequential', [0, 1], themeState.resolved));
	const bar = $derived(sc.css(0.75));
	const w = (v: number) => `${Math.max(0, Math.min(100, (v / (pic?.max ?? 1)) * 100))}%`;
	const CUT = { en: 'cut', zh: '截掉' };
</script>

{#if pic}
	<div class="dist">
		<p class="head">
			<strong>{t(settingsText(pic.s))}</strong>
			<span class="num">{fmtInt(pic.kept)} / {fmtInt(V)}</span>
			{t({ en: 'tokens', zh: '个词元' })}
			{t({ en: 'can be picked', zh: '可以被选中' })}
		</p>
		<ol class="rows">
			<li class="row hdr" aria-hidden="true">
				<span>{t({ en: 'Rank', zh: '名次' })}</span><span>{t({ en: 'Token', zh: '词元' })}</span><span>{t({ en: 'Chance', zh: '概率' })}</span><span class="r">{t({ en: 'After cutoffs', zh: '截断后' })}</span>
			</li>
			{#each pic.rows as r, i (r.id)}
				<li class="row" class:cut={r.final === 0} class:picked={pic.draw?.token === r.id}>
					<span class="rk num">{i + 1}</span>
					<span class="tok"><code>{r.text}</code></span>
					<span class="track">
						{#if pic.ghost}<span class="ghost" style="width: {w(r.p1)}"></span>{/if}
						<span class="fill" style="width: {w(r.pT)}; background: {r.final === 0 ? 'transparent' : bar}"></span>
						<span class="pv num">{fmtPct(r.pT)}</span>
					</span>
					<span class="fin num r">{r.final > 0 ? fmtPct(r.final) : t(CUT)}</span>
				</li>
				{#if pic.kLine === i + 1}
					<li class="line"><span>top-k = {pic.s.topK}{t({ en: ': keep the rows above', zh: '：保留线上方的行' })}</span></li>
				{/if}
				{#if pic.pLine === i + 1}
					<li class="line p">
						<span>top-p = {fmtSetting(pic.s.topP)}{t({ en: `: the rows above add up to ${fmtPct(r.cum)}`, zh: `：线上方各行加起来是 ${fmtPct(r.cum)}` })}</span>
					</li>
				{/if}
			{/each}
			<li class="row rest" class:cut={pic.rest.kept === 0}>
				<span class="rk"></span>
				<span class="tok">{t({ en: `the other ${fmtInt(V - pic.rows.length)}`, zh: `其余 ${fmtInt(V - pic.rows.length)} 个` })}</span>
				<span class="track"><span class="pv num">{fmtPct(pic.rest.pT)}</span></span>
				<span class="fin num r">{pic.rest.kept > 0 ? t({ en: `${fmtInt(pic.rest.kept)} kept`, zh: `保留 ${fmtInt(pic.rest.kept)} 个` }) : t(CUT)}</span>
			</li>
		</ol>
		<p class="note">
			{t({ en: 'Bar length: chance (the longest bar is', zh: '条的长度：概率（最长的条是' })} <span class="num">{fmtPct(pic.max)}</span>{t({ en: ')', zh: '）' })}{#if pic.ghost}; {t({ en: 'outline: at temperature 1', zh: '轮廓：温度为 1 时' })}{/if}{#if pic.nP > pic.rows.length && pic.s.topP < 1 && !pic.greedy}; {t({ en: `top-p keeps ${pic.nP} tokens, so its line falls below this list`, zh: `top-p 保留 ${pic.nP} 个词元，所以它的线在这张表下面` })}{/if}
		</p>
		{#if pic.draw}
			{@const d = pic.draw}
			<figure class="draw">
				<figcaption>{t({ en: 'Every kept chance laid end to end in vocabulary order, from 0 to 1: each piece is as long as its chance', zh: '把每个保留下来的概率按词表顺序首尾相接，从 0 排到 1：每一段的长度就是它的概率' })}</figcaption>
				<div class="strip">
					{#each d.strip as s (s.from)}
						<span class="seg" class:on={s.id === d.token} class:wide={s.to - s.from > 0.004} style="left: {s.from * 100}%; width: {(s.to - s.from) * 100}%; background: {bar}"></span>
					{/each}
					{#if Number.isFinite(d.u)}<span class="u" style="left: {d.u * 100}%"></span>{/if}
				</div>
				<p class="pick">
					{#if Number.isFinite(d.u)}u = <span class="num">{d.u.toFixed(3)}</span> → {/if}<code>{d.text}</code>
				</p>
			</figure>
		{/if}
	</div>
{/if}

<style>
	.dist {
		display: grid;
		gap: 8px;
		min-width: 0;
		max-width: 640px;
	}
	.head,
	.note,
	.pick {
		margin: 0;
		font-size: var(--fs-sm);
		color: var(--ink-2);
	}
	.note {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.rows {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 2px;
	}
	.row {
		display: grid;
		grid-template-columns: 2.2rem minmax(4rem, 7rem) minmax(0, 1fr) 5.4rem;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-sm);
		min-height: 24px;
	}
	.hdr {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.r {
		text-align: right;
	}
	.rk {
		color: var(--muted);
		text-align: right;
	}
	.tok {
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.track {
		position: relative;
		height: 18px;
		min-width: 0;
	}
	.ghost,
	.fill {
		position: absolute;
		left: 0;
		top: 2px;
		height: 14px;
		border-radius: 2px;
	}
	.ghost {
		border: 1.5px solid var(--ink-2);
		box-sizing: border-box;
	}
	.fill {
		top: 4px;
		height: 10px;
	}
	.cut .fill {
		border: 1.5px dashed var(--muted);
		box-sizing: border-box;
	}
	.pv {
		position: absolute;
		right: 0;
		top: 0;
		font-size: var(--fs-xs);
		color: var(--ink-2);
		background: var(--surface);
		padding: 0 2px;
		border-radius: 2px;
	}
	.cut .tok,
	.cut .fin {
		color: var(--muted);
	}
	.picked {
		outline: 2px solid var(--highlight);
		outline-offset: 1px;
		border-radius: 3px;
	}
	.line {
		border-top: 2px dashed var(--accent);
		font-size: var(--fs-xs);
		color: var(--ink-2);
		padding: 1px 0 3px 2.2rem;
	}
	.line.p {
		border-top-color: var(--highlight);
	}
	.rest .tok {
		font-style: italic;
		white-space: normal;
	}
	.draw {
		margin: 4px 0 0;
		display: grid;
		gap: 4px;
	}
	.draw figcaption {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.strip {
		position: relative;
		height: 22px;
		border: 1px solid var(--rule-strong);
		border-radius: 3px;
		overflow: hidden;
	}
	.seg {
		position: absolute;
		top: 0;
		bottom: 0;
		box-sizing: border-box;
	}
	.seg.wide {
		border-right: 1px solid var(--surface);
	}
	.seg.on {
		outline: 2px solid var(--highlight);
		outline-offset: -2px;
	}
	.u {
		position: absolute;
		top: -2px;
		bottom: -2px;
		width: 3px;
		margin-left: -1px;
		background: var(--ink);
	}
	@media (max-width: 420px) {
		.row {
			grid-template-columns: 1.6rem minmax(3.2rem, 5rem) minmax(0, 1fr) 4.6rem;
			gap: 6px;
		}
		.line {
			padding-left: 1.6rem;
		}
	}
</style>
