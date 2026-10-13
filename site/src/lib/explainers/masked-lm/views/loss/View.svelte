<!-- masked-lm 'loss' (meta.ts): the masked-LM loss row by row; graded rows get a −log p bar (one scale for every bar). -->
<script lang="ts">
	import { keyScroll } from '$lib/kit/keyscroll';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { fmt3, fmtPct, visibleToken } from '$lib/kit/format';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import type { MaskedRaw } from '../../adapters/masked-text/adapter.ts';
	import { lossOf, withDraw } from './meta.ts';

	let P: StageProps = $props();
	const S = $derived(lossOf(P));
	const bar = $derived(makeScale('sequential', [0, 1], themeState.resolved).css(0.72));
	const w = (v: number) => `${Math.max(0.5, (v / S.max) * 100)}%`;
	function send() {
		if (!S.draw) return;
		const raw = withDraw(P.rt.input as MaskedRaw, S.draw);
		if (P.setInput) P.setInput(raw);
		else P.rt.setInput(raw);
	}
</script>

<div class="ls">
	{#if P.props.draw}
		<div class="send">
			<button type="button" class="btn" disabled={!S.draw} aria-pressed={S.sent} onclick={send}>{t({ en: 'Send your row of the training draw to TinyBERT', zh: '把训练抽签里你那一行交给 TinyBERT' })}</button>
			<span class="small">{t(S.sent ? { en: 'TinyBERT now reads your text as training would show it.', zh: 'TinyBERT 现在读到的，就是训练时你的文字会呈现的样子。' } : S.draw ? { en: 'Row 0 of the batch is your text.', zh: '批次的第 0 行就是你的文字。' } : { en: 'The draw is still running.', zh: '抽签还在进行。' })}</span>
		</div>
	{/if}
	<div class="wrap" use:keyScroll>
		<table>
			<caption>{t({ en: `−log p of the right token at each position, in nats; bars share one scale, 0 to ${fmt3(S.max)}`, zh: `每个位置上正确词元的 −log p（单位：奈特）；所有条共用一个刻度，0 到 ${fmt3(S.max)}` })}</caption>
			<thead>
				<tr>
					<th scope="col">#</th>
					<th scope="col">{t({ en: 'Token → what TinyBERT reads', zh: '词元 → TinyBERT 读到的' })}</th>
					<th scope="col">{t({ en: 'p (right token)', zh: 'p（正确词元）' })}</th>
					<th scope="col">{t({ en: 'TinyBERT −log p', zh: 'TinyBERT 的 −log p' })}</th>
					{#if S.gpt}<th scope="col">{t({ en: 'TinyGPT −log p', zh: 'TinyGPT 的 −log p' })}</th>{/if}
				</tr>
			</thead>
			<tbody>
				{#each S.rows as r (r.k)}
					<tr class:graded={r.graded}>
						<td class="num">{r.k}</td>
						<td class="tok">{visibleToken(r.word)}{#if r.graded}{' → '}{t(r.reads)}{/if}</td>
						<td class="num">{fmtPct(r.p)}</td>
						<td>
							{#if r.graded}<span class="cell"><span class="bar" style="width: {w(r.nll)}; background: {bar}"></span><span class="num">{fmt3(r.nll)}</span></span>
							{:else}<span class="muted" aria-hidden="true">—</span><span class="visually-hidden">{t({ en: 'not graded', zh: '不批改' })}</span>{/if}
						</td>
						{#if S.gpt}
							<td>
								{#if r.clm !== null}<span class="cell"><span class="bar" style="width: {w(r.clm)}; background: {bar}"></span><span class="num">{fmt3(r.clm)}</span></span>
								{:else}<span class="muted" aria-hidden="true">—</span><span class="visually-hidden">{t({ en: 'nothing before it', zh: '前面没有词元' })}</span>{/if}
							</td>
						{/if}
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
	<p class="small">{t(S.caption)}</p>
</div>

<style>
	.ls { display: grid; grid-template-columns: minmax(0, 1fr); gap: 10px; min-width: 0; }
	.wrap { overflow-x: auto; max-width: 100%; position: relative; } /* clips the visually-hidden spans (position: absolute) too */
	table { border-collapse: collapse; font-size: var(--fs-sm); width: 100%; }
	caption { text-align: left; color: var(--ink-2); font-size: var(--fs-xs); padding-bottom: 4px; }
	th, td { padding: 3px 5px; text-align: left; border-bottom: 1px solid var(--rule); white-space: nowrap; }
	th { font-weight: 600; font-size: var(--fs-xs); color: var(--ink-2); }
	tr.graded td { background: var(--accent-soft); }
	.tok { font-family: var(--font-mono); }
	.cell { display: flex; align-items: center; gap: 6px; min-width: 84px; }
	.bar { display: inline-block; height: 10px; border-radius: 2px; flex: 0 0 auto; max-width: 70%; }
	.muted { color: var(--muted); }
	.small { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); }
	.send { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
	.btn { min-height: 32px; padding: 2px 12px; font-size: var(--fs-sm); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
</style>
