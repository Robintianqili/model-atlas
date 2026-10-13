<!--
  Lesson step 7: softmax of one row of masked scores (the followed query, one head), in three
  phases on a StepPlayer: the scores s (masked keys are −∞), e^(s − m), and the weights e/Σ.
  Every number comes from the live trace (L{b}.attn.masked); the weights are checked against
  L{b}.attn.weights. Clicking a key row picks the key for the formula below (lesson state).
-->
<script lang="ts">
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3, fmtPct } from '$lib/kit/format';
	import type { TGState } from './state.svelte';

	interface Props {
		LS: TGState;
		autoplay?: boolean;
		/** ms per phase */
		interval?: number;
		onend?: () => void;
	}
	let { LS, autoplay = false, interval = 2200, onend }: Props = $props();

	let phase = $state(0);
	const f = $derived(LS.run?.forward);
	const T = $derived(LS.T);
	const b = $derived(LS.block);
	const h = $derived(LS.head);
	const i = $derived(LS.pos);
	const k = $derived(Math.min(LS.key, i));
	const r = $derived.by(() => {
		if (!f) return null;
		const o = (h * T + i) * T;
		const s = f.t[`L${b}.attn.masked`].data.subarray(o, o + T);
		const w = f.t[`L${b}.attn.weights`].data.subarray(o, o + T);
		let m = -Infinity;
		for (let j = 0; j <= i; j++) m = Math.max(m, s[j]);
		const e = new Float64Array(T);
		let Z = 0;
		for (let j = 0; j <= i; j++) {
			e[j] = Math.exp(s[j] - m);
			Z += e[j];
		}
		let sMax = 0;
		for (let j = 0; j <= i; j++) sMax = Math.max(sMax, Math.abs(s[j]));
		let maxDiff = 0;
		for (let j = 0; j <= i; j++) maxDiff = Math.max(maxDiff, Math.abs(e[j] / Z - w[j]));
		return { s, w, e, m, Z, sMax: sMax || 1, maxDiff };
	});
	const sc = $derived(r ? makeScale('diverging', [-r.sMax, r.sMax], themeState.resolved) : null);
	const sq = $derived(makeScale('sequential', [0, 1], themeState.resolved));
	const labels = [
		{ en: 'The scores of this row. Keys after the query were set to −∞ by the causal mask.', zh: '这一行的分数。查询之后的键已被因果掩码设为 −∞。' },
		{ en: 'Subtract the largest score m and raise e to each: the largest becomes 1, everything is positive, and e^(−∞) = 0.', zh: '减去最大分数 m，再对每个取 e 的幂：最大的变成 1，全部为正，而 e^(−∞) = 0。' },
		{ en: 'Divide by the sum Σ: the weights are between 0 and 1 and add up to exactly 1.', zh: '除以总和 Σ：权重都在 0 和 1 之间，加起来正好等于 1。' }
	];
	const rows = $derived(Array.from({ length: T }, (_, j) => j));
	// long texts: keep the selected key's row in view inside the table box (not the page)
	let tw: HTMLElement | undefined = $state();
	$effect(() => {
		void k;
		void T;
		const box = tw;
		if (!box) return;
		requestAnimationFrame(() => {
			const tr = box.querySelector('tr.sel');
			if (!tr) return;
			const d = tr.getBoundingClientRect().top - box.getBoundingClientRect().top;
			if (d < 30 || d > box.clientHeight - 40) box.scrollTop += d - box.clientHeight / 2;
		});
	});
</script>

{#if r && sc}
	<div class="sr">
		<StepPlayer count={3} bind:step={phase} {labels} {interval} {autoplay} {onend} counter={{ en: 'Part {i} of {n}', zh: '第 {i} / {n} 部分' }} label={{ en: 'Softmax of one row, in three steps', zh: '对一行做 softmax，分三步' }}>
			{#snippet children(ph)}
				<div class="tw" bind:this={tw}>
					<table class="num">
						<caption class="visually-hidden">{t({ en: 'Softmax of the attention scores of one query', zh: '一个查询的注意力分数的 softmax' })}</caption>
						<thead>
							<tr>
								<th scope="col" class="kc">{t({ en: 'key', zh: '键' })}</th>
								<th scope="col">{t({ en: 'score s', zh: '分数 s' })}</th>
								<th scope="col" class:dim={ph < 1}>e<sup>s − m</sup></th>
								<th scope="col" class:dim={ph < 2}><span class="wide">{t({ en: 'weight', zh: '权重' })}{' '}</span>a = e / Σ</th>
							</tr>
						</thead>
						<tbody>
							{#each rows as j (j)}
								{@const fut = j > i}
								<tr class:fut class:sel={j === k}>
									<th scope="row" class="kc">
										{#if fut}<span class="tok">{LS.label(j)}</span>{:else}<button type="button" class="kb" aria-pressed={j === k} onclick={() => LS.selectKey(j)}><span class="tok">{LS.label(j)}</span></button>{/if}
									</th>
									{#if fut}
										<td class="hatch">−∞</td>
										<td class="hatch" class:dim={ph < 1}>{ph >= 1 ? '0' : ''}</td>
										<td class="hatch" class:dim={ph < 2}>{ph >= 2 ? '0' : ''}</td>
									{:else}
										<td>
											<span class="cell">
												<span class="sbar" aria-hidden="true"
													><span
														style="left: {r.s[j] < 0 ? 50 - (Math.abs(r.s[j]) / r.sMax) * 50 : 50}%; width: {(Math.abs(r.s[j]) / r.sMax) * 50}%; background: {sc.css(r.s[j])}"
													></span></span
												>{fmt3(r.s[j])}
											</span>
										</td>
										<td class:dim={ph < 1}>
											{#if ph >= 1}<span class="cell"><span class="pbar" aria-hidden="true"><span style="width: {r.e[j] * 100}%"></span></span>{fmt3(r.e[j])}</span>{/if}
										</td>
										<td class:dim={ph < 2}>
											{#if ph >= 2}<span class="cell"><span class="pbar w" aria-hidden="true"><span style="width: {r.w[j] * 100}%; background: {sq.css(r.w[j])}"></span></span>{fmtPct(r.w[j])}</span>{/if}
										</td>
									{/if}
								</tr>
							{/each}
						</tbody>
						<tfoot>
							<tr>
								<th scope="row" class="kc">Σ</th>
								<td class="muted">m = {fmt3(r.m)}</td>
								<td class:dim={ph < 1}>{#if ph >= 1}Σ = {fmt3(r.Z)}{/if}</td>
								<td class:dim={ph < 2}>{#if ph >= 2}100%{/if}</td>
							</tr>
						</tfoot>
					</table>
				</div>
			{/snippet}
		</StepPlayer>
		<p class="note muted">
			{t({
				en: `Query “${LS.label(i)}” (position ${i}), block ${b + 1}, head ${h + 1}. Click a row to pick a key. Largest difference from the model’s own weights: ${r.maxDiff.toExponential(1)}.`,
				zh: `查询“${LS.label(i)}”（位置 ${i}），第 ${b + 1} 块第 ${h + 1} 头。点击一行可选择一个键。与模型自己算出的权重的最大差异：${r.maxDiff.toExponential(1)}。`
			})}
		</p>
	</div>
{/if}

<style>
	/* a superscript in the table head stays at the 12 px floor */
	th sup {
		font-size: var(--t-xs);
	}
	.sr {
		display: grid;
		gap: 8px;
		grid-template-columns: minmax(0, 1fr);
	}
	.tw {
		max-height: 380px;
		overflow: auto;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
	}
	table {
		border-collapse: collapse;
		width: 100%;
		font-size: 12px;
	}
	th,
	td {
		padding: 2px 8px;
		text-align: right;
		border-bottom: 1px solid var(--surface-2);
		white-space: nowrap;
	}
	thead th {
		position: sticky;
		top: 0;
		background: var(--surface-2);
		font-weight: 650;
		z-index: 1;
	}
	.kc {
		text-align: left;
	}
	.tok {
		font-family: var(--font-mono);
		white-space: pre;
		font-variant-ligatures: none;
	}
	.kb {
		border: 0;
		background: none;
		padding: 0 2px;
		color: inherit;
		font: inherit;
		cursor: pointer;
		border-radius: 3px;
		min-height: 22px;
	}
	.kb:hover {
		background: var(--surface-3);
	}
	tr.sel {
		box-shadow: inset 3px 0 0 var(--k);
		background: color-mix(in srgb, var(--k) 8%, var(--surface));
	}
	.fut {
		color: var(--muted);
	}
	.hatch {
		background: repeating-linear-gradient(-45deg, var(--surface-2) 0 4px, var(--surface) 4px 8px);
	}
	/* a column header not reached yet: muted text (opacity would drop it below 4.5:1); empty cells stay faded */
	th.dim {
		color: var(--muted);
	}
	td.dim {
		opacity: 0.35;
	}
	.cell {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		justify-content: flex-end;
	}
	.sbar,
	.pbar {
		position: relative;
		display: inline-block;
		width: 60px;
		height: 8px;
		border-radius: 3px;
		background: var(--surface-3);
		overflow: hidden;
	}
	.sbar span {
		position: absolute;
		top: 0;
		bottom: 0;
	}
	.pbar span {
		display: block;
		height: 100%;
		background: var(--accent);
	}
	tfoot th,
	tfoot td {
		border-top: 1.5px solid var(--ink-2);
		position: sticky;
		bottom: 0;
		background: var(--surface);
	}
	.note {
		margin: 0;
		font-size: var(--fs-xs);
	}
	@media (max-width: 480px) {
		.sbar,
		.wide,
		td:nth-child(3) .pbar {
			display: none;
		}
		.pbar {
			width: 34px;
		}
		th,
		td {
			padding: 2px 4px;
		}
	}
</style>
