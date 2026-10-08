<!--
  loss-gradient-descent Capture for the local 'dials' adapter: the neuron's two dials, the weight w and the bias b
  (sliders, 0.05 apart, and the exact value), with the error score of the line they make, recomputed on every move
  (data.ts scoreOf: the mean of the eight squared errors), and three starting dials as buttons (short names in the
  compact copy beside the lesson). Reports { w, b }.
-->
<script lang="ts">
	import { colon, t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import type { CaptureProps } from '../../../../atlas/types/explainer.ts';
	import { B_RANGE, W_RANGE, dialsOf, scoreOf, TARGET, type Dials } from '../../data.ts';
	import { STARTS } from './adapter.ts';

	let { raw, onchange, compact = false }: CaptureProps<Dials> = $props();

	const d = $derived(dialsOf(raw));
	const score = $derived(scoreOf(d.w, d.b));
	const round = (v: number) => Math.round(v * 100) / 100;
	function set(k: 'w' | 'b', v: number, live = false) {
		onchange?.({ ...d, [k]: round(v) }, { live });
	}
	const dials = [
		{ k: 'w' as const, range: W_RANGE, name: { en: 'Weight w (the slope)', zh: '权重 w（斜率）' } },
		{ k: 'b' as const, range: B_RANGE, name: { en: 'Bias b (the height at x = 0)', zh: '偏置 b（x = 0 处的高度）' } }
	];
</script>

<div class="dials" class:compact>
	{#each dials as dl (dl.k)}
		<label class="row">
			<span class="name">{compact ? dl.k : t(dl.name)}</span>
			<input
				type="range"
				min={dl.range[0]}
				max={dl.range[1]}
				step="0.05"
				value={d[dl.k]}
				oninput={(e) => set(dl.k, Number((e.currentTarget as HTMLInputElement).value), true)}
				onchange={(e) => set(dl.k, Number((e.currentTarget as HTMLInputElement).value))}
			/>
			<output class="val num">{fmt3(d[dl.k])}</output>
		</label>
	{/each}
	<p class="score">
		{t({ en: 'Error score', zh: '误差分数' })}{colon()} <strong class="num">{fmt3(score)}</strong>
		{#if score < TARGET}<span class="ok">{t({ en: '(below the goal)', zh: '（低于目标）' })}</span>{/if}
	</p>
	<div class="starts" role="group" aria-label={t({ en: 'Start again from', zh: '重新开始的位置' })}>
		{#each STARTS as s (s.id)}
			<button type="button" class="btn" onclick={() => onchange?.({ ...s.raw })}>{t(compact ? s.short : s.label)}</button>
		{/each}
	</div>
</div>

<style>
	.dials {
		display: grid;
		gap: 8px;
		min-width: 0;
		max-width: 520px;
	}
	.row {
		display: grid;
		grid-template-columns: minmax(0, 11rem) minmax(0, 1fr) 4rem;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-sm);
	}
	.compact .row {
		grid-template-columns: 1.4rem minmax(0, 1fr) 3.4rem;
	}
	.name {
		color: var(--ink-2);
	}
	input[type='range'] {
		width: 100%;
		min-width: 0;
		accent-color: var(--accent);
	}
	.val {
		text-align: right;
	}
	.score {
		margin: 0;
		font-size: var(--fs-sm);
	}
	.ok {
		color: var(--live);
		margin-left: 4px;
	}
	.starts {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.starts .btn {
		font-size: var(--fs-xs);
	}
	@media (max-width: 420px) {
		.dials:not(.compact) .row {
			grid-template-columns: minmax(0, 1fr) 4rem;
		}
		.dials:not(.compact) .row .name {
			grid-column: 1 / -1;
		}
	}
</style>
