<!--
  regularization Capture for the local 'knobs' adapter: five sliders on fixed stops (data.ts), the data (noise, the share
  of points used for training), the network (units per hidden layer) and two regularizers (λ of the L2 penalty, p of
  dropout), with the value next to each. A slider reports on release (each change retrains the network), and the
  starting knobs as buttons (short names in the compact copy beside the lesson). Reports { noise, ratio, width, l2, dropout }.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/explainer.ts';
	import { DROPOUTS, LAMBDAS, NOISES, RATIOS, WIDTHS, knobsOf, splitOf, type Knobs } from '../../data.ts';
	import { STARTS } from './adapter.ts';

	let { raw, onchange, compact = false }: CaptureProps<Knobs> = $props();

	const k = $derived(knobsOf(raw));
	type Key = keyof Knobs;
	const knobs: { k: Key; stops: number[]; name: { en: string; zh: string }; short: { en: string; zh: string }; show: (v: number) => string }[] = [
		{ k: 'noise', stops: NOISES, name: { en: 'Noise in the data', zh: '数据里的噪声' }, short: { en: 'noise', zh: '噪声' }, show: String },
		{ k: 'ratio', stops: RATIOS, name: { en: 'Share of points for training', zh: '用来训练的点的比例' }, short: { en: 'train', zh: '训练' }, show: (v) => `${Math.round(v * 100)}%` },
		{ k: 'width', stops: WIDTHS, name: { en: 'Units per hidden layer', zh: '每个隐藏层的单元数' }, short: { en: 'units', zh: '单元' }, show: String },
		{ k: 'l2', stops: LAMBDAS, name: { en: 'L2 penalty λ', zh: 'L2 惩罚 λ' }, short: { en: 'λ', zh: 'λ' }, show: String },
		{ k: 'dropout', stops: DROPOUTS, name: { en: 'Dropout p', zh: 'Dropout p' }, short: { en: 'p', zh: 'p' }, show: String }
	];
	// the value a slider shows while it is dragged (committed on release)
	let drag = $state<Partial<Record<Key, number>>>({});
	const val = (key: Key) => drag[key] ?? k[key];
	function commit(key: Key, i: number, stops: number[]) {
		drag = {};
		onchange?.({ ...k, [key]: stops[i] });
	}
	const counts = $derived(splitOf(val('noise'), val('ratio')));
</script>

<div class="knobs" class:compact>
	{#each knobs as kn (kn.k)}
		<label class="row">
			<span class="name">{t(compact ? kn.short : kn.name)}</span>
			<input
				type="range"
				min="0"
				max={kn.stops.length - 1}
				step="1"
				value={kn.stops.indexOf(val(kn.k))}
				aria-valuetext={kn.show(val(kn.k))}
				oninput={(e) => (drag = { ...drag, [kn.k]: kn.stops[Number((e.currentTarget as HTMLInputElement).value)] })}
				onchange={(e) => commit(kn.k, Number((e.currentTarget as HTMLInputElement).value), kn.stops)}
			/>
			<output class="val num">{kn.show(val(kn.k))}</output>
		</label>
	{/each}
	<p class="count">
		{t({ en: `${counts.train.n} training points, ${counts.test.n} test points`, zh: `${counts.train.n} 个训练点，${counts.test.n} 个测试点` })}
	</p>
	<div class="starts" role="group" aria-label={t({ en: 'Start from', zh: '从这些设置开始' })}>
		{#each STARTS as s (s.id)}
			<button type="button" class="btn" onclick={() => onchange?.({ ...s.raw })}>{t(compact ? s.short : s.label)}</button>
		{/each}
	</div>
</div>

<style>
	.knobs {
		display: grid;
		gap: 6px;
		min-width: 0;
		max-width: 560px;
	}
	.row {
		display: grid;
		grid-template-columns: minmax(0, 13rem) minmax(0, 1fr) 3.6rem;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-sm);
	}
	.compact .row {
		grid-template-columns: 3.4rem minmax(0, 1fr) 3.4rem;
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
	.count {
		margin: 0;
		font-size: var(--fs-xs);
		color: var(--muted);
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
		.knobs:not(.compact) .row {
			grid-template-columns: minmax(0, 1fr) 3.6rem;
		}
		.knobs:not(.compact) .row .name {
			grid-column: 1 / -1;
		}
	}
</style>
