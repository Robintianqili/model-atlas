<!--
  cart-pole Capture (local adapter 'cart-pole'): the start state of the episode the lesson follows, four sliders
  (x, ẋ, θ, θ̇) and the presets. Reports { start: [x, ẋ, θ, θ̇] }. The compact copy beside the lesson keeps everything
  but the help lines.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { I18n } from '$lib/i18n/types';
	import { RANGES, startOf, type State } from '../../sim.ts';
	import { PRESETS } from './adapter.ts';

	interface Props { raw: unknown; onchange: (raw: unknown, o?: { live?: boolean }) => void; compact?: boolean }
	let { raw, onchange, compact = false }: Props = $props();
	const s = $derived(startOf(raw));
	const set = (i: number, v: number, live = false) => {
		const next = s.slice() as State;
		next[i] = Math.round(v * 100) / 100;
		onchange({ start: next }, { live });
	};
	const SLIDERS: [string, string, I18n][] = [
		['x', 'm', { en: 'cart position (0 = middle of the track)', zh: '小车位置（0 = 轨道中点）' }],
		['ẋ', 'm/s', { en: 'cart velocity (+ = moving right)', zh: '小车速度（+ = 向右）' }],
		['θ', 'rad', { en: 'pole angle from upright (+ = leaning right)', zh: '杆偏离竖直的角度（+ = 向右倾斜）' }],
		['θ̇', 'rad/s', { en: 'how fast the angle changes', zh: '角度变化的快慢' }]
	];
	const same = (a: State) => a.every((v, i) => v === s[i]);
</script>

<div class="cap" class:compact>
	<div class="row" role="group" aria-label={t({ en: 'Start states', zh: '起始状态' })}>
		{#each PRESETS as p (p.id)}<button type="button" class="btn" aria-pressed={same(p.raw.start)} onclick={() => onchange(p.raw)}>{t(p.label)}</button>{/each}
	</div>
	{#each SLIDERS as [sym, unit, help], i (sym)}
		<label class="sl">
			<span class="num">{sym} = {s[i]} {unit}</span>
			<input type="range" min={RANGES[i][0]} max={RANGES[i][1]} step="0.01" value={s[i]} oninput={(e) => set(i, Number(e.currentTarget.value), true)} onchange={(e) => set(i, Number(e.currentTarget.value))} />
			<span class="muted">{t(help)}</span>
		</label>
	{/each}
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	.sl { display: grid; grid-template-columns: 9.5em minmax(0, 1fr); gap: 0 8px; align-items: center; }
	.num { font-family: var(--font-mono, monospace); font-size: var(--fs-xs); }
	.sl .muted { grid-column: 2; font-size: var(--fs-xs); }
	.compact .sl .muted { display: none; }
</style>
