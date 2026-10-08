<!--
  hello-xor Capture for the local 'xor-point' adapter: two numbers x1 and x2 (quick 0/1 buttons, or any value in
  -3..3 with the sliders), plus the four XOR points as presets. Reports [x1, x2]; the adapter turns it into a [2] tensor.
  Props follow the W1-D1 Capture contract (atlas/ui/capture.ts CaptureProps: rt, raw, presets, onchange, compact),
  declared locally so this file does not depend on the UI package.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { I18n } from '$lib/i18n/types';
	import { toPoint } from './adapter.ts';

	interface Props {
		rt?: unknown;
		raw: unknown;
		presets?: { id: string; label: I18n; raw: unknown }[];
		onchange: (raw: unknown, o?: { live?: boolean }) => void;
		compact?: boolean;
	}
	let { raw, presets = [], onchange, compact = false }: Props = $props();

	const p = $derived(toPoint(raw));
	const set = (i: 0 | 1, v: number, live = false) => {
		const next: [number, number] = [p[0], p[1]];
		next[i] = Math.round(v * 100) / 100;
		onchange(next, { live });
	};
</script>

<div class="xor-cap" class:compact>
	{#each [0, 1] as const as i (i)}
		<div class="row">
			<span class="name">x{i + 1}</span>
			<button type="button" class:on={p[i] === 0} aria-pressed={p[i] === 0} onclick={() => set(i, 0)}>0</button>
			<button type="button" class:on={p[i] === 1} aria-pressed={p[i] === 1} onclick={() => set(i, 1)}>1</button>
			{#if !compact}
				<input
					type="range"
					min="-3"
					max="3"
					step="0.05"
					value={p[i]}
					aria-label={t({ en: `x${i + 1}, any value`, zh: `x${i + 1}，任意值` })}
					oninput={(e) => set(i, Number((e.currentTarget as HTMLInputElement).value), true)}
					onchange={(e) => set(i, Number((e.currentTarget as HTMLInputElement).value))}
				/>
			{/if}
			<output class="val">{p[i]}</output>
		</div>
	{/each}
	{#if presets.length && !compact}
		<div class="presets" role="group" aria-label={t({ en: 'The four XOR points', zh: '四个 XOR 点' })}>
			{#each presets as pr (pr.id)}
				<button type="button" onclick={() => onchange(toPoint(pr.raw))}>{t(pr.label)}</button>
			{/each}
		</div>
	{/if}
</div>

<style>
	.xor-cap {
		display: grid;
		gap: 0.5rem;
		min-width: 0;
	}
	.row {
		display: flex;
		align-items: center;
		gap: 0.4rem;
		flex-wrap: wrap;
	}
	.name {
		font-family: var(--font-mono, monospace);
		width: 1.8rem;
	}
	button {
		min-width: 2.2rem;
		min-height: 2.2rem;
		border: 1px solid var(--border, #8884);
		border-radius: 6px;
		background: var(--surface, transparent);
		color: inherit;
		cursor: pointer;
	}
	button.on {
		background: var(--accent-soft, #8882);
		border-color: var(--accent, currentColor);
		font-weight: 600;
	}
	input[type='range'] {
		flex: 1 1 8rem;
		min-width: 0;
	}
	.val {
		font-family: var(--font-mono, monospace);
		min-width: 3rem;
		text-align: right;
	}
	.presets {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
	}
	.presets button {
		padding: 0 0.6rem;
	}
</style>
