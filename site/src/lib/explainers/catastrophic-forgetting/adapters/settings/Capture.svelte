<!--
  catastrophic-forgetting Capture for the local 'settings' adapter: how the network trains on task B. Two sliders (the
  replay share, 0 to 50% in steps of 5%, with the number of stored task-A points it means; the pull λ, 0 to 1 in steps of
  0.1), two switches (freeze the first layer; a learning rate ten times smaller) and the presets as buttons (short names
  in the compact copy beside the lesson). Reports { replay, pull, freeze, slow }. Copied in shape from the
  loss-gradient-descent 'dials' widget.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/explainer.ts';
	import { MAX_PULL, MAX_REPLAY, replayCount, settingsOf, type Settings } from '../../data.ts';
	import { PRESETS } from './adapter.ts';

	let { raw, onchange, compact = false }: CaptureProps<Settings> = $props();

	const s = $derived(settingsOf(raw));
	// a multiple of the slider step, without float dust (0.15, not 0.15000000000000002)
	const round = (v: number, d: number) => Number((Math.round(v / d) * d).toFixed(2));
	function set(patch: Partial<Settings>, live = false) {
		onchange?.({ ...s, ...patch }, { live });
	}
	const pct = (v: number) => `${Math.round(v * 100)}%`;
	const stored = (k: number) => ({ en: `${k} stored task-A points`, zh: `存下 ${k} 个任务 A 的点` });
	const sliders = [
		{ k: 'replay' as const, max: MAX_REPLAY, step: 0.05, show: pct, short: { en: 'Replay', zh: '回放' }, name: { en: 'Replay: share of each batch from task A', zh: '回放：每批中来自任务 A 的比例' } },
		{ k: 'pull' as const, max: MAX_PULL, step: 0.1, show: (v: number) => String(Math.round(v * 10) / 10), short: { en: 'Pull λ', zh: '拉力 λ' }, name: { en: 'Pull λ toward the weights after task A', zh: '拉向任务 A 之后权重的拉力 λ' } }
	];
	const L = {
		freeze: { en: 'Freeze the first layer', zh: '冻结第一层' },
		slow: { en: 'Learning rate ten times smaller on task B', zh: '在任务 B 上把学习率缩小到十分之一' },
		presets: { en: 'Ready-made settings', zh: '预设' }
	};
</script>

<div class="settings" class:compact>
	{#each sliders as sl (sl.k)}
		<label class="row">
			<span class="name">{t(compact ? sl.short : sl.name)}</span>
			<input
				type="range"
				min="0"
				max={sl.max}
				step={sl.step}
				value={s[sl.k]}
				oninput={(e) => set({ [sl.k]: round(Number((e.currentTarget as HTMLInputElement).value), sl.step) }, true)}
				onchange={(e) => set({ [sl.k]: round(Number((e.currentTarget as HTMLInputElement).value), sl.step) })}
			/>
			<output class="val num">{sl.show(s[sl.k])}</output>
		</label>
		{#if sl.k === 'replay' && !compact}<p class="note">{t(stored(replayCount(s.replay)))}</p>{/if}
	{/each}
	<label class="check"><input type="checkbox" checked={s.freeze} onchange={(e) => set({ freeze: (e.currentTarget as HTMLInputElement).checked })} /> {t(L.freeze)}</label>
	<label class="check"><input type="checkbox" checked={s.slow} onchange={(e) => set({ slow: (e.currentTarget as HTMLInputElement).checked })} /> {t(L.slow)}</label>
	<div class="presets" role="group" aria-label={t(L.presets)}>
		{#each PRESETS as p (p.id)}
			<button type="button" class="btn" onclick={() => onchange?.({ ...p.raw })}>{t(compact ? p.short : p.label)}</button>
		{/each}
	</div>
</div>

<style>
	.settings {
		display: grid;
		gap: 8px;
		min-width: 0;
		max-width: 560px;
	}
	.row {
		display: grid;
		grid-template-columns: minmax(0, 15rem) minmax(0, 1fr) 3.4rem;
		align-items: center;
		gap: 8px;
		font-size: var(--fs-sm);
	}
	.compact .row {
		grid-template-columns: 4.6rem minmax(0, 1fr) 2.8rem;
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
	.note {
		margin: -4px 0 0;
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.check {
		display: flex;
		gap: 6px;
		align-items: center;
		font-size: var(--fs-sm);
	}
	.check input {
		accent-color: var(--accent);
	}
	.presets {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.presets .btn {
		font-size: var(--fs-xs);
	}
	@media (max-width: 420px) {
		.settings:not(.compact) .row {
			grid-template-columns: minmax(0, 1fr) 3.4rem;
		}
		.settings:not(.compact) .row .name {
			grid-column: 1 / -1;
		}
	}
</style>
