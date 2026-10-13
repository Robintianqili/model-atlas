<!--
  optimizers Capture for the local 'setup' adapter: the page's controls. The landscape of the race (three buttons) and
  its start (two sliders inside that landscape's window, 0.01 apart), the learning-rate schedule (three buttons and a
  warm-up switch) and the gradient-clipping switch of the cliff toy. Reports the whole Setup on every change.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import type { CaptureProps } from '../../../../atlas/types/explainer.ts';
	import { RACE_SURFACES, SCHEDULE_NAMES, setupOf, surfaceOf, type ScheduleId, type Setup } from '../../data.ts';

	let { raw, onchange, compact = false }: CaptureProps<Setup> = $props();

	const s = $derived(setupOf(raw));
	const surf = $derived(surfaceOf(s.surface));
	const round = (v: number) => Math.round(v * 100) / 100;
	const put = (o: Partial<Setup>, live = false) => onchange?.({ ...s, ...o }, { live });
	const pickSurface = (id: Setup['surface']) => {
		const n = surfaceOf(id);
		put({ surface: id, x: n.start[0], y: n.start[1] });
	};
	const SCHEDULES: ScheduleId[] = ['constant', 'step', 'cosine'];
	const T = {
		land: { en: 'Landscape', zh: '地形' },
		start: { en: 'Start', zh: '起点' },
		sched: { en: 'Learning-rate schedule', zh: '学习率调度' },
		warm: { en: 'Warm-up (first 20 steps)', zh: '预热（前 20 步）' },
		clip: { en: 'Gradient clipping (cliff toy)', zh: '梯度裁剪（悬崖玩具）' }
	};
</script>

<div class="setup" class:compact>
	<div class="group" role="group" aria-label={t(T.land)}>
		<span class="cap">{t(T.land)}</span>
		<div class="btns">
			{#each RACE_SURFACES as id (id)}
				<button type="button" class="btn" aria-pressed={s.surface === id} onclick={() => pickSurface(id)}>{t(surfaceOf(id).name)}</button>
			{/each}
		</div>
	</div>
	{#each [{ k: 'x' as const, r: surf.xr }, { k: 'y' as const, r: surf.yr }] as d (d.k)}
		<label class="row">
			<span class="name">{t(T.start)} {d.k}</span>
			<input
				type="range"
				min={d.r[0]}
				max={d.r[1]}
				step="0.01"
				value={s[d.k]}
				oninput={(e) => put({ [d.k]: round(Number((e.currentTarget as HTMLInputElement).value)) }, true)}
				onchange={(e) => put({ [d.k]: round(Number((e.currentTarget as HTMLInputElement).value)) })}
			/>
			<output class="val num">{fmt3(s[d.k])}</output>
		</label>
	{/each}
	<div class="group" role="group" aria-label={t(T.sched)}>
		<span class="cap">{t(T.sched)}</span>
		<div class="btns">
			{#each SCHEDULES as id (id)}
				<button type="button" class="btn" aria-pressed={s.schedule === id} onclick={() => put({ schedule: id })}>{t(SCHEDULE_NAMES[id])}</button>
			{/each}
		</div>
	</div>
	<label class="check"><input type="checkbox" checked={s.warmup} onchange={(e) => put({ warmup: (e.currentTarget as HTMLInputElement).checked })} /> {t(T.warm)}</label>
	<label class="check"><input type="checkbox" checked={s.clip} onchange={(e) => put({ clip: (e.currentTarget as HTMLInputElement).checked })} /> {t(T.clip)}</label>
</div>

<style>
	.setup {
		display: grid;
		gap: 8px;
		min-width: 0;
		max-width: 520px;
		font-size: var(--fs-sm);
	}
	.group {
		display: grid;
		gap: 4px;
		min-width: 0;
	}
	.cap {
		color: var(--ink-2);
	}
	.btns {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.btns .btn {
		font-size: var(--fs-xs);
	}
	.btn[aria-pressed='true'] {
		border-color: var(--accent);
		box-shadow: inset 0 0 0 1px var(--accent);
		font-weight: 600;
	}
	.row {
		display: grid;
		grid-template-columns: minmax(0, 5rem) minmax(0, 1fr) 3.6rem;
		align-items: center;
		gap: 8px;
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
	.check {
		display: flex;
		align-items: center;
		gap: 6px;
	}
	.check input {
		accent-color: var(--accent);
	}
</style>
