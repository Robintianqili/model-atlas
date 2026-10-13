<!--
  ppo-reacher Capture (local adapter 'reacher'): the target of the episode the lesson follows (presets and two sliders,
  kept in the upper half plane, 0.2 m to 1 m from the shoulder). PPO's clip ε is kept as it is: the task step's buttons
  set it, and the input section's note shows it. Reports { target: [x, y], eps }.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { epsOf, targetOf, type Pt } from '../../sim.ts';
	import { PRESETS } from './adapter.ts';

	interface Props { raw: unknown; onchange: (raw: unknown, o?: { live?: boolean }) => void; compact?: boolean }
	let { raw, onchange, compact = false }: Props = $props();
	const g = $derived(targetOf(raw));
	const eps = $derived(epsOf(raw));
	const set = (i: number, v: number, live = false) => {
		const next = g.slice() as Pt;
		next[i] = v;
		onchange({ target: targetOf({ target: next }), eps }, { live });
	};
</script>

<div class="cap" class:compact>
	<div class="row" role="group" aria-label={t({ en: 'Targets', zh: '目标' })}>
		{#each PRESETS as p (p.id)}<button type="button" class="btn" aria-pressed={p.raw.target[0] === g[0] && p.raw.target[1] === g[1]} onclick={() => onchange({ ...p.raw, eps })}>{t(p.label)}</button>{/each}
	</div>
	{#each [['x', -1, 1], ['y', 0, 1]] as [sym, lo, hi], i (sym)}
		<label class="sl">
			<span class="num">{t({ en: 'target', zh: '目标' })} {sym} = {g[i]} m</span>
			<input type="range" min={lo} max={hi} step="0.01" value={g[i]} oninput={(e) => set(i, Number(e.currentTarget.value), true)} onchange={(e) => set(i, Number(e.currentTarget.value))} />
		</label>
	{/each}
	<p class="muted help">{t({ en: 'The shoulder is at (0, 0); each link is 0.5 m. Targets stay above the shoulder line, where training puts them.', zh: '肩膀在 (0, 0)；每节臂长 0.5 m。目标只能放在肩膀那条线以上，和训练时一样。' })}</p>
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
	.sl { display: grid; grid-template-columns: 9.5em minmax(0, 1fr); gap: 0 8px; align-items: center; }
	.num { font-family: var(--font-mono, monospace); font-size: var(--fs-xs); }
	.help { margin: 0; font-size: var(--fs-xs); }
	.compact .help { display: none; }
</style>
