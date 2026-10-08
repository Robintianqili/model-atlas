<!--
  node2vec Capture (local adapter 'walk-graph'): node2vec's p and q as buttons, then the kit graph Capture as it is (the
  starting graphs, the member in focus, the club and its table to edit friendships), whose changes keep p and q.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { CaptureProps } from '../../../../atlas/types/index.ts';
	import GraphCapture from '../../../../atlas/adapters/graph/Capture.svelte';
	import { inputOf, PQ, type WalkRaw } from '../../walks.ts';

	let { raw, onchange, compact = false }: CaptureProps<WalkRaw> = $props();
	const g = $derived(inputOf(raw));
	const ROWS = [
		{ k: 'p' as const, label: { en: 'p (return): going back weighs 1/p', zh: 'p（返回）：走回去的权重是 1/p' } },
		{ k: 'q' as const, label: { en: 'q (in-out): moving farther weighs 1/q', zh: 'q（进出）：走得更远的权重是 1/q' } }
	];
	/** Kit issue 6 (workaround): the kit Club puts its clickable members (role="button") inside an svg with role="img",
	 *  which axe fails (nested-interactive, AK604). An svg that holds buttons is a group, as in the gcn package's Graph. */
	function buttonGroup(node: HTMLElement) {
		const fix = () => node.querySelectorAll('svg[role="img"]').forEach((s) => s.querySelector('[role="button"]') && s.setAttribute('role', 'group'));
		const mo = new MutationObserver(fix);
		fix();
		mo.observe(node, { childList: true, subtree: true });
		return { destroy: () => mo.disconnect() };
	}
</script>

<div class="cap" use:buttonGroup>
	{#each ROWS as r (r.k)}
		<div class="row" role="group" aria-label={t(r.label)}>
			<span>{t(r.label)}</span>
			{#each PQ as v (v)}<button type="button" class="btn" aria-pressed={g[r.k] === v} onclick={() => onchange?.({ ...g, [r.k]: v })}>{v}</button>{/each}
		</div>
	{/each}
	<GraphCapture raw={g} {compact} onchange={(x, o) => onchange?.({ ...x, p: g.p, q: g.q }, o)} />
</div>

<style>
	.cap { display: grid; gap: 8px; min-width: 0; font-size: var(--fs-sm); } .row { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
	.btn { min-height: 32px; min-width: 40px; padding: 2px 8px; font-size: var(--fs-xs); } .btn[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); }
</style>
