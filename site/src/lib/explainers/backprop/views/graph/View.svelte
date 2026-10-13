<!--
  'backprop/graph': the toy computation graph at a frame (meta.ts has the props). The Diagram draws the nodes and edges;
  under it, the line that says what this frame did and, at the check frame, the table of backpropagation against the
  numerical estimate. With props.stepper the view walks the source's own frames with buttons, reading the current tape
  each time, so an edit of the leaves (a new run) keeps the position and shows the new numbers.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { fmt3 } from '$lib/kit/format';
	import type { StageProps } from '../../../../atlas/types/view.ts';
	import { EDGES, NEURON } from '../../data.ts';
	import Diagram from './Diagram.svelte';
	import { captionOf, graphOf } from './meta.ts';

	let { rt, source, at, props }: StageProps = $props();

	/** the stepper's frame index (null: the step's own frame) */
	let idx = $state<number | null>(null);
	const tape = $derived(at?.tape ?? rt.sources[source]?.tape);
	const shown = $derived.by(() => {
		if (!props.stepper || idx === null || !tape || tape.length === 0) return { frame: at?.frame, state: at?.state };
		const i = Math.min(idx, tape.length - 1);
		return { frame: tape.frame(i), state: tape.state(i) };
	});
	const pic = $derived(graphOf(shown.frame, shown.state));
	const labels = $derived(
		pic ? EDGES.map((_, k) => (props.locals ? `× ${fmt3(pic.locals[k])}` : k === pic.edge ? `× ${fmt3(pic.local)}` : null)) : []
	);
	const caption = $derived(pic ? captionOf(pic) : null);

	let busy = $state(false);
	async function go(where: 'back' | 'next' | 'all' | 'reset' | 'check') {
		busy = true;
		try {
			await rt.ensure(source, 'last');
			const tp = rt.sources[source]?.tape;
			if (!tp || tp.length === 0) return;
			const cur = idx ?? at?.index ?? 0;
			const lastEdge = tp.find({ phase: 'edge', step: EDGES.length - 1 });
			if (where === 'reset') idx = 0;
			else if (where === 'back') idx = Math.max(0, Math.min(cur, lastEdge) - 1);
			else if (where === 'next') idx = Math.min(lastEdge, cur + 1);
			else if (where === 'all') idx = lastEdge;
			else idx = tp.find({ phase: 'check' });
		} finally {
			busy = false;
		}
	}
	const B = {
		reset: { en: 'Forward only', zh: '只做前向' },
		back: { en: 'Back one edge', zh: '退回一条边' },
		next: { en: 'Step one edge', zh: '走一条边' },
		all: { en: 'Run the whole backward pass', zh: '走完整个反向传播' },
		check: { en: 'Numerical check', zh: '数值检验' },
		controls: { en: 'Backward pass controls', zh: '反向传播控制' },
		fig: { en: 'The computation graph of L = (tanh(w·x + b) − y)²', zh: 'L = (tanh(w·x + b) − y)² 的计算图' },
		leaf: { en: 'leaf', zh: '叶子节点' },
		bp: { en: 'backpropagation', zh: '反向传播' },
		num: { en: 'numerical', zh: '数值估计' },
		diff: { en: 'difference', zh: '差' }
	};
</script>

{#if pic}
	<figure class="graph">
		{#if props.stepper}
			<div class="ctl" role="group" aria-label={t(B.controls)}>
				<button type="button" class="btn" disabled={busy} onclick={() => go('reset')}>{t(B.reset)}</button>
				<button type="button" class="btn" disabled={busy} onclick={() => go('back')}>{t(B.back)}</button>
				<button type="button" class="btn primary" disabled={busy} onclick={() => go('next')}>{t(B.next)}</button>
				<button type="button" class="btn" disabled={busy} onclick={() => go('all')}>{t(B.all)}</button>
				<button type="button" class="btn" disabled={busy} onclick={() => go('check')}>{t(B.check)}</button>
			</div>
		{/if}
		<Diagram def={NEURON} vals={pic.vals} grads={pic.grads} edges={pic.edges} {labels} badge={pic.acc ? EDGES[pic.edge].from : null} label={B.fig} />
		{#if caption && caption.en}<figcaption class="cap num" aria-live="polite">{t(caption)}</figcaption>{/if}
		{#if pic.check}
			<table class="chk">
				<thead><tr><th scope="col">{t(B.leaf)}</th><th scope="col">{t(B.bp)}</th><th scope="col">{t(B.num)}</th><th scope="col">{t(B.diff)}</th></tr></thead>
				<tbody>
					{#each pic.check as r (r.id)}
						<tr><th scope="row">{r.id}</th><td class="num">{fmt3(r.g)}</td><td class="num">{fmt3(r.n)}</td><td class="num">{fmt3(r.n - r.g)}</td></tr>
					{/each}
				</tbody>
			</table>
		{/if}
	</figure>
{/if}

<style>
	.graph {
		margin: 0;
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.ctl {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.ctl .btn {
		font-size: var(--fs-xs);
	}
	.cap {
		font-size: var(--fs-sm);
		color: var(--ink);
		overflow-wrap: anywhere;
	}
	.chk {
		border-collapse: collapse;
		font-size: var(--fs-sm);
		max-width: 520px;
	}
	.chk th,
	.chk td {
		padding: 2px 10px 2px 0;
		text-align: right;
		border-bottom: 1px solid var(--rule);
	}
	.chk th[scope='row'],
	.chk thead th:first-child {
		text-align: left;
	}
	.chk thead th {
		color: var(--ink-2);
		font-weight: 600;
	}
</style>
