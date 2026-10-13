<!-- behavior-cloning: record demonstrations. Tap an open cell to start a demonstration there, then make moves with the
     arrow buttons (or the arrow keys while a cell has focus); it is added when it reaches the goal or a pit, or with
     "Add it". Used by the Capture and by the stage's board (through StageProps.setInput). -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { ARROWS, DIR_NAMES, GOAL, isEnd, MAX_DEMOS, MAX_MOVES, OPEN, PIT, step, toRaw, worldOf, type BcRaw } from '../../world.ts';

	let { raw, onedit }: { raw: BcRaw; onedit: (r: BcRaw) => void } = $props();
	const W = $derived(worldOf(raw.map));
	let rec = $state<{ start: number; moves: number[] } | null>(null);
	const cells = $derived.by(() => {
		if (!rec) return [];
		const out = [rec.start];
		for (const a of rec.moves) out.push(step(W, out[out.length - 1], a));
		return out;
	});
	const cur = $derived(cells.length ? cells[cells.length - 1] : -1);
	function commit() {
		if (rec && rec.moves.length) onedit(toRaw({ ...raw, demos: [...raw.demos, rec] }));
		rec = null;
	}
	function move(a: number) {
		if (!rec || isEnd(W, cur) || rec.moves.length >= MAX_MOVES) return;
		const next = step(W, cur, a);
		rec = { start: rec.start, moves: [...rec.moves, a] };
		if (isEnd(W, next) || rec.moves.length >= MAX_MOVES) commit();
	}
	function key(e: KeyboardEvent) {
		const a = ['ArrowUp', 'ArrowRight', 'ArrowDown', 'ArrowLeft'].indexOf(e.key);
		if (a >= 0 && rec) (e.preventDefault(), move(a));
	}
	const NAME = [{ en: 'open', zh: '空地' }, { en: 'wall', zh: '墙' }, { en: 'goal', zh: '终点' }, { en: 'pit', zh: '陷阱' }];
	const status = $derived(
		rec
			? { en: `Recording from column ${rec.start % W.w}, row ${Math.floor(rec.start / W.w)}: ${rec.moves.length} moves so far.`, zh: `正在从第 ${rec.start % W.w} 列第 ${Math.floor(rec.start / W.w)} 行录制：已经走了 ${rec.moves.length} 步。` }
			: raw.demos.length >= MAX_DEMOS
				? { en: 'That is the most demonstrations this page keeps.', zh: '本页最多保留这么多条示范。' }
				: { en: 'Tap an open cell to start a demonstration there.', zh: '点一个空地格子，从那里开始一条示范。' }
	);
</script>

<div class="pad">
	<div class="grid" role="group" aria-label={t({ en: 'Demonstration grid', zh: '示范网格' })} style:grid-template-columns="repeat({W.w}, minmax(0, 2.1rem))">
		{#each W.cell as c, s (s)}
			<button type="button" class="cell c{c}" class:on={cells.includes(s)} disabled={c !== OPEN || (rec !== null && rec.moves.length > 0) || raw.demos.length >= MAX_DEMOS}
				onclick={() => (rec = { start: s, moves: [] })} onkeydown={key}
				aria-label={t({ en: `column ${s % W.w}, row ${Math.floor(s / W.w)}: ${NAME[c].en}`, zh: `第 ${s % W.w} 列第 ${Math.floor(s / W.w)} 行：${NAME[c].zh}` })}>{c === GOAL ? 'G' : c === PIT ? '×' : s === cur ? '●' : ''}</button>
		{/each}
	</div>
	<div class="row" role="group" aria-label={t({ en: 'Moves', zh: '动作' })}>
		{#each ARROWS as ar, a (a)}<button type="button" class="btn" disabled={!rec || isEnd(W, cur)} onclick={() => move(a)} onkeydown={key} aria-label={t(DIR_NAMES[a])}>{ar}</button>{/each}
		<button type="button" class="btn" disabled={!rec?.moves.length} onclick={commit}>{t({ en: 'Add it', zh: '加入' })}</button>
		<button type="button" class="btn ghost" disabled={!rec} onclick={() => (rec = null)}>{t({ en: 'Cancel', zh: '取消' })}</button>
	</div>
	<p class="muted small" role="status">{t(status)}</p>
</div>

<style>
	.pad { display: grid; gap: 6px; min-width: 0; }
	.grid { display: grid; gap: 2px; }
	.cell { aspect-ratio: 1; padding: 0; min-width: 0; border: 1px solid var(--rule-strong); border-radius: var(--r1); background: var(--surface); color: var(--ink); font: 700 var(--fs-xs) var(--font-mono); cursor: pointer; }
	.cell:disabled { cursor: default; }
	.c2 { background: var(--live-soft); border-color: var(--live); }
	.c3 { background: color-mix(in srgb, var(--div-neg) 22%, var(--surface)); color: var(--div-neg-text); }
	.c1 { background: var(--ink-2); }
	.cell.on { background: var(--accent-soft); border-color: var(--accent); color: var(--ink); }
	.row { display: flex; flex-wrap: wrap; gap: 4px; }
	.btn { min-height: 32px; min-width: 36px; padding: 2px 8px; font-size: var(--fs-xs); }
	.small { margin: 0; font-size: var(--fs-xs); }
</style>
