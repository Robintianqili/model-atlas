<!-- The maze editor: pick a tool, then tap cells. Used by the Capture and, through the edit bus, by the stage. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { toRaw, type GameRaw } from '../../world.ts';

	let { raw, onedit }: { raw: GameRaw; onedit: (r: GameRaw) => void } = $props();
	const TOOLS = [['#', '▩', 'Wall', '墙'], ['G', '+1', 'Goal', '终点'], ['P', '−1', 'Pit', '陷阱'], ['S', 'S', 'Start', '起点'], ['.', '·', 'Erase', '擦除']];
	const NAME: Record<string, [string, string]> = { '#': ['wall', '墙'], G: ['goal', '终点'], P: ['pit', '陷阱'], S: ['start', '起点'], '.': ['open', '空地'] };
	let tool = $state('#');
	function paint(x: number, y: number) {
		const rows = raw.rows.map((r) => [...r].map((c) => (tool === 'S' && c === 'S' ? '.' : c)));
		rows[y][x] = tool;
		onedit(toRaw({ ...raw, rows: rows.map((r) => r.join('')) }));
	}
</script>

<div class="ed">
	<div class="tools" role="group" aria-label={t({ en: 'Maze tools', zh: '迷宫工具' })}>
		{#each TOOLS as [c, mark, en, zh] (c)}
			<button type="button" aria-pressed={tool === c} onclick={() => (tool = c)}><b class="m c{c === '#' ? 'w' : c}">{mark}</b> {t({ en, zh })}</button>
		{/each}
	</div>
	<div class="grid" style:grid-template-columns="repeat({raw.rows[0].length}, minmax(0, 2.3rem))">
		{#each raw.rows as row, y (y)}
			{#each [...row] as c, x (x)}
				<button type="button" class="cell c{c === '#' ? 'w' : c}" onclick={() => paint(x, y)} aria-label={t({ en: `column ${x}, row ${y}: ${NAME[c][0]}`, zh: `第 ${x} 列第 ${y} 行：${NAME[c][1]}` })}>{c === 'G' ? '+1' : c === 'P' ? '−1' : c === 'S' ? 'S' : ''}</button>
			{/each}
		{/each}
	</div>
</div>

<style>
	.ed { display: grid; gap: 8px; min-width: 0; } .tools { display: flex; flex-wrap: wrap; gap: 4px; }
	.tools button, .cell { border: 1px solid var(--rule-strong); border-radius: var(--r1); background: var(--surface); color: var(--ink); cursor: pointer; font-size: var(--fs-xs); }
	.tools button { padding: 4px 8px; min-height: 32px; }
	.tools button[aria-pressed='true'] { border-color: var(--accent); background: var(--accent-soft); font-weight: 700; }
	.grid { display: grid; gap: 2px; } .cell { aspect-ratio: 1; padding: 0; font-weight: 700; font-family: var(--font-mono); min-width: 0; }
	.cw { background: var(--ink-2); color: var(--surface); } .cG { background: var(--live-soft); color: var(--live); border-color: var(--live); }
	.cP { background: color-mix(in srgb, var(--div-neg) 22%, var(--surface)); color: var(--div-neg-text); } .cS { color: var(--accent-strong); }
	.m { display: inline-block; min-width: 1.2em; text-align: center; border-radius: 3px; }
</style>
