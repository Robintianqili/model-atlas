<!-- snake-dqn: one snake board (SVG). The body fades from the head towards the tail (as the network's body plane does),
     the head is the accent, the food a dot. With q: the three moves from the head (straight on, left, right) as arrows,
     each labelled with its Q-value; `pick` is ringed. A move off the board ends in the margin around the grid. -->
<script lang="ts">
	import { fmt3 } from '../../../atlas/core/format.ts';
	import { type Game, SIZE, nextCell } from '../rules.ts';

	interface Props { g: Game; q?: number[] | null; pick?: number; label: string; small?: boolean }
	let { g, q = null, pick = -1, label, small = false }: Props = $props();
	const n = $derived(g.snake.length);
	const xy = (c: number) => [(c % SIZE) * 10 + 5, Math.floor(c / SIZE) * 10 + 5];
	const head = $derived(xy(g.snake[0]));
	const arrows = $derived(
		(q ?? []).map((v, a) => {
			const { dir } = nextCell(g, a), dx = [0, 1, 0, -1][dir], dy = [-1, 0, 1, 0][dir];
			return { a, v, x: head[0] + dx * 10, y: head[1] + dy * 10, dx, dy };
		})
	);
</script>

<svg class="board" class:small viewBox="-15 -10 120 110" role="img" aria-label={label}>
	<rect x="0" y="0" width="90" height="90" class="bg" />
	{#each Array.from({ length: SIZE - 1 }, (_, i) => (i + 1) * 10) as v (v)}<line x1={v} y1="0" x2={v} y2="90" class="grid" /><line x1="0" y1={v} x2="90" y2={v} class="grid" />{/each}
	{#if g.food >= 0}<circle cx={xy(g.food)[0]} cy={xy(g.food)[1]} r="3.4" class="food" />{/if}
	{#each g.snake.slice(1) as c, i (c)}<rect x={xy(c)[0] - 4.2} y={xy(c)[1] - 4.2} width="8.4" height="8.4" rx="1.6" class="body" style:opacity={0.3 + (0.7 * (n - i - 1)) / n} />{/each}
	<rect x={head[0] - 4.6} y={head[1] - 4.6} width="9.2" height="9.2" rx="2.4" class="head" class:dead={!!g.over} />
	{#each arrows as ar (ar.a)}
		<line x1={head[0] + ar.dx * 4.6} y1={head[1] + ar.dy * 4.6} x2={ar.x - ar.dx * 2.6} y2={ar.y - ar.dy * 2.6} class="arrow" class:pick={ar.a === pick} />
		<text x={ar.x} y={ar.y + 1.3} class="q" class:pick={ar.a === pick}>{fmt3(ar.v)}</text>
	{/each}
</svg>

<style>
	.board { width: min(100%, 340px); height: auto; display: block; }
	.board.small { width: min(100%, 220px); }
	.bg { fill: var(--surface-2); stroke: var(--rule-strong); stroke-width: 0.6; }
	.grid { stroke: var(--rule); stroke-width: 0.25; }
	.food { fill: var(--cat-3); }
	.body { fill: var(--ink-2); }
	.head { fill: var(--accent); }
	.head.dead { fill: var(--div-neg); }
	.arrow { stroke: var(--ink-2); stroke-width: 0.8; }
	.arrow.pick { stroke: var(--accent); stroke-width: 1.6; }
	.q { font-family: var(--font-mono); font-size: 3.8px; text-anchor: middle; fill: var(--ink); paint-order: stroke; stroke: var(--surface); stroke-width: 1.2px; }
	.q.pick { font-weight: 700; fill: var(--accent-strong); }
</style>
