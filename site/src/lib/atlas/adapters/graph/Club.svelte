<!--
  Club: a graph drawn twice, linked (K1, promoted from the gcn package): the node-link drawing (members at the offline
  layout positions of atlas/data/karate.ts) and, when `matrix` is given, the n × n table (A, A + I or Â) with the focus
  member's row outlined. Hovering a member lights its friendships in both. onnode(i) and oncell(i, j) make them clickable
  (focus a member, add or remove a friendship). Package views can use it: fill, ring, mark, edge and dim style members.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { I18n } from '$lib/i18n/types';
	import { N, POS, type GraphRaw } from '../../data/karate.ts';

	interface Props {
		g: GraphRaw;
		fill?: (i: number) => string | undefined;
		ring?: (i: number) => boolean;
		mark?: (i: number) => string;
		focus?: number;
		/** an edge's state: 'now' (thick), 'on' (highlighted) or '' */
		edge?: (a: number, b: number) => string;
		dim?: (i: number) => boolean;
		matrix?: { data: ArrayLike<number>; title: I18n } | null;
		onnode?: (i: number) => void;
		oncell?: (i: number, j: number) => void;
		label?: I18n;
	}
	let { g, fill, ring, mark, focus = -1, edge, dim, matrix = null, onnode, oncell, label }: Props = $props();
	let hover = $state(-1);
	const P = (i: number) => [6 + POS[i][0] * 110, 6 + POS[i][1] * 88];
	const lit = (a: number, b: number) => a === hover || b === hover;
	const max = $derived(matrix ? Math.max(1e-9, ...Array.from(matrix.data)) : 1);
	const cellFill = (v: number) => (v ? `color-mix(in srgb, var(--accent) ${Math.round(20 + 80 * Math.min(1, v / max))}%, var(--surface))` : 'var(--surface)');
	// only the non-zero cells over one background and one grid path; one handler finds the cell under the pointer
	let msvg = $state<SVGSVGElement>();
	const cells = $derived(matrix ? Array.from(matrix.data).flatMap((v, k) => (v ? [[Math.floor(k / N), k % N, v]] : [])) : []);
	const GRID = Array.from({ length: N + 1 }, (_, k) => `M0 ${k}H${N}M${k} 0V${N}`).join('');
	const cellAt = (e: MouseEvent): [number, number] | null =>
		((p) => (p.x >= 0 && p.x < N && p.y >= 0 && p.y < N ? [Math.floor(p.y), Math.floor(p.x)] : null))(new DOMPoint(e.clientX, e.clientY).matrixTransform(msvg?.getScreenCTM()?.inverse()));
</script>

<div class="gx" class:two={!!matrix}>
	<svg viewBox="0 0 100 100" role="img" aria-label={t(label ?? { en: 'The club: members and friendships', zh: '俱乐部：成员和朋友关系' })}>
		{#each g.edges as [a, b] (`${a}-${b}`)}
			{@const s = edge?.(a, b) ?? ''}
			<line x1={P(a)[0]} y1={P(a)[1]} x2={P(b)[0]} y2={P(b)[1]} class="e {s}" class:lit={lit(a, b)} />
		{/each}
		{#each POS as _, i (i)}
			<!-- svelte-ignore a11y_no_noninteractive_tabindex (a button only when onnode is given) -->
			<g
				transform="translate({P(i).join(',')})"
				class="n"
				class:dim={dim?.(i)}
				class:click={!!onnode}
				role={onnode ? 'button' : undefined}
				tabindex={onnode ? 0 : undefined}
				aria-label={onnode ? t({ en: `Member ${i}`, zh: `成员 ${i}` }) : undefined}
				onclick={() => onnode?.(i)}
				onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && onnode?.(i)}
				onpointerenter={() => (hover = i)}
				onpointerleave={() => (hover = -1)}
			>
				{#if ring?.(i)}<circle r="4.6" class="ring" />{/if}
				<circle r="3.2" style:fill={fill?.(i) ?? 'var(--surface)'} class:focus={i === focus} />
				<text y="1.1">{i}</text>
				{#if mark?.(i)}<text y="-4.4" class="mk">{mark(i)}</text>{/if}
			</g>
		{/each}
	</svg>
	{#if matrix}
		<figure>
			<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_noninteractive_element_interactions -->
			<svg
				bind:this={msvg}
				viewBox="-2.2 -1.9 {N + 2.7} {N + 2.4}"
				role="img"
				aria-label={t(matrix.title)}
				class:click={!!oncell}
				onpointerleave={() => (hover = -1)}
				onpointermove={(e) => (hover = cellAt(e)?.[0] ?? -1)}
				onclick={(e) => ((c) => c && c[0] !== c[1] && oncell?.(c[0], c[1]))(cellAt(e))}
			>
				<rect width={N} height={N} class="bg" />
				{#each cells as [i, j, v] (i * N + j)}<rect x={j} y={i} width="1" height="1" style:fill={cellFill(v)} class="c" />{/each}
				<path d={GRID} class="grid" />
				{#if hover >= 0}<rect y={hover} width={N} height="1" class="hl" /><rect x={hover} width="1" height={N} class="hl" />{/if}
				{#if focus >= 0}<rect x="0" y={focus} width={N} height="1" class="row" />{/if}
				{#each [0, 10, 20, 30] as k (k)}
					<text x={k + 0.5} y="-0.7" class="tk">{k}</text>
					{#if k !== focus}<text x="-0.5" y={k + 0.85} class="tk end">{k}</text>{/if}
				{/each}
				{#if focus >= 0}<text x="-0.5" y={focus + 0.85} class="tk end fr">{focus}</text>{/if}
			</svg>
			<figcaption class="muted">{t(matrix.title)}</figcaption>
		</figure>
	{/if}
</div>

<style>
	.gx {
		display: grid;
		gap: 10px;
		grid-template-columns: minmax(0, 1fr);
		min-width: 0;
	}
	.gx.two {
		grid-template-columns: repeat(auto-fit, minmax(min(100%, 240px), 1fr));
	}
	svg {
		width: 100%;
		height: auto;
		display: block;
		max-height: 420px;
	}
	figure {
		margin: 0;
		display: grid;
		gap: 4px;
	}
	figcaption {
		font-size: var(--fs-xs);
	}
	.e {
		stroke: var(--rule-strong);
		stroke-width: 0.35;
	}
	.e.on {
		stroke: var(--accent);
		stroke-width: 0.8;
	}
	.e.now {
		stroke: var(--accent-strong);
		stroke-width: 1.6;
	}
	.e.lit {
		stroke: var(--ink);
		stroke-width: 0.8;
	}
	.n circle {
		stroke: var(--ink-2);
		stroke-width: 0.35;
	}
	.n circle.focus {
		stroke: var(--ink);
		stroke-width: 1.1;
	}
	.n text {
		font: 600 2.6px var(--font-mono);
		text-anchor: middle;
		fill: var(--ink);
		pointer-events: none;
	}
	.n .mk {
		font-size: 3.4px;
		fill: var(--ink);
	}
	.n.dim {
		opacity: 0.3;
	}
	.click {
		cursor: pointer;
	}
	.n .ring {
		fill: none;
		stroke: var(--ink);
		stroke-width: 0.7;
		stroke-dasharray: 1.2 0.7;
	}
	.bg {
		fill: var(--surface);
	}
	.grid {
		fill: none;
		stroke: var(--rule);
		stroke-width: 0.04;
		pointer-events: none;
	}
	.hl {
		fill: none;
		stroke: var(--ink);
		stroke-width: 0.12;
		pointer-events: none;
	}
	.row {
		fill: none;
		stroke: var(--ink);
		stroke-width: 0.25;
		pointer-events: none;
	}
	.tk {
		font: 1.1px var(--font-mono);
		fill: var(--muted);
		text-anchor: middle;
	}
	.tk.end {
		text-anchor: end;
	}
	.tk.fr {
		fill: var(--ink);
		font-weight: 700;
	}
</style>
