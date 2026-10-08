<!--
  gnn-variants: one member and its senders, drawn around it (an ego graph). Each edge (and the self-loop) carries a
  label, usually its weight in one rule; `width` (0..1) thickens it. Friends keep their faction colour.
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { I18n } from '$lib/i18n/types';
	import { LABELS } from '../graph.ts';

	interface Props {
		i: number;
		nb: number[];
		label: (j: number) => string;
		width?: (j: number) => number;
		hot?: (j: number) => boolean;
		cut?: (j: number) => boolean;
		mark?: (j: number) => string;
		title: I18n;
		onnode?: (j: number) => void;
	}
	let { i, nb, label, width = () => 0.3, hot, cut, mark, title, onnode }: Props = $props();
	const n = $derived(Math.max(1, nb.length));
	const ang = (k: number) => -Math.PI / 2 + (2 * Math.PI * k) / n;
	const at = (a: number, r: number) => [50 + r * Math.cos(a), 51 + r * Math.sin(a)];
	const self = $derived(nb.length ? -Math.PI / 2 - Math.PI / n : -Math.PI / 2); // between the last and the first friend
	const fac = (j: number) => `color-mix(in srgb, var(${LABELS[j] ? '--div-pos' : '--div-neg'}) 45%, var(--surface))`;
	const sw = (j: number) => 0.35 + 2.4 * Math.max(0, Math.min(1, width(j)));
</script>

<!-- with clickable members the picture is a group of buttons, not an image (an image's children are not interactive) -->
<svg viewBox="0 0 100 102" role={onnode ? 'group' : 'img'} aria-label={t(title)}>
	<circle cx={at(self, 7)[0]} cy={at(self, 7)[1]} r="5" class="e loop" class:hot={hot?.(i)} style:stroke-width={sw(i)} />
	{#each nb as j, k (j)}
		<line x1="50" y1="51" x2={at(ang(k), 37)[0]} y2={at(ang(k), 37)[1]} class="e" class:hot={hot?.(j)} class:cut={cut?.(j)} style:stroke-width={sw(j)} />
	{/each}
	<text x={at(self, 16)[0]} y={at(self, 16)[1] + 1.2} class="w">{label(i)}</text>
	{#each nb as j, k (j)}
		{@const p = at(ang(k), nb.length > 7 && k % 2 ? 26 : 19)}
		<text x={p[0]} y={p[1] + 1.2} class="w">{label(j)}</text>
	{/each}
	{#each [i, ...nb] as j, k (j)}
		{@const p = k ? at(ang(k - 1), 37) : [50, 51]}
		<!-- svelte-ignore a11y_no_noninteractive_tabindex (a button only when onnode is given) -->
		<g transform="translate({p[0]},{p[1]})" class="n" class:me={!k} role={onnode && k ? 'button' : undefined} tabindex={onnode && k ? 0 : undefined}
			aria-label={onnode && k ? t({ en: `Follow member ${j}`, zh: `改为关注成员 ${j}` }) : undefined}
			onclick={() => k && onnode?.(j)} onkeydown={(e) => k && (e.key === 'Enter' || e.key === ' ') && onnode?.(j)}>
			<circle r={k ? 4.4 : 5.6} style:fill={k ? fac(j) : 'var(--accent-soft)'} />
			<text y="1.3">{j}</text>
			{#if mark?.(j)}<text y="-5.6" class="mk">{mark(j)}</text>{/if}
		</g>
	{/each}
</svg>

<style>
	svg { width: 100%; height: auto; display: block; max-height: 380px; }
	.e { stroke: var(--rule-strong); fill: none; } .e.hot { stroke: var(--accent-strong); } .e.cut { stroke-dasharray: 1.5 1.2; opacity: 0.5; }
	.w { font: 600 3.3px var(--font-mono); text-anchor: middle; fill: var(--ink); paint-order: stroke; stroke: var(--surface); stroke-width: 1.2px; }
	.n circle { stroke: var(--ink-2); stroke-width: 0.35; } .n.me circle { stroke: var(--ink); stroke-width: 0.9; }
	.n text { font: 600 3px var(--font-mono); text-anchor: middle; fill: var(--ink); pointer-events: none; }
	.n .mk { font-size: 4px; } .n[role='button'] { cursor: pointer; }
</style>
