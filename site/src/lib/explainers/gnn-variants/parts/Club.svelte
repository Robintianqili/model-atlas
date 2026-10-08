<!--
  gnn-variants: the whole club at its offline layout. The member in focus and its friendships are highlighted;
  onnode(i) makes members clickable (follow another member).
-->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { LABELS, N, neighbours, POS, type GraphRaw } from '../graph.ts';

	let { g, onnode }: { g: GraphRaw; onnode?: (i: number) => void } = $props();
	const P = (i: number) => [5 + POS[i][0] * 90, 5 + POS[i][1] * 90];
	const nb = $derived(neighbours(g, g.focus));
	const near = (i: number) => i === g.focus || nb.includes(i);
	const fac = (i: number) => `color-mix(in srgb, var(${LABELS[i] ? '--div-pos' : '--div-neg'}) ${near(i) ? 60 : 22}%, var(--surface))`;
</script>

<!-- with clickable members the picture is a group of buttons, not an image (an image's children are not interactive) -->
<svg viewBox="0 0 100 100" role={onnode ? 'group' : 'img'} aria-label={t({ en: `The club: ${N} members, ${g.edges.length} friendships; member ${g.focus} and its ${nb.length} friends are highlighted`, zh: `俱乐部：${N} 个成员，${g.edges.length} 段朋友关系；成员 ${g.focus} 和他的 ${nb.length} 个朋友被突出显示` })}>
	{#each g.edges as [a, b] (`${a}-${b}`)}
		<line x1={P(a)[0]} y1={P(a)[1]} x2={P(b)[0]} y2={P(b)[1]} class="e" class:on={a === g.focus || b === g.focus} />
	{/each}
	{#each POS as _, i (i)}
		<!-- svelte-ignore a11y_no_noninteractive_tabindex (a button only when onnode is given) -->
		<g transform="translate({P(i).join(',')})" class="n" class:me={i === g.focus} class:dim={!near(i)} role={onnode ? 'button' : undefined} tabindex={onnode ? 0 : undefined}
			aria-label={onnode ? t({ en: `Follow member ${i}`, zh: `关注成员 ${i}` }) : undefined}
			onclick={() => onnode?.(i)} onkeydown={(e) => (e.key === 'Enter' || e.key === ' ') && onnode?.(i)}>
			<circle r="3.1" style:fill={fac(i)} />
			<text y="1.05">{i}</text>
		</g>
	{/each}
</svg>

<style>
	svg { width: 100%; height: auto; display: block; max-height: 400px; }
	.e { stroke: var(--rule-strong); stroke-width: 0.3; } .e.on { stroke: var(--accent-strong); stroke-width: 1.1; }
	.n circle { stroke: var(--ink-2); stroke-width: 0.3; } .n.me circle { stroke: var(--ink); stroke-width: 1; }
	.n.dim { opacity: 0.55; } .n[role='button'] { cursor: pointer; }
	.n text { font: 600 2.5px var(--font-mono); text-anchor: middle; fill: var(--ink); pointer-events: none; }
</style>
