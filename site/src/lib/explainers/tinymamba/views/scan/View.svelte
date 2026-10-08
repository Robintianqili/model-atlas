<!-- tinymamba 'scan' (meta.ts): one column per token (chips on top; arcs from each letter to the ? that asks for it
     with the 'want' row), rows of numbers under them; the cursor column outlined; with props.full the whole state
     (channels × slots) at the cursor. Hover or focus a cell for its value. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import { makeScale } from '$lib/kit/color';
	import { fmt3 } from '$lib/kit/format';
	import { themeState } from '$lib/kit/theme.svelte';
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { scanOf } from './meta.ts';

	let P: StageProps = $props();
	const S = $derived(scanOf(P));
	let cw = $state(640);
	const LAB = 30;
	const W = $derived(Math.max(18, Math.min(36, Math.floor((cw - LAB - 4) / Math.max(1, S.toks.length)))));
	const H = (k: string) => (k === 'delta' ? 10 : k === 'got' ? 16 : 18);
	const arcs = $derived(S.grids.some((g) => g.key === 'want') ? S.links : []);
	const depth = (j: number) => 10 + 9 * Math.min(6, j);
	const top0 = $derived(30 + (arcs.length ? depth(arcs.length) : 0));
	const layout = $derived.by(() => {
		let y = top0;
		return S.grids.map((g) => {
			const top = y + (g.kind === 'text' ? 4 : 16);
			y = top + g.rows * H(g.key) + 8;
			const vals = g.cells.flat().filter(Number.isFinite);
			const m = Math.max(1e-6, ...vals.map(Math.abs));
			const scale = g.kind === 'text' ? null : makeScale(g.kind, g.kind === 'sequential' ? [0, m] : [-m, m], themeState.resolved);
			return { g, top, scale, h: H(g.key) };
		});
	});
	const height = $derived(layout.length ? layout[layout.length - 1].top + layout[layout.length - 1].g.rows * layout[layout.length - 1].h + 4 : top0);
	const x = (i: number) => LAB + i * W;
	const fullScale = $derived(S.full ? makeScale('diverging', [-1, Math.max(1e-6, ...S.full.flat().map(Math.abs))], themeState.resolved) : null);
	let box: HTMLDivElement | undefined = $state();
	$effect(() => {
		const c = S.cur;
		if (box && c !== null && box.scrollWidth > box.clientWidth) box.scrollLeft = Math.max(0, x(c) - box.clientWidth / 2);
	});
	let hover = $state('');
</script>

<div class="sc" bind:clientWidth={cw}>
	<div class="row">
		<div class="box" bind:this={box}>
			<svg width={LAB + S.toks.length * W + 4} {height} role="img" aria-label={t({ en: 'Your row, token by token', zh: '你的这一行，一个词元接一个词元' })} onpointerleave={() => (hover = '')}>
				{#each S.toks as c, i (i)}
					<rect x={x(i) + 1} y={4} width={W - 2} height={22} rx={4} class="chip" class:let={S.letters[i]} class:ask={c === '?'} />
					<text x={x(i) + W / 2} y={20} class="tok" text-anchor="middle">{c}</text>
				{/each}
				{#each arcs as a, j (j)}
					<path d={`M ${x(a.from) + W / 2} 27 C ${x(a.from) + W / 2} ${27 + depth(arcs.length - j)} ${x(a.to) + W / 2} ${27 + depth(arcs.length - j)} ${x(a.to) + W / 2} 27`} class="arc" style:stroke={`var(--cat-${j % 10})`} />
				{/each}
				{#each layout as L (L.g.key)}
					{#if L.g.kind === 'text'}<text x={2} y={L.top + L.h - 4} class="lab">{t(L.g.label)}</text>{:else}<text x={2} y={L.top - 5} class="lab">{t(L.g.label)}</text>{/if}
					{#each L.g.cells as r, k (k)}
						<text x={2} y={L.top + k * L.h + L.h - 4} class="rn">{L.g.rowNames[k]}</text>
						{#each r as v, i (i)}
							{#if L.g.kind === 'text'}
								<text x={x(i) + W / 2} y={L.top + k * L.h + L.h - 4} text-anchor="middle" class="txt {L.g.tone?.[i] ?? ''}" class:small={k === 1} style:fill={L.g.key === 'want' && S.wantIdx[i] >= 0 ? `var(--cat-${S.wantIdx[i] % 10})` : undefined}>{W < 30 && k === 1 ? (L.g.text?.[k][i] ?? '').replace(/\d+%/, '') : L.g.text?.[k][i]}</text>
							{:else}
								<rect x={x(i) + 1} y={L.top + k * L.h} width={W - 2} height={L.h - 1} fill={L.scale?.css(v)} class:todo={i > S.upto && (L.g.key === 'h' || L.g.key === 'dch')} role="presentation" onpointerenter={() => (hover = `${t(L.g.label)} · ${t({ en: 'token', zh: '词元' })} ${i} “${S.toks[i]}”${L.g.rowNames[k] ? ` · ${L.g.rowNames[k]}` : ''}: ${fmt3(v)}`)}><title>{fmt3(v)}</title></rect>
							{/if}
						{/each}
					{/each}
					{#each S.marks.filter((m) => m.key === L.g.key) as m (m.row * 1000 + m.col)}<rect x={x(m.col)} y={L.top + m.row * L.h - 1} width={W} height={L.h + 1} class="mark" />{/each}
					{#if S.cur !== null}<rect x={x(S.cur)} y={L.top - 1} width={W} height={L.g.rows * L.h + 1} class="cur" />{/if}
				{/each}
				{#if S.cur !== null}<rect x={x(S.cur)} y={2} width={W} height={26} class="cur" />{/if}
			</svg>
		</div>
		{#if S.full && fullScale}
			<div class="full">
				<span class="lab2">{t(S.fullAt < 0 ? { en: `The whole state of block ${S.b} before the first token: ${S.D} channels × ${S.N} slots, all 0`, zh: `读第一个词元之前第 ${S.b} 块的全部状态：${S.D} 个通道 × ${S.N} 个槽，全是 0` } : { en: `The whole state of block ${S.b} after token ${S.fullAt}: ${S.D} channels × ${S.N} slots`, zh: `第 ${S.fullAt} 个词元之后第 ${S.b} 块的全部状态：${S.D} 个通道 × ${S.N} 个槽` })}</span>
				<svg width={S.N * 18 + 2} height={S.D * 11 + 2} role="img" aria-label={t({ en: 'the whole state', zh: '全部状态' })}>
					{#each S.full as r, c (c)}
						{#each r as v, s (s)}<rect x={1 + s * 18} y={1 + c * 11} width={17} height={10} fill={fullScale.css(v)} class:mine={c === S.ch}><title>{`c${c} n${s}: ${fmt3(v)}`}</title></rect>{/each}
					{/each}
				</svg>
			</div>
		{/if}
	</div>
	{#if S.bars}<ProbBars items={S.bars.items} k={4} label={{ en: `Chances at the first ? that asks for a letter (token ${S.bars.at})`, zh: `第一个要字母的 ? 处（第 ${S.bars.at} 个词元）的概率` }} />{/if}
	{#if S.tries.length}<div class="tries">{#each S.tries as r (r.text)}<button type="button" class="btn" onclick={() => (P.setInput ? P.setInput(r.text) : P.rt.setInput(r.text))}>{t(r.label)} <code>{r.text}</code></button>{/each}</div>{/if}
	<p class="small">{hover || (S.caption ? t(S.caption) : t(S.hint))}</p>
</div>

<style>
	.sc { display: grid; grid-template-columns: minmax(0, 1fr); gap: 8px; min-width: 0; }
	.row { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: flex-start; min-width: 0; }
	.box { overflow-x: auto; min-width: 0; max-width: 100%; }
	svg { display: block; }
	.chip { fill: var(--surface-2); }
	.chip.let { fill: var(--accent-soft); stroke: var(--accent); }
	.chip.ask { fill: var(--surface-3); stroke: var(--ink-2); }
	.tok { font: 600 14px var(--font-mono, monospace); fill: var(--ink); }
	.arc { fill: none; stroke: var(--accent); stroke-width: 1.5; opacity: 0.8; }
	.lab { font-size: 12px; fill: var(--ink-2); }
	.rn { font: 10px var(--font-mono, monospace); fill: var(--muted); }
	.txt { font: 600 14px var(--font-mono, monospace); fill: var(--ink); }
	.txt.small { font-size: 10px; font-weight: 400; }
	.txt.ok { fill: var(--real); }
	.txt.bad { fill: var(--div-pos); }
	.cur { fill: none; stroke: var(--ink); stroke-width: 1.5; }
	.todo { opacity: 0.22; }
	.mark { fill: none; stroke: var(--accent); stroke-width: 2.5; }
	.tries { display: flex; flex-wrap: wrap; gap: 6px; }
	.btn { min-height: 32px; padding: 2px 10px; font-size: var(--fs-xs); }
	.mine { stroke: var(--ink); stroke-width: 1.2; }
	.full { display: grid; gap: 4px; justify-items: start; max-width: 260px; }
	.lab2, .small { margin: 0; font-size: var(--fs-sm); color: var(--ink-2); }
</style>
