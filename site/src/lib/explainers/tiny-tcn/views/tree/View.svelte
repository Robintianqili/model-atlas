<!-- tiny-tcn 'tree': the receptive field of one output with its real numbers (meta.ts lists the modes). Node colors use
     the kit's diverging scale, one symmetric domain per row (rows hold different quantities); the arithmetic of the
     picked node is the kit's TermTable. -->
<script lang="ts">
	import { t as tr } from '$lib/i18n/i18n.svelte';
	import { makeScale } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import TermTable from '../../../../atlas/ui/views/kit.parts/TermTable.svelte';
	import Num from '../../../../atlas/ui/views/kit.parts/Num.svelte';
	import { DILATIONS, RF, dataOf } from '../../data.ts';
	import { CH, nodeTerms } from '../../model.ts';
	import { featureOf, headAt, paramsAt, plainOf, tStar, treeOffsets } from '../../look.ts';

	let P: StageProps = $props();
	const W = RF + 1, CW = 26, X0 = 4, Y0 = 30, RH = 44;
	const mode = $derived(String(P.props.mode ?? 'tree'));
	const d = $derived(dataOf(P.rt.input));
	const prm = $derived(paramsAt(P.at));
	const pl = $derived(prm ? plainOf(prm, d) : null);
	const t = $derived(tStar(d));
	const times = $derived(Array.from({ length: W }, (_, i) => t - W + 1 + i));
	const first = $derived(mode === 'causal' ? 0 : mode === 'residual' ? 3 : 4);
	let picks: Record<string, { L: number; time: number }> = $state({});
	let feats: Record<string, number> = $state({});
	const sel = $derived(picks[mode] ?? { L: first, time: t });
	const ch = $derived(feats[mode] ?? (pl && sel.L >= 0 && sel.L < 4 ? featureOf(pl, sel.L, sel.time) : pl ? featureOf(pl, 3, t) : 0));
	const xOf = (time: number) => X0 + (time - (t - W + 1)) * CW + CW / 2;
	const yOf = (L: number) => Y0 + (4 - L) * RH;
	const inTree = (L: number, time: number) => {
		if (mode === 'causal') return (L === 0 && time === t) || (L === -1 && t - time <= 1 && t - time >= 0);
		if (L === 4) return time === t;
		return treeOffsets(L).includes(t - time);
	};
	const val = (L: number, time: number) => (!pl ? 0 : L < 0 ? d.c[time] : L === 4 ? pl.out[time] : pl.h[L][ch * pl.T + time]);
	const scales = $derived([-1, 0, 1, 2, 3, 4].map((L) => makeScale('diverging', [0, Math.max(1e-9, ...times.map((x) => Math.abs(val(L, x))))], themeState.resolved)));
	const sc = (L: number) => scales[L + 1];
	const kids = (L: number, time: number): [number, number][] => (L === 4 ? [[3, time]] : L >= 0 ? [[L - 1, time - DILATIONS[L]], [L - 1, time]] : []);
	const H = (en: string, zh: string) => tr({ en, zh });
	const rowName = (L: number) => (L < 0 ? H('changes', '变化量') : L === 4 ? H('head', '输出头') : H(`layer ${L + 1}`, `第 ${L + 1} 层`));
	const nodeName = (L: number, time: number) => (L < 0 ? H(`change at time ${time}`, `时刻 ${time} 的变化量`) : L === 4 ? H(`output at time ${time}`, `时刻 ${time} 的输出`) : H(`layer ${L + 1}, time ${time}, feature ${ch}`, `第 ${L + 1} 层，时刻 ${time}，特征 ${ch}`));
	const node = $derived(pl && prm && sel.L >= 0 && sel.L < 4 ? nodeTerms(prm, pl, d.c.subarray(0, d.split), sel.L, sel.time, ch) : null);
	const head = $derived(prm ? headAt(prm, d) : null);
	const pick = (L: number, time: number) => (picks[mode] = { L, time });
</script>

<div class="tv" class:ans={mode === 'reach'}>
	{#if !pl}
		<p class="muted">{H('Waiting for the weights…', '正在等权重……')}</p>
	{:else}
		<div class="feat" role="group" aria-label={H('Feature shown in layers 1 to 4', '第 1 到第 4 层显示的特征')}>
			<span class="small">{H('feature', '特征')}</span>
			{#each Array.from({ length: CH }, (_, i) => i) as i (i)}<button type="button" class="fb" aria-pressed={i === ch} onclick={() => (feats[mode] = i)}>{i}</button>{/each}
		</div>
		<svg viewBox="0 0 {X0 + W * CW + 8} {Y0 + 5 * RH + 26}" role="group" aria-label={H(`Nodes of the network over the last ${W} time steps`, `最近 ${W} 个时间步上网络的节点`)}>
			{#each [4, 3, 2, 1, 0, -1] as L (L)}
				<text class="rl" x="2" y={yOf(L) - 12}>{rowName(L)}{L >= 0 && L < 4 ? ` · d ${DILATIONS[L]}` : ''}</text>
				{#each times as time (time)}<circle class="off" cx={xOf(time)} cy={yOf(L)} r="4" />{/each}
			{/each}
			<g class="tree">
				{#each [4, 3, 2, 1, 0] as L (L)}
					{#each times.filter((x) => inTree(L, x)) as time (time)}
						{#each kids(L, time).filter(([l2, x2]) => inTree(l2, x2)) as [l2, x2] (l2 + ':' + x2)}<line x1={xOf(time)} y1={yOf(L)} x2={xOf(x2)} y2={yOf(l2)} />{/each}
					{/each}
				{/each}
				{#each [4, 3, 2, 1, 0, -1] as L (L)}
					{#each times.filter((x) => inTree(L, x)) as time (time)}
						{@const v = val(L, time)}
						<g role="button" tabindex="0" class="nd" class:on={sel.L === L && sel.time === time} aria-label="{nodeName(L, time)}: {v.toPrecision(3)}" aria-pressed={sel.L === L && sel.time === time}
							onclick={() => pick(L, time)} onkeydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(L, time); } }}>
							<circle cx={xOf(time)} cy={yOf(L)} r="9" style:fill={sc(L).css(v)} />
						</g>
					{/each}
				{/each}
			</g>
			{#each [t - RF + 1, t] as time (time)}<text class="ax" x={xOf(time)} y={Y0 + 5 * RH + 18} text-anchor="middle">{time}</text>{/each}
		</svg>
		<p class="small muted">{H(`Time runs left to right; blue below 0, orange above (each row its own scale). Strong nodes and lines: the tree of the output at time ${t}.`, `时间从左往右；蓝色小于 0，橙色大于 0（每行各用自己的刻度）。深色的节点和连线：时刻 ${t} 那个输出的树。`)}</p>
		<div class="rd">
			<p class="hd">{nodeName(sel.L, sel.time)}</p>
			{#if sel.L < 0}
				<p class="small">c = y<sub>{sel.time}</sub> − y<sub>{sel.time - 1}</sub> = <Num value={d.y[sel.time]} /> − <Num value={d.y[sel.time - 1]} /> = <Num value={d.c[sel.time]} /></p>
			{:else if sel.L === 4 && head}
				<TermTable terms={head.terms.map((x) => ({ x: x.x, w: x.w, p: x.x * x.w }))} idx={(i) => H(`feature ${i}`, `特征 ${i}`)} head={{ i: H('layer 4', '第 4 层'), x: H('value', '值'), w: H('weight', '权重'), p: H('product', '乘积') }} sum={head.sum} caption={{ en: 'The head: features times weights', zh: '输出头：特征乘权重' }} />
				<p class="small">Σ + {H('bias', '偏置')} <Num value={head.bias} /> = {H('predicted change', '预测的变化量')} <Num value={head.change} />; {H('forecast', '预测')} <Num value={head.last} /> + {#if head.change < 0}(<Num value={head.change} />){:else}<Num value={head.change} />{/if} = <b><Num value={head.forecast} /></b> ({H('true value', '真实值')} <Num value={head.truth} />)</p>
			{:else if node}
				<TermTable terms={node.terms.map((x) => ({ x: x.x, w: x.w, p: x.x * x.w }))} idx={(i) => `${sel.L ? H(`f${node.terms[i].ch}`, `特征${node.terms[i].ch}`) : 'c'} @ ${node.terms[i].time}`} head={{ i: H('input', '输入'), x: H('value', '值'), w: H('weight', '权重'), p: H('product', '乘积') }} sum={node.pre - node.bias} caption={{ en: 'The convolution: inputs times weights', zh: '卷积：输入乘权重' }} />
				<p class="small">Σ + {H('bias', '偏置')} <Num value={node.bias} /> = <Num value={node.pre} />; ReLU → <Num value={node.a} />{#if sel.L > 0}; {H('skip', '跳连')} <Num value={node.skip} /> + <Num value={node.a} /> = <b><Num value={node.h} /></b>{/if}</p>
			{/if}
		</div>
	{/if}
</div>

<style>
	.tv { display: grid; gap: 6px; min-width: 0; font-size: var(--fs-sm); }
	.feat { display: flex; flex-wrap: wrap; align-items: center; gap: 3px; }
	.fb { min-width: 28px; padding: 2px 6px; border: 1px solid var(--rule); border-radius: 4px; background: var(--paper); color: var(--ink); font: 12px var(--font-mono); cursor: pointer; }
	.fb[aria-pressed='true'] { border-color: var(--accent-strong); background: var(--accent-soft); font-weight: 700; }
	svg { width: 100%; max-width: 620px; height: auto; overflow: visible; }
	.rl { fill: var(--muted); font-size: 13px; }
	.ax { fill: var(--muted); font-size: 13px; font-family: var(--font-mono); }
	.off { fill: none; stroke: var(--rule); stroke-dasharray: 2 2; }
	.tree line { stroke: var(--ink); stroke-width: 1.3; }
	.nd circle { stroke: var(--ink); stroke-width: 1; cursor: pointer; }
	.nd.on circle { stroke-width: 3; }
	.nd:focus-visible { outline: none; }
	.nd:focus-visible circle { stroke: var(--accent-strong); stroke-width: 3; }
	.rd { display: grid; gap: 4px; min-width: 0; }
	.hd { margin: 0; font-weight: 600; }
	.small { margin: 0; font-size: var(--fs-xs); }
	.muted { color: var(--muted); }
	/* step 4's predict: the tree and the numbers answer it, so they stay hidden while the kit's predict gate makes the stage inert */
	:global([inert]) .ans .tree, :global([inert]) .ans .rd { visibility: hidden; }
</style>
