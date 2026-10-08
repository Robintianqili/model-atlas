<!-- gnn-variants 'agg' view: the member in focus and how each graph layer combines its friends (modes in meta.ts).
     A click on a member follows it (setInput): every source re-runs on the new focus. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { FACTION, LABELS, N, NAMES, VARIANTS, type Variant } from '../../graph.ts';
	import { focusOf, gcnRule, ginRule, isRight, rightOf, rowAfter, sageRule, sameStart, VARIANT_TEXT, weightOf } from '../../rules.ts';
	import Club from '../../parts/Club.svelte';
	import Ego from '../../parts/Ego.svelte';

	let P: StageProps = $props();
	const mode = $derived(String(P.props.mode ?? 'club'));
	const F = $derived(focusOf(P.rt));
	const T = $derived(P.rt.sources.fwd?.trace?.t);
	const go = (j: number) => (P.setInput ?? P.rt.setInput)({ ...F.g, focus: j });
	const m = $derived((P.at.frame?.metrics ?? {}) as Record<string, number>);
	const ph = $derived(P.at.frame?.phase ?? '');
	const L = (en: string, zh: string) => t({ en, zh });
	const f = (v: number | null | undefined) => (v === null || v === undefined ? '' : Number.isInteger(v) ? String(v) : fmt3(v));
	const v4 = (p: string) => [0, 1, 2, 3].map((k) => m[`${p}${k}`]);
	const W = (v: Variant) => (j: number) => weightOf(P.rt, v, j);
	const wmax = (v: Variant) => Math.max(1e-9, ...F.S.map((j) => W(v)(j) ?? 0));
	const egoLabel = (v: Variant) => (j: number) => (v === 'sage' && j === F.i ? L('own', '自己') : f(W(v)(j)));
	const fac = (c: number, pct = 55) => `color-mix(in srgb, var(${c ? '--div-pos' : '--div-neg'}) ${pct}%, var(--surface))`;
	/** gat frames: the score of every sender scored so far (score phase), else α */
	const frames = $derived(P.at.tape && mode === 'gat' ? Array.from({ length: P.at.index + 1 }, (_, k) => P.at.tape!.frame(k)) : []);
	const scores = $derived(new Map(frames.filter((x) => x.phase === 'score').map((x) => [x.metrics!.j, x.metrics!] as const)));
	const alphaOf = (j: number) => ((P.at.state as { w?: { data: ArrayLike<number> } } | undefined)?.w?.data[j] ?? NaN);
	const gatLabel = (j: number) => (ph === 'score' ? (scores.has(j) ? f(scores.get(j)!.e) : '') : f(alphaOf(j)));
	const sent = $derived(frames.filter((x) => x.phase === 'message').map((x) => x.metrics!));
	/** drop frames: the friends removed so far */
	const drops = $derived(P.at.tape && mode === 'drop' ? Array.from({ length: P.at.index + 1 }, (_, k) => P.at.tape!.frame(k).metrics!).filter((x) => x.d !== undefined) : []);
	const dState = $derived((P.at.state ?? {}) as { alpha?: { data: ArrayLike<number> } });
	const amax = $derived(Math.max(1e-9, ...F.nb.map((j) => dState.alpha?.data[j] ?? 0)));
	const same = $derived(sameStart(P.rt));
	/** embed: the head's dividing line of one network, in the plot's coordinates */
	const line = (v: Variant) => {
		const [Wh, b] = [P.rt.model?.weights[`${v}.head.weight`]?.data, P.rt.model?.weights[`${v}.head.bias`]?.data];
		if (!Wh || !b) return null;
		const [u, w, c] = [Wh[2] - Wh[0], Wh[3] - Wh[1], b[1] - b[0]];
		return (Math.abs(w) > Math.abs(u) ? [-1.2, 1.2].map((x) => [x, -(u * x + c) / w]) : [-1.2, 1.2].map((y) => [-(w * y + c) / u, y])).map(([x, y]) => [50 + x * 42, 50 - y * 42]);
	};
	const fact = (k: string) => P.rt.num(`fact:${k}`);
</script>

{#snippet table(rows: [string, (number | null)[]][], head: string[] = [])}
	<table class="num">
		{#if head.length}<thead><tr><th></th>{#each head as h (h)}<th colspan={head.length === 1 ? 4 : 1}>{h}</th>{/each}</tr></thead>{/if}
		<tbody>{#each rows as [lab, vals], r (r)}<tr class:sum={/^(Σ|=|tanh)/.test(lab)}><th>{lab}</th>{#each vals as v, k (k)}<td>{v === null ? '' : fmt3(v)}</td>{/each}</tr>{/each}</tbody>
	</table>
{/snippet}

<div class="av">
	{#if !T}<p class="muted">{L('Computing…', '计算中…')}</p>
	{:else if mode === 'club'}
		<Club g={F.g} onnode={go} />
		<p class="num small">{L(`Member ${F.i}: ${F.nb.length} friends (${F.nb.join(', ') || '–'}).`, `成员 ${F.i}：${F.nb.length} 个朋友（${F.nb.join('、') || '–'}）。`)}</p>
		<p class="legend small">{#each FACTION as fa, c (c)}<span><i style:background={fac(c)}></i>{L(`joined ${fa.en}`, `加入了${/^[A-Za-z]/.test(fa.zh ?? '') ? ' ' : ''}${fa.zh}一方`)}</span>{/each}</p>
	{:else if mode === 'gcn' || mode === 'sage' || mode === 'gin'}
		{@const v = mode as Variant}
		<Ego i={F.i} nb={F.nb} label={egoLabel(v)} width={(j) => (W(v)(j) ?? 0) / wmax(v)} title={VARIANT_TEXT[v]} onnode={go} />
		{#if v === 'gcn'}
			{@const R = gcnRule(P.rt)}
			{@render table([...F.S.map((j, k): [string, number[]] => [`${j} · ${f(W(v)(j))} ×`, R.weighted[k]]), ['Σ', R.sum], ['+ b', R.b], ['= z', R.z], ['tanh', R.h]], [L('Â × column of W', 'Â × W 的一列')])}
		{:else if v === 'sage'}
			{@const R = sageRule(P.rt)}
			{@render table([[L(`${F.i} · own`, `${F.i} · 自己`), R.own], ...F.nb.map((j, k): [string, number[]] => [`${j}`, R.fr[k]]), [L(`average of ${F.nb.length}`, `${F.nb.length} 个的平均`), R.mean], ['+ b', R.b], ['= z', R.z], ['tanh', R.h]], [L('column of W: own half for the member, friends’ half for its friends', 'W 的一列：成员本人用“自己”那一半，朋友用“朋友”那一半')])}
		{:else}
			{@const R = ginRule(P.rt)}
			{@render table([[`${F.i} × ${f(1 + R.eps)}`, R.own], ...F.nb.map((j, k): [string, number[]] => [`${j}`, R.fr[k]]), [L('Σ + b₁', 'Σ + b₁'), R.pre], ['ReLU', R.relu], [L('linear 2', '线性层 2'), R.out], ['tanh', R.h]], [L('column of W₁ (the MLP’s linear 1)', 'W₁ 的一列（MLP 的线性层 1）')])}
			<p class="note small">{L(`If every member started with the same single number 1, GraphSAGE’s average would be 1 for all ${N} members, while GIN’s sum is 1 + ε + friends: ${f(same.focus)} for member ${F.i}, ${same.kinds} different values across the club.`, `如果每个成员一开始都是同一个数 1，GraphSAGE 的平均对全部 ${N} 个成员都是 1，而 GIN 的和是 1 + ε + 朋友数：成员 ${F.i} 是 ${f(same.focus)}，全俱乐部共有 ${same.kinds} 种不同的值。`)}</p>
		{/if}
	{:else if mode === 'gat' && P.at.frame}
		<Ego i={F.i} nb={F.nb} label={gatLabel} width={(j) => (ph === 'score' ? 0.3 : alphaOf(j))} hot={(j) => (ph === 'message' && j === m.j) || (ph === 'score' && j === m.j)} title={VARIANT_TEXT.gat} onnode={go} />
		{#if ph === 'score'}
			<p class="num small">a_dst · z_{F.i} = {f(m.sd)}</p>
			{@render table([...scores.values()].map((s): [string, number[]] => [`${s.j}`, [s.ss, s.raw, s.e]]), ['a_src · z_j', `+ ${f(m.sd)}`, 'LeakyReLU'])}
		{:else if ph === 'softmax'}
			{@render table([...frames.filter((x) => x.phase === 'score').map((x): [string, number[]] => [`${x.metrics!.j}`, [x.metrics!.e, Math.exp(x.metrics!.e - m.max), alphaOf(x.metrics!.j)]]), ['Σ', [null, m.tot, 1] as number[]]], [L('score', '分数'), L('exp(score − max)', 'exp(分数 − 最大值)'), 'α'])}
		{:else}
			{#if sent.length}{@const z = sent.at(-1)!}<p class="num small">{L(`message of member ${z.j}: z_${z.j}`, `成员 ${z.j} 的消息：z_${z.j}`)} = ({[z.x0, z.x1, z.x2, z.x3].map((x) => fmt3(x)).join(', ')})</p>{/if}
			{@render table([...sent.map((s): [string, number[]] => [`${s.j} · ${f(s.a)} ×`, [s.x0, s.x1, s.x2, s.x3].map((x) => x * s.a)]), ['Σ', ph === 'activate' ? v4('s') : (sent.at(-1) ? [0, 1, 2, 3].map((k) => sent.at(-1)![`s${k}`]) : [])], ...(ph === 'activate' ? [['+ b', v4('b')], ['= y', v4('y')], ['tanh', v4('h')]] as [string, number[]][] : [])], [L('α × z_j', 'α × z_j')])}
		{/if}
	{:else if mode === 'side'}
		<div class="grid">
			{#each VARIANTS as v (v)}
				<figure>
					<figcaption>{t(VARIANT_TEXT[v])}</figcaption>
					<Ego i={F.i} nb={F.nb} label={egoLabel(v)} width={(j) => (W(v)(j) ?? 0) / wmax(v)} title={VARIANT_TEXT[v]} onnode={go} />
					<p class="num small">{L('row after layer 1', '第 1 层之后的行')}: ({rowAfter(P.rt, v).map((x) => fmt3(x)).join(', ')})</p>
				</figure>
			{/each}
		</div>
	{:else if mode === 'embed'}
		<div class="grid">
			{#each VARIANTS as v (v)}
				{@const e = T[`${v}.act2`]?.data}
				{@const ln = line(v)}
				{@const r = rightOf(P.rt, v)}
				<figure>
					<figcaption><b>{NAMES[v]}</b> · {L(`right: ${r.right} of ${r.rest}`, `分对：${r.rest} 人中 ${r.right} 人`)}</figcaption>
					<svg viewBox="0 0 100 100" class="sc" role="img" aria-label={L(`${NAMES[v]}: every member at its two numbers`, `${NAMES[v]}：每个成员放在他的两个数的位置`)}>
						<clipPath id="gv-box-{v}"><rect x="2" y="2" width="96" height="96" /></clipPath>
						<rect x="8" y="8" width="84" height="84" class="box" />
						{#if ln}<line x1={ln[0][0]} y1={ln[0][1]} x2={ln[1][0]} y2={ln[1][1]} class="cut" clip-path="url(#gv-box-{v})" />{/if}
						{#each LABELS as c, i (i)}
							{#if e}<g transform="translate({50 + e[i * 2] * 42},{50 - e[i * 2 + 1] * 42})">
								{#if i === F.i}<circle r="4.4" class="ring" />{/if}
								<circle r="2.6" style:fill={fac(c)} class="pt" />
								{#if !isRight(P.rt, v, i)}<text y="-3.4" class="mk">✕</text>{/if}
							</g>{/if}
						{/each}
					</svg>
				</figure>
			{/each}
		</div>
		<p class="legend small">{#each FACTION as fa, c (c)}<span><i style:background={fac(c)}></i>{L(`joined ${fa.en}`, `加入了${/^[A-Za-z]/.test(fa.zh ?? '') ? ' ' : ''}${fa.zh}一方`)}</span>{/each}<span>◯ {L(`member ${F.i}`, `成员 ${F.i}`)}</span><span>✕ {L('wrong side of the dashed line', '落在虚线错误的一侧')}</span></p>
	{:else if mode === 'acc'}
		<div class="bars">
			{#each VARIANTS as v (v)}
				{@const r = rightOf(P.rt, v)}
				<div class="bar">
					<b>{NAMES[v]}</b>
					<span class="track" title={L('this run (seed 0) on your graph; tick: the average over random starts; thin line: lowest to highest', '这次运行（种子 0）在你的图上；短竖线：多次随机初始化的平均；细横线：最低到最高')}>
						<i style:width="{(100 * r.right) / r.rest}%"></i>
						<s style:left="{100 * fact(`${v}Min`)}%" style:width="{100 * (fact(`${v}Max`) - fact(`${v}Min`))}%"></s>
						<u style:left="{100 * fact(`${v}Mean`)}%"></u>
					</span>
					<span class="num small">{L(`${r.right} of ${r.rest} · average ${fmt3(100 * fact(`${v}Mean`))}%`, `${r.rest} 人中 ${r.right} 人 · 平均 ${fmt3(100 * fact(`${v}Mean`))}%`)}</span>
				</div>
			{/each}
		</div>
		<p class="muted small">{L(`Bar: this trained network on your graph. Short upright tick: the average over ${fact('seeds')} random starts; thin line: the lowest to the highest of them.`, `长条：这个训练好的网络在你的图上的结果。短竖线：${fact('seeds')} 次随机初始化的平均；细横线：其中最低到最高。`)}</p>
	{:else if mode === 'drop' && P.at.frame}
		<Ego i={F.i} nb={F.nb} label={(j) => (j === F.i ? '' : f(dState.alpha?.data[j]))} width={(j) => (dState.alpha?.data[j] ?? 0) / amax} cut={(j) => ph === 'remove' && j === m.j}
			mark={(j) => (ph === 'verdict' ? (j === m.topA ? '★' : '') + (j === m.topE ? '◆' : '') : '')} title={{ en: 'Attention α per friend (GAT layer 1)', zh: '每个朋友的注意力 α（GAT 第 1 层）' }} onnode={go} />
		<p class="num small">{L(`Score gap of member ${F.i} (the Officer’s score − Mr. Hi’s; above 0 means the Officer’s side) with every friendship: ${f(m.gap)}`, `成员 ${F.i} 在所有朋友关系都在时的分数差（干事的分数 − Mr. Hi 的；大于 0 就是干事一侧）：${f(m.gap)}`)}</p>
		{@render table(drops.map((d): [string, number[]] => [`${d.j}${ph === 'verdict' ? (d.j === m.topA ? ' ★' : '') + (d.j === m.topE ? ' ◆' : '') : ''}`, [d.a, d.a2, d.after, d.d]]), [L('α layer 1', 'α 第 1 层'), L('α layer 2', 'α 第 2 层'), L('gap without it', '去掉后的分数差'), L('move', '变化')])}
		{#if ph === 'verdict' && m.members}<p class="small">★ {L('largest α in layer 1', '第 1 层 α 最大')} · ◆ {L('largest move', '变化最大')} · {L(`over the ${m.members} members with two or more friends, the largest move is at layer 1’s largest α for ${m.agree}, at layer 2’s for ${m.agree2}`, `在有两个及以上朋友的 ${m.members} 个成员中，变化最大的朋友恰好是第 1 层 α 最大的有 ${m.agree} 个，是第 2 层 α 最大的有 ${m.agree2} 个`)}</p>{/if}
	{:else}<p class="muted">{L('Computing…', '计算中…')}</p>{/if}
</div>

<style>
	.av { display: grid; gap: 8px; grid-template-columns: minmax(0, 1fr); min-width: 0; }
	.grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 250px), 1fr)); }
	figure { margin: 0; display: grid; gap: 4px; min-width: 0; } figcaption { font-size: var(--fs-xs); }
	table { border-collapse: collapse; font-size: var(--fs-xs); justify-self: start; max-width: 100%; display: block; overflow-x: auto; }
	th, td { padding: 2px 7px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; } th { text-align: left; font-weight: 500; }
	thead th { font-weight: 600; color: var(--muted); white-space: normal; text-align: left; } tr.sum td, tr.sum th { font-weight: 700; border-top: 1px solid var(--ink-2); }
	@media (max-width: 480px) { th, td { padding: 2px 4px; } }
	/* phones: the row labels may wrap and the numbers shrink a little, so all four columns fit at 360 px */
	@media (max-width: 420px) { table { font-size: 0.66rem; } th, td { padding: 2px 3px; } tbody th { white-space: normal; min-width: 3.2em; } }
	.small { margin: 0; font-size: var(--fs-xs); } .note { border-left: 3px solid var(--accent); padding-left: 8px; }
	.legend { display: flex; flex-wrap: wrap; gap: 12px; }
	.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: -1px; }
	.sc { width: 100%; max-width: 320px; height: auto; } .box { fill: none; stroke: var(--rule); stroke-width: 0.3; }
	.cut { stroke: var(--ink); stroke-width: 0.5; stroke-dasharray: 2 1.2; } .pt { stroke: var(--ink-2); stroke-width: 0.3; }
	.sc .mk { font: 600 3.4px var(--font-mono); text-anchor: middle; fill: var(--ink); } .ring { fill: none; stroke: var(--ink); stroke-width: 0.7; }
	.bars { display: grid; gap: 10px; } .bar { display: grid; grid-template-columns: 6.5em minmax(0, 1fr); gap: 2px 10px; align-items: center; }
	.bar .num { grid-column: 2; } .track { position: relative; height: 18px; background: var(--surface-2, var(--rule)); border-radius: 3px; }
	.track i { position: absolute; inset: 0 auto 0 0; background: var(--accent); border-radius: 3px; opacity: 0.75; }
	.track s { position: absolute; top: 8px; height: 2px; background: var(--ink); } .track u { position: absolute; top: -3px; width: 3px; height: 24px; background: var(--ink); }
</style>
