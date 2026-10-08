<!-- gcn 'graph' view: one drawing of the club, six modes (meta.ts). Clicks change the learner's input (setInput): the
     focus member, a friendship (adj) or the labelled members (pick); every source then re-runs on the new graph. -->
<script lang="ts">
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3 } from '../../../../atlas/core/format.ts';
	import { adjacency, FACTION, forwardGcn, KARATE, graphOf, hops, LABELS, N, neighbours, normalize, toggleEdge, toggleLabel, type GraphRaw, type Params } from '../../graph.ts';
	import Graph from '../../parts/Graph.svelte';
	import { cornerCount, retrained } from './meta.ts';

	let P: StageProps = $props();
	const mode = $derived(String(P.props.mode ?? 'adj'));
	const g = $derived(graphOf(P.rt.input));
	const T = $derived(P.rt.sources.fwd?.trace?.t);
	const set = (raw: GraphRaw) => (P.setInput ? P.setInput(raw) : P.rt.setInput(raw));
	const m = $derived(P.at.frame?.metrics ?? {});
	const ph = $derived(P.at.frame?.phase ?? '');
	const fac = (c: number, pct = 55) => `color-mix(in srgb, var(${c ? '--div-pos' : '--div-neg'}) ${pct}%, var(--surface))`;
	const focusNode = (i: number) => set({ ...g, focus: i });
	const senders = $derived([g.focus, ...neighbours(g, g.focus)]);
	const row = (id: string, r: number, w: number) => (T?.[id] ? Array.from(T[id].data.subarray(r * w, (r + 1) * w)) : []);
	const ah = (j: number) => T?.norm?.data[g.focus * N + j] ?? NaN;
	const deg = (j: number) => T?.['norm#deg']?.data[j] ?? NaN;
	const L = (en: string, zh: string) => t({ en, zh });
	const v4 = (p: string) => [0, 1, 2, 3].map((f) => m[`${p}${f}`]);
	/** as the narration prints a number: an exact integer as itself, anything else with 3 significant digits */
	const nf = (v: number) => (Number.isInteger(v) ? String(v) : fmt3(v));
	const tup = (v: number[]) => `(${v.map(nf).join(', ')})`;
	type Row = { lab: string; mul?: string; mid?: string; vals: number[]; sum?: boolean };
	/** mp: the table under the drawing, by phase: a header ([text, columns]) and rows */
	const table = $derived.by((): { head: [string, number][]; rows: Row[] } => {
		const xw = (j: number) => row('gc1#xw', j, 4);
		const none = { head: [], rows: [] };
		if (mode !== 'mp') return none;
		if (ph === 'norm')
			return { head: [[L('from · its d', '来自 · 它的 d'), 1], [`1 / √(d${sub(g.focus)} × dⱼ)`, 1], [`Â${sub(g.focus)}ⱼ`, 1]], rows: senders.map((j) => ({ lab: `${j} · d = ${deg(j)}`, mid: `1 / √(${deg(g.focus)} × ${deg(j)})`, vals: [ah(j)] })) };
		if (ph === 'transform') return { head: [[L('member', '成员'), 1], [L('its 4 numbers: its row of X Wᵀ = its column of W', '它的 4 个数：X Wᵀ 中它的行 = W 中它的列'), 4]], rows: senders.map((j) => ({ lab: `${j}`, vals: xw(j) })) };
		if (ph === 'message')
			return { head: [[L('from · Â ×', '来自 · Â ×'), 1], [L('the message: Â × its 4 numbers', '消息：Â × 它的 4 个数'), 4]], rows: [...senders.slice(0, m.k).map((j): Row => ({ lab: `${j} ·`, mul: `${nf(ah(j))} ×`, vals: xw(j).map((v) => v * ah(j)) })), { lab: 'Σ', vals: v4('s'), sum: true }] };
		if (ph === 'aggregate') return { head: [], rows: [{ lab: 'Σ', vals: v4('s') }, { lab: '+ b', vals: v4('b') }, { lab: '=', vals: v4('z'), sum: true }] };
		if (ph === 'activate') return { head: [], rows: [{ lab: 'Σ', vals: v4('s') }, { lab: '+ b', vals: v4('b') }, { lab: '= z', vals: v4('z') }, { lab: 'tanh(z)', vals: v4('h'), sum: true }] };
		return none;
	});
	const sent = (j: number) => senders.indexOf(j) >= 0 && (ph !== 'message' || senders.indexOf(j) < m.k);
	const mpEdge = (a: number, b: number) => (a !== g.focus && b !== g.focus ? '' : ph === 'message' && (a === m.j || b === m.j) ? 'now' : sent(a === g.focus ? b : a) ? 'on' : '');
	const d = $derived(hops(g, g.focus));
	const R = $derived(mode === 'pick' ? retrained(P) : null);
	const emb = $derived(mode === 'embed' && T?.act2 ? T.act2.data : null);
	const pred = (z: ArrayLike<number> | undefined, i: number) => (z ? (z[i * 2 + 1] > z[i * 2] ? 1 : 0) : -1);
	const count = (k: number) => d.filter((x) => x === k).length;
	/** embed on an edited graph: how many members sit on another side than with the recorded friendships */
	const edited = $derived(JSON.stringify(g.edges) !== JSON.stringify(KARATE.edges));
	const recZ = $derived(mode === 'embed' && edited && P.rt.model ? forwardGcn(P.rt.model.weights as Params, normalize(adjacency(KARATE), N).ahat).z : null);
	const moved = $derived(recZ && T?.head ? LABELS.filter((_, i) => pred(recZ, i) !== pred(T.head.data, i)).length : -1);
	const wrong = $derived(LABELS.map((c, i) => (pred(T?.head?.data, i) !== c ? i : -1)).filter((i) => i >= 0));
	/** members on the wrong side last, so they are drawn on top */
	const order = $derived([...LABELS.keys()].sort((a, b) => Number(wrong.includes(a)) - Number(wrong.includes(b))));
	const line = $derived.by(() => {
		const W = P.rt.model?.weights['head.weight']?.data;
		const b = P.rt.model?.weights['head.bias']?.data;
		if (!W || !b) return null;
		const [u, v, c] = [W[2] - W[0], W[3] - W[1], b[1] - b[0]]; // u·x + v·y + c = 0
		const pts = Math.abs(v) > Math.abs(u) ? [-1.1, 1.1].map((x) => [x, -(u * x + c) / v]) : [-1.1, 1.1].map((y) => [-(v * y + c) / u, y]);
		return pts.map(([x, y]) => [50 + x * 42, 50 - y * 42]);
	});
	const lean = $derived(mode === 'smooth' ? ((P.at.state as { lean?: { data: ArrayLike<number> } } | undefined)?.lean?.data ?? null) : null);
	const leanFill = (v: number) => `color-mix(in srgb, var(${v > 0 ? '--div-pos' : '--div-neg'}) ${Math.round(Math.min(1, Math.abs(v)) * 85)}%, var(--surface))`;
	/** smooth: the leaders' cosine similarity after every round of the run (the bars under the drawing) */
	const leaders = $derived(mode === 'smooth' && P.at.tape ? Array.from({ length: P.at.tape.length }, (_, k) => Number(P.at.tape!.frame(k).metrics?.leaders ?? NaN)) : []);
	function sub(i: number) {
		return String(i).replace(/\d/g, (c) => '₀₁₂₃₄₅₆₇₈₉'[Number(c)]);
	}
</script>

<div class="gv">
	{#if (mode === 'adj' || mode === 'x') && T?.adj && T.x}
		<Graph {g} focus={g.focus} fill={(i) => fac(LABELS[i], 35)} edge={(a, b) => (a === g.focus || b === g.focus ? 'on' : '')} onnode={focusNode}
			matrix={mode === 'x' ? { data: T.x.data, title: { en: 'X: row = member, column = feature. Member i has a single 1, in column i.', zh: 'X：行 = 成员，列 = 特征。成员 i 只有一个 1，在第 i 列。' } } : { data: T.adj.data, title: { en: 'A: row = member, column = friend. Click a square to add or remove that friendship.', zh: 'A：行 = 成员，列 = 朋友。点一个格子可以加上或去掉这段朋友关系。' } }}
			oncell={mode === 'adj' ? (i, j) => set(toggleEdge(g, i, j)) : undefined} />
	{:else if mode === 'mp'}
		<Graph {g} focus={g.focus} fill={(i) => (i === g.focus ? 'var(--accent-soft)' : senders.includes(i) ? fac(LABELS[i]) : undefined)} dim={(i) => !senders.includes(i)} edge={mpEdge}
			mark={(i) => (ph === 'norm' && senders.includes(i) ? fmt3(ah(i), 2) : '')} onnode={focusNode}
			matrix={ph === 'loop' && T?.['norm#loops'] ? { data: T['norm#loops'].data, title: { en: 'A + I: the 1s on the diagonal are the self-loops', zh: 'A + I：对角线上的 1 就是自环' } } : ph === 'norm' && T?.norm ? { data: T.norm.data, title: { en: 'Â: the weight of every edge, self-loops included', zh: 'Â：每条边（包括自环）的权重' } } : null} />
		{#if ph === 'loop'}
			<p class="num small">{L(`Member ${g.focus}: ${m.friends} friend${m.friends === 1 ? '' : 's'} + itself → d = ${m.d}`, `成员 ${g.focus}：${m.friends} 个朋友 + 自己 → d = ${m.d}`)}</p>
		{/if}
		{#if table.rows.length}
			<table class="num">
				{#if table.head.length}<thead><tr>{#each table.head as [h, span], c (c)}<th colspan={span}>{h}</th>{/each}</tr></thead>{/if}
				<tbody>{#each table.rows as r, k (k)}<tr class:sum={r.sum}><th>{r.lab}{#if r.mul}{' '}<span class="mul">{r.mul}</span>{/if}</th>{#if r.mid}<td class="mid">{r.mid}</td>{/if}{#each r.vals as v, f (f)}<td>{nf(v)}</td>{/each}</tr>{/each}</tbody>
			</table>
		{/if}
		{#if ph === 'message' && m.j !== undefined}
			<p class="num small now">{L(`Message ${m.k} of ${m.of}, from member ${m.j}: ${nf(m.a)} × ${tup(v4('x'))} = ${tup(v4('m'))}`, `第 ${m.k} 条（共 ${m.of} 条），来自成员 ${m.j}：${nf(m.a)} × ${tup(v4('x'))} = ${tup(v4('m'))}`)}</p>
		{/if}
	{:else if mode === 'hops'}
		<Graph {g} focus={g.focus} onnode={focusNode} dim={(i) => d[i] < 0 || d[i] > 2} edge={(a, b) => (Math.max(d[a], d[b]) <= 2 && d[a] >= 0 && d[b] >= 0 && Math.min(d[a], d[b]) <= 1 ? 'on' : '')}
			fill={(i) => (d[i] === 0 ? 'var(--accent)' : d[i] === 1 ? 'color-mix(in srgb, var(--accent) 55%, var(--surface))' : d[i] === 2 ? 'color-mix(in srgb, var(--accent) 22%, var(--surface))' : undefined)} />
		<p class="muted small">{L('Dark: the member in focus · middle: its friends (layer 1) · light: friends of friends (layer 2) · faded: out of reach after two layers.', '深色：关注的成员 · 中等：它的朋友（第 1 层）· 浅色：朋友的朋友（第 2 层）· 变淡：两层之后还够不到。')}</p>
		<p class="num small">{L(`Member ${g.focus} hears from ${count(1)} friends and ${count(2)} friends of friends: ${count(1) + count(2)} of the ${N} members.`, `成员 ${g.focus} 听到 ${count(1)} 个朋友和 ${count(2)} 个朋友的朋友：${N} 个成员中的 ${count(1) + count(2)} 个。`)}
			{#if T?.act2}{L(` Layer 2 gives it the numbers ${tup(row('act2', g.focus, 2))}.`, `第 2 层给它的数是 ${tup(row('act2', g.focus, 2))}。`)}{/if}</p>
	{:else if mode === 'embed' && emb}
		<svg viewBox="0 0 100 100" class="sc" role="img" aria-label={L('Every member at its two numbers', '每个成员放在它的两个数的位置')}>
			<clipPath id="gcn-box"><rect x="4" y="4" width="92" height="92" /></clipPath>
			<rect x="8" y="8" width="84" height="84" class="box" />
			{#if line}<line x1={line[0][0]} y1={line[0][1]} x2={line[1][0]} y2={line[1][1]} class="cut" clip-path="url(#gcn-box)" />{/if}
			{#each order as i (i)}
				{@const c = LABELS[i]}
				{@const x = 50 + emb[i * 2] * 42}
				{@const y = 50 - emb[i * 2 + 1] * 42}
				<g transform="translate({x},{y})">
					{#if KARATE.labelled.includes(i)}<circle r="4.2" class="ring" />{/if}
					<circle r="2.8" style:fill={fac(c)} class="pt" />
					<text y="1">{i}</text>
					{#if pred(T?.head?.data, i) !== c}<text y="-3.6" class="mk">✕</text>{/if}
				</g>
			{/each}
			<text x="50" y="99" class="ax">{L('number 0 →', '第 0 个数 →')}</text>
			<text x="2" y="50" class="ax" transform="rotate(-90 2 50)">{L('number 1 →', '第 1 个数 →')}</text>
		</svg>
		<p class="num small">{L(`Dashed line: where the head's two scores tie. On the wrong side: ${wrong.join(', ') || 'nobody'}.`, `虚线：输出头两个分数相等的地方。落在错误一侧的：${wrong.join('、') || '没有'}。`)}</p>
		{#if moved >= 0}<p class="num small">{L(`Compared with the recorded club, ${moved} member${moved === 1 ? '' : 's'} changed sides: the weights were trained on the recorded friendships, and each member’s own column of W still carries what they taught it.`, `和记录的俱乐部相比，有 ${moved} 个成员换到了另一侧：权重是在记录的朋友关系上训练出来的，W 中每个成员自己的那一列仍然带着那些朋友关系教给它的东西。`)}</p>{/if}
		<p class="muted small">{L(`${cornerCount(emb)} members sit near a corner (both numbers below −0.8 or both above 0.8), so many dots there overlap.`, `有 ${cornerCount(emb)} 个成员挤在角落附近（两个数都小于 −0.8 或都大于 0.8），所以那里的很多点叠在一起。`)}</p>
	{:else if mode === 'pick'}
		{#if R}
			<Graph {g} focus={-1} fill={(i) => fac(pred(R.z, i))} ring={(i) => g.labelled.includes(i)} mark={(i) => (!g.labelled.includes(i) && pred(R.z, i) !== LABELS[i] ? '✕' : '')} onnode={(i) => set(toggleLabel(g, i))}
				label={{ en: 'Click members to label them; the network retrains', zh: '点成员给它们加标签；网络会重新训练' }} />
			<p class="num small">{L(`Labelled: ${g.labelled.join(', ') || '–'} · step ${R.step} · right: ${R.right} of ${R.rest}`, `已标注：${g.labelled.join('、') || '–'} · 第 ${R.step} 步 · 分对：${R.rest} 人中 ${R.right} 人`)}</p>
			<p class="muted small">{L('Colour: the side the retrained network puts each member on; ✕: not the side that member joined.', '颜色：重新训练的网络把每个成员分到哪一派；✕：不是这个成员实际加入的一派。')}</p>
		{:else}<p class="muted">{L('Training the network…', '正在训练网络…')}</p>{/if}
	{:else if mode === 'smooth' && lean}
		<Graph {g} focus={-1} fill={(i) => leanFill(lean[i])} ring={(i) => i === 0 || i === 33} />
		<p class="muted small">{m.k === 0 ? L('Before any averaging, only the two leaders have a colour.', '还没有平均时，只有两个领头人有颜色。') : L(`After ${m.k} round${m.k === 1 ? '' : 's'} of averaging.`, `平均 ${m.k} 轮之后。`)} {L("Colour: does a member's row look more like Mr. Hi's (blue) or the Officer's (orange)? Pale = hard to tell.", '颜色：一个成员的行更像 Mr. Hi 的（蓝）还是干事的（橙）？越淡越难分辨。')}</p>
		{#if leaders.length}
			<figure class="bars">
				<svg viewBox="0 0 {leaders.length * 10} 31" role="img" aria-label={L(`Cosine similarity of the rows of members 0 and 33 after each round: ${leaders.map((v) => fmt3(v)).join(', ')}`, `每一轮之后成员 0 和 33 的行的余弦相似度：${leaders.map((v) => fmt3(v)).join('、')}`)}>
					<line x1="0" x2={leaders.length * 10} y1="24" y2="24" class="base" />
					{#each leaders as v, k (k)}
						<g class:cur={k === m.k}><rect x={k * 10 + 2} y={24 - 18 * Math.max(0, v)} width="6" height={18 * Math.max(0, v)} /><text x={k * 10 + 5} y={22.6 - 18 * Math.max(0, v)}>{fmt3(v)}</text><text x={k * 10 + 5} y="29.5">{k}</text></g>
					{/each}
				</svg>
				<figcaption class="muted small">{L('Bars: how alike the rows of members 0 and 33 are (cosine similarity; 1 = the same direction) after 0, 1, … rounds; the dark bar is the round shown.', '柱子：平均 0、1、… 轮之后，成员 0 和 33 的行有多像（余弦相似度；1 表示方向完全相同）；深色的柱子是当前这一轮。')}</figcaption>
			</figure>
		{/if}
	{:else}<p class="muted">{L('Computing…', '计算中…')}</p>{/if}
	{#if mode === 'adj' || mode === 'x' || mode === 'embed' || mode === 'pick'}
		<p class="legend small">{#each FACTION as f, c (c)}<span><i style:background={fac(c)}></i>{L(`${f.en}'s side`, `${f.zh}一方`)}</span>{/each}{#if mode === 'embed' || mode === 'pick'}<span>◌ {L('labelled', '已标注')}</span><span>✕ {L('wrong side', '分错了')}</span>{/if}</p>
	{/if}
</div>

<style>
	.gv { display: grid; gap: 8px; grid-template-columns: minmax(0, 1fr); min-width: 0; }
	table { border-collapse: collapse; font-size: var(--fs-xs); justify-self: start; max-width: 100%; display: block; overflow-x: auto; }
	th, td { padding: 2px 6px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; } th { text-align: left; font-weight: 500; }
	thead th { font-weight: 400; color: var(--muted); white-space: normal; vertical-align: bottom; } td.mid { color: var(--muted); }
	tr.sum td, tr.sum th { font-weight: 700; border-top: 1px solid var(--ink-2); }
	.small { margin: 0; font-size: var(--fs-xs); } .legend { display: flex; flex-wrap: wrap; gap: 12px; }
	.now { overflow-wrap: anywhere; }
	.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 4px; vertical-align: -1px; }
	.sc { width: 100%; max-width: 420px; height: auto; } .box { fill: none; stroke: var(--rule); stroke-width: 0.3; }
	.cut { stroke: var(--ink); stroke-width: 0.5; stroke-dasharray: 2 1.2; } .pt { stroke: var(--ink-2); stroke-width: 0.3; }
	.sc text { font: 600 2.4px var(--font-mono); text-anchor: middle; fill: var(--ink); } .sc .mk { font-size: 3.4px; }
	.sc .ax { font: 3px var(--font-body); fill: var(--muted); } .ring { fill: none; stroke: var(--ink); stroke-width: 0.6; stroke-dasharray: 1.2 0.7; }
	.bars { margin: 0; display: grid; gap: 2px; } .bars svg { width: 100%; max-width: 420px; height: auto; display: block; }
	.bars rect { fill: color-mix(in srgb, var(--accent) 30%, var(--surface)); } .bars .cur rect { fill: var(--accent); }
	.bars text { font: 2.3px var(--font-mono); text-anchor: middle; fill: var(--muted); } .bars .cur text { fill: var(--ink); font-weight: 700; }
	.bars .base { stroke: var(--rule-strong); stroke-width: 0.2; }
	.mul { white-space: nowrap; }
	@media (max-width: 480px) { th, td { padding: 2px 3px; } tbody th .mul { display: block; } }
</style>
