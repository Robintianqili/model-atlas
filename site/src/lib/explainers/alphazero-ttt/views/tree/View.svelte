<!-- alphazero-ttt 'tree': the search tree at a frame of the `search` process, drawn as rows of boxes (one row per move
     ahead; a box's width is the share of simulations that went through it; this simulation's path in the accent), and
     what the frame does: the PUCT scores of a choice (select), the new position the network looks at (expand), or the
     value going back up the path (backup). props.limits: the tree's size next to the whole game tree from here. -->
<script lang="ts">
	import FormulaCard from '$lib/kit/FormulaCard.svelte';
	import { t } from '$lib/i18n/i18n.svelte';
	import type { StageProps } from '../../../../atlas/types/index.ts';
	import { fmt3, fmtInt } from '../../../../atlas/core/format.ts';
	import Board from '../../parts/Board.svelte';
	import { ARROW, CELL, L, RESULT, markOf, toMove, treeSize } from '../../rules.ts';
	import { boardAt, posOf, stateOf } from '../../picture.ts';
	import { score } from '../../search.ts';
	import { alignTex } from '../../../../atlas/ui/views/kit.parts/tex.ts';

	let P: StageProps = $props();
	const s = $derived(stateOf(P.at));
	const f = $derived(P.at.frame);
	const m = $derived(f?.metrics ?? {});
	const root = $derived(posOf(P.rt));
	const kids = $derived.by(() => {
		const k: number[][] = [];
		for (let i = 0; i < (s?.count ?? 0); i++) k[i] = [];
		for (let i = 1; i < (s?.count ?? 0); i++) k[s!.parent.data[i]].push(i);
		return k;
	});
	const onPath = $derived(new Set(s?.path ?? []));
	const boxes = $derived.by(() => {
		const out: { node: number; x: number; w: number; d: number }[] = [];
		if (!s) return out;
		const rec = (node: number, x: number, w: number, d: number) => {
			out.push({ node, x, w, d });
			const ks = kids[node].filter((k) => s.N.data[k] > 0 || onPath.has(k)), wt = (k: number) => Math.max(s.N.data[k], onPath.has(k) ? 0.6 : 0);
			const tot = ks.reduce((a, k) => a + wt(k), 0);
			let cx = x;
			for (const k of ks) { rec(k, cx, (w * wt(k)) / tot, d + 1); cx += (w * wt(k)) / tot; }
		};
		rec(0, 0, 600, 0);
		return out;
	});
	const depth = $derived(Math.max(1, ...boxes.map((b) => b.d + 1)));
	const node = $derived(Number(m.node ?? 0));
	const nb = $derived(s ? boardAt(s, root, f?.phase === 'select' ? node : Number(m.node ?? 0)) : root);
	const leafP = $derived(new Map(s && f?.phase === 'expand' ? kids[node].map((k) => [s.move.data[k], s.P.data[k]] as [number, number]) : []));
	const table = $derived(s && f?.phase === 'select' ? kids[node].map((k) => ({ k, cell: s.move.data[k], ...score(s.N.data, s.W.data, s.P.data, node, k, Number(m.c)) })) : []);
	const sq = (v: number) => String(+v.toFixed(4));
	const tex = $derived(alignTex((P.at.tape?.phases?.select as { formula?: { tex: string } } | undefined)?.formula?.tex ?? ''));
	const path = $derived(f?.phase === 'backup' && s ? (s.path ?? []).map((x, i) => ({ x, cell: s.move.data[x], n: s.N.data[x], q: s.N.data[x] > 0 ? s.W.data[x] / s.N.data[x] : 0, d: i })) : []);
	const changes = $derived(new Map((f?.changes ?? []).filter((c) => c.field === 'W' || c.field === 'N').map((c) => [`${c.field}${c.index?.[0]}`, c.from ?? 0])));
	const say = (o: { en: string; zh: string }) => t(o);
</script>

<div class="tv">
	<div class="ice" style:height="{depth * 26}px" role="img" aria-label={say(L('The search tree: one row per move ahead; a box’s width is its share of the simulations', '搜索树：每往后一步一行；方块的宽度是经过它的模拟所占的份额'))}>
		{#each boxes as bx (bx.node)}
			<span class:path={onPath.has(bx.node)} style:left="{bx.x / 6}%" style:width="{bx.w / 6}%" style:top="{bx.d * 26}px">{bx.node === 0 ? say(L('board', '棋盘')) : ARROW[s!.move.data[bx.node]]} {s!.N.data[bx.node]}</span>
		{/each}
	</div>
	<p class="small">{say(L(`Rows: moves ahead. Width: share of the ${s ? s.N.data[0] : 0} visits so far. ${f?.phase === 'choose' ? 'Highlighted: the move it plays.' : 'Highlighted: this simulation’s path (drawn even before its first visit).'} ↖ … ↘ name the cell of each move.`, `每一行：往后一步。宽度：到目前为止 ${s ? s.N.data[0] : 0} 次访问里所占的份额。${f?.phase === 'choose' ? '高亮的是它下的那一步。' : '高亮的是这次模拟走的路径（还没被访问过也画出来）。'}↖ … ↘ 表示每一步下在哪个格子。`))}</p>
	{#if P.props.limits && s}
		<p class="note">{say(L(`This tree: ${fmtInt(s.count)} positions; the network looked at ${fmtInt(m.nets ?? 0)} of them (the others are finished games, which need no network, or branches no simulation has reached yet). The whole game tree from this board: ${fmtInt(treeSize(root))} positions.`, `这棵树：${fmtInt(s.count)} 个局面，其中 ${fmtInt(m.nets ?? 0)} 个让网络看过（其余的是已经结束的棋局，不用网络；或者是还没有模拟走到的分支）。从这块棋盘出发的完整博弈树：${fmtInt(treeSize(root))} 个局面。`))}</p>
	{/if}
	{#if f?.phase === 'select' && s}
		<div class="row">
			<Board b={nb} small cells={(i) => ({ ring: i === m.cell })} label={say(L('The position where the search chooses', '搜索做选择的局面'))} />
			<p class="small">{say(L(`Simulation ${m.sim}, ${m.depth ? `${m.depth} move${m.depth > 1 ? 's' : ''} ahead` : 'at the board you set'}: ${markOf(toMove(nb))} chooses.`, `第 ${m.sim} 次模拟，${m.depth ? `往后 ${m.depth} 步` : '在你摆的棋盘上'}：轮到 ${markOf(toMove(nb))} 选。`))}</p>
		</div>
		<table class="num">
			<thead><tr><th>{say(L('move', '走法'))}</th><th>P</th><th>n</th><th>Q</th><th>{say(L('bonus', '奖励分'))}</th><th>U</th></tr></thead>
			<tbody>{#each table as r (r.k)}<tr class:pick={r.cell === m.cell}><td>{ARROW[r.cell]} <span class="nm">{t(CELL[r.cell])}</span></td><td>{fmt3(r.p)}</td><td>{r.n}</td><td>{fmt3(r.q)}</td><td>{fmt3(r.bonus)}</td><td><b>{fmt3(r.u)}</b></td></tr>{/each}</tbody>
		</table>
		<FormulaCard {tex} values={{ q: m.q, c: sq(m.c), p: m.p, N: sq(m.N), n: sq(m.n), u: m.u }} provenance={P.provenance} />
	{:else if f?.phase === 'expand' && s}
		<div class="row">
			<Board b={nb} cells={(i) => (leafP.has(i) ? { text: fmt3(leafP.get(i)!), fill: leafP.get(i) } : { ring: i === m.cell })} label={say(L('The new position', '新的局面'))} />
			<p>{m.terminal ? say(L(`The game is over here: ${RESULT[Number(m.v) + 1].en} for ${markOf(m.mover)}, so v = ${fmt3(m.v)}. No network needed.`, `这里棋局已经结束：对 ${markOf(m.mover)} 来说是${RESULT[Number(m.v) + 1].zh}，所以 v = ${fmt3(m.v)}，不用网络。`)) : say(L(`New position, ${markOf(m.mover)} to move. The network: v = ${fmt3(m.v)}, and a prior for each of the ${m.kids} moves (on the board).`, `新局面，轮到 ${markOf(m.mover)}。网络给出：v = ${fmt3(m.v)}，以及 ${m.kids} 种走法各自的先验（标在棋盘上）。`))}</p>
		</div>
	{:else if f?.phase === 'backup' && s}
		<table class="num">
			<thead><tr><th>{say(L('position', '局面'))}</th><th>n</th><th>{say(L('Q (for the player who chose it)', 'Q（对做选择的一方）'))}</th></tr></thead>
			<tbody>{#each path as r (r.x)}{@const n0 = changes.get(`N${r.x}`) ?? r.n}{@const w0 = changes.get(`W${r.x}`) ?? 0}<tr><td>{r.d === 0 ? say(L('the board you set', '你摆的棋盘')) : `${ARROW[r.cell]} ${t(CELL[r.cell])}`}</td><td>{n0} → {r.n}</td><td class="answer">{r.d === 0 ? '' : `${n0 > 0 ? fmt3(w0 / n0) : '0'} → ${fmt3(r.q)}`}</td></tr>{/each}</tbody>
		</table>
	{/if}
</div>

<style>
	.tv { display: grid; gap: 8px; min-width: 0; }
	.ice { position: relative; }
	.ice span { position: absolute; height: 22px; box-sizing: border-box; overflow: hidden; white-space: nowrap; text-align: center; font: 12px/20px var(--font-mono); color: var(--ink); background: var(--surface-2); border: 1px solid var(--rule-strong); border-radius: 3px; }
	.ice span.path { background: var(--accent-soft); border: 2px solid var(--accent); line-height: 18px; }
	.small, .note { margin: 0; font-size: var(--fs-xs); color: var(--ink-2); }
	.note { padding: 6px 8px; border-left: 3px solid var(--accent); color: var(--ink); }
	.row { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
	.row p { flex: 1 1 200px; }
	.row p { margin: 0; }
	table { width: 100%; border-collapse: collapse; font-size: var(--fs-xs); }
	th, td { padding: 2px 6px; text-align: right; border-bottom: 1px solid var(--rule); }
	th:first-child, td:first-child { text-align: left; }
	tr.pick td { background: var(--accent-soft); font-weight: 700; }
	@media (max-width: 480px) { .nm { display: none; } }
	/* while a predict question gates the stage (StageHost makes it inert), the new Q values would give the answer away */
	:global([inert]) .answer { visibility: hidden; }
</style>
