// tinymoe: process 'dispatch' (kind C): one block's routing of the learner's text played character by character under
// an expert capacity, then the load-balancing loss of that routing.
//   input  ids [T], p [T][E] (the block's router chances, read from the trace: B<b>.route), the block, k
//   start     every expert empty; each may take cap = ceil(cf · k · T / E) slots (cf: the capacity factor)
//   route     one frame per character: its k experts in order of their chances; an expert with room takes the slot
//             (its gate goes into `kept`), a full one drops it (first come, first served: the page's rule, routing.ts
//             dispatchAll; real systems differ in the order)
//   balance   f_e = share of the k·T slots sent to expert e (before any drop), P_e = the mean chance of e,
//             balancing loss = E · Σ f_e · P_e (Switch Transformer; 1 when perfectly balanced)
// Gates here are the kept chances rescaled in float64. pipeline/tinymoe/reference.py is its Python twin (atlas parity).
//   npm run atlas -- run tinymoe dispatch --frames 0..8
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { sym } from '../../atlas/core/tex.ts';
import { dispatchAll } from './routing.ts';

export type DispatchInput = { ids: number[]; p: number[][]; block: number; k: number };
export type DispatchOpt = { cf: number };
export type DispatchState = { load: Tensor; kept: Tensor };
type Run = ProcessRun<DispatchState, DispatchOpt>;

const L = (en: string, zh: string) => ({ en, zh });

export const dispatch: ProcessDef<DispatchInput, DispatchOpt, DispatchState> = {
	id: 'dispatch',
	kind: 'algorithm',
	title: L('Sending characters to experts with a capacity', '在容量限制下把字符送给专家'),
	phases: {
		start: { label: L('Empty experts', '空的专家'), kind: 'input', narration: { what: L('Each expert may take {frame.cap|int} slots.', '每个专家最多接 {frame.cap|int} 个位置。') } },
		route: {
			label: L('Route one character', '分派一个字符'),
			kind: 'act',
			narration: { what: L('Character {frame.t|int} (counting from 0) goes to experts {frame.a|int} and {frame.b|int}; {frame.dropped|int} slots dropped so far.', '第 {frame.t|int} 个字符（从 0 数起）去专家 {frame.a|int} 和 {frame.b|int}；到目前为止丢掉了 {frame.dropped|int} 个位置。') }
		},
		balance: {
			label: L('Balancing loss', '负载均衡损失'),
			kind: 'loss',
			formula: {
				tex: `${sym('out', '\\mathcal{L}_{\\text{bal}}')} = E \\sum_e ${sym('act', 'f')}_e\\, ${sym('param', 'P')}_e = {{E}} \\times ({{fp0}} + {{fp1}} + {{fp2}} + {{fp3}}) = {{aux}}`,
				values: (f) => f.metrics ?? {}
			},
			narration: { what: L('The balancing loss of this routing is {frame.aux|n4}.', '这次分派的负载均衡损失是 {frame.aux|n4}。') }
		}
	},
	schema: {
		load: { kind: 'tensor', label: L('slots taken per expert', '每个专家接下的位置数'), shape: ['E'], scale: 'sequential', tol: 'exact' },
		kept: { kind: 'tensor', label: L('gates kept (characters × experts)', '保留的门控（字符 × 专家）'), shape: ['T', 'E'], scale: 'sequential', tol: 1e-12 }
	},
	defaults: { cf: 1 },
	*run(input, ctx): Run {
		const { ids, p, k } = input;
		const n = ids.length;
		if (!n || p.length !== n) return;
		const E = p[0].length;
		const r = dispatchAll(p, k, ctx.opt.cf);
		const kept = new Float64Array(n * E);
		const st = (load: number[]) => ({ load: { shape: [E], data: Float64Array.from(load) }, kept: { shape: [n, E], data: kept.slice() } });
		yield { phase: 'start', state: st(new Array(E).fill(0)), metrics: { T: n, E, k, cap: r.cap, cf: ctx.opt.cf, block: input.block } };
		for (let t = 0; t < n; t++) {
			const s = r.steps[t];
			s.i.forEach((j, m) => s.ok[m] && (kept[t * E + j] = s.g[m]));
			const [a, b] = [s.i[0], s.i[1] ?? -1];
			yield { phase: 'route', step: t, state: st(s.load), metrics: { t, id: ids[t], a, b, ga: s.g[0], gb: s.g[1] ?? 0, pa: p[t][a], pb: p[t][b] ?? 0, okA: s.ok[0], okB: s.ok[1] ?? 0,
				loadA: s.load[a], loadB: s.load[b] ?? 0, cap: r.cap, dropped: s.dropped, lost: s.lost, slots: k * (t + 1) } };
		}
		const m: Record<string, number> = { T: n, E, k, cap: r.cap, dropped: r.dropped, lost: r.lost, dropShare: r.dropped / (k * n), lostShare: r.lost / n };
		let aux = 0;
		let busiest = 0;
		for (let j = 0; j < E; j++) {
			let P = 0;
			for (let t = 0; t < n; t++) P += p[t][j] / n;
			const f = r.sent[j] / (k * n);
			Object.assign(m, { [`f${j}`]: f, [`P${j}`]: P, [`fp${j}`]: f * P, [`n${j}`]: r.sent[j], [`l${j}`]: r.load[j] });
			aux += E * f * P;
			if (f > m[`f${busiest}`]) busiest = j;
		}
		yield { phase: 'balance', step: n, state: st(r.load), metrics: { ...m, aux, busiest, fBusiest: m[`f${busiest}`] } };
	}
};
