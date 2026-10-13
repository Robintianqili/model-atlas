// View 'tinymoe/route': what each mode draws, as plain data (pure TS; View.svelte draws it, meta.ts describes it).
//   token  one character in one block: the 4 router scores, their chances, the 2 experts kept and their gates
//   grid   every character with its layer × expert grid: which 2 of the 4 experts each block chose, shaded by the gate
//   load   the 'dispatch' process: slots each expert has taken against its capacity, the characters so far with
//          their experts kept or dropped; at the balance frame also f (share of slots sent) and P (mean chance)
//   dense  the measured comparison with the dense twin and with the MoE trained without the balancing loss (facts)
import type { StageProps } from '../../../../atlas/types/index.ts';
import { topK } from '../../ops/moe.ts';
import { cfg, chances, charOf, focusT, routeOf, row, type RT } from '../../routing.ts';

type P = Omit<StageProps, 'ctx'>;
export type Mode = 'token' | 'grid' | 'load' | 'dense';
export const modeOf = (p: P) => ((p.props?.mode as Mode) ?? 'grid');

export function tokenOf(p: P) {
	const rt = p.rt as RT;
	const b = Number(p.props?.block ?? 1);
	const t = Number(p.props?.t ?? focusT(rt));
	const r = routeOf(rt, b, t);
	const zTop = Math.max(...r.z);
	const ex = r.z.map((v) => Math.exp(v - zTop));
	return { block: b, ...r, zTop, ex, sum: ex.reduce((x, y) => x + y, 0) };
}

/** Every character with, per block, its gates over the experts (0 = not chosen). */
export function gridOf(p: P) {
	const rt = p.rt as RT;
	const { n, L, E } = cfg(rt);
	return { E, L, chars: Array.from({ length: n }, (_, t) => ({ t, c: charOf(rt, t), g: Array.from({ length: L || 0 }, (_, b) => row(rt, `B${b + 1}.moe#gate`, t)) })) };
}

type M = Record<string, number>;
/** The dispatch frame: per expert its slots taken, sent so far, the cap; per character so far, its experts kept or not. */
export function loadOf(p: P) {
	const rt = p.rt as RT;
	const f = p.at.frame;
	const m = (f?.metrics ?? {}) as M;
	const st = (f?.state ?? {}) as { load?: { data: ArrayLike<number> }; kept?: { data: ArrayLike<number> } };
	const block = Number(p.props?.block ?? 1);
	const { E, K } = cfg(rt);
	const pr = chances(rt, block);
	const upto = f?.phase === 'balance' ? pr.length - 1 : f?.phase === 'route' ? Number(m.t) : -1;
	const sent = new Array<number>(E || 0).fill(0);
	const chars = pr.slice(0, upto + 1).map((r, t) => {
		const { i } = topK(r, K || 2);
		return { t, c: charOf(rt, t), picks: i.map((j) => (sent[j]++, { e: j, ok: (st.kept?.data[t * E + j] ?? 0) > 0 })) };
	});
	const load = Array.from({ length: E || 0 }, (_, e) => st.load?.data[e] ?? 0);
	return { block, cap: m.cap ?? NaN, load, sent, chars, balance: f?.phase === 'balance' ? m : null, dropped: m.dropped ?? 0 };
}

/** The measured comparison (facts.json). */
export function denseOf(p: P) {
	const fact = (k: string) => p.rt.num(`fact:${k}`);
	return [
		{ id: 'moe', name: { en: 'MoE (this page)', zh: 'MoE（本页）' }, total: fact('params'), active: fact('active'), loss: fact('valLoss') },
		{ id: 'dense', name: { en: 'Dense MLP twin', zh: '稠密 MLP 对照' }, total: fact('denseParams'), active: fact('denseParams'), loss: fact('valLossDense') },
		{ id: 'noaux', name: { en: 'MoE, no balancing loss', zh: 'MoE，不加均衡损失' }, total: fact('params'), active: fact('active'), loss: fact('valLossNoaux') }
	];
}
