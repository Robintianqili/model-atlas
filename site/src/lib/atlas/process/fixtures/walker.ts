// A synthetic kind-C process for the tape tests (acceptance 1): every draw kind (int, normal, a fork's next), two
// 'changes'-tracked tensors, snapshot fields, metrics, and the 'set' control. `forgetChange: true` makes it mutate Q
// once without reporting it, which the determinism harness must catch.
import type { Tensor } from '../../types/core.ts';
import type { Control, Frame, ProcessDef, RunCtx } from '../../types/process.ts';

export type WalkerOpt = { steps: number; cells: number; alpha: number; forgetChange?: boolean };
export type WalkerState = { pos: number; action: number; noise: number; visits: Tensor; Q: Tensor; best: number };

export const walker: ProcessDef<{ start: number }, WalkerOpt, WalkerState> = {
	id: 'walker',
	kind: 'algorithm',
	title: { en: 'A random walk that learns', zh: '会学习的随机游走' },
	phases: {
		act: { label: { en: 'Choose a move', zh: '选择一步' }, kind: 'act' },
		update: {
			label: { en: 'Update the table', zh: '更新表格' },
			kind: 'update',
			formula: { tex: 'Q \\leftarrow Q + {{alpha}}\\,(r - Q)', values: (f) => ({ alpha: f.metrics?.alpha ?? NaN }) }
		}
	},
	schema: {
		pos: { kind: 'int', label: { en: 'Position', zh: '位置' }, tol: 'exact' },
		action: { kind: 'int', label: { en: 'Move', zh: '移动' }, tol: 'exact' },
		noise: { kind: 'scalar', label: { en: 'Noise', zh: '噪声' }, tol: 1e-12 },
		visits: { kind: 'tensor', label: { en: 'Visits per cell', zh: '每格访问次数' }, track: 'changes', tol: 'exact' },
		Q: { kind: 'tensor', label: { en: 'Q table', zh: 'Q 表' }, track: 'changes', tol: 1e-12 },
		best: { kind: 'scalar', label: { en: 'Best value', zh: '最大值' }, derive: (s: WalkerState) => (s.Q ? Math.max(...s.Q.data) : NaN) }
	},
	defaults: { steps: 400, cells: 9, alpha: 0.5 },
	*run(input: { start: number }, ctx: RunCtx<WalkerOpt>): Generator<Frame<WalkerState>, void, Control<WalkerOpt> | undefined> {
		const G = ctx.opt.cells;
		const visits: Tensor = { shape: [G], data: new Float64Array(G) };
		const Q: Tensor = { shape: [G, 2], data: new Float64Array(G * 2) };
		const env = ctx.rng.fork('env');
		let pos = input.start % G;
		visits.data[pos] = 1;
		yield { phase: 'act', step: 0, state: { pos, action: -1, noise: 0, visits, Q }, metrics: { pos } };
		for (let s = 0; s < ctx.opt.steps && !ctx.signal.cancelled; s++) {
			const action = ctx.rng.int(2);
			const noise = ctx.rng.normal();
			const slip = env.next() < 0.1;
			const move = slip ? -(action * 2 - 1) : action * 2 - 1;
			pos = (pos + move + G) % G;
			const before = visits.data[pos];
			visits.data[pos] = before + 1;
			const c = yield {
				phase: 'act',
				step: s,
				state: { pos, action, noise },
				changes: [{ field: 'visits', index: [pos], from: before, to: before + 1 }],
				metrics: { pos, noise, slip: slip ? 1 : 0 }
			};
			if (c?.type === 'set') Object.assign(ctx.opt, c.opt);
			const r = pos === G - 1 ? 1 : 0;
			const i = pos * 2 + action;
			const old = Q.data[i];
			const next = old + ctx.opt.alpha * (r + 0.01 * noise - old);
			Q.data[i] = next;
			if (ctx.opt.forgetChange && s === 100) Q.data[0] += 0.25; // the bug: an in-place edit with no change entry
			yield {
				phase: 'update',
				step: s,
				state: {},
				changes: [{ field: 'Q', index: [pos, action], from: old, to: next }],
				metrics: { r, q: next, alpha: ctx.opt.alpha }
			};
		}
	}
};
