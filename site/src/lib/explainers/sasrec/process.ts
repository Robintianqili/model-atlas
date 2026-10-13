// sasrec process (kind C): logs, the training logs, made by world.ts. Frame 0: the click histories of the fixed inputs
// (presets and non-empty edges: export.py's reference cases, so parity runs on the ids the page uses), −1 pads. Then one
// frame per chunk of sessions: their product ids in click order [c, 13], −1 pads. Session s depends only on s
// (world.ts session). `npm run atlas -- run sasrec logs --opt '{"n":10000}' --save` writes what train.py reads.
import type { ProcessDef, ProcessRun } from '../../atlas/types/index.ts';
import { FIXED, MAX_CLICKS, MAX_SESSION, session } from './world.ts';

const I18 = (en: string, zh: string) => ({ en, zh });
type I32 = { shape: number[]; data: Int32Array; dtype: 'i32' };
export type LogsOpt = { n: number; chunk: number };
type LogsState = { fixed: I32; seqs: I32 };
const sum = (a: ArrayLike<number>) => Array.from(a).reduce((x, y) => x + y, 0);
const i32 = (shape: number[], data: Int32Array): I32 => ({ shape, data, dtype: 'i32' });

export const logs: ProcessDef<unknown, LogsOpt, LogsState> = {
	id: 'logs',
	kind: 'algorithm',
	title: I18('Write the training logs', '生成训练日志'),
	phases: { fixed: { label: I18('The fixed inputs', '固定输入'), kind: 'input' }, chunk: { label: I18('A chunk of sessions', '一批会话'), kind: 'input' } },
	schema: {
		fixed: { kind: 'tensor', label: I18('Clicks of the fixed inputs (−1 pads)', '固定输入的点击（−1 是填充）'), role: 'data', tol: 'exact' },
		seqs: { kind: 'tensor', label: I18('Sessions: product ids in click order (−1 pads)', '会话：按点击顺序的商品编号（−1 是填充）'), role: 'data', tol: 'exact' }
	},
	defaults: { n: 24, chunk: 1000 },
	*run(_input, ctx): ProcessRun<LogsState, LogsOpt> {
		const F = FIXED.length, fx = new Int32Array(F * MAX_CLICKS).fill(-1);
		FIXED.forEach((f, k) => fx.set(f.raw.clicks, k * MAX_CLICKS));
		yield { phase: 'fixed', state: { fixed: i32([F, MAX_CLICKS], fx) } as LogsState, metrics: { count: F, sum: sum(fx) } };
		const { n, chunk } = ctx.opt;
		for (let from = 0, k = 0; from < n && !ctx.signal.cancelled; from += chunk, k++) {
			const c = Math.min(chunk, n - from), seqs = new Int32Array(c * MAX_SESSION).fill(-1);
			let clicks = 0;
			for (let r = 0; r < c; r++) {
				const s = session(from + r);
				seqs.set(s, r * MAX_SESSION);
				clicks += s.length;
			}
			yield { phase: 'chunk', step: k, state: { seqs: i32([c, MAX_SESSION], seqs) } as LogsState, metrics: { from, count: c, clicks, sum: sum(seqs) } };
		}
	}
};
