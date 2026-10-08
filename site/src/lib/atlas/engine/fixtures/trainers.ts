// Test fixtures for the engine: a small classifier trainer (record-valued params, grads, Adam moments, batch, acts) and
// an "arena" process that yields subarray views of one big buffer and mutates a live 'changes' table in place, so the
// transfer rules (exact-size copies, nothing detached, copies made at yield time) are exercised.
import type { Control, Frame, ProcessDef, RunCtx, Tensor, TrainOpt, TrainerDef } from '../../types/index.ts';
import { input, type Var } from '../../autograd/var.ts';
import { crossEntropy } from '../../autograd/loss.ts';
import { MLP } from '../../autograd/nn.ts';
import { moons, sampleBatch, type Points2D } from '../../data/points2d.ts';
import { createRng } from '../../core/rng.ts';

export type ClsBatch = { x: Var; y: Int32Array };

export const DATA: Points2D = moons(64, createRng(1));

export const clsTrainer: TrainerDef<Points2D, MLP, ClsBatch, TrainOpt> = {
	id: 'cls',
	title: { en: 'Two moons', zh: '两个月牙' },
	defaults: { seed: 1, lr: 0.05, batch: 16, steps: 60, snapshotEvery: 5, optimizer: 'adam' },
	model: (rng) => new MLP('net', [2, 8, 8, 2], 'tanh', rng),
	batch(data, _step, rng, opt) {
		const b = sampleBatch(data, opt.batch, rng);
		return { x: input(b.x), y: b.label };
	},
	loss(m, b, rec) {
		return { loss: crossEntropy(m.forward(b.x, rec), b.y) };
	}
};

export type ArenaOpt = { steps: number };
export type ArenaState = { view: Tensor; table: Tensor; total: number; twice: Tensor };

/** Yields subarray views of a 4096-double arena (and the same view twice in one frame), a 'changes'-tracked table that it
 *  mutates right after yielding, and a derived field. Throws if anything detached its arena. */
export const arenaProcess: ProcessDef<null, ArenaOpt, ArenaState> = {
	id: 'arena',
	kind: 'algorithm',
	title: { en: 'Arena views', zh: '缓冲区视图' },
	phases: { tick: { label: { en: 'Tick', zh: '一步' }, kind: 'update' } },
	schema: {
		view: { kind: 'tensor', label: { en: 'View', zh: '视图' } },
		twice: { kind: 'tensor', label: { en: 'Same view', zh: '同一视图' } },
		table: { kind: 'tensor', label: { en: 'Table', zh: '表' }, track: 'changes' },
		total: { kind: 'scalar', label: { en: 'Total', zh: '总和' }, derive: (s: ArenaState) => {
			if (!s.table) return NaN;
			let t = 0;
			for (const v of s.table.data) t += v;
			return t;
		} }
	},
	defaults: { steps: 50 },
	*run(_input: null, ctx: RunCtx<ArenaOpt>): Generator<Frame<ArenaState>, void, Control<ArenaOpt> | undefined> {
		const arena = new Float64Array(4096);
		const table: Tensor = { shape: [4], data: new Float64Array(4) };
		for (let k = 0; k < ctx.opt.steps && !ctx.signal.cancelled; k++) {
			if (arena.buffer.byteLength !== 4096 * 8) throw new Error('the arena was detached');
			for (let i = 0; i < 8; i++) arena[k * 8 + i] = ctx.rng.next();
			const view: Tensor = { shape: [8], data: arena.subarray(k * 8, k * 8 + 8) };
			const cell = ctx.rng.int(4);
			const from = table.data[cell];
			table.data[cell] = from + 1;
			yield { phase: 'tick', step: k, state: { view, twice: view, table }, changes: [{ field: 'table', index: [cell], from, to: from + 1 }], metrics: { cell } };
			// mutate the live objects right after yielding: a packer that copied late would send these values
			arena.fill(-1, k * 8, k * 8 + 8);
		}
	}
};
