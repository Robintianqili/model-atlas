// mlp-playground: processes made of trainingProcess(fit) frames (nothing is computed another way).
//   play    one run of 'fit' as a process source: the runtime re-opens process sources when the learner changes the
//           points, so the lesson retrains (a finished 'train' source keeps its old weights: a kit issue, see the
//           notes); no points = no frames (the stage shows the empty note). opt.fine records forward / loss / backward /
//           update frames; init frames carry the activation (field 'act') for the derived heatmaps.
//   lrPair  two runs in lockstep from the same weights at learning rates opt.lr and opt.lrHi: the first run's frames,
//           with the second's loss and accuracy as the metrics hiLoss and hiAccuracy.
import type { Control, ProcessDef, ProcessRun, TrainState } from '../../atlas/types/index.ts';
import { trainingProcess } from '../../atlas/train/trainer.ts';
import { fit, type FitOpt } from './trainer.ts';
import { dataOf } from './net.ts';

const inner = trainingProcess(fit);
type PlayOpt = FitOpt & { fine?: boolean };
type PairOpt = FitOpt & { lrHi: number };

export const play: ProcessDef<unknown, PlayOpt, TrainState> = {
	...inner,
	id: 'play',
	schema: { ...inner.schema, act: { kind: 'text', label: { en: 'Activation', zh: '激活函数' } } },
	defaults: { ...fit.defaults, fine: false },
	*run(input, ctx): ProcessRun<TrainState, PlayOpt> {
		const opt = { ...fit.defaults, ...ctx.opt } as PlayOpt;
		if (dataOf(input, opt).n === 0) return;
		const g = inner.run(input, { ...ctx, detail: opt.fine ? 'fine' : ctx.detail });
		let c: Control<PlayOpt> | undefined;
		for (let r = g.next(); !r.done; r = g.next(c)) {
			if (r.value.phase === 'init') r.value.state.act = opt.act;
			c = yield r.value;
		}
	}
};

export const lrPair: ProcessDef<unknown, PairOpt, TrainState> = {
	...inner,
	id: 'lrPair',
	title: { en: 'The same training at two learning rates', zh: '同样的训练，两种学习率' },
	defaults: { ...fit.defaults, lrHi: 3 },
	*run(input, ctx): ProcessRun<TrainState, PairOpt> {
		const opt = { ...fit.defaults, ...ctx.opt } as PairOpt;
		if (dataOf(input, opt).n === 0) return;
		const a = inner.run(input, { ...ctx, detail: 'coarse' });
		const b = inner.run(input, { ...ctx, detail: 'coarse', opt: { ...ctx.opt, lr: opt.lrHi } });
		for (let ra = a.next(); !ra.done; ra = a.next()) {
			const rb = b.next();
			const m = rb.done ? undefined : rb.value.metrics;
			if (m?.loss !== undefined) ra.value.metrics = { ...ra.value.metrics, hiLoss: m.loss };
			if (m?.accuracy !== undefined) ra.value.metrics = { ...ra.value.metrics, hiAccuracy: m.accuracy };
			yield ra.value;
		}
	}
};
