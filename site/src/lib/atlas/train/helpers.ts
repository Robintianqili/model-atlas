// Narration helpers for training tapes (design §4.5, W1-B1).
import type { At, Tensor, TrainState } from '../types/index.ts';

export type UpdateVals = { old: number; g: number; lr: number; new: number };

function flatIndex(shape: number[], index: number | number[]): number {
	if (typeof index === 'number') {
		if (index < 0 || index >= shape.reduce((p, d) => p * d, 1)) throw new Error(`updateOf: index ${index} out of range for [${shape}]`);
		return index;
	}
	if (index.length !== shape.length) throw new Error(`updateOf: index [${index}] does not match shape [${shape}]`);
	return index.reduce((acc, k, d) => {
		if (k < 0 || k >= shape[d]) throw new Error(`updateOf: index [${index}] out of range for [${shape}]`);
		return acc * shape[d] + k;
	}, 0);
}

/**
 * { old, g, lr, new } for one weight at an 'update' frame of a fine-detail train source, so a narration can say
 * "old − lr × g = new" with the numbers that ran. It throws unless that rule is what ran: plain SGD without weight decay
 * (Adam and momentum move weights differently, so narration cannot claim the SGD rule for them).
 */
export function updateOf(at: At, key: string, index: number | number[]): UpdateVals {
	const s = at.state as Partial<TrainState> | undefined;
	if (!s) throw new Error('updateOf: no state at this cursor');
	const optimizer = s.optimizer;
	if (optimizer !== 'sgd') {
		throw new Error(`updateOf: this source trains with '${String(optimizer)}'; "old − lr × g = new" only describes plain SGD`);
	}
	const prev = (s.prev as Record<string, Tensor> | undefined)?.[key];
	const cur = s.params?.[key];
	const g = s.grads?.[key];
	if (!prev || !cur || !g) throw new Error(`updateOf: needs an 'update' frame of a fine-detail train source with '${key}' in prev, params and grads`);
	const wd = at.frame?.metrics?.weightDecay ?? 0;
	if (wd !== 0) throw new Error(`updateOf: weight decay ${wd} is on, so the update is not old − lr × g`);
	const lr = typeof s.lr === 'number' ? s.lr : (at.frame?.metrics?.lr ?? NaN);
	const i = flatIndex(cur.shape, index);
	const old = prev.data[i];
	const gi = g.data[i];
	const nw = cur.data[i];
	const expect = old + -lr * gi;
	if (!(Math.abs(expect - nw) <= 1e-12 * Math.max(1, Math.abs(nw)))) {
		throw new Error(`updateOf: '${key}'[${index}] went ${old} → ${nw}, but old − lr × g = ${expect} (lr ${lr}); the frames do not belong to one SGD step`);
	}
	return { old, g: gi, lr, new: nw };
}
