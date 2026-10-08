// Losses (design §4.5, W1-B1). Mean reduction, PyTorch semantics, built from differentiable ops (double grads work).
import type { Tensor } from '../types/index.ts';
import { Var, input, raw } from './var.ts';
import * as op from './ops.ts';

function asVar(t: Var | Tensor | number[]): Var {
	if (t instanceof Var) return t;
	if (Array.isArray(t)) return input({ shape: [t.length], data: t });
	return input(t);
}

/** Mean squared error: mean((pred − target)²) (torch.nn.functional.mse_loss). */
export function mse(pred: Var, target: Var | Tensor | number[]): Var {
	const d = op.sub(pred, asVar(target));
	return op.mean(op.mul(d, d));
}

/**
 * Cross-entropy from logits [N, C] (or [C] with one target) and integer class targets:
 * −mean_i log softmax(z_i)[t_i] (torch.nn.functional.cross_entropy).
 */
export function crossEntropy(logits: Var, targets: ArrayLike<number>): Var {
	const z = logits.shape.length === 1 ? op.reshape(logits, [1, logits.shape[0]]) : logits;
	if (z.shape.length !== 2) throw new Error(`crossEntropy: logits must be [N, C], got [${logits.shape}]`);
	const [N, C] = z.shape;
	if (targets.length !== N) throw new Error(`crossEntropy: ${targets.length} targets for ${N} rows`);
	const onehot = new Float64Array(N * C);
	for (let i = 0; i < N; i++) {
		const t = targets[i];
		if (!Number.isInteger(t) || t < 0 || t >= C) throw new Error(`crossEntropy: target ${t} is not a class index in [0, ${C})`);
		onehot[i * C + t] = 1;
	}
	const picked = op.sum(op.mul(op.logSoftmax(z, -1), raw(onehot, [N, C])), -1);
	return op.neg(op.mean(picked));
}

/**
 * Binary cross-entropy from logits: mean(max(z, 0) − z·y + log(1 + e^{−|z|}))
 * (torch.nn.functional.binary_cross_entropy_with_logits, written in its stable form).
 */
export function bceWithLogits(logits: Var, target: Var | Tensor | number[]): Var {
	const y = asVar(target);
	const softplusNeg = op.log(op.add(op.exp(op.neg(op.abs(logits))), 1));
	return op.mean(op.add(op.sub(op.relu(logits), op.mul(logits, y)), softplusNeg));
}

/**
 * InfoNCE (CLIP) on an N×N logit matrix whose diagonal holds the matching pairs:
 * symmetric (default): (CE(L, arange) + CE(Lᵀ, arange)) / 2; otherwise CE over rows only.
 */
export function infoNCE(logits: Var, o: { symmetric?: boolean } = {}): Var {
	if (logits.shape.length !== 2 || logits.shape[0] !== logits.shape[1]) throw new Error(`infoNCE: logits must be N×N, got [${logits.shape}]`);
	const N = logits.shape[0];
	const idx = Array.from({ length: N }, (_, i) => i);
	const rows = crossEntropy(logits, idx);
	if (o.symmetric === false) return rows;
	const cols = crossEntropy(op.transpose(logits), idx);
	return op.scale(op.add(rows, cols), 0.5);
}
