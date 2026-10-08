// Shared helpers for the autograd tests (W1-B1): fixture loading, the op dispatcher matching
// pipeline/atlas_fixtures/autograd/make.py case names, and the project's relative error.
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Var } from '../../../src/lib/atlas/autograd/var.ts';
import * as op from '../../../src/lib/atlas/autograd/ops.ts';
import { bceWithLogits, crossEntropy, infoNCE, mse } from '../../../src/lib/atlas/autograd/loss.ts';
import { lstmCell } from '../../../src/lib/atlas/autograd/nn.ts';

export const FIXTURES = dirname(fileURLToPath(import.meta.url));

export function loadFixture<T = unknown>(name: string): T {
	const text = readFileSync(join(FIXTURES, name), 'utf8');
	return JSON.parse(text.charCodeAt(0) === 0xfeff ? text.slice(1) : text) as T;
}

export interface J {
	shape: number[];
	data: number[];
}

export interface OpCase {
	name: string;
	op: string;
	attrs: Record<string, unknown>;
	inputs: J[];
	y: J;
	r: J;
	grads: J[];
	v: J[];
	ggrads: J[];
}

/** max over i of |a_i − b_i| / max(1, |b_i|); NaN or a length mismatch gives Infinity. */
export function relErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
	if (a.length !== b.length) return Infinity;
	let m = 0;
	for (let i = 0; i < a.length; i++) {
		const e = Math.abs(a[i] - b[i]) / Math.max(1, Math.abs(b[i]));
		if (Number.isNaN(e)) return Infinity;
		if (e > m) m = e;
	}
	return m;
}

type A = Record<string, unknown>;
const n = (v: unknown) => v as number;

/** The TS call for each fixture op (same names and attrs as make.py). */
export const DISPATCH: Record<string, (xs: Var[], a: A) => Var> = {
	add: ([a, b]) => op.add(a, b),
	sub: ([a, b]) => op.sub(a, b),
	mul: ([a, b]) => op.mul(a, b),
	div: ([a, b]) => op.div(a, b),
	neg: ([a]) => op.neg(a),
	scale: ([a], o) => op.scale(a, n(o.k)),
	pow: ([a], o) => op.pow(a, n(o.p)),
	square: ([a]) => op.square(a),
	sqrt: ([a]) => op.sqrt(a),
	exp: ([a]) => op.exp(a),
	log: ([a]) => op.log(a),
	tanh: ([a]) => op.tanh(a),
	sigmoid: ([a]) => op.sigmoid(a),
	relu: ([a]) => op.relu(a),
	silu: ([a]) => op.silu(a),
	gelu: ([a], o) => op.gelu(a, (o.approx as 'none' | 'tanh') ?? 'none'),
	erf: ([a]) => op.erf(a),
	sin: ([a]) => op.sin(a),
	cos: ([a]) => op.cos(a),
	abs: ([a]) => op.abs(a),
	matmul: ([a, b]) => op.matmul(a, b),
	linear: ([x, w, b]) => op.linear(x, w, b ?? null),
	sum: ([a], o) => op.sum(a, o.axis as number | number[] | null, o.keepdim as boolean),
	mean: ([a], o) => op.mean(a, o.axis as number | number[] | null, o.keepdim as boolean),
	reshape: ([a], o) => op.reshape(a, o.shape as number[]),
	transpose: ([a], o) => op.transpose(a, n(o.dim0), n(o.dim1)),
	broadcastTo: ([a], o) => op.broadcastTo(a, o.shape as number[]),
	sumTo: ([a], o) => op.sumTo(a, o.shape as number[]),
	concat: (xs, o) => op.concat(xs, n(o.axis)),
	stack: (xs, o) => op.stack(xs, n(o.axis)),
	slice: ([a], o) => op.slice(a, n(o.axis), n(o.start), o.end as number | undefined),
	gather: ([a], o) => op.gather(a, n(o.axis), o.index as number[]),
	scatterAdd: ([g], o) => op.scatterAdd(g, n(o.axis), o.index as number[], n(o.n)),
	gatherTanh: ([w], o) => op.tanh(op.gather(w, n(o.axis), o.index as number[])),
	logsumexp: ([a], o) => op.logsumexp(a, n(o.axis), o.keepdim as boolean),
	softmax: ([a], o) => op.softmax(a, n(o.axis)),
	logSoftmax: ([a], o) => op.logSoftmax(a, n(o.axis)),
	conv1d: ([x, w, b], o) => op.conv1d(x, w, b ?? null, o as op.Conv1dOpts),
	layerNorm: ([x, w, b], o) => op.layerNorm(x, w ?? null, b ?? null, n(o.eps)),
	mse: ([a, b]) => mse(a, b),
	crossEntropy: ([z], o) => crossEntropy(z, o.targets as number[]),
	bceWithLogits: ([z, y]) => bceWithLogits(z, y),
	infoNCE: ([L], o) => infoNCE(L, { symmetric: o.symmetric as boolean }),
	lstmcell: ([x, h0, c0, wih, whh, bih, bhh]) => {
		const s = lstmCell(x, { h: h0, c: c0 }, { weight_ih: wih, weight_hh: whh, bias_ih: bih, bias_hh: bhh });
		return op.concat([s.h, s.c], -1);
	}
};
