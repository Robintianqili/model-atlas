// adversarial-tinycnn: process 'attack' (kind C). FGSM and PGD on the learner's drawing against TinyCNN, one frame per
// event, every number computed here in float64 on the autograd core (net.ts):
//   clean  the drawing, its class y (TinyCNN's own answer) and the loss L = −log p(y)
//   grad   g = dL/dx, one number per pixel (reverse mode through the network back to the input)
//   sign   s = sign(g): +1, −1 or 0 per pixel
//   fgsm   x' = clip(x + ε·s, 0, 1) and TinyCNN's answer on it
//   pgd    k = 1..K: x_k = clip(x_{k−1} + α·sign(dL/dx at x_{k−1}), max(x − ε, 0), min(x + ε, 1)), α = ε/4, no random start
//   scan   FGSM's x' for every ε on a grid (0.005 to 0.3) with the same s: the smallest ε that flips the answer
// ε comes from the learner's raw input ({ ..., eps }, set by the stage's slider), else EPS. pipeline/adversarial-tinycnn/
// reference.py is the same process in PyTorch float64 (torch.autograd): `atlas parity` compares the frames.
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { N, argmax, logitsOf, lossGrad, softmax, type Weights } from './net.ts';

/** the default ε (pixel values run from 0 to 1) */
export const EPS = 0.1;
export type AttackIn = { x: ArrayLike<number>; eps: number; w: Weights | null };
export type AttackOpt = { steps: number; alphaFrac: number; scanStep: number; scanMax: number };
export type AttackState = { x: Tensor; p0: Tensor; g: Tensor; s: Tensor; adv: Tensor; p: Tensor; scanP: Tensor; scanTop: Tensor };

/** ε of a raw input (the stage's slider writes it), else the default */
export function epsOf(raw: unknown): number {
	const e = (raw as { eps?: unknown } | null)?.eps;
	return typeof e === 'number' && Number.isFinite(e) && e >= 0 ? e : EPS;
}

const img = (d: ArrayLike<number>): Tensor => ({ shape: [1, 28, 28], data: Float64Array.from(d) });
const vec = (d: ArrayLike<number>): Tensor => ({ shape: [d.length], data: Float64Array.from(d) });
const T = (en: string, zh: string) => ({ en, zh });

/** the loss −log p(y) from logits, as cross-entropy computes it (log-sum-exp minus the logit) */
function ce(z: ArrayLike<number>, y: number): number {
	let m = -Infinity;
	for (let i = 0; i < z.length; i++) m = Math.max(m, z[i]);
	let s = 0;
	for (let i = 0; i < z.length; i++) s += Math.exp(z[i] - m);
	return m + Math.log(s) - z[y];
}

/** TinyCNN's answer on x: probabilities, top class, its probability, p(y) and the loss for y */
function judge(x: ArrayLike<number>, y: number, w: Weights) {
	const z = logitsOf(x, w), p = softmax(z), top = argmax(p);
	return { p, top, ptop: p[top], py: p[y], loss: ce(z, y), flipped: top === y ? 0 : 1 };
}

export const attack: ProcessDef<AttackIn, AttackOpt, AttackState> = {
	id: 'attack',
	kind: 'algorithm',
	title: T('FGSM and PGD on your drawing', '在你的画上做 FGSM 和 PGD'),
	phases: {
		clean: { label: T('Your drawing and the loss', '你的画和损失'), kind: 'loss', formula: { tex: 'L = -\\log p(y) = -\\log {{py}} = {{loss}}', values: (f) => ({ py: f.metrics!.py, loss: f.metrics!.loss }) } },
		grad: { label: T('The gradient for every pixel', '每个像素的梯度'), kind: 'backward' },
		sign: { label: T('Keep only the sign', '只留下符号'), kind: 'other' },
		fgsm: {
			label: T('One step of size ε (FGSM)', '走一步，步长 ε（FGSM）'),
			kind: 'update',
			formula: { tex: "x' = \\operatorname{clip}_{[0,1]}\\big(x + \\varepsilon \\cdot \\operatorname{sign}(\\nabla_x L)\\big),\\quad \\varepsilon = {{eps}}", values: (f) => ({ eps: f.metrics!.eps }) }
		},
		pgd: { label: T('PGD: small steps, clipped', 'PGD：小步走，再裁剪'), kind: 'update' },
		scan: { label: T('Every ε at once', '一次试遍所有 ε'), kind: 'observe' }
	},
	schema: {
		x: { kind: 'tensor', label: T('Your drawing x', '你的画 x'), shape: [1, 28, 28], role: 'data', tol: 'exact' },
		p0: { kind: 'tensor', label: T('Class shares on your drawing', '你的画上的类别份额'), shape: [10], role: 'output', scale: 'sequential', tol: 1e-9 },
		g: { kind: 'tensor', label: T('Gradient of the loss for every pixel', '损失对每个像素的梯度'), shape: [1, 28, 28], role: 'grad', scale: 'diverging', tol: 1e-9 },
		s: { kind: 'tensor', label: T('Its sign', '它的符号'), shape: [1, 28, 28], role: 'grad', scale: 'diverging', tol: 'exact' },
		adv: { kind: 'tensor', label: T('The changed drawing x′', '改过的画 x′'), shape: [1, 28, 28], role: 'data', tol: 1e-12 },
		p: { kind: 'tensor', label: T('Class shares on the changed drawing', '改过的画上的类别份额'), shape: [10], role: 'output', scale: 'sequential', tol: 1e-9 },
		scanP: { kind: 'tensor', label: T('p(y) for every ε', '每个 ε 下的 p(y)'), shape: ['E'], role: 'output', tol: 1e-9 },
		scanTop: { kind: 'tensor', label: T('The answer for every ε', '每个 ε 下的答案'), shape: ['E'], role: 'output', tol: 'exact' }
	},
	defaults: { steps: 10, alphaFrac: 0.25, scanStep: 0.005, scanMax: 0.3 },
	metricTol: 1e-9,
	*run(inp, ctx): ProcessRun<AttackState, AttackOpt> {
		const w = inp.w;
		if (!w) throw new Error('adversarial-tinycnn: the TinyCNN weights are not loaded');
		const x = Float64Array.from(inp.x);
		if (x.length !== N) throw new Error(`adversarial-tinycnn: expected ${N} pixels, got ${x.length}`);
		const eps = inp.eps;
		const { steps, alphaFrac, scanStep, scanMax } = ctx.opt;
		const alpha = alphaFrac * eps;

		const z0 = logitsOf(x, w), p0 = softmax(z0), y = argmax(p0), r = argmax(p0, y);
		const { loss, g } = lossGrad(x, y, w);
		let ink = 0;
		for (const v of x) if (v > 0) ink++;
		yield { phase: 'clean', step: 0, state: { x: img(x), p0: vec(p0), p: vec(p0) }, metrics: { y, py: p0[y], loss, r, pr: p0[r], eps, ink } };

		let gmax = 0, at = 0, sum = 0, onInk = 0;
		for (let i = 0; i < N; i++) {
			const a = Math.abs(g[i]);
			sum += a;
			if (x[i] > 0) onInk += a;
			if (a > gmax) (gmax = a), (at = i);
		}
		yield { phase: 'grad', step: 0, state: { g: img(g) }, metrics: { gmax, row: Math.floor(at / 28), col: at % 28, gAt: g[at], onInk: sum > 0 ? onInk / sum : 0, inkShare: ink / N } };

		const s = Float64Array.from(g, Math.sign);
		let nPos = 0, nNeg = 0;
		for (const v of s) v > 0 ? nPos++ : v < 0 && nNeg++;
		yield { phase: 'sign', step: 0, state: { s: img(s) }, metrics: { nPos, nNeg, nZero: N - nPos - nNeg } };

		const adv = new Float64Array(N);
		let clipped = 0, moved = 0;
		for (let i = 0; i < N; i++) {
			const v = x[i] + eps * s[i];
			adv[i] = Math.min(1, Math.max(0, v));
			if (adv[i] !== v) clipped++;
			moved += Math.abs(adv[i] - x[i]);
		}
		const f = judge(adv, y, w);
		yield { phase: 'fgsm', step: 0, state: { adv: img(adv), p: vec(f.p) }, metrics: { eps, y, top: f.top, ptop: f.ptop, py: f.py, loss: f.loss, flipped: f.flipped, clipped, meanMove: moved / N } };

		let xk = x, first = 0;
		for (let k = 1; k <= steps; k++) {
			const gk = k === 1 ? g : lossGrad(xk, y, w).g;
			const next = new Float64Array(N);
			for (let i = 0; i < N; i++) next[i] = Math.min(Math.min(x[i] + eps, 1), Math.max(Math.max(x[i] - eps, 0), xk[i] + alpha * Math.sign(gk[i])));
			xk = next;
			const j = judge(xk, y, w);
			if (j.flipped && !first) first = k;
			yield { phase: 'pgd', step: k, state: { adv: img(xk), p: vec(j.p) }, metrics: { k, steps, eps, alpha, y, top: j.top, ptop: j.ptop, py: j.py, loss: j.loss, flipped: j.flipped, firstFlip: first } };
		}

		const n = Math.round(scanMax / scanStep);
		const scanP = new Float64Array(n), scanTop = new Float64Array(n);
		let minEps = -1;
		for (let k = 1; k <= n; k++) {
			const e = k * scanStep, xe = new Float64Array(N);
			for (let i = 0; i < N; i++) xe[i] = Math.min(1, Math.max(0, x[i] + e * s[i]));
			const j = judge(xe, y, w);
			scanP[k - 1] = j.py;
			scanTop[k - 1] = j.top;
			if (j.flipped && minEps < 0) minEps = e;
		}
		yield { phase: 'scan', step: 0, state: { scanP: vec(scanP), scanTop: vec(scanTop) }, metrics: { eps, y, minEps, found: minEps > 0 ? 1 : 0, scanStep, scanMax, flipped: f.flipped, pgdFlip: first } };
	}
};
