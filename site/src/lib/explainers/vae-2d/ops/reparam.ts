// vae-2d local op 'reparam' (the reparameterization trick): z = mu + sigma · eps with sigma = exp(½ · logvar), cell by
// cell. Inputs [mu, logvar, eps], all [Z]; output z [Z]; sub-step 'sigma' [Z]. The randomness is the input eps (drawn
// outside the model, from N(0, 1)), so the op itself is plain arithmetic and parity holds as for any layer.
// Python mirror: pipeline/vae-2d/ops_reparam.py (mu + torch.exp(0.5 * logvar) * eps, as model.py's forward()).
import type { OpDef, Tensor, UnitExplanation } from '../../../atlas/types/index.ts';
import { flatIndex } from '../../../atlas/core/tensor.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, outArray, store, t, unitBase } from '../../../atlas/ops/common.ts';

type NoP = Record<string, never>;

export const reparam: OpDef<NoP> = {
	op: 'reparam',
	version: 1,
	attrs: {},
	category: 'merge',
	label: t('Sample z = μ + σ·ε', '采样 z = μ + σ·ε'),
	subs: [{ name: 'sigma', label: t('σ = exp(½ · log σ²)', 'σ = exp(½ · log σ²)') }],
	inferShape(ins) {
		if (ins.length !== 3) throw new Error('reparam: needs three inputs (mu, logvar, eps)');
		if (ins.some((s) => s.length !== 1 || s[0] !== ins[0][0])) throw new Error('reparam: mu, logvar and eps must be vectors of one length');
		return ins[0].slice();
	},
	axes: () => ['f'],
	forward(xs, _p, ctx) {
		const [mu, lv, eps] = xs;
		const n = mu.data.length;
		const z = outArray(n, mu, lv, eps);
		const s = outArray(n, mu, lv, eps);
		for (let i = 0; i < n; i++) {
			const sigma = Math.exp(0.5 * lv.data[i]);
			s[i] = sigma;
			z[i] = mu.data[i] + sigma * eps.data[i];
		}
		ctx.emit('sigma', { shape: [n], data: s });
		return { shape: [n], data: z };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'first' },
	explain(q): UnitExplanation {
		const f = flatIndex(q.out.shape, q.unit);
		const [mu, lv, eps] = (q.ins as Tensor[]).map((x) => x.data[f]);
		const sigma = Math.exp(0.5 * lv);
		const z = store(mu + sigma * eps, q.out);
		const e = unitBase('elementwise', q.layer, q.unit, z, q.out.data[f]);
		e.reads = q.layer.in.map((name) => ({ tensor: name, cells: [f] }));
		// shown as a sum of two parts: mu, and eps scaled by sigma
		e.terms = [
			{ x: mu, w: 1, p: mu },
			{ x: eps, w: sigma, p: sigma * eps }
		];
		e.values = { i: f, mu, logvar: lv, sigma: store(sigma, q.out), eps, a: mu, b: sigma * eps, z };
		return e;
	},
	formula: `${sym('out', 'z')} = ${sym('in', '\\mu')} + ${sym('in', '\\sigma')}\\,${sym('in', '\\varepsilon')} = {{mu}} + {{sigma}} \\times {{eps}} = {{z}}`,
	narration: {
		what: t(
			'The encoder does not give one point but a small cloud: a centre μ and a spread σ = exp(½ · log σ²) for each of the {n|int} latent numbers. To pick one point z from that cloud, the model draws ε from a standard bell curve (mostly between −2 and 2) and computes z = μ + σ · ε.',
			'编码器给出的不是一个点，而是一小团云：对 {n|int} 个隐变量中的每一个，有一个中心 μ 和一个散布 σ = exp(½ · log σ²)。为了从这团云里取出一个点 z，模型从标准钟形曲线里抽一个 ε（大多在 −2 到 2 之间），再算 z = μ + σ · ε。'
		),
		how: t('Each number of z is its μ plus its σ times its ε.', 'z 的每个数都等于它的 μ 加上它的 σ 乘以它的 ε。'),
		why: t(
			'The randomness lives in ε alone, so z changes smoothly with μ and σ, and training can still adjust the encoder through them (the reparameterization trick).',
			'随机性只在 ε 里，所以 z 随 μ 和 σ 平滑地变化，训练仍然能通过它们调整编码器（重参数化技巧）。'
		),
		look: t('z lands near μ: how far depends on σ and on the draw ε.', 'z 落在 μ 附近：离多远取决于 σ 和这次抽到的 ε。'),
		unit: t('z = {mu} + {sigma} × {eps} = {z}.', 'z = {mu} + {sigma} × {eps} = {z}。')
	},
	vars: (layer, spec) => ({ ...baseVars(layer, spec), n: typeof layer.outShape[0] === 'number' ? layer.outShape[0] : '?' }),
	zoom: 'elementwise',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'z = mu + torch.exp(0.5 * logvar) * eps',
	terms: ['reparameterization']
};

export default reparam;
