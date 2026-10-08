// ddpm-unet local op 'time_embed': the timestep t [1] as dim numbers [dim, 1, 1] (one number per channel, so 1×1
// convolutions can mix them and `add` can put them on every pixel of a feature map): sin(t·f_k) for the first half,
// cos(t·f_k) for the second, f_k = 10000^(−k / half) (DDPM's sinusoidal embedding). Computed in f64, stored as float32.
// Python mirror: pipeline/ddpm-unet/ops_time_embed.py (model.py time_embed).
import type { OpDef, UnitExplanation } from '../../../atlas/types/index.ts';
import { sym } from '../../../atlas/core/tex.ts';
import { baseVars, outArray, store, t, unitBase } from '../../../atlas/ops/common.ts';

type P = { dim: number };

/** channel i of the embedding of t: which function, the frequency f, the angle t·f and the value */
export function wave(tv: number, i: number, dim: number) {
	const half = dim / 2, k = i % half;
	const f = Math.exp((-Math.log(10000) * k) / half);
	const arg = tv * f;
	return { k, f, arg, cos: i >= half ? 1 : 0, y: i >= half ? Math.cos(arg) : Math.sin(arg) };
}

export const timeEmbed: OpDef<P> = {
	op: 'time_embed',
	version: 1,
	attrs: { dim: { type: 'int', required: true, doc: 'number of channels (even): dim / 2 sines, then dim / 2 cosines' } },
	category: 'embedding',
	label: t('Timestep as waves (sin, cos)', '把时间步变成波（sin、cos）'),
	inferShape(ins, p) {
		if (ins.length !== 1 || ins[0].length !== 1) throw new Error('time_embed: input must be t of shape [1]');
		if (!(p.dim > 0 && p.dim % 2 === 0)) throw new Error('time_embed: dim must be a positive even number');
		return [p.dim, 1, 1];
	},
	axes: () => ['c', 'y', 'x'],
	forward(xs, p) {
		const y = outArray(p.dim, xs[0]);
		for (let i = 0; i < p.dim; i++) y[i] = wave(xs[0].data[0], i, p.dim).y;
		return { shape: [p.dim, 1, 1], data: y };
	},
	unit: { axes: (out) => out.slice(), scan: 'none', pick: 'first' },
	explain(q): UnitExplanation {
		const i = q.unit[0];
		const w = wave(q.ins[0].data[0], i, q.p.dim);
		const y = store(w.y, q.out);
		const e = unitBase('none', q.layer, q.unit, y, q.out.data[i]);
		e.reads = [{ tensor: q.layer.in[0], cells: [0] }];
		e.values = { t: q.ins[0].data[0], i, k: w.k, half: q.p.dim / 2, f: w.f, arg: w.arg, cos: w.cos, y };
		return e;
	},
	formula: `f_k = 10000^{-k/{{half}}} = {{f}},\\quad ${sym('in', 't')}\\, f_k = {{t}} \\times {{f}} = {{arg}},\\quad ${sym('out', 'e')}_{ {{i}} } = {{y}}`,
	narration: {
		what: t(
			'The network must know how noisy its input is, so the step number t becomes {outShape} numbers: sines of t at {half|int} speeds, then cosines at the same speeds.',
			'网络必须知道输入里有多少噪声，所以步数 t 被变成 {outShape} 个数：t 在 {half|int} 种速度下的正弦，再加上同样速度下的余弦。'
		),
		how: t('Channel k (and k + {half|int}) turns t·f_k into a wave, f_k = 10000^(−k/{half|int}): fast waves for small k, slow ones for large k.', '第 k 个通道（以及第 k + {half|int} 个）把 t·f_k 变成波，f_k = 10000^(−k/{half|int})：k 小的波快，k 大的波慢。'),
		why: t('Nearby steps get similar numbers and far ones different numbers, which a layer can read easily.', '相近的步数得到相近的数，相距远的步数得到不同的数，下一层很容易读懂。'),
		look: t('Each number is between −1 and 1.', '每个数都在 −1 和 1 之间。'),
		unit: t('Channel {i|int}: angle t·f = {arg}, value {y}.', '第 {i|int} 个通道：角度 t·f = {arg}，值 {y}。')
	},
	vars: (layer, spec) => ({ ...baseVars(layer, spec), half: Number(layer.params?.dim ?? NaN) / 2 }),
	zoom: 'none',
	color: 'diverging',
	parity: { bar: 1e-4 },
	pytorch: () => 'torch.cat([torch.sin(t * f), torch.cos(t * f)])',
	terms: ['timestep-embedding']
};

export default timeEmbed;
