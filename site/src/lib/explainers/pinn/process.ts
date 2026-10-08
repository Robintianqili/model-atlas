// pinn: process 'probe' (kind C): the slope u′ and the bend u″ of the plain network at one time t, by autodiff and by
// finite differences with shrinking steps h. It first trains the plain network exactly as the 'plain' source does (same
// trainer, options and seed: the same weights; claims.test.ts checks it), then yields
//   'autodiff'  u(t), u′(t), u″(t) from grad(…, { createGraph: true }) twice
//   'fd'        one frame per h: u(t ± h), the central differences and their gaps to the autodiff values
// kind 'training' so the e2e step cap (?atlas-e2e=1) shortens its inner training exactly as it shortens the plain source.
import type { ProcessDef, ProcessRun, Tensor } from '../../atlas/types/index.ts';
import { trainingProcess } from '../../atlas/train/trainer.ts';
import { withGradMode } from '../../atlas/autograd/var.ts';
import { netAt, plain, type PinnOpt } from './trainer.ts';
import { derivs, valuesAt } from './osc.ts';

export type ProbeOpt = PinnOpt & { at: number; hs: number[] };
const inner = trainingProcess(plain);
const L = (en: string, zh: string) => ({ en, zh });
const SUP: Record<string, string> = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
/** a step size as the learner reads it: 0.001, 10⁻⁴ ("…" when it is not known yet) */
export const hText = (h: number) => (!Number.isFinite(h) || h <= 0 ? '…' : h >= 1e-3 ? String(h) : `10⁻${String(Math.round(-Math.log10(h))).replace(/\d/g, (d) => SUP[d])}`);

export const probe: ProcessDef<unknown, ProbeOpt, { h: number }> = {
	id: 'probe',
	kind: 'training',
	title: L('Slope and bend of the plain network: autodiff and finite differences', '普通网络的斜率和弯曲：自动微分与有限差分'),
	phases: {
		autodiff: { label: L('Autodiff', '自动微分'), kind: 'backward', formula: { tex: "u'(t) = {{d1}};\\quad u''(t) = {{d2}}", values: (f) => f.metrics ?? {} } },
		fd: {
			label: L('Finite differences', '有限差分'),
			kind: 'other',
			formula: {
				// its own aligned block, one short line each, so the numbers stay on a 390 px phone
				tex: "\\begin{aligned} h &= 10^{ {{hExp}} } \\\\ u' &\\approx \\tfrac{u(t+h) - u(t-h)}{2h} \\\\ &= {{fd1}} \\\\ u'' &\\approx \\tfrac{u(t+h) - 2u(t) + u(t-h)}{h^2} \\\\ &= {{fd2}} \\\\ u'_{\\text{autodiff}} &= {{d1}} \\\\ u''_{\\text{autodiff}} &= {{d2}} \\end{aligned}",
				values: (f) => f.metrics ?? {}
			}
		}
	},
	schema: { h: { kind: 'scalar', label: L('step h', '步长 h'), tol: 1e-12 } },
	defaults: { ...plain.defaults, at: 0.7, hs: [0.1, 0.01, 0.001, 1e-4, 1e-5, 1e-6, 1e-7] },
	*run(raw, ctx): ProcessRun<{ h: number }, ProbeOpt> {
		const opt = { ...probe.defaults, ...ctx.opt } as ProbeOpt;
		let params: Record<string, Tensor> | undefined;
		for (const f of inner.run(raw, { ...ctx, opt, detail: 'coarse' })) if (f.state.params) params = f.state.params;
		if (!params) return;
		const m = netAt(params, opt);
		const t = opt.at;
		const d = withGradMode(true, () => derivs(m, [t], false));
		const [u, d1, d2] = [d.u.data[0], d.du.data[0], d.d2u.data[0]];
		const rows = opt.hs.map((h) => {
			const [up, um] = valuesAt(m, [t + h, t - h]);
			const fd1 = (up - um) / (2 * h);
			const fd2 = (up - 2 * u + um) / (h * h);
			return { t, h, hExp: Math.round(Math.log10(h)), u, up, um, fd1, fd2, d1, d2, e1: Math.abs(fd1 - d1), e2: Math.abs(fd2 - d2) };
		});
		// every frame also carries the smallest step's gap (hTiny, eTiny), so a step at any h can narrate it
		const tiny = rows[rows.length - 1];
		const sum = { hTiny: tiny?.h ?? NaN, eTiny: tiny?.e2 ?? NaN };
		yield { phase: 'autodiff', step: 0, state: { h: 0 }, metrics: { t, u, d1, d2, ...sum } };
		for (let i = 0; i < rows.length; i++) yield { phase: 'fd', step: i + 1, state: { h: rows[i].h }, caption: L(`h = ${hText(rows[i].h)}`, `h = ${hText(rows[i].h)}`), metrics: { ...rows[i], ...sum } };
	}
};
