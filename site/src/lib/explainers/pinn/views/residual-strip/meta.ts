// View 'pinn/residual-strip' (kind B): the network's curve u(t) over the true curve (dashed) and the learner's points,
// and under it a strip of 200 cells, one per physics time, colored by the residual (u″ + c·u′ + k·u) / k of the run's
// law: the background color where the curve obeys the law, deep blue or orange where it breaks it. Props:
//   net, strip, truth, points, split   draw the network's curve / the strip / the true curve / the points / the shaded
//                      time after t = 0.4 (default true)
//   gap                shade the band between the network's curve and the true curve after t = 0.4 (default false)
//   choices            [{ source, label }]: buttons that switch to another source (the view runs it to its end)
// Numbers: the step's frame, or a chosen source's last frame; the curve and the residual are the derived field 'curves'.
import type { I18n, Tensor } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtInt } from '../../../../atlas/core/format.ts';
import { N, SPLIT } from '../../osc.ts';

export type Choice = { source: string; label: I18n };

/** What the view draws for `source` (the step's source at its cursor, else that source's last frame). */
export function fitPic(p: Pick<StageProps, 'rt' | 'source' | 'at'>, source = p.source) {
	const tape = p.rt.sources[source]?.tape;
	let i = source === p.source ? (p.at?.index ?? -1) : tape ? tape.length - 1 : -1;
	if (!tape || i < 0 || i >= tape.length) return null;
	// the latest frame with weights (snapshot or init) at or before it
	while (i > 0 && tape.frame(i).phase !== 'snapshot' && tape.frame(i).phase !== 'init') i--;
	let cv: Tensor | undefined;
	try {
		cv = (tape.state(i) as { curves?: Tensor }).curves;
	} catch {
		cv = undefined;
	}
	const f = tape.frame(i);
	const opt = (tape.opt ?? {}) as { lam?: number; c?: number; physics?: boolean };
	const ok = !!cv && cv.data.length === 4 * N;
	return { source, step: f.step ?? 0, u: ok ? cv!.data.subarray(0, N) : null, r: ok ? cv!.data.subarray(3 * N, 4 * N) : null, err: f.metrics?.err ?? NaN, data: f.metrics?.data === 0 ? 0 : (f.metrics?.data ?? NaN), lam: opt.lam, c: opt.c, pinn: opt.physics !== false };
}
export type FitPic = NonNullable<ReturnType<typeof fitPic>>;

export function maxAbs(a: ArrayLike<number> | null): number {
	let m = 0;
	for (let i = 0; a && i < a.length; i++) m = Math.max(m, Math.abs(a[i]));
	return m;
}

/** mean square of a residual row: the physics loss of those weights against the run's own law */
export const physOf = (r: ArrayLike<number> | null): number => (r && r.length ? Array.from(r).reduce((s, v) => s + v * v, 0) / r.length : NaN);

/** what the view draws, as text; `props` as the step gives them (net, strip, points: false hide those parts) */
export function describeFit(pic: FitPic | null, props: Record<string, unknown> = {}): I18n {
	const on = (k: string) => props[k] !== false;
	if (!on('net')) return { en: `The true curve of the damped spring (dashed), from t = 0 to 1${on('points') ? ', with your points' : ''}.`, zh: `阻尼弹簧的真实曲线（虚线），从 t = 0 到 1${on('points') ? '，以及你的数据点' : ''}。` };
	if (!pic || !pic.u) return { en: 'The network has no curve to show yet.', zh: '网络还没有可以显示的曲线。' };
	const err = Number.isFinite(pic.err) ? { en: ` After t = ${SPLIT} it is ${fmt3(pic.err)} from the true curve on average${props.gap ? ' (the shaded band)' : ''}.`, zh: `t = ${SPLIT} 之后，它与真实曲线平均相差 ${fmt3(pic.err)}${props.gap ? '（阴影带）' : ''}。` } : { en: '', zh: '' };
	const res = on('strip') && pic.r ? { en: ` Under it, the residual at 200 times: at most ${fmt3(maxAbs(pic.r))} in size; physics loss ${fmt3(physOf(pic.r))}.`, zh: `下方是 200 个时刻上的残差：大小最多 ${fmt3(maxAbs(pic.r))}；物理损失 ${fmt3(physOf(pic.r))}。` } : { en: '', zh: '' };
	return {
		en: `The ${pic.pinn ? 'PINN' : 'plain network'}'s curve after ${fmtInt(pic.step)} training steps, over the dashed true curve and the points.${err.en}${res.en}`,
		zh: `${pic.pinn ? 'PINN' : '普通网络'}训练 ${fmtInt(pic.step)} 步后的曲线，画在虚线的真实曲线和数据点上。${err.zh}${res.zh}`
	};
}

const meta = {
	key: 'pinn/residual-strip',
	title: { en: 'The curve and its residual', zh: '曲线和它的残差' },
	kinds: ['train'],
	describe: (p: StageProps): I18n => describeFit(fitPic(p), p.props ?? {}),
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { seriesPresetRaw }, { trainingProcess }, { pinn }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../../../atlas/adapters/series/adapter.ts'), import('../../../../atlas/train/trainer.ts'), import('../../trainer.ts')]);
		const raw = seriesPresetRaw('oscillator');
		const tape = record(trainingProcess(pinn), raw, { steps: 60 }, { seed: 1 });
		const rt = fakeRuntime({ slug: 'pinn', input: raw, tapes: { pinn: tape } });
		return { rt, source: 'pinn', at: atFor(rt, 'pinn', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
