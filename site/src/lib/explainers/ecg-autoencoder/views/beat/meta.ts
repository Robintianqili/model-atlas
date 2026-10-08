// View 'ecg-autoencoder/beat': one heartbeat over its 140 points. props.mode: input (the beat, where its parts usually
// are), rebuild (the beat and the autoencoder's rebuild, the 4-number code), odd (on the 'data' source: training beats
// in the first panel; the odd beat ODD and its rebuild in the second), error (+ the error at every point as bars, the largest
// ringed), score (+ the score's formula with live numbers, on a line with the test set's medians and the threshold),
// draw (error + the verdict at the learner's threshold). Pure helpers here, the drawing is View.svelte.
import type { I18n, StageProps, Tensor, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { I, N, beatOf } from '../../beats.ts';
import { ODD, recOf, rebuildOf, regionOf, thrOf } from '../../ecg.ts';

export type Mode = 'input' | 'rebuild' | 'odd' | 'error' | 'score' | 'draw';
const MODES: Mode[] = ['input', 'rebuild', 'odd', 'error', 'score', 'draw'];

export function beatData(p: Omit<StageProps, 'ctx'>) {
	const mode = (MODES.includes(p.props.mode as Mode) ? p.props.mode : 'rebuild') as Mode;
	const rec = recOf(p.rt);
	const g = (p.at.state as { beats?: Tensor } | undefined)?.beats;
	const gallery = mode === 'odd' && g ? Array.from({ length: g.shape[0] }, (_, i) => g.data.subarray(i * N, (i + 1) * N)) : [];
	const odd = mode === 'odd' && p.rt.model ? { ...rebuildOf(p.rt.model, ODD.kind, ODD.seed), twin: rebuildOf(p.rt.model, 'normal', ODD.seed) } : null;
	const max = rec ? Array.from(rec.x).reduce((b, v, i, a) => (v > a[b] ? i : b), 0) : 0;
	return { mode, rec, gallery, odd, max, thr: thrOf(p.rt.input) };
}

export function describeBeat(p: Omit<StageProps, 'ctx'>): I18n {
	const d = beatData(p), r = d.rec;
	if (d.mode === 'odd') {
		if (!d.odd) return I('Training beats and an odd beat: the model is not loaded yet.', '训练心跳和一个异常心跳：模型还没有载入。');
		return I(`First panel: ${d.gallery.length} normal training beats. Second panel: a beat with a raised ST segment and its rebuild: score ${fmt3(d.odd.score)}; the same beat without the raise scores ${fmt3(d.odd.twin.score)}.`, `第一张图：${d.gallery.length} 个正常的训练心跳。第二张图：一个 ST 段抬高的心跳和它的重建，分数 ${fmt3(d.odd.score)}；同一个心跳去掉抬高后分数是 ${fmt3(d.odd.twin.score)}。`);
	}
	if (!r) return I('No beat yet: pick one or draw one.', '还没有心跳：选一个或画一个。');
	if (d.mode === 'input') return I(`A beat of ${N} heights; the tallest, ${fmt3(r.x[d.max])}, is at point ${d.max}.`, `一个由 ${N} 个高度组成的心跳；最高的是第 ${d.max} 个点，${fmt3(r.x[d.max])}。`);
	const base = I(`Your beat and its rebuild from the code (${r.code.map(fmt3).join(', ')}).`, `你的心跳，以及从编码（${r.code.map(fmt3).join('，')}）重建出的心跳。`);
	if (d.mode === 'rebuild') return base;
	const w = regionOf(r.worst);
	const err = I(` The largest error, ${fmt3(r.e[r.worst])}, is at point ${r.worst} (${w.en}); the score is ${fmt3(r.score)}.`, `最大的误差 ${fmt3(r.e[r.worst])} 在第 ${r.worst} 个点（${w.zh}）；分数是 ${fmt3(r.score)}。`);
	const verdict = d.mode === 'draw' ? (r.score >= d.thr ? I(` At the threshold ${fmt3(d.thr)} it is flagged as odd.`, `在阈值 ${fmt3(d.thr)} 下，它被标为异常。`) : I(` At the threshold ${fmt3(d.thr)} it passes as normal.`, `在阈值 ${fmt3(d.thr)} 下，它被当作正常。`)) : I('', '');
	return I(base.en + err.en + verdict.en, (base.zh ?? '') + (err.zh ?? '') + (verdict.zh ?? ''));
}

const meta = {
	key: 'ecg-autoencoder/beat',
	title: I('A heartbeat and its rebuild', '一次心跳和它的重建'),
	kinds: ['trace', 'process'],
	describe: describeBeat,
	async demo() {
		// made-up rebuild: the beat smoothed over 5 points, and a made-up code
		const x = beatOf('stUp', 9), xr = x.map((_, i) => { let s = 0, n = 0; for (let j = Math.max(0, i - 2); j <= Math.min(N - 1, i + 2); j++) (s += x[j]), n++; return s / n; });
		const t = (shape: number[], data: Float32Array) => ({ shape, data });
		const rt = fakeRuntime({ slug: 'ecg-autoencoder', trace: { order: ['input', 'code', 'out'], t: { input: t([N], x), code: t([4], Float32Array.of(1.2, -0.4, 0.3, 2.1)), out: t([N], xr) } } });
		return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'error' } };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
