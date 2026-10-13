// pointnet view 'cloud': the model's input points (centred and scaled) on the plane; the drawing is View.svelte.
//   points    the points, the cross at 0 and the circle of radius 1, with the first table rows
//   order     the table order and a shuffled order side by side, each with the network's answer (predict step)
//   feature   one feature of the shared MLP at every point (sequential ramp), its winner ringed; ◀ ▶ step features
//   critical  the critical points (they win at least one feature of the max), larger when they win more;
//             props.task adds the task's buttons (remove the critical points; put the points back)
//   stray     the points and the same points plus one stray point, each with the network's answer (limits step)
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { crit, label, T, variants, argmax } from '../../cloud.ts';
import { normalize, presetPoints } from '../../shapes.ts';

type P = Omit<StageProps, 'ctx'>;

export function describeCloud(p: P): I18n {
	const x = T(p.rt, 'input');
	const c = crit(p.rt);
	if (!x || !c) return { en: 'No points yet.', zh: '还没有点。' };
	const mode = String(p.props.mode ?? 'points');
	const sm = T(p.rt, 'softmax')?.data;
	const top = sm ? label(argmax(sm)) : { en: '…', zh: '…' };
	const head = { en: `${c.n} points, centred and scaled; the network answers ${top.en}.`, zh: `${c.n} 个点，已居中并缩放；网络的答案是${top.zh}。` };
	let more = { en: '', zh: '' };
	const V = mode === 'order' || mode === 'stray' ? variants(p.rt) : null;
	if (mode === 'order' && V?.shuf) more = { en: ` Shuffled, it answers ${label(V.shuf.top).en}.`, zh: ` 打乱之后，答案是${label(V.shuf.top).zh}。` };
	if (mode === 'stray' && V?.stray) more = { en: ` With one stray point it answers ${label(V.stray.top).en}.`, zh: ` 加上一个离群点后，答案是${label(V.stray.top).zh}。` };
	if (mode === 'critical') more = { en: ` ${c.pts.length} of them are critical points (points ${c.pts.join(', ')}).`, zh: ` 其中 ${c.pts.length} 个是关键点（点 ${c.pts.join('、')}）。` };
	if (mode === 'feature') {
		const f = Number(p.props.f ?? 0);
		more = c.win[f] >= 0 ? { en: ` Feature ${f} is largest at point ${c.win[f]}.`, zh: ` 特征 ${f} 在点 ${c.win[f]} 上最大。` } : { en: ` Feature ${f} is 0 at every point.`, zh: ` 特征 ${f} 在每个点上都是 0。` };
	}
	return { en: head.en + more.en, zh: head.zh + more.zh };
}

const meta = {
	key: 'pointnet/cloud',
	title: { en: 'The points on the plane', zh: '平面上的点' },
	kinds: ['trace'],
	describe: describeCloud,
	async demo() {
		const m = normalize(presetPoints('square').flatMap((q) => [q.x, q.y]));
		const n = m.n, F = 4;
		const relu2 = { shape: [n, F], data: Float32Array.from({ length: n * F }, (_, k) => Math.max(0, m.xy[2 * Math.floor(k / F) + (k % 2)] * (k % F < 2 ? 1 : -1))) };
		const rt = fakeRuntime({ slug: 'pointnet', trace: { order: ['input', 'relu2'], t: { input: { shape: [n, 2], data: m.xy }, relu2 } } });
		return { rt, source: 'fwd', at: atFor(rt, 'fwd', 0), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'critical' } };
	},
	minHeight: 380
} satisfies ViewMeta;

export default meta;
