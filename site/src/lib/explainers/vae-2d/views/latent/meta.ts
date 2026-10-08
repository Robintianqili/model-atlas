// View 'vae-2d/latent': the 2-D latent space. props.mode:
//   'cloud'   a close-up around the learner's μ: the 1σ and 2σ rings; with props.sample, ε, z = μ + σ·ε and a button to
//             draw another ε (the raw input's `draw`)
//   'map'     the μ of the test drawings (latent.json, precomputed), coloured by kind, and the learner's μ
//   'drag'    the map with a ring the learner drags; the decoder redraws the ring's z live (the raw input's `probe`)
//   'interp'  a straight walk from the learner's μ to the μ of a preset drawing of another kind, decoded at 7 points
// Every drawing on the stage is decoded live by the exported decoder (vae.ts decode); only the dots are precomputed.
import type { ExplainerRuntime, I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { K, className, codeOf, extraOf, latentOf, muOf, near, zhAfter, type Latent, type XY } from '../../vae.ts';

export type Mode = 'cloud' | 'map' | 'drag' | 'interp';
const presetsOf = (rt: ExplainerRuntime) => (rt as ExplainerRuntime & { presets?: { id: string; raw: unknown }[] }).presets ?? [];

/** the kind the walk goes to: the raw input's `target`, else the nearest other kind's centre */
export function targetOf(rt: ExplainerRuntime, lat: Latent, mu: XY): { c: number; mu: XY } {
	const own = near(lat, mu).order[0];
	const asked = lat.classes.indexOf(extraOf(rt.input).target ?? '');
	const dist = (c: number) => (lat.centres[c][0] - mu[0]) ** 2 + (lat.centres[c][1] - mu[1]) ** 2;
	const c = asked >= 0 ? asked : lat.classes.map((_, k) => k).filter((k) => k !== own).sort((a, b) => dist(a) - dist(b))[0];
	const pre = presetsOf(rt).find((p) => p.id === lat.classes[c]);
	return { c, mu: rt.model && pre ? muOf(rt.model, pre.raw) : lat.centres[c] };
}

export function latentPic(p: Omit<StageProps, 'ctx'>) {
	const lat = latentOf(p.rt);
	const code = codeOf(p.rt);
	if (!lat || !code.mu.every(Number.isFinite)) return null;
	const mode = (['cloud', 'map', 'drag', 'interp'].includes(p.props.mode as string) ? p.props.mode : 'map') as Mode;
	const probe = extraOf(p.rt.input).probe ?? code.mu;
	return { mode, lat, ...code, sample: !!p.props.sample, draw: extraOf(p.rt.input).draw ?? 0, probe, nearMu: near(lat, code.mu), nearProbe: near(lat, probe), target: mode === 'interp' ? targetOf(p.rt, lat, code.mu) : null };
}

const nm = (rt: ExplainerRuntime, c: number, lang: 'en' | 'zh') => className(rt, c)[lang];
const pair = (v: XY) => `(${fmt3(v[0])}, ${fmt3(v[1])})`;

export function describeLatent(p: Omit<StageProps, 'ctx'>): I18n {
	const m = latentPic(p);
	if (!m) return { en: 'The latent map: nothing to show yet.', zh: '隐空间地图：还没有可显示的内容。' };
	const top = (n: ReturnType<typeof near>, lang: 'en' | 'zh') => n.order.filter((c) => n.counts[c] > 0).map((c) => (lang === 'en' ? `${n.counts[c]} ${nm(p.rt, c, lang)}` : `${n.counts[c]} 幅${zhAfter(nm(p.rt, c, lang))}`)).join(lang === 'en' ? ', ' : '、');
	if (m.mode === 'cloud')
		return {
			en: `Close-up around your μ = ${pair(m.mu)} with σ = ${pair(m.sigma)}: rings at 1σ and 2σ.${m.sample ? ` ε = ${pair(m.eps)} gives z = ${pair(m.z)}.` : ''}`,
			zh: `你的 μ = ${pair(m.mu)} 附近的特写，σ = ${pair(m.sigma)}：画出 1σ 和 2σ 的圈。${m.sample ? `ε = ${pair(m.eps)}，得到 z = ${pair(m.z)}。` : ''}`
		};
	if (m.mode === 'drag')
		return { en: `The ring is at z = ${pair(m.probe)}; its ${K} nearest test drawings: ${top(m.nearProbe, 'en')}. The redrawing beside it is decoded from that z.`, zh: `圆圈在 z = ${pair(m.probe)}；离它最近的 ${K} 幅测试画：${top(m.nearProbe, 'zh')}。旁边的重画就是从这个 z 解码出来的。` };
	if (m.mode === 'interp' && m.target)
		return { en: `A straight walk from your μ ${pair(m.mu)} to a ${nm(p.rt, m.target.c, 'en')} drawing at ${pair(m.target.mu)}, decoded at 7 evenly spaced points.`, zh: `从你的 μ ${pair(m.mu)} 沿直线走到一幅${zhAfter(nm(p.rt, m.target.c, 'zh'))}的 μ ${pair(m.target.mu)}，在 7 个等距的点上解码。` };
	return {
		en: `${m.lat.x.length} test drawings on the map, coloured by kind. Your μ ${pair(m.mu)}; its ${K} nearest test drawings: ${top(m.nearMu, 'en')}.`,
		zh: `地图上的 ${m.lat.x.length} 幅测试画，按类别着色。你的 μ ${pair(m.mu)}；离它最近的 ${K} 幅测试画：${top(m.nearMu, 'zh')}。`
	};
}

const meta = {
	key: 'vae-2d/latent',
	title: { en: 'The latent map', zh: '隐空间地图' },
	kinds: ['trace'],
	describe: describeLatent,
	async demo() {
		// made-up map: five blobs of 20 points; made-up code numbers (no model: the decoded drawings are left out)
		const lat: Latent = { classes: ['star', 'ladder', 'umbrella', 'eyeglasses', 't-shirt'], x: [], y: [], c: [], centres: [[0.6, 0.6], [-0.5, 1.2], [-1.2, -0.3], [1.4, -0.4], [-0.2, -1]], presets: {} };
		lat.centres.forEach(([cx, cy], c) => { for (let i = 0; i < 20; i++) (lat.x.push(cx + 0.4 * Math.cos(i * 2.4)), lat.y.push(cy + 0.4 * Math.sin(i * 1.7)), lat.c.push(c)); });
		const v = (a: number, b: number) => ({ shape: [2], data: Float32Array.of(a, b) });
		const trace = { order: ['mu', 'logvar', 'eps', 'z'], t: { mu: v(0.6, 0.55), logvar: v(-5, -5.4), 'z#sigma': v(0.082, 0.067), eps: v(-0.7, 0.5), z: v(0.543, 0.584) } };
		const spec = { slug: 'vae-2d', version: 'demo', layers: [], kinds: lat.classes.map((id, k) => ({ id, en: id, zh: ['星星', '梯子', '雨伞', '眼镜', 'T 恤'][k] })) } as never;
		const rt = Object.assign(fakeRuntime({ slug: 'vae-2d', trace, spec }), { assets: { 'latent.json': lat } });
		return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'map' } };
	},
	minHeight: 380
} satisfies ViewMeta;

export default meta;
