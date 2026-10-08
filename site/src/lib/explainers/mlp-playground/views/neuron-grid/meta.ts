// View 'mlp-playground/neuron-grid' (kind B): every neuron of the playground MLP as a heatmap over the plane, from the
// derived field `grids` (TF Playground's layout idea, Apache-2.0, credited in the manifest; our own code). Props:
//   show     'net' (default): inputs -> hidden neurons -> output, edges colored by weight sign, thicker for larger
//            weights; 'layer': one hidden layer's neurons with their 0-lines; 'neuron': one neuron, large
//   layer, unit   which hidden layer (0-based) and neuron;  point: a data point to ring
//   choices  [{ source, label }]: buttons that switch to another source (the view runs it to its end)
// Numbers: the step's frame, or a chosen source's last frame. Hidden neurons: −1 blue to +1 orange (tanh); output: the
// share for class 1 in the class colors, with the data points.
import type { I18n, Tensor } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmtInt, fmtPct } from '../../../../atlas/core/format.ts';
import { dataOf, dec, evalPoints, layersOf, sizesOf, type Act, type Params } from '../../net.ts';

export type Choice = { source: string; label: I18n };

/** What the view draws for `source` (the step's own source at its cursor, else that source's last frame); grids:
 *  [N, R, R], every hidden neuron, then P(class 1). */
export function netPic(p: Pick<StageProps, 'rt' | 'source' | 'at'>, source = p.source) {
	const tape = p.rt.sources[source]?.tape;
	let s: Record<string, unknown> | undefined;
	let loss = NaN;
	try {
		s = (source === p.source ? p.at?.state : tape && tape.length ? tape.state(tape.length - 1) : undefined) as Record<string, unknown> | undefined;
		// the loss of these weights on all the points (a snapshot frame's metric; an init frame has none)
		loss = (source === p.source ? p.at?.frame : tape && tape.length ? tape.frame(tape.length - 1) : undefined)?.metrics?.loss ?? NaN;
	} catch {
		s = undefined;
	}
	const params = s?.params as Params | undefined;
	if (!params || !layersOf(params).length) return null;
	const act: Act = s!.act === 'relu' ? 'relu' : 'tanh';
	const points = dataOf(p.rt.input, (tape?.opt ?? null) as { data?: unknown } | null);
	let grids: Tensor | null = null;
	try {
		grids = (s!.grids as Tensor | undefined) ?? null;
	} catch {
		grids = null;
	}
	return {
		source,
		step: typeof s!.step === 'number' ? (s!.step as number) : 0,
		sizes: sizesOf(params),
		act,
		grids,
		layers: layersOf(params),
		points,
		ev: points.n ? evalPoints(params, act, points) : null,
		loss
	};
}

/** The text form of what the view draws (screen readers, "Show as text"). */
export type NetPic = NonNullable<ReturnType<typeof netPic>>;
export function describePic(pic: NetPic | null): I18n {
	if (!pic) return { en: 'The network has no weights to show yet.', zh: '网络还没有可以显示的权重。' };
	const arch = pic.sizes.join(' → ');
	const hidden = pic.sizes.slice(1, -1).reduce((a, b) => a + b, 0);
	const acc = pic.ev
		? { en: ` ${fmtPct(pic.ev.accuracy)} of the ${fmtInt(pic.points.n)} points lie on their own class's side.`, zh: `${fmtInt(pic.points.n)} 个点中有 ${fmtPct(pic.ev.accuracy)} 落在自己那一类的一侧。` }
		: { en: '', zh: '' };
	if (Number.isFinite(pic.loss)) (acc.en += ` The loss is ${dec(pic.loss)}.`), (acc.zh += `损失是 ${dec(pic.loss)}。`);
	return {
		en: `The network ${arch} after ${fmtInt(pic.step)} training steps. Each of its ${hidden} hidden neurons is drawn as a square over the plane: blue where it outputs −1, orange where it outputs +1. The output square shows the share for class 1 everywhere, with the data points.${acc.en}`,
		zh: `网络 ${arch}，训练了 ${fmtInt(pic.step)} 步。${hidden} 个隐藏神经元各画成平面上的一个方块：输出 −1 的地方是蓝色，输出 +1 的地方是橙色。输出方块显示各处属于第 1 类的比例，并画出数据点。${acc.zh}`
	};
}

const meta = {
	key: 'mlp-playground/neuron-grid',
	title: { en: 'Every neuron over the plane', zh: '每个神经元在平面上的样子' },
	kinds: ['train', 'process'],
	describe: (p: StageProps): I18n => describePic(netPic(p, typeof p.props?.source === 'string' ? (p.props.source as string) : p.source)),
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { presetRaw }, { play }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../../../atlas/adapters/points2d/adapter.ts'), import('../../process.ts')]);
		const raw = presetRaw('moons');
		const tape = record(play, raw, { steps: 120 }, { seed: 1 });
		const rt = fakeRuntime({ slug: 'mlp-playground', input: raw, tapes: { run: tape } });
		return { rt, source: 'run', at: atFor(rt, 'run', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 340
} satisfies ViewMeta;

export default meta;
