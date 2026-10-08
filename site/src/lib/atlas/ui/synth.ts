// A manifest for a spec that has no explainer package (the Lesson Lab's ?spec=models/<dir>, e.g. the shipped TinyCNN and
// TinyGPT assets). Everything in it comes from the spec: the input adapter of its first declared input, the output view,
// and a predict-first question whose answer is the spec's own layer count. Pure TS, so Node tests use it too.
import type { I18n, PredictQuestion } from '../types/core.ts';
import type { AtlasSpec } from '../types/spec.ts';
import type { ExplainerManifest } from '../types/explainer.ts';

export interface SynthOpts {
	/** the slug the page uses (default spec.slug) */
	slug?: string;
	/** asset folder relative to static/ ('models/tinycnn') */
	dir: string;
	/** presets file name in that folder, when it exists */
	presets?: string | null;
	/** files loaded on every visit (default spec.json + weights.bin) */
	auto?: string[];
	/** adapter options (text-bpe: { tokenizer }) */
	options?: Record<string, unknown>;
	/** the first input */
	start?: unknown;
}

const nameOf = (s: AtlasSpec): I18n => s.name ?? { en: s.slug, zh: s.slug };

/** A question with a real answer: how many layers come after the input (read from the spec). */
export function layerCountQuestion(s: AtlasSpec): PredictQuestion {
	const ops = s.layers.filter((l) => l.op !== 'input');
	const n = ops.length;
	const name = nameOf(s);
	const opts = [...new Set([Math.max(1, Math.round(n / 3)), n, n * 3])].sort((a, b) => a - b);
	const kinds = [...new Set(ops.map((l) => l.op))];
	return {
		id: `${s.slug}-layer-count`,
		question: { en: `How many layers does ${name.en} pass your input through?`, zh: `${name.zh ?? name.en} 让你的输入经过多少层？` },
		options: opts.map((k) => ({ en: `${k}`, zh: `${k}` })),
		answer: opts.indexOf(n),
		explain: {
			en: `${n} layers after the input, of ${kinds.length} kinds: ${kinds.join(', ')}. Small models are deep enough to be interesting and small enough to show every number.`,
			zh: `输入之后有 ${n} 层，共 ${kinds.length} 种：${kinds.join('、')}。小模型的深度足够有趣，又小到能把每个数都展示出来。`
		}
	};
}

export function synthManifest(spec: AtlasSpec, o: SynthOpts): ExplainerManifest {
	const name = nameOf(spec);
	const inp = spec.inputs?.[0];
	const n = spec.layers.filter((l) => l.op !== 'input').length;
	const classifier = Array.isArray(spec.classes) || spec.layers[spec.layers.length - 1]?.op === 'softmax';
	const m: ExplainerManifest = {
		slug: o.slug ?? spec.slug,
		type: 'model',
		kinds: ['trace'],
		quality: 'draft',
		assets: { dir: o.dir, auto: o.auto ?? ['spec.json', 'weights.bin'] },
		page: {
			goals: [
				{ en: `Follow your own input through all ${n} layers of ${name.en}, one step at a time.`, zh: `一步一步跟着你自己的输入走过 ${name.zh ?? name.en} 的全部 ${n} 层。` },
				{ en: 'Read the arithmetic behind one number of each kind of layer.', zh: '读懂每种层里一个数背后的计算。' }
			],
			prequestion: layerCountQuestion(spec)
		},
		credits: []
	};
	if (inp) {
		m.input = { adapter: inp.adapter };
		if (o.presets) m.input.presets = o.presets;
		if (o.options) m.input.options = o.options;
		if (o.start !== undefined) m.input.start = o.start;
	}
	if (classifier) m.output = { view: 'kit.probbars' };
	return m;
}
