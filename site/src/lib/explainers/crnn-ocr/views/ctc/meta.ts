// View 'crnn-ocr/ctc': the strip of writing, the top symbol of each of its 32 columns and what CTC makes of them
// (modes in data.ts). The narration's numbers are computed by the lesson from the same data.ts functions.
import type { StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { fmtPct } from '../../../../atlas/core/format.ts';
import { SYMBOLS } from '../../glyphs.ts';
import { pathFacts, stripOf } from './data.ts';

type P = Omit<StageProps, 'ctx'>;

export function describeCtc(p: P) {
	const s = stripOf(p);
	if (!s) return { en: 'CTC decoding: nothing written yet.', zh: 'CTC 解码：还没有写字。' };
	const shown = s.best.slice(0, s.upTo + 1).map((k) => SYMBOLS[k]).join(' ');
	if (s.mode === 'chances') {
		const n = s.best.filter((k) => k === 0).length;
		return { en: `Chances of the ${s.q.V} symbols (rows) in each of the ${s.q.T} columns (columns) under the strip; the blank is the top symbol in ${n} columns.`, zh: `长条下面是 ${s.q.T} 列（横向）中每一列 ${s.q.V} 个符号（纵向）的概率；有 ${n} 列排第一的是空白。` };
	}
	const f = pathFacts(s, s.best);
	return {
		en: `Top symbol of columns 0 to ${s.upTo} (– is the blank): ${shown}. All ${s.q.T} columns read “${s.text}”; this path has a chance of ${fmtPct(Math.exp(f.logP))}, and all ${f.paths} paths that spell the text together ${fmtPct(Math.exp(f.logText))}.`,
		zh: `第 0 到第 ${s.upTo} 列排第一的符号（– 是空白）：${shown}。全部 ${s.q.T} 列读作“${s.text}”；这条路径的概率是 ${fmtPct(Math.exp(f.logP))}，拼出这段文字的全部 ${f.paths} 条路径合起来是 ${fmtPct(Math.exp(f.logText))}。`
	};
}

const meta = {
	key: 'crnn-ocr/ctc',
	title: { en: 'Columns to text with CTC', zh: '用 CTC 把列变成文字' },
	kinds: ['trace', 'process'],
	describe: describeCtc,
	async demo() {
		// made-up data: 8 columns, 3 symbols (blank, "0", "1"), a strip with two bright stripes
		const [T, V] = [8, 3];
		const top = [0, 2, 2, 0, 2, 0, 1, 0];
		const probs = Float32Array.from({ length: T * V }, (_, i) => (i % V === top[Math.floor(i / V)] ? 0.8 : 0.1));
		const ink = { shape: [1, 16, 16], data: Float32Array.from({ length: 256 }, (_, i) => (i % 16 === 3 || i % 16 === 9 ? 1 : 0)) };
		const rt = fakeRuntime({ slug: 'crnn-ocr', pre: { tensors: { input: ink }, steps: [], empty: false }, trace: { order: ['probs'], t: { probs: { shape: [T, V], data: probs } } } });
		return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'paths' } };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
