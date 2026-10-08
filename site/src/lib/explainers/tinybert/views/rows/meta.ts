// tinybert view 'rows': the three learned rows that make one position's input to block 1, and their sum: the token's row
// (layer tok), its position's row (pos), its segment's row (seg) and tok + pos + seg (embed). props.at: the position
// (default: the [MASK] position); props.f: the feature outlined in all four strips, with its value in each label
// (default: the largest one of the sum, the one step 3 works out). One colour scale for all four strips.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { bigFeature, gap, row, T } from '../../bert.ts';
import { demoProps } from '../../demo.ts';

type P = Omit<StageProps, 'ctx'>;

export function rowsOf(p: P) {
	const g = gap(p.rt);
	const q = typeof p.props.at === 'number' ? p.props.at : g.m;
	const f = typeof p.props.f === 'number' ? p.props.f : bigFeature(p.rt);
	const segB = g.seg[q] === 1;
	const parts = [
		{ id: 'tok', label: { en: `token row: “${g.word(g.ids[q])}”`, zh: `词元行：“${g.word(g.ids[q])}”` } },
		{ id: 'pos', label: { en: `position row: position ${q}`, zh: `位置行：位置 ${q}` } },
		{ id: 'seg', label: { en: `segment row: ${segB ? 'B' : 'A'}`, zh: `段行：${segB ? 'B' : 'A'}` } },
		{ id: 'embed', label: { en: '= the sum, block 1’s input', zh: '= 三者之和，第 1 块的输入' } }
	].map((x) => {
		const data = row(T(p.rt, x.id), q);
		const v = data ? data[f] : NaN;
		const shown = fmt3(v, x.id === 'embed' ? 3 : 4); // as the narration prints them: the parts with 4 digits, the sum with 3
		const label: I18n = { en: `${x.label.en} · feature ${f}: ${shown}`, zh: `${x.label.zh} · 特征 ${f}：${shown}` };
		return { id: x.id, label, data, v };
	});
	let a = 0;
	for (const x of parts) for (const v of Array.from(x.data ?? [])) a = Math.max(a, Math.abs(v));
	return { q, f, parts, d: parts[0].data?.length ?? 0, domain: [-(a || 1), a || 1] as [number, number] };
}

const meta = {
	key: 'tinybert/rows',
	title: { en: 'Three rows added up', zh: '三行相加' },
	kinds: ['trace'],
	describe: (p: P): I18n => {
		const r = rowsOf(p);
		if (!r.parts[3].data) return { en: 'The input rows: nothing to show yet.', zh: '输入行：还没有可显示的内容。' };
		return { en: `Position ${r.q}, ${r.d} numbers per row (feature ${r.f} outlined, counting from 0): ${r.parts.map((x) => x.label.en).join('; ')}.`, zh: `位置 ${r.q}，每行 ${r.d} 个数（框出的是特征 ${r.f}，从 0 数起）：${r.parts.map((x) => x.label.zh).join('；')}。` };
	},
	demo: async () => demoProps(),
	minHeight: 220
} satisfies ViewMeta;

export default meta;
