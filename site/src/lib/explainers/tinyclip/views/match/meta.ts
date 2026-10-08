// tinyclip view 'match' (props.mode: pairs | words | vectors | pair | matrix | captions; data.ts reads the numbers).
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { nce } from '../../process.ts';
import { PARTNERS } from '../../scenes.ts';
import { rawOf } from '../../adapters/shapes-captions/adapter.ts';
import { captionsData, matrixData, pairsData, vectorsData, wordsData } from './data.ts';

const SHOW: Record<string, I18n> = {
	L: { en: 'scores', zh: '分数' },
	P: { en: 'row shares (which caption fits each drawing)', zh: '行份额（每幅画配哪句描述）' },
	Q: { en: 'column shares (which drawing fits each caption)', zh: '列份额（每句描述配哪幅画）' },
	G: { en: 'gradient of the loss on each score', zh: '损失对每个分数的梯度' }
};

export function describeMatch(p: Omit<StageProps, 'ctx'>): I18n {
	const mode = String(p.props.mode ?? 'matrix');
	if (mode === 'pairs') {
		const items = pairsData(p);
		return { en: `${items.length} training pairs: ${items.map((x) => `“${x.caption}”`).join(', ')}.`, zh: `${items.length} 对训练用的图文对：${items.map((x) => `“${x.caption}”`).join('，')}。` };
	}
	if (mode === 'words') {
		const w = wordsData(p.rt);
		return { en: `The caption “${w.caption}”: ${w.words.length} word rows of ${w.width} numbers, their average, and the caption vector.`, zh: `描述“${w.caption}”：${w.words.length} 个词行，每行 ${w.width} 个数，它们的平均，以及描述向量。` };
	}
	if (mode === 'vectors' || mode === 'pair') {
		const v = vectorsData(p.rt);
		return { en: `Image vector of length ${fmt3(v.lenA)} and caption vector of length ${fmt3(v.lenB)}, both scaled to length 1; their products add up to the cosine ${fmt3(v.cos)}, and × ${fmt3(v.scale)} gives the score ${fmt3(v.score)}.`, zh: `图像向量长度 ${fmt3(v.lenA)}，描述向量长度 ${fmt3(v.lenB)}，都缩放到长度 1；它们的乘积加起来是余弦 ${fmt3(v.cos)}，乘以 ${fmt3(v.scale)} 得到分数 ${fmt3(v.score)}。` };
	}
	if (mode === 'captions') {
		const rows = captionsData(p.rt, p.props.texts as string[] | undefined);
		return { en: `Your drawing against ${rows.length} captions: ${rows.map((r) => `“${r.text}” score ${fmt3(r.score)}, share ${fmtPct(r.share)}`).join('; ')}.`, zh: `你的画对 ${rows.length} 句描述：${rows.map((r) => `“${r.text}” 分数 ${fmt3(r.score)}，份额 ${fmtPct(r.share)}`).join('；')}。` };
	}
	const m = matrixData(p);
	if (!m.M) return { en: 'The batch table: nothing to show yet.', zh: '这一批的分数表：还没有可显示的内容。' };
	const diag = Array.from({ length: m.N }, (_, i) => fmt3(m.M![i * m.N + i]));
	return { en: `A ${m.N}×${m.N} table of ${SHOW[m.show].en}; rows are drawings, columns are captions; the diagonal (true pairs): ${diag.join(', ')}.`, zh: `一张 ${m.N}×${m.N} 的${SHOW[m.show].zh}表；行是画，列是描述；对角线（正确的对）：${diag.join('，')}。` };
}

const meta = {
	key: 'tinyclip/match',
	title: { en: 'Drawings and captions, matched', zh: '画与描述的匹配' },
	kinds: ['trace', 'process'],
	describe: describeMatch,
	async demo() {
		// made-up unit vectors (illustrative): three pairs in two dimensions
		const tape = record(nce, { N: 3, D: 2, scale: 10, I: [1, 0, 0, 1, 0.6, 0.8], T: [1, 0, 0.6, 0.8, 0, 1] }, {}, { seed: 1 });
		const rt = fakeRuntime({ slug: 'tinyclip', input: rawOf(PARTNERS[0]), tapes: { nce: tape } });
		return { rt, source: 'nce', at: atFor(rt, 'nce', 0), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'matrix', show: 'L' } };
	},
	minHeight: 360
} satisfies ViewMeta;

export { SHOW };
export default meta;
