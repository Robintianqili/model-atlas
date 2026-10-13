// View 'lora/counter': the parameter counter at the frame's rank r (metrics.r; the learner's rank before a frame
// exists). A to-scale picture of the toy layer: W0 (16 × 16 numbers, frozen) next to B (16 × r) and A (r × 16), one
// square per number; then a table of real layer sizes: the numbers full fine-tuning trains (d_out·d_in) against LoRA's
// r·(d_in + d_out), and their ratio. The last row is all of GPT-2 small. No props.
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmtInt } from '../../../../atlas/core/format.ts';
import { GPT2_PARAMS, LAYERS, gpt2Lora, loraCount, pctText, rankAt } from '../../data.ts';

export interface CountRow {
	id: string;
	label: I18n;
	shape: string;
	full: number;
	lora: number;
	share: number;
}

/** The table's rows at rank r (pure; also the text form). */
export function countRows(r: number): CountRow[] {
	const rows: CountRow[] = LAYERS.map((l) => {
		const lora = loraCount(r, l.dIn, l.dOut);
		return { id: l.id, label: l.label, shape: `${fmtInt(l.dOut)} × ${fmtInt(l.dIn)}`, full: l.dIn * l.dOut, lora, share: lora / (l.dIn * l.dOut) };
	});
	const g = gpt2Lora(r);
	rows.push({ id: 'gpt2', label: { en: 'GPT-2 small, the whole model (72 matrices adapted)', zh: 'GPT-2 small 整个模型（适配 72 个矩阵）' }, shape: '', full: GPT2_PARAMS, lora: g, share: g / GPT2_PARAMS });
	return rows;
}

export function describeCount(r: number): I18n {
	const rows = countRows(r);
	const line = (lang: 'en' | 'zh') => rows.map((x) => `${x.label[lang]}${x.shape ? ` (${x.shape})` : ''}: ${fmtInt(x.full)} → ${fmtInt(x.lora)} (${pctText(x.share)})`).join('; ');
	return {
		en: `Rank r = ${r}. Numbers full fine-tuning trains, then LoRA's r·(d_in + d_out), then the share: ${line('en')}.`,
		zh: `秩 r = ${r}。全量微调要训练的数、LoRA 的 r·(d_in + d_out)，以及两者之比：${line('zh')}。`
	};
}

const meta = {
	key: 'lora/counter',
	title: { en: 'The parameter counter', zh: '参数计数器' },
	kinds: ['process'],
	describe: (p: StageProps) => describeCount(rankAt(p)),
	demo: async () => {
		const { fakeRuntime } = await import('../../../../atlas/runtime/fake.ts');
		const rt = fakeRuntime({ slug: 'lora', input: { r: 2 }, tapes: {} });
		return { rt, source: 'fit', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 380
} satisfies ViewMeta;

export default meta;
