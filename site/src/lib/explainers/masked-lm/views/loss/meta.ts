// masked-lm view 'loss': the masked-LM loss on the learner's text, one table row per text token: what TinyBERT reads
// there, whether the position is graded (chosen), TinyBERT's probability for the original token and its −log p (a bar to
// scale, graded rows only), and the mean over the graded rows: the loss.
//   props.gpt   adds TinyGPT's −log p for every token after the first (the causal loss on the same text) and its mean
//   props.draw  a button that sends row 0 of the 'batch' draw (the step's source, at its last frame) to TinyBERT
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { contentOf, vocab, type MaskedRaw } from '../../adapters/masked-text/adapter.ts';
import { clmLoss, mlmLoss, text } from '../../text.ts';
import { demoProps } from '../../demo.ts';

type P = Omit<StageProps, 'ctx'>;
export const READS: I18n[] = [{ en: 'itself', zh: '它自己' }, { en: '[MASK]', zh: '[MASK]' }, { en: 'random', zh: '随机' }, { en: 'kept', zh: '保留' }];

/** Row 0 of the batch draw at the step's frame (the learner's text): its fates and random ids; null before the draw. */
export function drawOf(p: P): { fate: number[]; swap: number[] } | null {
	const n = contentOf(p.rt.input as MaskedRaw).length;
	const st = (p.at.tape && p.at.index >= 0 ? p.at.tape.state(p.at.index) : null) as { fate?: { data: ArrayLike<number> }; shown?: { data: ArrayLike<number> } } | null;
	if (!n || !st?.fate || !st.shown || p.at.frame?.phase !== 'done') return null;
	const fate = Array.from(st.fate.data).slice(0, n);
	return { fate, swap: fate.map((f, k) => (f === 2 ? Number(st.shown!.data[k]) : -1)) };
}

export function lossOf(p: P) {
	const x = text(p.rt);
	const L = mlmLoss(p.rt);
	const G = p.props.gpt ? clmLoss(p.rt) : null;
	const rows = L.rows.map((r) => ({ ...r, word: vocab(r.id), reads: (r.fate === 2 ? { en: `${READS[2].en}: ${vocab(x.shown[r.k])}`, zh: `${READS[2].zh}：${vocab(x.shown[r.k])}` } : READS[r.fate]) as I18n, graded: r.fate > 0, clm: G?.rows.find((q) => q.k === r.k)?.nll ?? null }));
	const max = Math.max(1, ...rows.filter((r) => r.graded).map((r) => r.nll), ...rows.map((r) => r.clm ?? 0));
	const draw = p.props.draw ? drawOf(p) : null;
	const caption: I18n = !rows.length
		? { en: 'No text yet.', zh: '还没有文字。' }
		: { en: `The masked-LM loss is the mean −log p over the ${L.graded.length} graded rows: ${L.graded.length ? `${fmt3(L.loss)} nats` : 'none, nothing is chosen'}. The ${L.seen.length} other rows add nothing.${G ? ` TinyGPT’s causal loss grades every token after the first, ${G.rows.length} rows: ${G.rows.length ? `${fmt3(G.loss)} nats` : 'none'}.` : ''}`, zh: `掩码语言模型的损失是 ${L.graded.length} 个被批改行的 −log p 平均值：${L.graded.length ? `${fmt3(L.loss)} 奈特` : '没有，因为没有选中任何词元'}。其余 ${L.seen.length} 行不计入。${G ? `TinyGPT 的因果损失批改第一个之后的每个词元，共 ${G.rows.length} 行：${G.rows.length ? `${fmt3(G.loss)} 奈特` : '没有'}。` : ''}` };
	return { rows, max, loss: L.loss, nGraded: L.graded.length, gpt: G, draw, sent: x.fromDraw, caption };
}

/** The raw input with row 0 of the draw: what training would show TinyBERT for the learner's text. */
export function withDraw(raw: MaskedRaw, d: { fate: number[]; swap: number[] }): MaskedRaw {
	const k = d.fate.findIndex((f) => f > 0);
	return { ...(raw ?? {}), fate: d.fate, swap: d.swap, picked: true, fromDraw: true, focus: k };
}

const meta = {
	key: 'masked-lm/loss',
	title: { en: 'The loss, position by position', zh: '逐个位置的损失' },
	kinds: ['trace'],
	describe: (p: P): I18n => {
		const S = lossOf(p);
		const r = S.rows.map((x) => ({ en: `token ${x.k} “${x.word}”, reads ${x.reads.en}: ${x.graded ? `graded, p ${fmtPct(x.p)}, −log p ${fmt3(x.nll)}` : 'not graded'}`, zh: `第 ${x.k} 个词元“${x.word}”，读到${x.reads.zh}：${x.graded ? `批改，p ${fmtPct(x.p)}，−log p ${fmt3(x.nll)}` : '不批改'}` }));
		return { en: `${r.map((x) => x.en).join('; ')}. ${S.caption.en}`, zh: `${r.map((x) => x.zh).join('；')}。${S.caption.zh}` };
	},
	demo: async () => demoProps({ gpt: true }),
	minHeight: 320
} satisfies ViewMeta;

export default meta;
