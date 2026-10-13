// self-supervised-objectives: every objective as a mask pattern on the learner's sentence or on the drawing (pure TS). A
// pattern says, token by token or patch by patch, what the model sees, what is hidden from it and what it must predict.
// Causal LM and masked LM are what TinyGPT and TinyBERT were trained on (their steps are live); the rest are drawings
// (provenance illustrative): computed on the learner's text with each paper's settings (patterns.ts), no model trained.
import type { I18n, Tensor } from '../../atlas/types/index.ts';
import { visibleToken } from '../../atlas/core/format.ts';
import { contentOf, maskOf, splitAt, vocab, PRESETS, type MaskedRaw } from './adapters/masked-text/adapter.ts';
import { bartSpans, cropViews, jepaBlock, maePatches, SENTINELS, t5Spans } from './patterns.ts';

/** seen: the model reads it; gone: hidden from it; target: what it must predict; mark: a special or sentinel token */
export type Role = 'seen' | 'gone' | 'target' | 'mark';
export type Chip = { text: string; role: Role };
export type Line = { label: I18n; chips: Chip[] };
export type Picture = { label: I18n; mode: 'mae' | 'crop' | 'noise'; hidden?: boolean[]; crop?: { x: number; y: number; size: number; flip: boolean }; target?: boolean };
export interface Objective {
	id: string;
	name: I18n;
	/** the method that made it known, and a model of this site that uses it */
	example: I18n;
	sees: I18n;
	predicts: I18n;
	live: boolean;
	href?: string;
	lines: Line[];
	pictures: Picture[];
}

const f = (en: string, zh: string): I18n => ({ en, zh });
const chip = (id: number, role: Role): Chip => ({ text: visibleToken(vocab(id)), role });
const mark = (text: string, role: Role = 'mark'): Chip => ({ text, role });
/** A frame state of the 'hide' process: fate 0 not chosen, 1 [MASK], 2 random, 3 kept (process.ts FATE); shown: the ids.
 *  A chosen token that stays as it is is drawn as seen (the model reads it) and is still in the target row. */
type HideState = { fate?: Tensor; shown?: Tensor } | null | undefined;

/** A sentence from another example text, for the NotNext pair. */
function otherSentence(c: number[]): number[] {
	const p = PRESETS.find((q) => q.id !== 'park' && q.raw.ids.join() !== c.join()) ?? PRESETS[1];
	return p.raw.ids;
}

export function objectives(raw: MaskedRaw, hide?: HideState): Objective[] {
	const c = contentOf(raw);
	const n = c.length;
	const k = maskOf(raw);
	const j = splitAt(c);
	const A = j >= 0 ? c.slice(0, j + 1) : c;
	const B = j >= 0 ? c.slice(j + 1) : [];
	const t5 = t5Spans(n);
	const bart = bartSpans(n);
	const mae = maePatches();
	const fate = (i: number) => (hide?.fate ? Number(hide.fate.data[i]) : i === k ? 1 : 0);
	// MLM: the 80/10/10 draw of the 'hide' process when its frame is given, else the one token hidden in the lesson
	const mlmIn = c.map((id, i) => {
		const x = fate(i);
		return x === 1 ? mark('[MASK]', 'gone') : x === 2 ? chip(Number(hide?.shown?.data[i] ?? id), 'gone') : chip(id, 'seen');
	});
	const mlmOut = c.flatMap((id, i) => (fate(i) > 0 ? [chip(id, 'target')] : []));
	const t5In: Chip[] = [];
	const t5Out: Chip[] = [];
	c.forEach((id, i) => {
		const s = t5.span[i];
		if (s < 0) return void t5In.push(chip(id, 'seen'));
		if (i === 0 || t5.span[i - 1] !== s) (t5In.push(mark(`<${SENTINELS[s]}>`)), t5Out.push(mark(`<${SENTINELS[s]}>`)));
		t5Out.push(chip(id, 'target'));
	});
	if (t5.spans) t5Out.push(mark(`<${SENTINELS[t5.spans]}>`));
	const bartIn: Chip[] = [];
	c.forEach((id, i) => {
		if (!bart.masked[i]) bartIn.push(chip(id, 'seen'));
		else if (i === 0 || !bart.masked[i - 1]) bartIn.push(mark('[MASK]', 'gone'));
	});
	// FIM: the middle is the hidden token and its two neighbours; the start and the end come first, the middle last
	const lo = Math.max(0, k - 1);
	const hi = Math.min(n, k + 2);
	const fim = n ? [mark('<PRE>'), ...c.slice(0, lo).map((id) => chip(id, 'seen')), mark('<SUF>'), ...c.slice(hi).map((id) => chip(id, 'seen')), mark('<MID>'), ...c.slice(lo, hi).map((id) => chip(id, 'target'))] : [];
	const nsp = (b: number[], label: string, l: I18n): Line => ({ label: l, chips: [mark('[CLS]'), ...A.map((id) => chip(id, 'seen')), mark('[SEP]'), ...b.map((id) => chip(id, 'seen')), mark('[SEP]'), mark(label, 'target')] });
	return [
		{
			id: 'clm', live: true, href: '/m/tinygpt/',
			name: f('Causal language modelling (next token)', '因果语言建模（预测下一个词元）'),
			example: f('GPT; TinyGPT on this site', 'GPT；本站的 TinyGPT'),
			sees: f('the tokens before the position', '该位置之前的词元'),
			predicts: f('the next token, at every position at once', '下一个词元，每个位置同时预测'),
			lines: [{ label: f('predicting one token', '预测其中一个词元'), chips: c.map((id, i) => chip(id, i < k ? 'seen' : i === k ? 'target' : 'gone')) }],
			pictures: []
		},
		{
			id: 'mlm', live: true, href: '/m/tinybert/',
			name: f('Masked language modelling', '掩码语言建模'),
			example: f('BERT; TinyBERT on this site', 'BERT；本站的 TinyBERT'),
			sees: f('the whole text, with some tokens replaced', '整段文字，其中一些词元被替换'),
			predicts: f('the chosen tokens only', '只预测被选中的词元'),
			lines: [{ label: f('input', '输入'), chips: [mark('[CLS]'), ...mlmIn, mark('[SEP]')] }, { label: f('target', '目标'), chips: mlmOut }],
			pictures: []
		},
		{
			id: 'nsp', live: false,
			name: f('Next-sentence prediction', '下一句预测'),
			example: f('the original BERT', '最初的 BERT'),
			sees: f('two sentences, A and B', '两句话：A 和 B'),
			predicts: f('one label: whether B came right after A', '一个标签：B 是否紧跟在 A 后面'),
			lines: [...(B.length ? [nsp(B, 'IsNext', f('a real pair', '真实的一对'))] : []), nsp(otherSentence(c), 'NotNext', f('B from another text', 'B 来自另一段文字'))],
			pictures: []
		},
		{
			id: 'span', live: false,
			name: f('Span corruption', '片段破坏'),
			example: f('T5', 'T5'),
			sees: f('the text with each hidden span replaced by one sentinel', '每个被藏片段换成一个哨兵词元后的文字'),
			predicts: f('only the hidden spans, each after its sentinel', '只预测被藏的片段，各自跟在自己的哨兵后面'),
			lines: [{ label: f('input', '输入'), chips: t5In }, { label: f('target', '目标'), chips: t5Out }],
			pictures: []
		},
		{
			id: 'denoise', live: false,
			name: f('Denoising (text infilling)', '去噪（文本填充）'),
			example: f('BART', 'BART'),
			sees: f('the text with spans cut out, one [MASK] per span', '挖掉片段的文字，每个片段只剩一个 [MASK]'),
			predicts: f('the whole original text', '完整的原文'),
			lines: [{ label: f('input', '输入'), chips: bartIn }, { label: f('target', '目标'), chips: c.map((id) => chip(id, 'target')) }],
			pictures: []
		},
		{
			id: 'fim', live: false,
			name: f('Fill in the middle', '中间填充'),
			example: f('code models', '代码模型'),
			sees: f('the start and the end, moved in front', '开头和结尾，挪到最前面'),
			predicts: f('the middle, token by token', '中间部分，一个词元一个词元地预测'),
			lines: [{ label: f('reordered text', '重新排列的文字'), chips: fim }],
			pictures: []
		},
		{
			id: 'mae', live: false,
			name: f('Masked autoencoding', '掩码自编码'),
			example: f('MAE', 'MAE'),
			sees: f('the visible patches only', '只有没被藏的图块'),
			predicts: f('the pixels of the hidden patches', '被藏图块的像素'),
			lines: [],
			pictures: [{ label: f('input', '输入'), mode: 'mae', hidden: mae.hidden }, { label: f('target', '目标'), mode: 'mae', hidden: mae.hidden.map((h) => !h), target: true }]
		},
		{
			id: 'jepa', live: false,
			name: f('Predicting hidden patches as numbers', '把被藏图块预测成一串数'),
			example: f('JEPA', 'JEPA'),
			sees: f('the patches outside a hidden block of neighbouring patches', '一块相邻图块被藏起后，其余的图块'),
			predicts: f('another network’s summary numbers for the hidden block, not its pixels', '另一个网络为被藏的那块图块算出的概括数值，而不是像素'),
			lines: [],
			pictures: [{ label: f('input (the hatched block is the target)', '输入（打斜线的那块是目标）'), mode: 'mae', hidden: jepaBlock().hidden }]
		},
		{
			id: 'contrastive', live: false, href: '/m/tinyclip/',
			name: f('Contrastive learning', '对比学习'),
			example: f('SimCLR; TinyCLIP on this site pairs pictures with captions', 'SimCLR；本站的 TinyCLIP 把图片和文字配对'),
			sees: f('two altered views of one image, and other images', '同一张图的两个变形视图，以及别的图片'),
			predicts: f('which views belong together; no hidden part has to come back', '哪些视图属于同一张图；不需要找回任何被藏的部分'),
			lines: [],
			pictures: cropViews().map((cr, i) => ({ label: i ? f('view 2 (mirrored)', '视图 2（镜像）') : f('view 1', '视图 1'), mode: 'crop' as const, crop: cr }))
		},
		{
			id: 'diffusion', live: false,
			name: f('Denoising (added noise)', '去噪（加噪声）'),
			example: f('diffusion models', '扩散模型'),
			sees: f('every pixel, with noise added', '加了噪声的每一个像素'),
			predicts: f('the noise that was added', '被加上的噪声'),
			lines: [],
			pictures: [{ label: f('input', '输入'), mode: 'noise' }]
		}
	];
}
