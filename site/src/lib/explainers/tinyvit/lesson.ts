// tinyvit: the guided lesson (design §7 outline, 11 steps). Every number is a {placeholder}: structure from the spec
// (dims()), live values from the trace of the learner's drawing, measured ones from facts.json ({fact:...}).
//   1 pixels → 2 patches (scan over the 16) → 3 patch embedding (49 → 48) → 4 [CLS] + positions → 5 q·k for [CLS]
//   (predict) → 6 the attention row over the drawing, softmax worked out (task) → 7 mix V → 8 MLP → 9 blocks 2-4 →
//   10 classes (label smoothing) → 11 TinyCNN on the same drawing (limits). Claims that are not computed live are
//   tested in claims.test.ts. Teaching review (R1): notes/review-tinyvit.md.
import type { ExplainerRuntime, LessonDef, Tensor } from '../../atlas/types/index.ts';
import { outputLabel, outputProbs } from '../../atlas/lesson/facts.ts';
import { num, pick } from '../../atlas/lesson/say.ts';
import { GRID, PATCH, SIDE, gridOf, inkedPatches } from './views/patch-grid/grid.ts';

type RT = ExplainerRuntime;
const T = (rt: RT, id: string, src = 'fwd'): Tensor | undefined => rt.sources[src]?.trace?.t[id];
const argmax = (a: ArrayLike<number>, from = 0, to = a.length) => {
	let b = from;
	for (let i = from + 1; i < to; i++) if (a[i] > a[b]) b = i;
	return b;
};

/** Structural numbers, read from the spec (never typed into the text). */
export function dims(rt: RT) {
	const L = (id: string) => rt.spec?.layers.find((l) => l.id === id);
	const side = Number(rt.spec?.inputs?.[0]?.shape[2] ?? NaN);
	const p = (L('patch')?.params ?? {}) as { kernel?: number; outChannels?: number };
	const k = p.kernel ?? NaN;
	const grid = side / k;
	const s = (L('B1.scores')?.params ?? {}) as { nHead?: number; dHead?: number };
	const blocks = rt.spec?.layers.filter((l) => /^B\d+\.attn$/.test(l.id)).length ?? NaN;
	const cnnK = Number((rt.sources.cnn?.model?.spec.layers[1]?.params as { kernel?: number } | undefined)?.kernel ?? NaN);
	return { side, k, grid, n: grid * grid, px: k * k, d: p.outChannels ?? NaN, rows: grid * grid + 1, heads: s.nHead ?? NaN, dh: s.dHead ?? NaN, blocks, hidden: Number(L('B1.fc1')?.outShape[1] ?? NaN), classes: Array.isArray(rt.spec?.classes) ? rt.spec.classes.length : NaN, cnnK };
}
/** the block-1 head the lesson zooms into (stats.py picks it; facts.json) */
const H1 = (rt: RT) => rt.num('fact:head');
const inked = (rt: RT) => {
	const x = T(rt, 'input');
	return x ? inkedPatches(x.data).filter(Boolean).length : NaN;
};
/** the [CLS] row's scores in head h of block b, over the 17 rows */
const clsScores = (rt: RT, h: number, layer = 'B1.scores') => {
	const s = T(rt, layer);
	if (!s) return null;
	const [, Tq, Tk] = s.shape;
	return s.data.subarray(h * Tq * Tk, h * Tq * Tk + Tk);
};
/** softmax of the [CLS] row's block-1 scores in head h, worked out: e^score of the top patch, and the sum over all rows */
export function softmaxWorked(rt: RT, h: number) {
	const s = clsScores(rt, h);
	if (!s) return null;
	const e = Array.from(s, Math.exp);
	const top = argmax(e, 1) - 1; // a patch (row top + 1)
	const eSum = e.reduce((a, b) => a + b, 0);
	return { top, sTop: s[top + 1], eTop: e[top + 1], eSum, w: e[top + 1] / eSum };
}
/** the learner drew (strokes, not a preset) */
const drew = (rt: RT) => {
	const s = (rt.input as { strokes?: unknown[] } | null)?.strokes;
	return Array.isArray(s) && s.length > 0;
};
/** top answer of a trace source: label and probability */
function topOf(rt: RT, src: string) {
	const st = rt.sources[src];
	const id = st?.model?.spec.outputs?.[0] ?? 'softmax';
	const t = st?.trace?.t[id];
	if (!t) return { label: '…', p: NaN, i: -1 };
	const p = outputProbs(t);
	const i = argmax(p);
	const l = outputLabel(st?.model?.spec ?? null, i);
	return { label: l, p: p[i], i };
}
/** label smoothing, as whole percentages (printed exactly, "91%" and "1%"): the training target of the right class and of
 *  each other class; NaN unless they are whole percentages (claims.test.ts checks 0.91 = 1 - 0.1 + 0.1 / 10) */
const wholePct = (v: number) => (Math.abs(v * 100 - Math.round(v * 100)) < 1e-9 ? Math.round(v * 100) : NaN);
const smooth = (rt: RT) => ({ target: wholePct(rt.num('fact:smoothTarget')), restPct: wholePct((1 - rt.num('fact:smoothTarget')) / (dims(rt).classes - 1)) });
/** row names for the token views: [CLS], then p0..p15 (patch t is row t + 1) */
const ROWS = ['[CLS]', ...Array.from({ length: 16 }, (_, t) => `p${t}`)];
const attn = (block: number, head: (rt: RT) => number, pickBlock = false) => (rt: RT) => ({ mode: 'attention', block, head: head(rt), pickBlock });
const gridAt = (rt: RT, block: number, head: number) => gridOf({ rt, source: 'fwd', at: { index: 0 }, props: { mode: 'attention', block, head } });

const lesson: LessonDef = {
	title: { en: 'TinyViT: a drawing read as patches with attention', zh: 'TinyViT：用注意力把画当作图块来读' },
	summary: {
		en: 'Cut your drawing into 16 patches, follow the [CLS] row as it looks at them, mixes what it sees and gives the answer, then compare with TinyCNN on the same drawing.',
		zh: '把你的画切成 16 个图块，跟着 [CLS] 行看向它们、混合看到的内容并给出答案，再和 TinyCNN 在同一幅画上比较。'
	},
	steps: [
		{
			auto: 'input',
			override: {
				title: { en: 'Your drawing becomes numbers', zh: '你的画变成数字' },
				look: { en: 'The coloured cells are the ink (near 1), the blank ones are 0: TinyCNN gets exactly these numbers too.', zh: '有颜色的格子是墨（接近 1），空白的格子是 0：TinyCNN 拿到的也正是这些数。' }
			}
		},
		{
			id: 'patches',
			source: 'patches',
			at: 'first',
			play: { from: 'first', to: 'last', interval: 450 },
			view: { key: 'tinyvit/patch-grid', props: () => ({ mode: 'patches' }) },
			provenance: 'live',
			title: { en: 'Cut into patches', zh: '切成图块' },
			narration: {
				en: 'TinyCNN slides a small {cnnK}×{cnnK} window over the pixels. A Vision Transformer (ViT) instead cuts the {side}×{side} grid into {grid}×{grid} = {n} patches of {k}×{k} = {px} pixels and treats each patch as one token, the way a language model treats a word. {inked} of the {n} patches of your drawing have ink (at least one pixel of {fact:inkThreshold|n2} or more).',
				zh: 'TinyCNN 用一个 {cnnK}×{cnnK} 的小窗口在像素上滑动。视觉 Transformer（ViT）则把 {side}×{side} 的网格切成 {grid}×{grid} = {n} 个图块，每块 {k}×{k} = {px} 个像素，并把每个图块当作一个词元，就像语言模型对待一个词。你的画的 {n} 个图块里有 {inked} 个有墨（至少有一个像素不小于 {fact:inkThreshold|n2}）。',
				vars: (rt) => ({ ...dims(rt), inked: inked(rt) })
			},
			look: {
				en: 'The outlined patch visits all {n} patches, row by row; on the right are its {px} numbers. “Output k of {n}” under the drawing is patch k − 1, because patches count from 0.',
				zh: '加框的图块逐行走遍全部 {n} 个图块；右边是它的 {px} 个数。画下面的“输出 k / {n}”就是第 k − 1 个图块，因为图块从 0 数起。',
				vars: (rt) => dims(rt)
			},
			terms: ['token']
		},
		{
			id: 'embed',
			source: 'fwd',
			at: { layer: 'patch' },
			select: (rt) => {
				const x = T(rt, 'input');
				const y = T(rt, 'patch');
				if (!x || !y) return { layer: 'patch', unit: [0, 0, 0] };
				const ink = Array.from({ length: 16 }, (_, k) => gridOf({ rt, source: 'fwd', at: { index: 0 }, props: { mode: 'patches', patch: k } })!.pixels.reduce((a, b) => a + b, 0));
				const p = argmax(ink);
				const vals = Array.from({ length: y.shape[0] }, (_, c) => Math.abs(y.data[c * 16 + p]));
				return { layer: 'patch', unit: [argmax(vals), Math.floor(p / 4), p % 4] };
			},
			view: 'zoom.weighted-sum',
			provenance: 'live',
			title: { en: 'Each patch becomes a row of numbers', zh: '每个图块变成一行数' },
			narration: {
				en: 'Each patch’s {px} pixels become {d} numbers. Each number is a weighted sum of all {px} pixels with its own learned weights, plus a bias, and every patch uses the same weights: a convolution whose {k}×{k} window of weights (kernel) jumps {k} pixels at a time (stride), so patches never overlap. Patch {p} (row {pr}, column {pc}, counting from 0; {which}) gives number {c}, which the stage calls channel {c}: its {px} products add up to {dot}, plus the bias {b}, makes {y}.',
				zh: '每个图块的 {px} 个像素变成 {d} 个数。每个数都是全部 {px} 个像素的加权和，用它自己的一组学来的权重，再加一个偏置；每个图块用的都是同一组权重。这就是一个卷积：它的 {k}×{k} 权重窗口（卷积核）每次跳 {k} 个像素（步长），所以图块互不重叠。第 {p} 个图块（第 {pr} 行、第 {pc} 列，从 0 数起；{which}）的第 {c} 个数，也就是画面上的通道 {c}：{px} 个乘积加起来是 {dot}，再加上偏置 {b}，得到 {y}。',
				vars: (rt) => {
					const u = rt.sel.unit ?? [];
					const y = T(rt, 'patch');
					const x = T(rt, 'input');
					const W = rt.model?.weights['patch.weight'];
					const B = rt.model?.weights['patch.bias'];
					let dot = NaN;
					if (x && W && u.length === 3) {
						dot = 0;
						for (let i = 0; i < PATCH; i++) for (let j = 0; j < PATCH; j++) dot += x.data[(u[1] * PATCH + i) * SIDE + u[2] * PATCH + j] * W.data[u[0] * PATCH * PATCH + i * PATCH + j];
					}
					const which = inked(rt) > 0 ? { en: 'the one with the most ink', zh: '墨最多的那个' } : { en: 'the pad is blank, so all patches are alike', zh: '画板是空的，所有图块都一样' };
					return { ...dims(rt), which, c: u[0] ?? NaN, p: u[1] * GRID + u[2], pr: u[1] ?? NaN, pc: u[2] ?? NaN, dot, b: B ? B.data[u[0]] : NaN, y: y ? y.data[(u[0] * GRID + u[1]) * GRID + u[2]] : NaN };
				}
			},
			look: { en: 'The pixels with ink: their products decide the sum.', zh: '有墨的像素：它们的乘积决定这个和。' },
			terms: ['convolution', 'kernel', 'stride', 'embedding']
		},
		{
			id: 'tokens',
			source: 'fwd',
			at: { layer: 'embed' },
			select: () => ({ layer: 'embed', unit: [0, 0] }),
			view: { key: 'layer.tensor', props: () => ({ tokens: ROWS }) },
			provenance: 'live',
			title: { en: 'One more row, [CLS], and positions', zh: '多一行 [CLS]，再加上位置' },
			narration: {
				en: 'Each patch is now a row of {d} numbers, a token. In front of the {n} patch rows the model puts one more learned row, [CLS] (short for “class”): it holds no pixels, is the same for every drawing, and the answer will be read from it at the end. Then a learned position row (a positional encoding) is added to each of the {rows} rows, because attention by itself cannot tell where a patch came from.',
				zh: '现在每个图块是一行 {d} 个数，也就是一个词元。模型在 {n} 个图块行前面再放一行学来的数，叫 [CLS]（“class”，类别的缩写）：它不含任何像素，对每幅画都一样，最后的答案就从它读出。然后给 {rows} 行中的每一行加上一行学来的位置数（位置编码），因为注意力本身分不清一个图块来自哪里。',
				vars: (rt) => dims(rt)
			},
			look: { en: 'Row 0 at the top is the [CLS] row; each colour is one of its {d} numbers.', zh: '最上面的第 0 行就是 [CLS] 行；每个颜色格是它的 {d} 个数之一。', vars: (rt) => dims(rt) },
			terms: ['token', 'positional-encoding']
		},
		{
			id: 'qk',
			source: 'fwd',
			at: { layer: 'B1.scores' },
			select: (rt) => {
				const h = H1(rt);
				const s = clsScores(rt, h);
				return { layer: 'B1.scores', unit: [h, 0, s ? argmax(s, 1) : 1] };
			},
			view: { key: 'zoom.weighted-sum', props: () => ({ tokens: ROWS }) },
			provenance: 'live',
			title: { en: 'The [CLS] row asks: query · key', zh: '[CLS] 行提问：查询 · 键' },
			predict: {
				id: 'tinyvit-cls-look',
				question: {
					en: 'The [CLS] row holds no pixels, so in block 1 its query is the same for every drawing. Will it look at the same patches for every drawing?',
					zh: '[CLS] 行不含像素，所以在第 1 块里，它的查询对每幅画都一样。它会对每幅画都看向同样的图块吗？'
				},
				options: [
					{ en: 'Yes: the same query looks at the same patches', zh: '会：同样的查询看向同样的图块' },
					{ en: 'No: the keys come from the patches, so the weights follow the drawing', zh: '不会：键来自图块，所以权重跟着画走' },
					{ en: 'It looks only at itself', zh: '它只看它自己' }
				],
				answer: 1,
				explain: {
					en: 'A score is query · key, and every key is computed from its own patch. The query stays fixed, but the keys change with your drawing, and so do the weights. In this head, patches with ink get clearly more weight than empty ones; the next step shows how much.',
					zh: '分数是查询 · 键，而每个键都是从它自己的图块算出来的。查询不变，但键随你的画而变，权重也就跟着变。在这个头里，有墨的图块明显比空图块得到更多权重；下一步会显示多多少。'
				}
			},
			narration: {
				en: 'Block 1 rescales each row (layer normalization), then three learned weighted sums turn each row into a query (what it looks for), a key (what it offers) and a value (what it passes on), each cut into {heads} heads of {dh} numbers. In head {h} (counting from 0), the [CLS] row’s score for a row is its query times that row’s key, number by number, added up and divided by √{dh}, which is × {inv|n3}. For patch {j} (row {r}; row 0 is [CLS]) that gives {s}, its highest score.',
				zh: '第 1 块先把每一行重新缩放（层归一化），再用三组学来的加权和把每一行变成查询（它要找什么）、键（它能提供什么）和值（它要传递什么），每个都分成 {heads} 个头、每头 {dh} 个数。在头 {h}（从 0 数起）里，[CLS] 行给某一行的分数，是它的查询和那一行的键逐个相乘、相加，再除以 √{dh}，也就是乘以 {inv|n3}。第 {j} 个图块在第 {r} 行（第 0 行是 [CLS]），它得到 {s}，是最高的分数。',
				vars: (rt) => {
					const u = rt.sel.unit ?? [];
					const s = clsScores(rt, u[0]);
					const D = dims(rt);
					return { ...D, inv: 1 / Math.sqrt(D.dh), h: u[0], j: u[2] - 1, r: u[2], s: s ? s[u[2]] : NaN };
				}
			},
			look: { en: 'The {dh} products of query and key: their sum, times {inv|n3}, is the score.', zh: '查询和键的这 {dh} 个乘积：它们的和乘以 {inv|n3} 就是分数。', vars: (rt) => ({ ...dims(rt), inv: 1 / Math.sqrt(dims(rt).dh) }) },
			terms: ['layer-norm', 'query-key-value', 'attention-head', 'attention']
		},
		{
			id: 'row',
			source: 'fwd',
			at: { layer: 'B1.attn' },
			view: { key: 'tinyvit/patch-grid', props: attn(1, H1) },
			provenance: 'live',
			title: { en: 'Where the [CLS] row looks', zh: '[CLS] 行看向哪里' },
			narration: {
				en: 'Softmax turns head {h}’s {rows} scores into weights that add up to 1: it raises e ({e|n4}) to the power of each score and divides by the sum of all {rows} results. Patch {top}’s score {sTop|n4} gives e to the {sTop|n4} = {eTop|n4}, and {eTop|n4} ÷ {eSum|n4} = {topW|pct}, the largest weight. {share} Over the {fact:testN} test drawings, a patch with ink gets on average {fact:headLean|n2} times the weight of an empty one in this head.',
				zh: 'Softmax 把头 {h} 的 {rows} 个分数变成加起来等于 1 的权重：对每个分数 s 求 e 的 s 次方（e = {e|n4}），再除以全部 {rows} 个结果的和。第 {top} 个图块的分数 {sTop|n4} 给出 e 的 {sTop|n4} 次方 = {eTop|n4}，{eTop|n4} ÷ {eSum|n4} = {topW|pct}，是最大的权重。{share}在 {fact:testN} 幅测试画上平均来看，这个头里一个有墨的图块得到的权重是空图块的 {fact:headLean|n2} 倍。',
				vars: (rt) => {
					const h = H1(rt);
					const g = gridAt(rt, 1, h);
					const sm = softmaxWorked(rt, h);
					const n = inked(rt);
					const self = num(g?.self ?? NaN, 'pct');
					const ink = num(g?.inkShare ?? NaN, 'pct');
					const t = (v: unknown, lang: 'en' | 'zh') => (typeof v === 'object' && v ? (v as Record<string, string>)[lang] : String(v));
					const share = pick(
						n > 0,
						{ en: `The ${n} patches with ink get ${t(ink, 'en')} in all, the [CLS] row itself ${t(self, 'en')}.`, zh: `${n} 个有墨的图块一共得到 ${t(ink, 'zh')}，[CLS] 行自己得到 ${t(self, 'zh')}。` },
						{ en: `Your pad is blank, so no patch has ink; the [CLS] row keeps ${t(self, 'en')} for itself.`, zh: `画板是空的，没有图块有墨；[CLS] 行留给自己 ${t(self, 'zh')}。` }
					);
					return { ...dims(rt), h, e: Math.E, top: sm?.top ?? NaN, sTop: sm?.sTop ?? NaN, eTop: sm?.eTop ?? NaN, eSum: sm?.eSum ?? NaN, topW: sm?.w ?? NaN, share: g && sm ? share : NaN };
				}
			},
			look: { en: 'Each patch shows its weight; darker patches get more, the dashed one the most; ● marks ink.', zh: '每个图块上写着它的权重；颜色越深得到越多，虚线框的最多；● 表示有墨。' },
			task: {
				prompt: {
					en: 'Draw one straight line: press “Change it here” under Your input and draw on the pad (press Clear first if it already has strokes). A line inks only some of the patches. Do those patches still get most of the [CLS] row’s attention?',
					zh: '画一条直线：点“你的输入”下面的“在这里修改”，在画板上画（如果画板上已经有笔画，先点“清空”）。一条线只给一部分图块上墨。这些图块还能得到 [CLS] 行的大部分注意力吗？'
				},
				done: (rt) => drew(rt) && inked(rt) >= 1 && inked(rt) <= 10,
				waiting: { en: 'Not yet: draw one straight line on the pad (a preset does not count).', zh: '还没有：在画板上画一条直线（选预设不算）。' },
				hint: { en: 'A single stroke covers only part of the grid. Add up the weights of the patches marked ●, or read the narration: it updates as you draw.', zh: '一笔只覆盖网格的一部分。把标着 ● 的图块的权重加起来，或者看讲解：它会随着你画而更新。' }
			},
			linger: 12000,
			terms: ['softmax', 'attention']
		},
		{
			id: 'mix',
			source: 'fwd',
			at: { layer: 'B1.mix' },
			select: (rt) => {
				const h = H1(rt);
				const m = T(rt, 'B1.mix');
				const dh = dims(rt).dh;
				if (!m || !Number.isFinite(h)) return { layer: 'B1.mix', unit: [0, 0] };
				const abs = Array.from(m.data.subarray(h * dh, (h + 1) * dh), Math.abs);
				return { layer: 'B1.mix', unit: [0, h * dh + argmax(abs)] };
			},
			view: { key: 'zoom.weighted-sum', props: () => ({ tokens: ROWS }) },
			provenance: 'live',
			title: { en: 'Mix what it sees', zh: '混合看到的内容' },
			narration: {
				en: 'The [CLS] row now collects what it looks at. In each head, every row’s value is multiplied by that row’s weight from the last step (column a) and the {rows} results are added up. Number {f} of the [CLS] row’s result is {y} (counting from 0; head {h} fills numbers {f0} to {f1}). A linear layer merges the {heads} heads back into {d} numbers, which are added to the row the [CLS] row started from (a residual connection), so it keeps what it had and gains what it read.',
				zh: '现在 [CLS] 行把它看到的内容收集起来。在每个头里，每一行的值乘以这一行在上一步得到的权重（a 列），再把 {rows} 个结果加起来。[CLS] 行结果的第 {f} 个数是 {y}（从 0 数起；头 {h} 负责第 {f0} 到第 {f1} 个数）。一个线性层把 {heads} 个头重新合成 {d} 个数，再加回 [CLS] 行原来的那一行（残差连接），所以它保留了原有的内容，又得到了读到的内容。',
				vars: (rt) => {
					const u = rt.sel.unit ?? [];
					const D = dims(rt);
					const m = T(rt, 'B1.mix');
					const h = Math.floor(u[1] / D.dh);
					return { ...D, h, f: u[1], f0: h * D.dh, f1: (h + 1) * D.dh - 1, y: m ? m.data[u[1]] : NaN };
				}
			},
			look: { en: 'Column v is each row’s value, column a its weight: rows with weight near 0 add almost nothing.', zh: 'v 列是每一行的值，a 列是它的权重：权重接近 0 的行几乎没有贡献。' },
			terms: ['query-key-value', 'residual-connection']
		},
		{
			id: 'mlp',
			source: 'fwd',
			at: { layer: 'B1.gelu' },
			select: (rt) => {
				const f = T(rt, 'B1.fc1');
				let b = 0;
				if (f) for (let i = 1; i < f.shape[1]; i++) if (f.data[i] < f.data[b]) b = i;
				return { layer: 'B1.gelu', unit: [0, b] };
			},
			view: { key: 'zoom.elementwise', props: () => ({ tokens: ROWS }) },
			provenance: 'live',
			title: { en: 'Then each row thinks alone: the MLP', zh: '然后每行各自思考：MLP' },
			narration: {
				en: 'Next, every row passes on its own through a small two-layer network (an MLP): {d} numbers grow to {hidden}, GELU lets large positive ones through almost unchanged and shrinks negative ones toward 0, and a second layer brings them back to {d}, added to the row again. Attention moves information between rows; the MLP works on each row by itself. In the [CLS] row, {neg} of the {hidden} numbers are negative before GELU; the most negative, {x}, becomes {y}.',
				zh: '接着，每一行各自通过一个小的两层网络（MLP）：{d} 个数先扩展成 {hidden} 个，GELU 让较大的正数几乎原样通过，把负数往 0 压缩，第二层再把它们变回 {d} 个，并再次加回这一行。注意力在行与行之间传递信息；MLP 则对每一行单独加工。在 [CLS] 行里，经过 GELU 之前 {hidden} 个数中有 {neg} 个是负的；最负的 {x} 变成了 {y}。',
				vars: (rt) => {
					const f = T(rt, 'B1.fc1');
					const g = T(rt, 'B1.gelu');
					const k = rt.sel.unit?.[1] ?? 0;
					const D = dims(rt);
					return { ...D, neg: f ? Array.from(f.data.subarray(0, D.hidden)).filter((v) => v < 0).length : NaN, x: f ? f.data[k] : NaN, y: g ? g.data[k] : NaN };
				}
			},
			look: { en: 'The point on the GELU curve: a negative number comes out closer to 0.', zh: 'GELU 曲线上的点：负数出来时更接近 0。' },
			terms: ['mlp', 'activation-function']
		},
		{
			id: 'blocks',
			source: 'fwd',
			at: (rt) => ({ layer: `B${dims(rt).blocks}.attn` }),
			view: { key: 'tinyvit/patch-grid', props: (rt) => ({ mode: 'attention', block: dims(rt).blocks, head: 0, pickBlock: true }) },
			provenance: 'live',
			title: { en: 'The other blocks do the same again', zh: '其余的块把同样的事再做几遍' },
			narration: {
				en: 'Blocks 2 to {blocks} repeat the same recipe (normalize, attend, mix, MLP) with their own weights, so the [CLS] row reads the patches {blocks} times, each time with what it has already gathered. {here} Over the {fact:testN} test drawings, block {blocks} puts {fact:b4InkShare|pct} of the [CLS] row’s attention on patches with ink, while only {fact:inkedShare|pct} of all patches have ink.',
				zh: '第 2 到第 {blocks} 块用各自的权重重复同样的步骤（归一化、注意力、混合、MLP），所以 [CLS] 行一共读 {blocks} 遍图块，每一遍都带着之前收集到的内容。{here}在 {fact:testN} 幅测试画上，第 {blocks} 块把 [CLS] 行 {fact:b4InkShare|pct} 的注意力放在有墨的图块上，而所有图块中只有 {fact:inkedShare|pct} 有墨。',
				vars: (rt) => {
					const D = dims(rt);
					const ink = num(gridAt(rt, D.blocks, 0)?.inkShare ?? NaN, 'pct');
					const t = (lang: 'en' | 'zh') => (typeof ink === 'object' ? ink[lang] : String(ink));
					const here = pick(
						inked(rt) > 0,
						{ en: `In head 0 of block ${D.blocks}, ${t('en')} of its attention lands on patches with ink.`, zh: `在第 ${D.blocks} 块的头 0 里，${t('zh')} 的注意力落在有墨的图块上。` },
						{ en: 'Your pad is blank, so here no patch has ink.', zh: '画板是空的，所以这里没有图块有墨。' }
					);
					return { ...D, here };
				}
			},
			look: { en: 'Switch blocks and heads: each head looks at its own places. Every patch row attends to the others too, so later blocks read patches that already carry news from other patches.', zh: '切换块和头：每个头看向各自的地方。每个图块行也在关注其他行，所以后面的块读到的图块已经带着其他图块的信息。' },
			terms: ['transformer', 'attention-head']
		},
		{
			id: 'classes',
			source: 'fwd',
			at: { layer: 'softmax' },
			view: 'kit.probbars',
			provenance: 'live',
			title: { en: 'The answer comes from the [CLS] row', zh: '答案来自 [CLS] 行' },
			narration: {
				en: 'After block {blocks} and a last layer normalization, only the [CLS] row’s {d} numbers go on; the {n} patch rows are dropped. A linear layer turns them into {classes} class scores, and softmax turns those into shares: TinyViT says “{top.label}” with {top.p|pct}, then “{runnerUp.label}” with {runnerUp.p|pct}. It was trained with label smoothing, which sets the target for the right class to {target}% instead of certainty and {restPct}% for each other class, so even its surest answers stay near that: at most {fact:maxTopP|pct} on the {fact:testN} test drawings.',
				zh: '经过第 {blocks} 块和最后一次层归一化之后，只有 [CLS] 行的 {d} 个数继续往下走；{n} 个图块行都被丢掉。一个线性层把它们变成 {classes} 个类别分数，softmax 再把这些分数变成份额：TinyViT 的回答是“{top.label}”，{top.p|pct}；其次是“{runnerUp.label}”，{runnerUp.p|pct}。训练时用了标签平滑：正确类别的目标是 {target}%，而不是百分之百，其余每个类别是 {restPct}%；所以即使最有把握的回答也停在这附近：在 {fact:testN} 幅测试画上最高只有 {fact:maxTopP|pct}。',
				vars: (rt) => ({ ...dims(rt), ...smooth(rt) })
			},
			look: { en: 'The longest bar is TinyViT’s answer; on a clear drawing it stops near {target}%, and the other bars sit near {restPct}%.', zh: '最长的条就是 TinyViT 的回答；画得清楚时它停在 {target}% 附近，其他条在 {restPct}% 附近。', vars: (rt) => smooth(rt) },
			terms: ['logit', 'softmax', 'classifier']
		},
		{
			id: 'compare',
			source: 'fwd',
			at: { layer: 'softmax' },
			view: 'compare',
			compare: { a: 'fwd', b: 'cnn', label: [{ en: 'TinyViT', zh: 'TinyViT' }, { en: 'TinyCNN', zh: 'TinyCNN' }] },
			provenance: 'live',
			kindOf: 'limits',
			title: { en: 'TinyCNN on the same drawing', zh: '同一幅画上的 TinyCNN' },
			narration: {
				en: 'TinyCNN reads the same numbers with small sliding windows: here it says “{cnn}” ({cnnP|pct}), TinyViT “{vit}” ({vitP|pct}). On the {fact:testN} test drawings they agree {fact:agree|pct} of the time; TinyViT is right {fact:testAcc|pct} of the time, TinyCNN {fact:cnnTestAcc|pct}, a small gap between two different training recipes, so it does not show which design is better. But TinyViT needs {fact:params} learned numbers, about {ratio} times TinyCNN’s {fact:cnnParams}: a convolution is built to look at neighbouring pixels together, while attention can look anywhere and has to learn from data where to look.',
				zh: 'TinyCNN 用小的滑动窗口读同样的数：这里它说“{cnn}”（{cnnP|pct}），TinyViT 说“{vit}”（{vitP|pct}）。在 {fact:testN} 幅测试画上，两者有 {fact:agree|pct} 的时候答案相同；TinyViT 的正确率是 {fact:testAcc|pct}，TinyCNN 是 {fact:cnnTestAcc|pct}。两者的训练方法不同，这点小差距说明不了哪种设计更好。但 TinyViT 需要 {fact:params} 个学来的数，大约是 TinyCNN（{fact:cnnParams} 个）的 {ratio} 倍：卷积天生就把相邻的像素放在一起看，而注意力可以看任何地方，必须从数据里学会该看哪里。',
				vars: (rt) => {
					const a = topOf(rt, 'fwd');
					const b = topOf(rt, 'cnn');
					return { vit: a.label, vitP: a.p, cnn: b.label, cnnP: b.p, ratio: Math.round(rt.num('fact:params') / rt.num('fact:cnnParams')) };
				}
			},
			look: {
				en: 'A and B: the same drawing, two kinds of model. TinyViT’s top bar stops near {target}% because of label smoothing; TinyCNN was trained without it, so its top bar can reach certainty. A longer bar does not mean a more accurate model.',
				zh: 'A 和 B：同一幅画，两种模型。因为标签平滑，TinyViT 最长的条停在 {target}% 附近；TinyCNN 训练时没有用它，所以它最长的条可以到“完全确定”。条更长并不说明模型更准。',
				vars: (rt) => smooth(rt)
			},
			task: {
				prompt: { en: 'Find a drawing on which the two models give different answers.', zh: '找一幅让两个模型给出不同答案的画。' },
				done: (rt) => topOf(rt, 'fwd').i >= 0 && topOf(rt, 'fwd').i !== topOf(rt, 'cnn').i,
				waiting: { en: 'Not yet: both models still give the same answer.', zh: '还没有：两个模型的回答还是一样的。' },
				hint: { en: 'Try the ladder or envelope preset (classes neither model knows), or draw a shape between two classes.', zh: '试试“梯子”或“信封”预设（两个模型都不认识这两类），或者画一个介于两类之间的形状。' }
			},
			terms: ['convolution', 'attention', 'parameter']
		}
	],
	fixtures: [
		{ id: 'line', label: { en: 'one straight line', zh: '一条直线' }, raw: { strokes: [{ points: [{ x: 0.1, y: 0.2 }, { x: 0.9, y: 0.8 }] }] } },
		{ id: 'hline', label: { en: 'a horizontal line', zh: '一条横线' }, raw: { strokes: [{ points: [{ x: 0.2, y: 0.5 }, { x: 0.8, y: 0.5 }] }] } },
		{ id: 'vline', label: { en: 'a vertical line', zh: '一条竖线' }, raw: { strokes: [{ points: [{ x: 0.5, y: 0.2 }, { x: 0.5, y: 0.8 }] }] } },
		{ id: 'corner', label: { en: 'a diagonal from corner to corner', zh: '一条从角到角的对角线' }, raw: { strokes: [{ points: [{ x: 0.05, y: 0.05 }, { x: 0.95, y: 0.95 }] }] } }
	]
};

export default lesson;
