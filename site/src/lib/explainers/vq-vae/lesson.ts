// vq-vae: the guided lesson (10 steps). Every number is a {placeholder}: structure from the spec (vq.ts dims), live
// values from the trace of the learner's drawing and from the decoder (vq.ts), measured ones from facts.json ({fact:...}).
//   1 patches → 2 the encoder: z_e per patch → 3 the codebook → 4 which code? (predict) → 5 the drawing as indices →
//   6 the decoder redraws → 7 swap a code (task: add a stroke) → 8 training: the straight-through estimator →
//   9 codebook usage → 10 limits: rough pieces and dead codes. Claims not computed live: claims.test.ts.
import type { ExplainerRuntime, I18n, LessonDef, TmplVal } from '../../atlas/types/index.ts';
import { fmt3, fmtNum, fmtSetting } from '../../atlas/core/format.ts';
import { T, blankOf, cellOf, cellVec, codesOf, dims, distOf, gradAt, lossOf, rankOf, swapOf } from './vq.ts';

type RT = ExplainerRuntime;
const words = (v: I18n) => v as unknown as TmplVal;
/** a list of numbers as one placeholder: "0.12, −0.5, …" (3 significant digits, the typographic minus) */
const list = (xs: number[]): TmplVal => (xs.length && xs.every(Number.isFinite) ? xs.map(fmt3).join(', ') : NaN);
/** "row 1, column 2" of cell c (counting from 0 is said in the narration) */
const at = (rt: RT, c: number, G = dims(rt).G) => words({ en: `row ${Math.floor(c / G)}, column ${c % G}`, zh: `第 ${Math.floor(c / G)} 行第 ${c % G} 列` });
const ink = (t: ArrayLike<number> | undefined) => (t ? Array.from(t).filter((v) => v > 0.5).length : NaN);
/** the code indices of the drawing: how many differ, how many patches sit on the blank code */
function used(rt: RT) {
	const codes = codesOf(rt), blank = rt.model ? blankOf(rt.model) : NaN, none = !codes.length;
	return { nUsed: none ? NaN : new Set(codes).size, nBlank: none ? NaN : codes.filter((k) => k === blank).length, blank };
}
/** the outlined patch: where it is, its z_e, its nearest and second-nearest codes */
function snap(rt: RT) {
	const c = cellOf(rt), r = rankOf(distOf(rt, c));
	return { where: at(rt, c), ze: list(cellVec(T(rt, 'ze'), c)), k: r[0]?.k ?? NaN, d: r[0]?.d ?? NaN, k2: r[1]?.k ?? NaN, d2: r[1]?.d ?? NaN };
}
export const ADD = 5;
const LADDER = [[{ x: 0.35, y: 0.1 }, { x: 0.35, y: 0.9 }], [{ x: 0.65, y: 0.1 }, { x: 0.65, y: 0.9 }], [{ x: 0.35, y: 0.3 }, { x: 0.65, y: 0.3 }], [{ x: 0.35, y: 0.5 }, { x: 0.65, y: 0.5 }], [{ x: 0.35, y: 0.7 }, { x: 0.65, y: 0.7 }]]; // the task: a swap that adds at least this many pixels above half ink to its block

const lesson: LessonDef = {
	title: { en: 'A VQ-VAE: a drawing as 16 code numbers', zh: 'VQ-VAE：一幅画变成 16 个码字编号' },
	summary: {
		en: 'Cut your drawing into patches, turn each into a few numbers, snap them to the nearest code, redraw from the indices, swap a code, then see how training gets past the snap, how the codes are used, and the limits.',
		zh: '把你的画切成图块，每块变成几个数，再吸附到最近的码字；从编号重画，换一个码字；然后看训练怎样越过“吸附”这一步、码字用得怎样，以及局限。'
	},
	steps: [
		{
			id: 'patches', source: 'fwd', at: { layer: 'input' }, view: { key: 'vq-vae/grid', props: () => ({ mode: 'patches' }) }, provenance: 'live',
			title: { en: 'Your drawing, cut into patches', zh: '你的画，切成图块' },
			narration: {
				en: 'The model sees your drawing as {n} numbers, 0 for paper and 1 for ink. A VQ-VAE (vector-quantized autoencoder) cuts them into a {G}×{G} grid of patches, {P}×{P} pixels each, and describes every patch by one code picked from a list of {K}. The VAE page squeezes a whole drawing into 2 numbers; here each patch keeps its own code. It learned from {fact:trainN} drawings of {fact:classesN} kinds.',
				zh: '模型看到的你的画是 {n} 个数：0 是纸，1 是墨。VQ-VAE（向量量化自编码器）把它们切成 {G}×{G} 的图块网格，每块 {P}×{P} 个像素，并用一张 {K} 项清单里的一个码字来描述每个图块。VAE 那一页把整幅画挤成 2 个数；这里每个图块都有自己的码字。这个模型是从 {fact:classesN} 类、共 {fact:trainN} 幅画里学出来的。',
				vars: (rt) => dims(rt)
			},
			look: { en: 'The outlined patch, at first the one with the most ink; pick another patch to follow it through the next steps instead.', zh: '框出的图块，一开始是墨最多的那块；点别的图块，后面几步就跟着那一块走。' },
			terms: ['autoencoder', 'vector-quantization']
		},
		{
			id: 'encode', source: 'fwd', at: { layer: 'enc1' },
			select: (rt) => {
				const c = cellOf(rt), h = T(rt, 'enc1'), { G } = dims(rt), v = h ? cellVec(h, c) : [0];
				return { layer: 'enc1', unit: [v.reduce((b, x, i) => (x > v[b] ? i : b), 0), Math.floor(c / G), c % G] };
			},
			view: 'zoom.weighted-sum', provenance: 'live',
			title: { en: 'The encoder: a few numbers per patch', zh: '编码器：每个图块变成几个数' },
			narration: {
				en: 'The encoder reads one patch at a time. A {P}×{P} kernel moving {P} pixels per step (its stride) takes a weighted sum of a patch’s {block} pixels; {h} kernels, each followed by ReLU (negative sums become 0), give {h} numbers per patch. A second layer mixes them into just {D}, called z_e. The outlined patch, at {where} (counting from 0), gets z_e = ({ze}).',
				zh: '编码器一次读一个图块。一个 {P}×{P} 的卷积核每次移动 {P} 个像素（这就是步幅），对一个图块的 {block} 个像素求加权和；{h} 个卷积核各自再接 ReLU（负的和变成 0），每个图块得到 {h} 个数。第二层再把这 {h} 个数混合成 {D} 个，记作 z_e。框出的图块在{where}（从 0 数起），它的 z_e = ({ze})。',
				vars: (rt) => ({ ...dims(rt), ...snap(rt) })
			},
			look: { en: 'One kernel’s {block} products with the outlined patch, and their sum.', zh: '一个卷积核和框出图块的 {block} 个乘积，以及它们的和。', vars: (rt) => dims(rt) },
			terms: ['encoder-decoder', 'convolution', 'kernel', 'stride', 'relu']
		},
		{
			id: 'codebook', source: 'fwd', at: { layer: 'zq' }, view: { key: 'vq-vae/codebook', props: () => ({ mode: 'gallery' }) }, provenance: 'live',
			title: { en: 'The codebook: a list of stock pieces', zh: '码本：一套现成的小块' },
			narration: {
				en: 'Training also learned a codebook: {K} codes, numbered 0 to {last}, each a list of {D} numbers like z_e. To see what a code means, put it into one patch, give every other patch the code of empty paper, code {blank}, and let the decoder (the half of the model that redraws from codes) draw it. Each code is a small piece of stroke, not a whole kind of drawing; some look faint because the decoder also reads each patch’s neighbours, here all empty. Your drawing uses {nUsed} of them.',
				zh: '训练还学出了一个码本：{K} 个码字，编号从 0 到 {last}，每个码字和 z_e 一样，是 {D} 个数。想看一个码字代表什么，就把它放进一个图块，其余图块都放空白纸的码字，即码字 {blank}，再让解码器（模型里根据码字重画的那一半）画出来。每个码字只是一小段笔画，而不是一整类画；有些看起来很淡，因为解码器还会读每个图块的邻居，而这里的邻居全是空白。你的画用到了其中 {nUsed} 个。',
				vars: (rt) => ({ ...dims(rt), last: dims(rt).K - 1, ...used(rt) })
			},
			look: { en: 'Outlined: the codes your drawing uses; ×n says how many of its patches use each.', zh: '加框的是你的画用到的码字；×n 表示你的画里有几个图块用了它。' },
			terms: ['codebook']
		},
		{
			id: 'snap', source: 'fwd', at: { layer: 'zq' }, view: { key: 'vq-vae/nearest', props: () => ({ mode: 'snap' }) }, provenance: 'live',
			title: { en: 'Which code will this patch snap to?', zh: '这个图块会吸附到哪个码字？' },
			predict: {
				id: 'vq-vae-snap',
				question: { en: 'Which of the codes will the outlined patch snap to?', zh: '框出的图块会吸附到哪个码字？' },
				options: [
					{ en: 'The code whose numbers are nearest to z_e: the smallest sum of squared differences', zh: '数值离 z_e 最近的码字：差的平方和最小' },
					{ en: 'The code whose numbers add up to the most', zh: '各个数加起来最大的码字' },
					{ en: 'A new code made just for this patch', zh: '专为这个图块新造的一个码字' }
				],
				answer: 0,
				explain: {
					en: 'The quantizer works out the squared distance Σ (z_e − e_k)² from z_e to every code e_k and keeps the smallest; the stage lists them all, nearest first, and works the winning sum out number by number. Nothing new is made: after training the codebook is fixed.',
					zh: '量化器算出 z_e 到每个码字 e_k 的距离平方 Σ (z_e − e_k)²，留下最小的那个；舞台按从近到远列出全部距离，并把胜出的那个和逐个数算出来。不会新造任何东西：训练结束后码本就固定了。'
				}
			},
			narration: {
				en: 'The outlined patch is now z_e = ({ze}). The quantizer must replace these {D} numbers by one of the {K} codes, and from then on the patch is nothing but that code’s index. Commit to a rule before you look.',
				zh: '框出的图块现在是 z_e = ({ze})。量化器必须把这 {D} 个数换成 {K} 个码字之一，从此这个图块就只剩下那个码字的编号。先选定一条规则，再看舞台。',
				vars: (rt) => ({ ...dims(rt), ...snap(rt) })
			},
			look: { en: 'The list of distances, nearest first, and the table that adds up the winner’s.', zh: '从近到远的距离列表，以及把胜出码字的距离逐项相加的表格。' },
			terms: ['vector-quantization']
		},
		{
			id: 'indices', source: 'fwd', at: { layer: 'zq' }, view: { key: 'vq-vae/grid', props: () => ({ mode: 'codes' }) }, provenance: 'live',
			title: { en: 'Your drawing as 16 numbers', zh: '你的画变成 16 个数' },
			narration: {
				en: 'The outlined patch snapped to code {k}, at squared distance {d}; the runner-up, code {k2}, is at {d2}. Every patch snaps the same way, so your drawing is now a {G}×{G} grid of indices with {nUsed} different codes; {blanks}. Each index is one of {K}, {bitsPer} bits, so the whole drawing takes {bits} bits instead of {n} pixels. A PixelCNN (a planned page) learns to write such grids.',
				zh: '框出的图块吸附到码字 {k}，距离平方 {d}；第二近的是码字 {k2}，距离平方 {d2}。每个图块都这样吸附，于是你的画变成一个 {G}×{G} 的编号网格，用到 {nUsed} 种码字；{blanks}。每个编号是 {K} 选一，即 {bitsPer} 比特，整幅画只要 {bits} 比特，而不是 {n} 个像素。PixelCNN（计划中的一页）学的就是怎样写出这样的网格。',
				vars: (rt) => {
					const u = used(rt);
					const b = u.nBlank === 1 ? { en: `1 patch sits on the blank code ${u.blank}`, zh: `1 个图块用的是空白码字 ${u.blank}` } : { en: `${fmtNum(u.nBlank)} patches sit on the blank code ${u.blank}`, zh: `${fmtNum(u.nBlank)} 个图块用的是空白码字 ${u.blank}` };
					return { ...dims(rt), ...snap(rt), ...u, blanks: Number.isFinite(u.nBlank) ? words(b) : NaN };
				}
			},
			look: { en: 'The grid of indices: dashed cells hold the blank code.', zh: '编号网格：虚线格子里是空白码字。' },
			terms: ['codebook']
		},
		{
			id: 'decode', source: 'fwd', at: { layer: 'out' }, view: { key: 'vq-vae/grid', props: () => ({ mode: 'redraw' }) }, provenance: 'live',
			title: { en: 'The decoder redraws from the indices', zh: '解码器从编号重画' },
			narration: {
				en: 'The decoder never sees z_e, only the codes: each index is replaced by its code’s {D} numbers (z_q). A 3×3 kernel reads each code together with its neighbours, two more layers make {block} scores per patch, the scores are laid out as the patch’s {P}×{P} block, and a sigmoid turns each into ink between 0 and 1. The redrawing has {inkOut} pixels above half ink; your drawing has {inkIn}.',
				zh: '解码器看不到 z_e，只看到码字：每个编号被换成它那个码字的 {D} 个数（z_q）。一个 3×3 卷积核连同邻居一起读每个码字，再经过两层，每个图块得到 {block} 个分数；这些分数排成这个图块的 {P}×{P} 小块，sigmoid 再把每个分数变成 0 到 1 之间的墨量。重画里墨量超过一半的像素有 {inkOut} 个；你的画有 {inkIn} 个。',
				vars: (rt) => ({ ...dims(rt), inkOut: ink(T(rt, 'out')?.data), inkIn: ink(T(rt, 'input')?.data) })
			},
			look: { en: 'Green frame: your drawing. Magenta frame: the redrawing from the indices.', zh: '绿框：你的画。品红框：从编号重画出的画。' },
			terms: ['encoder-decoder', 'activation-function']
		},
		{
			id: 'swap', source: 'fwd', at: { layer: 'out' }, view: { key: 'vq-vae/grid', props: () => ({ mode: 'swap' }) }, provenance: 'live',
			title: { en: 'Swap one code', zh: '换掉一个码字' },
			narration: {
				en: 'Each patch is just an index, so you can change it. Pick a patch and a code in the stage, and the decoder redraws with that one index changed. {result}',
				zh: '每个图块只是一个编号，所以你可以改它。在舞台里选一个图块和一个码字，解码器就用只改了这一个编号的网格重画。{result}',
				vars: (rt) => {
					const s = swapOf(rt);
					if (!s) return { result: words({ en: 'No swap yet: the stage shows the redrawing of step 6.', zh: '还没有替换：舞台显示的是第 6 步的重画。' }) };
					return {
						result: words({
							en: `You put code ${s.to} where code ${s.from} was: in that block, the pixels above half ink went from ${fmtNum(s.ink0)} to ${fmtNum(s.ink1)}. Added up over the pixels, the ink changed by ${fmt3(s.inside)} inside the block and ${fmt3(s.outside)} outside it, because the decoder’s 3×3 kernel also reads each patch’s neighbours.`,
							zh: `你把码字 ${s.to} 放到了原来码字 ${s.from} 的位置：这一块里墨量超过一半的像素从 ${fmtNum(s.ink0)} 个变成 ${fmtNum(s.ink1)} 个。把各像素的变化加起来，块内是 ${fmt3(s.inside)}，块外是 ${fmt3(s.outside)}，因为解码器的 3×3 卷积核也会读每个图块的邻居。`
						})
					};
				}
			},
			look: { en: 'Orange in the change map: ink the swap added; blue: ink it removed.', zh: '变化图里的橙色：替换添上的墨；蓝色：去掉的墨。' },
			task: {
				prompt: { en: 'Find a code that adds a stroke: put a code into an empty patch so that its block gains at least 5 pixels above half ink.', zh: '找一个能添上一笔的码字：把一个码字放进空图块，让这一块里墨量超过一半的像素至少多出 5 个。' },
				done: (rt) => (swapOf(rt)?.added ?? 0) >= ADD,
				waiting: { en: 'Not yet: no swap has added 5 inked pixels to its block.', zh: '还没有：还没有哪次替换让那一块多出 5 个有墨像素。' },
				hint: { en: 'Pick a blank patch next to your strokes, then a code whose picture shows a bold line.', zh: '选一个挨着笔画的空白图块，再选一个图里有粗线的码字。' }
			},
			terms: ['codebook']
		},
		{
			id: 'train', source: 'fwd', at: { layer: 'zq' }, view: { key: 'vq-vae/nearest', props: () => ({ mode: 'train' }) }, provenance: 'live',
			title: { en: 'Training: copy the gradient past the snap', zh: '训练：梯度怎样越过“吸附”' },
			narration: {
				en: 'Snapping has no useful slope: nudge z_e a little and the same code still wins, so no gradient would reach the encoder. The straight-through estimator copies the gradient the loss sends to z_q onto z_e unchanged; for the outlined patch it is ({g}). A commitment loss, weighted {beta}, pulls z_e towards its code, and each code is kept at a moving average of the z_e that snapped to it, in which each new batch counts {ema}.',
				zh: '吸附这一步没有可用的斜率：z_e 稍微挪一点，胜出的还是同一个码字，所以梯度到不了编码器。直通估计器把损失传给 z_q 的梯度原样复制给 z_e；对框出的图块，这个梯度是 ({g})。承诺损失（权重 {beta}）把 z_e 拉向它的码字；每个码字则保持为吸附到它的那些 z_e 的移动平均，每一批新数据在其中占 {ema}。',
				vars: (rt) => ({ g: list(gradAt(rt, cellOf(rt))), beta: fmtSetting(rt.num('fact:beta')), ema: `${fmtSetting(100 * rt.num('fact:emaStep'))}%` })
			},
			look: { en: 'The last two columns are equal: that copy is the whole trick.', zh: '最后两列完全相同：这一次复制就是全部的诀窍。' },
			terms: ['straight-through', 'gradient', 'commitment-loss', 'moving-average']
		},
		{
			id: 'usage', source: 'fwd', at: { layer: 'zq' }, view: { key: 'vq-vae/codebook', props: () => ({ mode: 'usage' }) }, provenance: 'live',
			title: { en: 'How the codes are used', zh: '码字用得怎样' },
			narration: {
				en: 'Over the {fact:testPatches} patches of the test drawings, all {fact:usedN} codes are used, but very unevenly: the blank code takes {fact:blankShare|pct} of the patches, and {fact:rareN} codes are used by fewer than one patch in a hundred, the rarest by {fact:rareMin|pct}. A typical test drawing uses {fact:distinctMedian} different codes; yours uses {nUsed}.',
				zh: '在测试画的 {fact:testPatches} 个图块里，{fact:usedN} 个码字全都用到了，但很不均匀：空白码字占了 {fact:blankShare|pct} 的图块，有 {fact:rareN} 个码字每一百个图块里用不到一次，最少的只占 {fact:rareMin|pct}。一幅典型的测试画用到 {fact:distinctMedian} 种码字；你的画用到 {nUsed} 种。',
				vars: (rt) => used(rt)
			},
			look: { en: 'Bold rows: the codes your drawing uses, with how many of its patches use each.', zh: '加粗的行：你的画用到的码字，以及它有几个图块用了它。' },
			terms: ['codebook']
		},
		{
			id: 'limits', source: 'fwd', at: { layer: 'out' }, view: { key: 'vq-vae/grid', props: () => ({ mode: 'limits' }) }, provenance: 'live', kindOf: 'limits',
			title: { en: 'Limits: rough pieces and dead codes', zh: '局限：粗糙的小块和死码字' },
			narration: {
				en: 'Each patch keeps only which of {K} pieces it is nearest to, so strokes come back shifted and smudged, as the error map shows. This redrawing’s error is {rec} per pixel, {cmp} the test average of {fact:testRecPx}. More codes help only a little: with {fact:k128Codes} codes the test error falls to {fact:k128RecPx}. And codes can die: trained the textbook way, from tiny random codes, a {fact:k128Codes}-code book used only {fact:plainK128Used} of them; the rest were dead codes, never picked.',
				zh: '每个图块只记住它最接近 {K} 种小块里的哪一种，所以笔画回来时会偏移、发糊，误差图标出了位置。这次重画的误差是每像素 {rec}，{cmp}测试画的平均值 {fact:testRecPx}。码字多了也只好一点：用 {fact:k128Codes} 个码字，测试误差降到 {fact:k128RecPx}。码字还会“死掉”：按教科书的做法从很小的随机码字开始训练，{fact:k128Codes} 个码字的码本只用上了 {fact:plainK128Used} 个，其余都是从没被选中的死码字。',
				vars: (rt) => {
					const rec = lossOf(rt)?.rec ?? NaN;
					return { ...dims(rt), rec, cmp: Number.isFinite(rec) ? words(rec < rt.num('fact:testRecPx') ? { en: 'below', zh: '低于' } : { en: 'above', zh: '高于' }) : NaN };
				}
			},
			look: { en: 'The error map: stronger colour where the redrawing misses your ink or adds ink you did not draw.', zh: '误差图：重画漏掉你的墨、或多出你没画的墨的地方，颜色更浓。' },
			terms: ['dead-code', 'bce']
		}
	],
	fixtures: [
		{ id: 'ladder-swap', label: { en: 'a ladder of strokes, code 6 put into the blank patch at row 1, column 0', zh: '用笔画画的梯子，把码字 6 放进第 1 行第 0 列的空白图块' }, raw: { strokes: LADDER, swap: { cell: 4, code: 6 } } },
		{ id: 'ladder-corner', label: { en: 'a ladder of strokes, the empty corner patch outlined', zh: '用笔画画的梯子，框出空白的角落图块' }, raw: { strokes: LADDER, cell: 0 } }
	]
};

export default lesson;
