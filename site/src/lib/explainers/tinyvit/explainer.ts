// tinyvit: the package manifest (design §4.9, §7). A tiny Vision Transformer on TinyCNN's Quick, Draw! drawings:
//   fwd      the exported model (pipeline/tinyvit/export.py) traced on the learner's drawing
//   patches  a scan over the patch convolution: one frame per patch (16), for the "cut into patches" step
//   cnn      TinyCNN (static/models/tinycnn, read through legacy.ts) on the same 784 numbers, for the compare step
// Measured numbers (accuracy, agreement, where the [CLS] row looks) are in facts.json (pipeline/tinyvit/stats.py).
import type { ExplainerManifest } from '../../atlas/types/index.ts';

const manifest: ExplainerManifest = {
	slug: 'tinyvit',
	type: 'model',
	kinds: ['trace'],
	quality: 'reviewed',
	assets: { dir: 'models/tinyvit', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'drawing28', presets: 'presets.json' },
	output: { view: 'kit.probbars' },
	sources: {
		fwd: { kind: 'trace' },
		patches: { kind: 'scan', of: 'fwd', layer: 'patch', unit: () => [0] },
		cnn: { kind: 'trace', assets: 'models/tinycnn' }
	},
	overview: { layout: 'stack' },
	page: {
		goals: [
			{ en: 'Say how a 28×28 drawing becomes 17 rows of 48 numbers: 16 patches plus the [CLS] row.', zh: '说出一幅 28×28 的画如何变成 17 行、每行 48 个数：16 个图块加上 [CLS] 行。' },
			{ en: 'Work out one attention score (query · key ÷ √12) and say how softmax turns 17 scores into weights.', zh: '算出一个注意力分数（查询 · 键 ÷ √12），并说出 softmax 如何把 17 个分数变成权重。' },
			{ en: 'Read where the [CLS] row looks on your drawing, and say why the answer is read from that row.', zh: '读出 [CLS] 行在你的画上看向哪里，并说出为什么答案从这一行读出。' },
			{ en: 'Compare TinyViT with TinyCNN on accuracy and size, and say what a convolution builds in that attention has to learn.', zh: '比较 TinyViT 和 TinyCNN 的准确率和大小，并说出卷积天生具备、而注意力必须学会的是什么。' }
		],
		prequestion: {
			id: 'tinyvit-pre',
			question: {
				en: 'TinyCNN’s first layer looks at 3×3 pixels at a time. In its first block, how much of your drawing can TinyViT’s [CLS] row read?',
				zh: 'TinyCNN 的第一层一次只看 3×3 个像素。在第一个块里，TinyViT 的 [CLS] 行能读到你的画的多少？'
			},
			options: [
				{ en: 'A small window, like a convolution', zh: '一个小窗口，和卷积一样' },
				{ en: 'Only the patches next to it', zh: '只有它旁边的图块' },
				{ en: 'All 16 patches at once', zh: '一次读全部 16 个图块' }
			],
			answer: 2,
			explain: {
				en: 'Attention compares a row with every row, near or far, so the [CLS] row can read every patch in one step. Learned weights decide how much each patch counts.',
				zh: '注意力把一行和所有行比较，不管远近，所以 [CLS] 行一步就能读到每个图块。学来的权重决定每个图块占多少分量。'
			}
		},
		tryThis: [
			{
				id: 'ladder',
				title: { en: 'A ladder', zh: '一架梯子' },
				setup: { preset: 'heldout-ladder' },
				predict: { en: 'TinyViT never saw a ladder. What will it answer?', zh: 'TinyViT 从没见过梯子。它会怎么回答？' },
				look: { en: 'It still splits all its confidence among its 10 classes.', zh: '它仍然把全部信心分给它的 10 个类别。' },
				goto: { step: 'classes' }
			},
			{
				id: 'dot',
				title: { en: 'One dot', zh: '一个点' },
				setup: { input: { strokes: [{ points: [{ x: 0.5, y: 0.5 }] }] } },
				predict: { en: 'A dot is scaled up into the middle of the grid, where four patches meet. Where will the [CLS] row look?', zh: '一个点会被放大到网格中间，正好落在四个图块的交界处。[CLS] 行会看向哪里？' },
				look: { en: 'Compare the weights of the four inked patches (●) with those of the empty ones.', zh: '比较四个有墨的图块（●）和空图块的权重。' },
				goto: { step: 'row' }
			}
		],
		quiz: [
			{
				q: { en: 'How many rows enter the first block?', zh: '进入第一个块的有多少行？' },
				options: [
					{ en: '16', zh: '16' },
					{ en: '17', zh: '17' },
					{ en: '784', zh: '784' }
				],
				answer: 1,
				explain: { en: 'One row per patch (16) plus the [CLS] row.', zh: '每个图块一行（16 行），再加上 [CLS] 行。' }
			},
			{
				q: { en: 'Which row does the class head read?', zh: '类别头读的是哪一行？' },
				options: [
					{ en: 'The [CLS] row', zh: '[CLS] 行' },
					{ en: 'The patch with the most ink', zh: '墨最多的那个图块' },
					{ en: 'The average of all rows', zh: '所有行的平均' }
				],
				answer: 0,
				explain: { en: 'Only row 0, the [CLS] row, goes to the class head; the patch rows are dropped.', zh: '只有第 0 行，也就是 [CLS] 行，进入类别头；图块行都被丢掉。' }
			}
		],
		misconception: {
			claim: { en: 'Attention only looks at nearby patches, like a convolution.', zh: '注意力只看附近的图块，和卷积一样。' },
			truth: { en: 'Every row can read every other row in one step; learned weights, not distance, decide how much.', zh: '每一行一步就能读到所有其他行；决定分量的是学来的权重，而不是距离。' },
			step: 'row'
		},
		lessonNote: {
			en: 'Eleven steps: your drawing becomes 17 rows, the [CLS] row looks at the patches, mixes what it sees, and gives the answer; then TinyCNN on the same drawing.',
			zh: '十一步：你的画变成 17 行，[CLS] 行看向各个图块、混合看到的内容并给出答案；最后和 TinyCNN 在同一幅画上比较。'
		}
	},
	credits: [
		{
			name: 'Quick, Draw! dataset',
			url: 'https://github.com/googlecreativelab/quickdraw-dataset',
			license: 'CC BY 4.0',
			usedFor: 'TinyViT training and test drawings (TinyCNN’s 10 classes and split) and the preset drawings',
			attribution: 'Quick, Draw! dataset by Google Creative Lab'
		}
	],
	catalog: {
		params: { en: '116,938', zh: '116,938' },
		size: { en: '457 KB of weights', zh: '权重 457 KB' }
	}
};

export default manifest;
