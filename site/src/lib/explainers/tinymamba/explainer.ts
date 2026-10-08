// tinymamba: the package manifest (design §4.9). A tiny selective state-space model (Mamba-style: input-dependent Δ, B
// and C, a diagonal A; 2 blocks) trained on the selective-copy task of task.ts: remember the letters among the dots
// and give them back at the question marks.
//   fwd   the exported model (pipeline/tinymamba/export.py) on the learner's row: Δ, B, C and the state h of each block
//   scan  block 1's scan worked out token by token in float64 (process.ts), for the channel that separates letters
//         from dots most; pipeline/tinymamba/reference.py is its Python twin
// Measured numbers (accuracy on new rows, with more letters, over longer gaps) are in facts.json (stats.py).
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { scanInput } from './tm.ts';

const L = (en: string, zh: string) => ({ en, zh });

const manifest: ExplainerManifest = {
	slug: 'tinymamba',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/tinymamba', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'seq', presets: 'presets.json', start: '.c..a....e..???' },
	output: { view: 'tinymamba/scan', props: { rows: ['want', 'got'] } },
	sources: {
		fwd: { kind: 'trace' },
		scan: { kind: 'process', process: 'scan', run: 'main', input: (rt) => scanInput(rt, 1) }
	},
	overview: { layout: 'stack' },
	page: {
		goals: [
			L('Say what the selective-copy task asks and why it needs a memory.', '说出选择性复制任务要做什么，以及为什么它需要记忆。'),
			L('Explain how the step size Δ decides, token by token, how much of the state to keep and how much to write.', '解释步长 Δ 如何逐个词元决定保留多少状态、写入多少新内容。'),
			L('Work out one scan step h = exp(Δ·A)·h + Δ·B·u with real numbers, and say what B and C do.', '用真实的数算出一步扫描 h = exp(Δ·A)·h + Δ·B·u，并说出 B 和 C 的作用。'),
			L('Compare the memory a fixed-size state needs with the memory attention needs, and name what a fixed state cannot do.', '比较固定大小的状态和注意力各需要多少记忆，并说出固定状态做不到什么。')
		],
		prequestion: {
			id: 'tinymamba-pre',
			question: L('TinyGPT looks back at every earlier token. This model reads the row once, left to right, and keeps only a small, fixed set of numbers. Can it still give back letters it saw long before?', 'TinyGPT 会回头看前面的每一个词元。这个模型从左到右只读一遍，只保留一小组固定数量的数。它还能找回很早以前见过的字母吗？'),
			options: [L('No: without looking back it forgets everything', '不能：不回头看就会全部忘掉'), L('Yes, if it learns what to keep in its small state and what to let pass', '能，只要它学会在小小的状态里保留什么、放过什么'), L('Yes, it secretly stores the whole row', '能，它偷偷存下了整行')],
			answer: 1,
			explain: L('It keeps a state of a fixed size and updates it at every token. Each token decides how much to keep and how much to write (the step size Δ), so the letters can be held while the dots pass by. A fixed state also has limits: step 10 shows them.', '它保留一份大小固定的状态，并在每个词元处更新。每个词元自己决定保留多少、写入多少（步长 Δ），所以点经过时字母能一直留着。固定的状态也有极限：第 10 步会展示。')
		},
		tryThis: [
			{ id: 'twice', title: L('A letter twice', '同一个字母两次'), setup: { preset: 'twice' }, predict: L('The row has “d” twice, then “b”. Will the three ? give back d, d, b, in that order?', '这一行先有两个“d”，再有一个“b”。三个 ? 会按顺序找回 d、d、b 吗？'), look: L('Each ? needs its own place in the order, even when two letters are the same: read the line under the ?.', '即使两个字母一样，每个 ? 也要找回顺序里属于自己的那一个：看 ? 下面那一行。'), goto: { step: 'output' } },
			{ id: 'far', title: L('One letter, far back', '一个字母，离得很远'), setup: { preset: 'far' }, predict: L('“b” comes first, then more dots than any training row had, then one ?. Will the model still say “b”?', '先是“b”，然后是比任何训练行都多的点，最后一个 ?。模型还会说“b”吗？'), look: L('Watch the followed channel along the dots: where Δ stays near 0, its state hardly fades.', '沿着这些点看被跟踪的通道：Δ 一直接近 0 的地方，它的状态几乎不会消退。'), goto: { step: 'whole' } }
		],
		misconception: {
			claim: L('A model without attention just forgets what came early in the row.', '没有注意力的模型会把前面的内容忘掉。'),
			truth: L('A selective state can hold early tokens for a long time: where Δ is near 0 the state is kept almost as it is. What it cannot do is hold more than its fixed size allows.', '选择性的状态可以把前面的词元保留很久：Δ 接近 0 的地方，状态几乎原样保留。它做不到的是保存超出固定大小所能容纳的内容。'),
			step: 'limits'
		},
		sections: {
			intros: {
				overview: L('This is Mamba’s selective scan in a small model: Δ, B and C come from each token, A is diagonal and D is a shortcut. A full Mamba block also has a short convolution before the scan and a gate after it; this model leaves both out so that the scan stays in view, and uses layer normalization where Mamba usually uses RMSNorm.', '这是放在一个小模型里的 Mamba 选择性扫描：Δ、B、C 由每个词元算出，A 是对角的，D 是一条捷径。完整的 Mamba 块在扫描之前还有一个短卷积，之后还有一个门控；这个模型把两者都去掉了，好让扫描本身一目了然，并且用层归一化代替了 Mamba 通常使用的 RMSNorm。')
			}
		},
		lessonNote: L('Ten steps: the task, tokens as vectors, a state updated token by token, Δ deciding what to keep, B writing and C reading, one scan step with real numbers, the state over the whole row, the output, the memory compared with attention, and what a fixed state cannot hold.', '十步：任务、词元变成向量、逐个词元更新的状态、Δ 决定保留什么、B 写入与 C 读出、用真实的数算一步扫描、整行读下来的状态、输出、和注意力比较记忆，以及固定状态装不下什么。')
	},
	credits: [
		{ name: 'Mamba: Linear-Time Sequence Modeling with Selective State Spaces (ideas)', url: 'https://arxiv.org/abs/2312.00752', license: 'ideas only (paper; no code or text used)', usedFor: 'the selective scan (input-dependent Δ, B, C; diagonal A = −exp(A_log); softplus Δ), its initialization and the selective-copying task; code written fresh', attribution: 'Albert Gu and Tri Dao (2023)' },
		{ name: 'The Hidden Attention of Mamba Models (ideas)', url: 'https://arxiv.org/abs/2403.01590', license: 'ideas only (paper; no code or text used)', usedFor: 'reading the scan of one channel as a matrix of look-back weights (step 9); code written fresh', attribution: 'Ameen Ali, Itamar Zimerman and Lior Wolf (2024)' }
	],
	catalog: {
		params: L('2,474', '2,474'),
		size: L('9.7 KB of weights', '权重 9.7 KB'),
		runtime: L('Own JS forward pass (checked against PyTorch)', '自研 JS 前向计算（已与 PyTorch 核对）')
	},
	glossary: [
		{ id: 'state-space-model', term: L('state-space model (SSM)', '状态空间模型（SSM）'), def: L('A sequence model that reads one token at a time and carries a state of fixed size from token to token, updated by a simple rule. Mamba is a selective one: the update depends on the token.', '一次读一个词元、把一份大小固定的状态从一个词元带到下一个词元的序列模型，状态按一条简单的规则更新。Mamba 是“选择性”的：更新方式取决于当前词元。'), related: ['token', 'attention'] },
		{ id: 'step-size', term: L('step size Δ (delta)', '步长 Δ（delta）'), def: L('A number of at least 0 that each token computes for each channel. Near 0 the state is kept and nothing is written; large, the old state fades and the token is written in.', '每个词元为每个通道算出的一个不小于 0 的数。接近 0 时状态被保留、什么都不写入；很大时旧状态淡去、写入这个词元。'), related: ['state-space-model'] },
		{ id: 'kv-cache', term: L('key-value cache', '键值缓存'), def: L('The keys and values a Transformer keeps for every earlier token so that new tokens can look back at them. It grows with every token.', 'Transformer 为前面每个词元保存的键和值，让新的词元能回头看它们。每多一个词元，它就变大一点。'), related: ['attention', 'query-key-value'] }
	]
};

export default manifest;
