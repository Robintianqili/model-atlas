// tinymoe: the package manifest. A tiny character-level Transformer (2 blocks, 32 numbers per character) whose MLPs are
// mixtures of 4 experts with top-2 routing, trained on TinyStories, read on the learner's own text:
//   fwd       the exported model (pipeline/tinymoe/export.py) traced on the learner's text: router scores and chances,
//             the gates (B<b>.moe#gate) and the outputs of the experts that ran (B<b>.moe#e)
//   dispatch  block 1's routing of the text played character by character under an expert capacity, then the
//             load-balancing loss (process.ts; pipeline/tinymoe/reference.py is its Python twin)
// Measured numbers (held-out loss of the MoE, its dense twin and the MoE trained without the balancing loss; expert
// shares and dropped slots on held-out stories) are in facts.json (train.py, stats.py).
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { chances, cfg, idsOf } from './routing.ts';

const L = (en: string, zh: string) => ({ en, zh });

const manifest: ExplainerManifest = {
	slug: 'tinymoe',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/tinymoe', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'text-presets', presets: 'presets.json', start: 'Once upon a time, there was a little girl named Lily.' },
	output: { view: 'kit.probbars' },
	sources: {
		fwd: { kind: 'trace' },
		dispatch: { kind: 'process', process: 'dispatch', opt: { cf: 1 }, input: (rt) => ({ ids: idsOf(rt), p: chances(rt, 1), block: 1, k: cfg(rt).K }) }
	},
	overview: { layout: 'pipeline' },
	page: {
		goals: [
			L('Say what the router, the experts and the gates of a mixture-of-experts layer do.', '说出混合专家层里路由器、专家和门控各做什么。'),
			L('Work out one character’s route with real numbers: scores, softmax, the top 2, and the gate-weighted sum.', '用真实的数算出一个字符的分派：分数、softmax、前 2 名，以及按门控加权的和。'),
			L('Explain why training adds a load-balancing loss, and what an expert’s capacity does to an unbalanced batch.', '解释训练为什么要加负载均衡损失，以及专家容量会对不均衡的一批数据造成什么影响。'),
			L('Compare a mixture of experts with a dense MLP that uses about as many parameters per character.', '把混合专家和每个字符用到的参数差不多的稠密 MLP 作比较。')
		],
		prequestion: {
			id: 'tinymoe-pre',
			question: L('This model has 4 experts in each block, but every character uses only 2 of them. What decides which 2?', '这个模型每一块有 4 个专家，但每个字符只用其中 2 个。由什么决定用哪 2 个？'),
			options: [L('A fixed rule, such as letters to some experts and spaces to others', '一条固定的规则，比如字母给某些专家、空格给另一些'), L('A small learned layer, the router, that scores every expert for each character', '一个学来的小层，也就是路由器，它为每个字符给每个专家打分'), L('Chance: two experts are drawn at random', '运气：随机抽两个专家')],
			answer: 1,
			explain: L('A router, a linear layer trained with the rest of the model, scores the 4 experts for each character; the 2 best scores win. Nobody wrote a rule, yet step 6 shows that the trained router does sort characters by kind.', '路由器是一个和模型其余部分一起训练的线性层，它为每个字符给 4 个专家打分，得分最高的 2 个胜出。没有人写过规则，但第 6 步会看到，训练好的路由器确实会按字符的种类来分。')
		},
		tryThis: [
			{ id: 'same', title: L('One letter, over and over', '同一个字母，一遍又一遍'), setup: { preset: 'same' }, predict: L('Every character is “a”. How many of the slots will the experts have to drop?', '每个字符都是“a”。专家们要丢掉多少个位置？'), look: L('Count the × marks under the characters and read the bars against the dashed capacity line.', '数一数字符下面的 ×，再对照虚线容量线看那些条。'), goto: { step: 'capacity' } },
			{ id: 'spaces', title: L('Many short words', '很多短词'), setup: { preset: 'spaces' }, predict: L('Short words mean many spaces and periods. Will one expert of block 2 get busy?', '短词意味着很多空格和句号。第 2 块会有某个专家变得很忙吗？'), look: L('Look at the outlined characters: the ones block 2 sent to its space expert.', '看加框的字符：第 2 块送给它的“空格专家”的那些。'), goto: { step: 'who' } }
		],
		misconception: {
			claim: L('A mixture of experts mixes all of its experts, like attention mixes all positions.', '混合专家会把所有专家都混在一起，就像注意力混合所有位置那样。'),
			truth: L('Only the top-k experts run for each token (here each character runs 2 of 4); the others are skipped entirely. That is what makes a large MoE cheap to run.', '每个词元只运行前 k 个专家（这里每个字符运行 4 个里的 2 个），其余的完全跳过。正是这一点让大的 MoE 运行起来很省。'),
			step: 'pick'
		},
		metrics: {
			val_loss: { label: L('Loss on held-out stories (this MoE)', '在没见过的故事上的损失（本 MoE）'), fmt: 'n4' },
			val_loss_dense: { label: L('… the dense twin', '…… 稠密对照模型'), fmt: 'n4' },
			val_loss_noaux: { label: L('… the MoE without the balancing loss', '…… 不加均衡损失的 MoE'), fmt: 'n4' }
		},
		sections: { hide: ['inspector'] },
		lessonNote: L('Ten steps: experts instead of one MLP, the router’s scores, softmax, the top 2, the weighted sum, who handles what, the balancing loss, capacity, a dense twin, and the next character.', '十步：用专家代替一个 MLP、路由器的分数、softmax、前 2 名、加权求和、谁负责什么、负载均衡损失、容量、稠密对照模型，以及下一个字符。')
	},
	credits: [
		{
			name: 'TinyStories (TinyStoriesV2-GPT4)',
			url: 'https://huggingface.co/datasets/roneneldan/TinyStories',
			license: 'CDLA-Sharing-1.0',
			usedFor: 'Training text (39.3 million characters from the start of the train split) and held-out validation text for the routing statistics of the tinymoe model',
			attribution: "Ronen Eldan and Yuanzhi Li, 'TinyStories: How Small Can Language Models Be and Still Speak Coherent English?' (2023)"
		},
		{ name: 'Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (ideas)', url: 'https://arxiv.org/abs/1701.06538', license: 'ideas only (paper; no code or text used)', usedFor: 'the mixture-of-experts layer with a learned top-k router; code written fresh', attribution: 'Noam Shazeer et al. (2017)' },
		{ name: 'GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (ideas)', url: 'https://arxiv.org/abs/2006.16668', license: 'ideas only (paper; no code or text used)', usedFor: 'top-2 routing in a Transformer and expert capacity (slots dropped when an expert is full); code written fresh', attribution: 'Dmitry Lepikhin et al. (2020)' },
		{ name: 'Switch Transformers (ideas)', url: 'https://arxiv.org/abs/2101.03961', license: 'ideas only (paper; no code or text used)', usedFor: 'the load-balancing loss E · Σ f·P with weight 0.01 and the capacity factor; code written fresh', attribution: 'William Fedus, Barret Zoph and Noam Shazeer (2021)' },
		{ name: 'Mixtral of Experts (ideas)', url: 'https://arxiv.org/abs/2401.04088', license: 'ideas only (paper; no code or text used)', usedFor: 'top-2 of the experts with a softmax over the kept scores (the gates), and training without dropping tokens; code written fresh', attribution: 'Albert Q. Jiang et al. (2024)' }
	],
	glossary: [
		{ id: 'mixture-of-experts', term: L('mixture of experts (MoE)', '混合专家（MoE）'), def: L('A layer made of several smaller networks (the experts) and a router that sends each token to only a few of them. The model holds many parameters but uses only some for each token.', '由几个较小的网络（专家）和一个路由器组成的层，路由器只把每个词元送给其中几个专家。模型的参数很多，但每个词元只用到一部分。'), related: ['router', 'mlp'] },
		{ id: 'router', term: L('router (gating network)', '路由器（门控网络）'), def: L('The small learned layer of a mixture of experts that scores every expert for each token; the top-scoring experts process the token, weighted by their gates.', '混合专家里那个学来的小层：它为每个词元给每个专家打分，得分最高的几个专家处理这个词元，并按各自的门控加权。'), related: ['mixture-of-experts', 'softmax'] },
		{ id: 'load-balancing-loss', term: L('load-balancing loss', '负载均衡损失'), def: L('An extra term added to the training loss of a mixture of experts that grows when a few experts get most of the tokens, so the router learns to spread the work.', '加在混合专家训练损失上的额外一项：少数专家拿到大部分词元时它会变大，于是路由器学会把活分散开。'), related: ['mixture-of-experts', 'loss'] },
		{ id: 'expert-capacity', term: L('expert capacity', '专家容量'), def: L('The most tokens an expert may take in one batch, set by a capacity factor. Tokens sent to a full expert are dropped for that expert.', '一个专家在一批数据里最多能接的词元数，由容量因子决定。送到已满专家的词元会被这个专家丢掉。'), related: ['mixture-of-experts'] }
	],
	catalog: {
		hook: L('Each character picks 2 of 4 experts. See who handles what, and what a full expert drops.', '每个字符从 4 个专家里选 2 个。看看谁负责什么，以及专家满了会丢掉什么。'),
		shows: L(
			'The router’s 4 scores for your last character, softmax worked out, the top 2 experts and their gate-weighted sum, a layer × expert grid under every character of your text, which characters each expert takes, the load-balancing loss, the slots dropped when an expert is full, a dense MLP with about as many parameters per character compared, and the next-character chances.',
			'路由器给你最后一个字符打的 4 个分数、算出来的 softmax、前 2 名专家和它们按门控加权的和、你文字每个字符下方的层×专家网格、每个专家接哪些字符、负载均衡损失、专家满了时被丢掉的位置、和每个字符用到的参数差不多的稠密 MLP 的比较，以及下一个字符的概率。'
		),
		params: L('49,418 (32,650 used per character)', '49,418（每个字符用到 32,650）'),
		size: L('193 KB of weights', '权重 193 KB'),
		runtime: L('Own JS forward pass (checked against PyTorch)', '自研 JS 前向计算（已与 PyTorch 核对）')
	}
};

export default manifest;
