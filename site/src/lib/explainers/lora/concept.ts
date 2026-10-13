// lora: the concept page around the lesson (K2 concept packages; atlas/ui/concept/panels.ts ConceptDef).
// hook (text), toy, real and compare are StepDefs on the lesson's own runtime and sources, shown as free-standing stages.
import type { I18n, StepDef } from '../../atlas/types/index.ts';
import { ALPHA, D, GPT2_PARAMS, MB, floorOf, gpt2Lora, memoryOf, metricsOf, pctText, rankAt } from './data.ts';

export const hook: I18n = {
	en: 'A trained model can be taught a new task without touching its weights. LoRA freezes every weight matrix and learns, next to it, a small change built from two thin matrices. Here you train one on a real layer, choose its rank and count what it saves, from this page\'s toy up to GPT-2 small.',
	zh: '一个训练好的模型，可以不动它的权重就学会新任务。LoRA 冻结每个权重矩阵，在旁边学一个由两个细长矩阵组成的小改变量。在这里，你会在一个真实的层上训练它，选择它的秩，并算一算它能省下多少，从本页的玩具层一直到 GPT-2 small。'
};

export const toy: StepDef = {
	id: 'toy',
	title: { en: 'Pick a rank and train the adapter', zh: '选一个秩，训练适配器' },
	source: 'fit',
	at: 'last',
	view: 'lora/matrices',
	play: { from: 'first', to: 'last', interval: 120, unit: { en: 'update', zh: '次更新' } },
	provenance: 'live',
	narration: {
		en: 'Rank {r}: B is {d} × {r} and A is {r} × {d}, scaled by α/r = {scale}. After {n} updates the loss is {loss}, down from {base} for the base layer and {gap} above {floor}, the best this rank can do. Change the rank and the run starts again from B = 0.',
		zh: '秩 {r}：B 是 {d} × {r}，A 是 {r} × {d}，再乘以 α/r = {scale}。更新 {n} 次之后损失是 {loss}：比基础层的 {base} 低，只比这个秩的最低值 {floor} 高 {gap}。换一个秩，运行就从 B = 0 重新开始。',
		vars: (rt, at) => {
			const m = metricsOf(at);
			const r = rankAt({ rt, at });
			return { r, d: D, scale: m.scale ?? ALPHA / r, n: m.step, loss: m.loss, base: m.base, floor: floorOf(r), gap: m.loss - floorOf(r) };
		}
	},
	look: { en: 'Press play: B starts gray and grows; W0 never changes.', zh: '按下播放：B 从灰色开始长出颜色；W0 始终不变。' }
};

export const real: StepDef = {
	id: 'real',
	title: { en: 'A real model: GPT-2 small', zh: '真实的模型：GPT-2 small' },
	source: 'fit',
	at: { phase: 'start' },
	view: 'lora/memory',
	provenance: 'live',
	narration: {
		en: 'GPT-2 small has {params} parameters. LoRA at rank {r} on its attention and MLP matrices trains {lora} of them, {share}. Its training memory falls from {full} MB to {loraMB} MB, not by the same factor: the frozen weights are still there.',
		zh: 'GPT-2 small 有 {params} 个参数。在它的注意力和 MLP 矩阵上做秩为 {r} 的 LoRA，只训练其中 {lora} 个，占 {share}。训练内存从 {full} MB 降到 {loraMB} MB，降幅却远没有那么大：冻结的权重还在。',
		vars: (rt, at) => {
			const r = rankAt({ rt, at });
			const m = memoryOf(r);
			return { params: GPT2_PARAMS, r, lora: gpt2Lora(r), share: pctText(gpt2Lora(r) / GPT2_PARAMS), full: Math.round(MB(m.full.total)), loraMB: Math.round(MB(m.lora.total)) };
		}
	}
};

export const compare: StepDef = {
	id: 'compare',
	title: { en: 'Four ranks and full fine-tuning', zh: '四种秩与全量微调' },
	source: 'ranks',
	at: 'last',
	view: {
		key: 'loss-curve',
		props: () => ({
			metrics: ['l1', 'l2', 'l4', 'l8', 'lfull'],
			log: true,
			labels: { l1: { en: 'rank 1', zh: '秩 1' }, l2: { en: 'rank 2', zh: '秩 2' }, l4: { en: 'rank 4', zh: '秩 4' }, l8: { en: 'rank 8', zh: '秩 8' }, lfull: { en: 'full fine-tuning', zh: '全量微调' } }
		})
	},
	provenance: 'live',
	narration: {
		en: 'The same start, five runs. After {n} updates: rank 1 {l1}, rank 2 {l2}, rank 4 {l4}, rank 8 {l8}, full fine-tuning {lfull}. The big drop comes from rank 1 to rank 2, because the needed change has two strong directions; every rank after that buys much less.',
		zh: '同样的起点，五次运行。更新 {n} 次之后：秩 1 为 {l1}，秩 2 为 {l2}，秩 4 为 {l4}，秩 8 为 {l8}，全量微调为 {lfull}。最大的下降发生在秩 1 到秩 2 之间，因为所需的改变量有两个强方向；之后每增加一点秩，收获都小得多。',
		vars: (_rt, at) => {
			const m = metricsOf(at);
			return { n: m.step, l1: m.l1, l2: m.l2, l4: m.l4, l8: m.l8, lfull: m.lfull };
		}
	}
};

export const pitfalls = [
	{
		claim: { en: 'A higher rank is always better.', zh: '秩越高越好。' },
		truth: {
			en: 'Only up to the rank the change needs. Here the change has two strong directions: every rank above 2 trains more numbers for a much smaller drop in the loss than the step from 1 to 2.',
			zh: '只到改变量所需的秩为止。这里的改变量有两个强方向：秩超过 2 以后，多训练的数换来的损失下降，比从 1 到 2 那一步小得多。'
		},
		step: 'spectrum'
	},
	{
		claim: { en: 'B starts at zero, so it cannot learn.', zh: 'B 从零开始，所以它学不动。' },
		truth: {
			en: 'B\'s gradient is not zero at the start; A\'s is. The first update moves B, and from then on both learn.',
			zh: 'B 的梯度在开始时并不是零，是 A 的梯度为零。第一次更新就让 B 动起来，此后两者都在学习。'
		},
		step: 'first'
	},
	{
		claim: { en: 'A LoRA model is slower at inference, because of the extra matrices.', zh: 'LoRA 模型推理更慢，因为多了额外的矩阵。' },
		truth: {
			en: 'After training, (α/r)·B·A can be added into W0 once. The merged layer is one matrix of the old size and gives the same outputs, so it runs as fast as the base model.',
			zh: '训练结束后，可以把 (α/r)·B·A 一次性加到 W0 上。合并后的层是一个与原来同样大小的矩阵，输出完全相同，所以运行速度和基础模型一样。'
		},
		step: 'train'
	}
];

export const further = [
	{ label: { en: 'Transfer learning on this site: reuse a trained model for a new task', zh: '本站的迁移学习：把训练好的模型用到新任务上' }, href: '/train/transfer-learning/' },
	{ label: { en: 'Hu et al., LoRA: Low-Rank Adaptation of Large Language Models (paper)', zh: 'Hu 等，LoRA：大型语言模型的低秩适配（论文）' }, href: 'https://arxiv.org/abs/2106.09685' },
	{ label: { en: 'Dettmers et al., QLoRA: Efficient Finetuning of Quantized LLMs (paper)', zh: 'Dettmers 等，QLoRA：量化大模型的高效微调（论文）' }, href: 'https://arxiv.org/abs/2305.14314' },
	{ label: { en: 'Liu et al., DoRA: Weight-Decomposed Low-Rank Adaptation (paper)', zh: 'Liu 等，DoRA：权重分解的低秩适配（论文）' }, href: 'https://arxiv.org/abs/2402.09353' },
	{ label: { en: 'Hugging Face PEFT: LoRA in practice (documentation)', zh: 'Hugging Face PEFT：LoRA 的实际用法（文档）' }, href: 'https://huggingface.co/docs/peft' }
];
