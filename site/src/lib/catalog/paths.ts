/**
 * Curated learning paths shown on the home page. Steps point at catalog models or concepts;
 * planned steps are shown dimmed so the path also works as a roadmap.
 */
import type { LearningPath } from './types.ts';

export const PATHS: LearningPath[] = [
	{
		id: 'start-here',
		title: { en: 'Start here: how a network sees', zh: '从这里开始：网络如何“看”' },
		blurb: {
			en: 'Draw something, follow it through a small CNN, then teach the CNN new classes.',
			zh: '画点东西，跟着它穿过一个小 CNN，再教这个 CNN 认识新类别。'
		},
		steps: [
			{ kind: 'model', slug: 'tinycnn' },
			{ kind: 'concept', slug: 'transfer-learning' },
			{ kind: 'concept', slug: 'loss-gradient-descent' },
			{ kind: 'concept', slug: 'backprop' },
			{ kind: 'model', slug: 'adversarial-tinycnn' }
		]
	},
	{
		id: 'language-models',
		title: { en: 'How language models work', zh: '语言模型如何工作' },
		blurb: {
			en: 'From characters to tokens to a Transformer that predicts the next one.',
			zh: '从字符到词元，再到预测下一个词元的 Transformer。'
		},
		steps: [
			{ kind: 'concept', slug: 'tokenization' },
			{ kind: 'model', slug: 'tinygpt' },
			{ kind: 'concept', slug: 'next-token-prediction' },
			{ kind: 'concept', slug: 'decoding' },
			{ kind: 'model', slug: 'gpt-2' }
		]
	},
	{
		id: 'llm-lifecycle',
		title: { en: 'The life of a language model', zh: '一个语言模型的一生' },
		blurb: {
			en: 'Pre-training, fine-tuning, alignment, compression and serving, in order.',
			zh: '按顺序走过预训练、微调、对齐、压缩与部署。'
		},
		steps: [
			{ kind: 'concept', slug: 'next-token-prediction' },
			{ kind: 'concept', slug: 'scaling-laws' },
			{ kind: 'concept', slug: 'instruction-tuning' },
			{ kind: 'concept', slug: 'lora' },
			{ kind: 'concept', slug: 'dpo' },
			{ kind: 'concept', slug: 'quantization' },
			{ kind: 'concept', slug: 'kv-cache' }
		]
	},
	{
		id: 'vision-path',
		title: { en: 'Vision, from CNNs to ViTs', zh: '视觉：从 CNN 到 ViT' },
		blurb: {
			en: 'Convolutions, residual networks, patches and attention, and models that learn without labels.',
			zh: '卷积、残差网络、图块与注意力，以及无需标签就能学习的模型。'
		},
		steps: [
			{ kind: 'model', slug: 'tinycnn' },
			{ kind: 'model', slug: 'resnet-18' },
			{ kind: 'model', slug: 'tinyvit' },
			{ kind: 'concept', slug: 'contrastive-learning' },
			{ kind: 'model', slug: 'dinov2' }
		]
	},
	{
		id: 'foundations',
		title: { en: 'Foundations of training', zh: '训练基础' },
		blurb: {
			en: 'The order a textbook would use: loss, gradients, optimizers, then the tricks that make deep nets train.',
			zh: '按教材的顺序：损失、梯度、优化器，再到让深层网络能训练起来的各种技巧。'
		},
		steps: [
			{ kind: 'concept', slug: 'loss-gradient-descent' },
			{ kind: 'concept', slug: 'backprop' },
			{ kind: 'model', slug: 'mlp-playground' },
			{ kind: 'concept', slug: 'optimizers' },
			{ kind: 'concept', slug: 'regularization' },
			{ kind: 'concept', slug: 'init-normalization' },
			{ kind: 'concept', slug: 'residual-connections' }
		]
	}
];
