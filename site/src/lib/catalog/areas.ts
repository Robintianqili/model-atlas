import type { Area, AreaSlug } from './types.ts';

/** The 11 application areas, in display order (brief §4.1). */
export const AREAS: Area[] = [
	{
		slug: 'vision',
		n: 1,
		name: { en: 'Vision', zh: '视觉' },
		blurb: {
			en: 'Images, video and 3D: how networks find edges, shapes and objects.',
			zh: '图像、视频与三维：网络如何找到边缘、形状和物体。'
		},
		examples: { en: 'CNNs, ViTs, segmentation, detection', zh: 'CNN、ViT、分割、检测' }
	},
	{
		slug: 'language',
		n: 2,
		name: { en: 'Language', zh: '语言' },
		blurb: {
			en: 'Text: tokens, embeddings, attention and predicting the next word.',
			zh: '文本：词元、嵌入、注意力，以及预测下一个词。'
		},
		examples: { en: 'GPT, BERT, RNNs, word2vec', zh: 'GPT、BERT、RNN、word2vec' }
	},
	{
		slug: 'audio',
		n: 3,
		name: { en: 'Speech and audio', zh: '语音与音频' },
		blurb: {
			en: 'Sound: from a waveform to a spectrogram to words, pitch and music.',
			zh: '声音：从波形到频谱图，再到文字、音高与音乐。'
		},
		examples: { en: 'keyword spotting, Whisper, TTS, codecs', zh: '关键词识别、Whisper、语音合成、音频编解码' }
	},
	{
		slug: 'time-series',
		n: 4,
		name: { en: 'Time series', zh: '时间序列' },
		blurb: {
			en: 'Numbers measured over time: forecasting the next values and spotting anomalies.',
			zh: '随时间测量的数值：预测未来的值，发现异常。'
		},
		examples: { en: 'LSTM, TCN, autoencoders, Kalman filter', zh: 'LSTM、TCN、自编码器、卡尔曼滤波' }
	},
	{
		slug: 'recommendation',
		n: 5,
		name: { en: 'Recommendation', zh: '推荐' },
		blurb: {
			en: 'Ratings and clicks: predicting what a person will like next.',
			zh: '评分与点击：预测一个人接下来会喜欢什么。'
		},
		examples: { en: 'matrix factorization, two-tower, SASRec', zh: '矩阵分解、双塔模型、SASRec' }
	},
	{
		slug: 'tabular',
		n: 6,
		name: { en: 'Tabular and classic ML', zh: '表格数据与经典机器学习' },
		blurb: {
			en: 'Rows and columns, and the classic methods that still work well there.',
			zh: '行与列的数据，以及在这里依然好用的经典方法。'
		},
		examples: { en: 'MLP, trees, SVM, k-means, t-SNE', zh: 'MLP、决策树、SVM、k-means、t-SNE' }
	},
	{
		slug: 'rl',
		n: 7,
		name: { en: 'Reinforcement learning and control', zh: '强化学习与控制' },
		blurb: {
			en: 'Agents that learn from trial, error and reward: bandits, games, control and robots.',
			zh: '通过试错与奖励学习的智能体：多臂老虎机、游戏、控制与机器人。'
		},
		examples: { en: 'Q-learning, policy gradient, DQN, MCTS', zh: 'Q 学习、策略梯度、DQN、蒙特卡洛树搜索' }
	},
	{
		slug: 'graphs',
		n: 8,
		name: { en: 'Graphs', zh: '图' },
		blurb: {
			en: 'Molecules, social networks and meshes: nodes passing messages to their neighbors.',
			zh: '分子、社交网络与网格：节点向邻居传递消息。'
		},
		examples: { en: 'GCN, GAT, GraphSAGE, node2vec', zh: 'GCN、GAT、GraphSAGE、node2vec' }
	},
	{
		slug: 'generative',
		n: 9,
		name: { en: 'Generative models', zh: '生成模型' },
		blurb: {
			en: 'Models that make new samples: GANs, VAEs, flows and diffusion.',
			zh: '生成新样本的模型：GAN、VAE、流模型与扩散模型。'
		},
		examples: { en: 'flow matching, GAN, VAE, DDPM', zh: '流匹配、GAN、VAE、DDPM' }
	},
	{
		slug: 'multimodal',
		n: 10,
		name: { en: 'Multimodal', zh: '多模态' },
		blurb: {
			en: 'Images, text and sound mapped into one shared space.',
			zh: '把图像、文本和声音映射到同一个共享空间。'
		},
		examples: { en: 'CLIP, vision-language models, CLAP', zh: 'CLIP、视觉语言模型、CLAP' }
	},
	{
		slug: 'science',
		n: 11,
		name: { en: 'Science', zh: '科学' },
		blurb: {
			en: 'Physics, proteins and molecules: networks as tools for science.',
			zh: '物理、蛋白质与分子：作为科学工具的神经网络。'
		},
		examples: { en: 'PINN, protein language models, neural operators', zh: 'PINN、蛋白质语言模型、神经算子' }
	}
];

export const AREA_BY_SLUG: Record<AreaSlug, Area> = Object.fromEntries(AREAS.map((a) => [a.slug, a])) as Record<
	AreaSlug,
	Area
>;

export function isAreaSlug(s: string): s is AreaSlug {
	return s in AREA_BY_SLUG;
}
