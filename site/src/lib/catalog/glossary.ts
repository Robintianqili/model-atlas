/**
 * Bilingual glossary: the single source of terminology for both languages.
 * Use <Term id="softmax">softmax</Term> in pages; the tooltip and /glossary/ read from here.
 * Ids are lowercase-hyphenated and stable (they are URL anchors: /glossary/#softmax).
 */
import type { GlossaryTerm, I18n } from './types.ts';

const f = (en: string, zh: string): I18n => ({ en, zh });

export const GLOSSARY: GlossaryTerm[] = [
	// ---- data and tensors
	{ id: 'tensor', term: f('tensor', '张量'), def: f('A grid of numbers with any number of axes: a vector has 1 axis, a matrix 2, an image with color channels 3.', '任意维数的数字网格：向量有 1 个轴，矩阵有 2 个，带颜色通道的图像有 3 个。'), related: ['shape'] },
	{ id: 'shape', term: f('shape', '形状'), def: f('The size of each axis of a tensor, written like [8×26×26] for 8 channels of 26×26.', '张量每个轴的大小，例如 [8×26×26] 表示 8 个 26×26 的通道。'), related: ['tensor', 'channel'] },
	{ id: 'channel', term: f('channel', '通道'), def: f('One 2D slice of an image-like tensor. RGB images have 3; a conv layer outputs one channel per kernel.', '类图像张量中的一张二维切片。RGB 图像有 3 个通道；卷积层的每个卷积核输出一个通道。'), related: ['feature-map', 'kernel'] },
	{ id: 'feature-map', term: f('feature map', '特征图'), def: f('The output channel of a conv layer: where in the input one kernel’s pattern was found, and how strongly.', '卷积层输出的一个通道：表示某个卷积核的模式在输入的哪里出现、有多强。'), related: ['convolution', 'channel'] },
	{ id: 'batch', term: f('batch', '批次'), def: f('A group of examples processed together in one training step.', '在一个训练步骤中一起处理的一组样本。'), related: ['epoch', 'sgd'] },
	{ id: 'epoch', term: f('epoch', '轮次（epoch）'), def: f('One full pass over the training data.', '把训练数据完整过一遍。'), related: ['batch'] },
	{ id: 'dataset-split', term: f('train / validation / test split', '训练 / 验证 / 测试划分'), def: f('Training data fits the parameters, validation data picks settings, and test data is used once to measure the final model.', '训练集用来拟合参数，验证集用来挑选设置，测试集只在最后用一次来衡量模型。'), related: ['overfitting'] },
	{ id: 'label', term: f('label', '标签'), def: f('The correct answer attached to a training example, such as the class "cat".', '附在训练样本上的正确答案，比如类别“猫”。'), related: ['classifier'] },

	// ---- network building blocks
	{ id: 'neuron', term: f('neuron (unit)', '神经元（单元）'), def: f('One number computed as a weighted sum of inputs plus a bias, usually followed by an activation function.', '由输入的加权和加上偏置得到的一个数，通常再经过激活函数。'), related: ['weight', 'bias', 'activation-function'] },
	{ id: 'layer', term: f('layer', '层'), def: f('One step of a network that turns an input tensor into an output tensor, such as a convolution or an attention block.', '网络中把输入张量变成输出张量的一步，例如卷积或注意力模块。'), related: ['tensor'] },
	{ id: 'parameter', term: f('parameter', '参数'), def: f('A number inside the model that training adjusts, such as a weight or a bias.', '模型内部、由训练来调整的数，例如权重或偏置。'), related: ['weight', 'bias'] },
	{ id: 'weight', term: f('weight', '权重'), def: f('A parameter that multiplies an input. Large positive or negative weights mean that input matters a lot.', '与输入相乘的参数。很大的正或负权重表示该输入很重要。'), related: ['parameter'] },
	{ id: 'bias', term: f('bias', '偏置'), def: f('A parameter added after the weighted sum; it shifts the output up or down.', '在加权和之后加上的参数，让输出整体上移或下移。'), related: ['parameter'] },
	{ id: 'activation', term: f('activation', '激活值'), def: f('The numbers a layer outputs for a given input. They change with every input; parameters do not.', '某层对某个输入给出的输出数字。激活值随输入变化，而参数不会。'), related: ['layer', 'parameter'] },
	{ id: 'activation-function', term: f('activation function', '激活函数'), def: f('A simple nonlinear function applied to each number, such as ReLU. Without it, stacked layers collapse into one linear map.', '逐元素作用的简单非线性函数，如 ReLU。没有它，多层叠加会退化成一个线性映射。'), related: ['relu'] },
	{ id: 'relu', term: f('ReLU', 'ReLU'), def: f('max(0, x): keeps positive values and turns negative values into zero.', 'max(0, x)：保留正数，把负数变成 0。'), related: ['activation-function'] },
	{ id: 'mlp', term: f('MLP (multilayer perceptron)', 'MLP（多层感知机）'), def: f('Fully connected layers with activation functions in between. Inside a Transformer it widens each vector, applies a nonlinearity, and narrows it again.', '由全连接层和中间的激活函数组成。在 Transformer 中，它把每个向量先扩宽、经过非线性、再收窄。'), related: ['dense-layer'] },
	{ id: 'dense-layer', term: f('dense (fully connected) layer', '全连接层'), def: f('A layer where every output is a weighted sum of every input: y = Wx + b.', '每个输出都是所有输入的加权和：y = Wx + b。'), related: ['weight', 'mlp'] },

	// ---- CNN
	{ id: 'convolution', term: f('convolution', '卷积'), def: f('Sliding a small grid of weights (a kernel) over the input and taking a weighted sum at every position.', '把一小块权重网格（卷积核）在输入上滑动，在每个位置做加权求和。'), related: ['kernel', 'stride', 'padding', 'feature-map'] },
	{ id: 'kernel', term: f('kernel (filter)', '卷积核（滤波器）'), def: f('The small grid of weights a convolution slides over its input, for example 3×3. Each kernel looks for one pattern.', '卷积在输入上滑动的小块权重，例如 3×3。每个卷积核寻找一种模式。'), related: ['convolution'] },
	{ id: 'stride', term: f('stride', '步长'), def: f('How many positions the kernel moves each step. Stride 2 halves the output size.', '卷积核每次移动的格数。步长为 2 时输出尺寸减半。'), related: ['convolution'] },
	{ id: 'padding', term: f('padding', '填充'), def: f('Extra values (usually zeros) added around the input border so the kernel can also be centered on edge pixels.', '在输入边缘外补上的值（通常为 0），使卷积核也能以边缘像素为中心。'), related: ['convolution'] },
	{ id: 'pooling', term: f('pooling', '池化'), def: f('Shrinking a feature map by summarizing small windows; max pooling keeps the largest value of each window.', '通过汇总小窗口来缩小特征图；最大池化保留每个窗口中的最大值。'), related: ['feature-map'] },
	{ id: 'receptive-field', term: f('receptive field', '感受野'), def: f('The region of the input that can affect one output value. It grows with every conv and pooling layer.', '能影响某个输出值的输入区域。每经过一层卷积或池化，它都会变大。'), related: ['convolution'] },
	{ id: 'flatten', term: f('flatten', '展平'), def: f('Reshaping a multi-axis tensor into one long vector, keeping every value.', '把多轴张量重排成一条长向量，所有数值保持不变。'), related: ['shape'] },

	// ---- outputs
	{ id: 'logit', term: f('logit', 'logit（对数几率）'), def: f('A raw score for one class or token before softmax. Only differences between logits matter.', 'softmax 之前某个类别或词元的原始分数。只有 logit 之间的差值才重要。'), related: ['softmax'] },
	{ id: 'softmax', term: f('softmax', 'softmax'), def: f('Turns a list of scores into probabilities: exponentiate each score, then divide by the total so they sum to 1.', '把一组分数变成概率：对每个分数取指数，再除以总和，使它们加起来等于 1。'), related: ['logit', 'temperature'] },
	{ id: 'classifier', term: f('classifier', '分类器'), def: f('A model that outputs one probability per class and predicts the most likely class.', '为每个类别输出一个概率、并预测最可能类别的模型。'), related: ['softmax', 'label'] },
	{ id: 'temperature', term: f('temperature', '温度'), def: f('A number that divides the logits before softmax. Below 1 sharpens the distribution; above 1 flattens it.', '在 softmax 之前用来除 logits 的数。小于 1 让分布更尖锐，大于 1 让分布更平坦。'), related: ['softmax', 'sampling'] },
	{ id: 'sampling', term: f('sampling (top-k, top-p)', '采样（top-k、top-p）'), def: f('Picking the next token at random from the predicted probabilities, often only among the k most likely (top-k) or the smallest set whose probabilities sum to p (top-p).', '按预测概率随机选下一个词元，通常只在最可能的 k 个中选（top-k），或在概率累计达到 p 的最小集合中选（top-p）。'), related: ['temperature'] },

	// ---- language
	{ id: 'token', term: f('token', '词元（token）'), def: f('The unit a language model reads and writes: a character, a piece of a word, a whole word or a single byte.', '语言模型读写的基本单位：一个字符、一个词片、一个完整的词或一个字节。'), related: ['tokenizer', 'vocabulary'] },
	{ id: 'tokenizer', term: f('tokenizer', '分词器'), def: f('The fixed procedure that splits text into tokens and maps each token to an integer ID.', '把文本切成词元、并把每个词元映射为整数 ID 的固定流程。'), related: ['token', 'bpe'] },
	{ id: 'vocabulary', term: f('vocabulary', '词表'), def: f('The list of all tokens a model knows. Its size sets the number of output logits.', '模型认识的所有词元的列表。它的大小决定了输出 logits 的个数。'), related: ['token'] },
	{ id: 'bpe', term: f('byte-pair encoding (BPE)', '字节对编码（BPE）'), def: f('A tokenizer training method that starts from characters or bytes and repeatedly merges the most frequent adjacent pair into a new token.', '一种分词器训练方法：从字符或字节开始，反复把最常见的相邻对合并成新词元。'), related: ['tokenizer'] },
	{ id: 'wordpiece', term: f('WordPiece', 'WordPiece'), def: f('The subword tokenizer of BERT: split text into words, then cut each word into the longest pieces found in the vocabulary. Pieces inside a word start with ##; a word that cannot be spelled becomes [UNK].', 'BERT 使用的子词分词方法：先切成单词，再把每个词切成词表里能找到的最长片段。词中间的片段以 ## 开头；拼不出来的词变成 [UNK]。'), related: ['tokenizer', 'bpe'] },
	{ id: 'utf-8', term: f('UTF-8', 'UTF-8'), def: f('The standard way to store text as bytes: ASCII characters take 1 byte, most accented letters 2, Chinese characters 3 and emoji 4.', '把文本存成字节的标准方式：ASCII 字符占 1 个字节，大多数带重音的字母占 2 个，汉字占 3 个，表情符号占 4 个。'), related: ['token', 'bpe'] },
	{ id: 'embedding', term: f('embedding', '嵌入'), def: f('A learned vector for each token (or item); looking it up is just picking one row of a weight matrix.', '为每个词元（或物品）学到的向量；查找嵌入就是取权重矩阵中的一行。'), related: ['token', 'positional-encoding'] },
	{ id: 'positional-encoding', term: f('positional encoding', '位置编码'), def: f('Information added to token vectors so the model knows the order of tokens; attention alone ignores order.', '加到词元向量上的位置信息，让模型知道词元顺序；仅靠注意力无法区分顺序。'), related: ['embedding', 'attention'] },
	{ id: 'context-window', term: f('context window', '上下文窗口'), def: f('The maximum number of tokens a model can look at at once.', '模型一次最多能看到的词元数量。'), related: ['token'] },
	{ id: 'autoregressive', term: f('autoregressive generation', '自回归生成'), def: f('Generating one token at a time, each time feeding the tokens produced so far back in as input.', '一次生成一个词元，每次都把已生成的词元作为输入再喂回模型。'), related: ['next-token-prediction'] },
	{ id: 'next-token-prediction', term: f('next-token prediction', '下一个词元预测'), def: f('The training task of language models like GPT: given the text so far, predict a probability for every possible next token.', 'GPT 等语言模型的训练任务：给定已有文本，为每一个可能的下一个词元预测概率。'), related: ['cross-entropy', 'autoregressive'] },

	// ---- attention and transformers
	{ id: 'attention', term: f('attention', '注意力'), def: f('A layer where each position builds its output as a weighted average of other positions’ values, with weights from query–key similarity.', '每个位置用其他位置的值的加权平均来构造输出，权重来自查询与键的相似度。'), related: ['query-key-value', 'attention-head', 'causal-mask'] },
	{ id: 'query-key-value', term: f('query, key, value', '查询、键、值（Q/K/V）'), def: f('Three vectors made from each token. A query is compared with every key to get attention weights, which then mix the values.', '由每个词元生成的三个向量。查询与每个键比较得到注意力权重，再用这些权重混合值向量。'), related: ['attention'] },
	{ id: 'attention-head', term: f('attention head', '注意力头'), def: f('One independent attention computation with its own Q, K and V projections; layers run several heads in parallel.', '拥有独立 Q、K、V 投影的一次注意力计算；每层并行运行多个头。'), related: ['attention'] },
	{ id: 'causal-mask', term: f('causal mask', '因果掩码'), def: f('A rule that stops each token from attending to later tokens, so the model cannot peek at the answer.', '禁止每个词元关注其后面的词元，使模型无法偷看答案。'), related: ['attention', 'next-token-prediction'] },
	{ id: 'transformer', term: f('Transformer', 'Transformer'), def: f('A network built from repeated blocks of attention and MLP, each wrapped with layer norm and a residual connection.', '由反复堆叠的注意力与 MLP 模块组成的网络，每个模块都带有层归一化与残差连接。'), related: ['attention', 'mlp', 'residual-connection', 'layer-norm'] },
	{ id: 'residual-connection', term: f('residual connection', '残差连接'), def: f('Adding a block’s input to its output (x + f(x)), so information and gradients can skip through deep networks.', '把模块的输入加到输出上（x + f(x)），让信息和梯度能穿过很深的网络。'), related: ['transformer'] },
	{ id: 'layer-norm', term: f('layer normalization', '层归一化（LayerNorm）'), def: f('Rescaling each vector to mean 0 and variance 1, then applying a learned scale and shift.', '把每个向量缩放到均值 0、方差 1，再乘以可学习的缩放并加上偏移。'), related: ['transformer'] },

	// ---- training
	{ id: 'loss', term: f('loss', '损失'), def: f('A single number that measures how wrong the model is on some examples. Training tries to make it small.', '衡量模型在一些样本上错得有多离谱的一个数。训练的目标是让它变小。'), related: ['cross-entropy', 'gradient'] },
	{ id: 'cross-entropy', term: f('cross-entropy', '交叉熵'), def: f('The usual classification loss: −log of the probability the model gave to the correct answer.', '常用的分类损失：模型赋予正确答案的概率取负对数。'), related: ['loss', 'perplexity'] },
	{ id: 'perplexity', term: f('perplexity', '困惑度'), def: f('exp(average cross-entropy). Roughly how many tokens the model is choosing between at each step; lower is better.', 'exp(平均交叉熵)。大致表示模型每一步在多少个词元之间犹豫；越低越好。'), related: ['cross-entropy'] },
	{ id: 'gradient', term: f('gradient', '梯度'), def: f('For every parameter, how much the loss would change if that parameter were nudged up a little.', '对每个参数而言：把它稍微调大一点，损失会改变多少。'), related: ['backpropagation', 'gradient-descent'] },
	{ id: 'backpropagation', term: f('backpropagation', '反向传播'), def: f('Computing all gradients efficiently by applying the chain rule from the loss backward through the layers.', '从损失出发、沿各层反向应用链式法则，高效地计算所有梯度。'), related: ['gradient'] },
	{ id: 'gradient-descent', term: f('gradient descent', '梯度下降'), def: f('Training by repeatedly moving each parameter a small step against its gradient.', '反复让每个参数沿梯度的反方向走一小步来进行训练。'), related: ['learning-rate', 'sgd'] },
	{ id: 'sgd', term: f('stochastic gradient descent (SGD)', '随机梯度下降（SGD）'), def: f('Gradient descent where each step uses a small random batch instead of the whole dataset.', '每一步只用一个随机小批次（而不是整个数据集）的梯度下降。'), related: ['gradient-descent', 'optimizer'] },
	{ id: 'optimizer', term: f('optimizer', '优化器'), def: f('The rule that turns gradients into parameter updates, such as SGD, momentum or Adam.', '把梯度变成参数更新的规则，如 SGD、动量法或 Adam。'), related: ['sgd', 'learning-rate'] },
	{ id: 'learning-rate', term: f('learning rate', '学习率'), def: f('The step size of each update. Too large and training diverges; too small and it crawls.', '每次更新的步长。太大训练会发散，太小训练会很慢。'), related: ['optimizer'] },
	{ id: 'overfitting', term: f('overfitting', '过拟合'), def: f('When a model memorizes its training data and does worse on new data.', '模型记住了训练数据，却在新数据上表现变差。'), related: ['regularization', 'dataset-split'] },
	{ id: 'regularization', term: f('regularization', '正则化'), def: f('Any technique that discourages overfitting, such as weight decay, dropout or data augmentation.', '任何抑制过拟合的技巧，如权重衰减、dropout 或数据增强。'), related: ['overfitting', 'dropout'] },
	{ id: 'dropout', term: f('dropout', 'Dropout（随机失活）'), def: f('During training, randomly set some activations to zero so the network cannot rely on any single unit.', '训练时随机把一部分激活值置零，使网络不能依赖任何单个神经元。'), related: ['regularization'] },

	// ---- pre-training and fine-tuning
	{ id: 'pre-training', term: f('pre-training', '预训练'), def: f('Training a model on a large, general task (such as next-token prediction on lots of text) before adapting it to specific tasks.', '在适配具体任务之前，先在大规模通用任务（如在大量文本上预测下一个词元）上训练模型。'), related: ['fine-tuning', 'self-supervised'] },
	{ id: 'self-supervised', term: f('self-supervised learning', '自监督学习'), def: f('Learning from unlabeled data by hiding part of each example and predicting it from the rest.', '把每个样本的一部分藏起来、用其余部分去预测，从而在无标签数据上学习。'), related: ['pre-training'] },
	{ id: 'fine-tuning', term: f('fine-tuning', '微调'), def: f('Continuing to train a pre-trained model on a smaller, task-specific dataset.', '在较小的、特定任务的数据集上继续训练一个预训练模型。'), related: ['pre-training', 'transfer-learning', 'frozen-layer'] },
	{ id: 'transfer-learning', term: f('transfer learning', '迁移学习'), def: f('Reusing what a model learned on one task to learn a new task faster or with less data.', '把模型在一个任务上学到的东西用在新任务上，学得更快或所需数据更少。'), related: ['fine-tuning', 'feature-extraction'] },
	{ id: 'feature-extraction', term: f('feature extraction', '特征提取'), def: f('Transfer learning where the pre-trained layers are frozen and only a new output head is trained on their outputs.', '一种迁移学习方式：冻结预训练层，只在其输出上训练一个新的输出头。'), related: ['transfer-learning', 'frozen-layer', 'linear-probe'] },
	{ id: 'frozen-layer', term: f('frozen layer', '冻结层'), def: f('A layer whose parameters are kept fixed during training: it still computes outputs but receives no updates.', '训练时参数保持不变的层：它仍然计算输出，但不接受更新。'), related: ['fine-tuning', 'feature-extraction'] },
	{ id: 'linear-probe', term: f('linear probe', '线性探针'), def: f('A single linear layer trained on frozen features to test what information they contain.', '在冻结特征上训练的单个线性层，用于检验这些特征包含什么信息。'), related: ['feature-extraction'] },
	{ id: 'head', term: f('head (output head)', '输出头'), def: f('The last layer(s) that turn a model’s features into task outputs, such as class logits. Fine-tuning often replaces it.', '把模型特征变成任务输出（如类别 logits）的最后一层或几层。微调时常常替换它。'), related: ['transfer-learning'] },
	{ id: 'lora', term: f('LoRA (low-rank adaptation)', 'LoRA（低秩适配）'), def: f('Fine-tuning by adding a small trainable low-rank update BA to a frozen weight matrix instead of changing the whole matrix.', '在冻结的权重矩阵上加一个小的可训练低秩更新 BA 来微调，而不是改动整个矩阵。'), related: ['fine-tuning', 'frozen-layer'] },
	{ id: 'quantization', term: f('quantization', '量化'), def: f('Storing numbers with fewer bits (for example 8-bit integers instead of 32-bit floats) to save memory and time, at some loss of precision.', '用更少的比特存储数字（例如用 8 位整数代替 32 位浮点数），以损失少量精度换取内存和时间。'), related: ['precision'] },
	{ id: 'precision', term: f('numeric precision (fp32, fp16, int8)', '数值精度（fp32、fp16、int8）'), def: f('How many bits each number uses. fp32 is the usual training format; fp16 and int8 are smaller and less exact.', '每个数字使用多少比特。fp32 是常用的训练格式；fp16 与 int8 更小但不够精确。'), related: ['quantization'] },

	// ---- running models
	{ id: 'forward-pass', term: f('forward pass', '前向传播'), def: f('Running an input through all layers in order to get the output.', '把输入按顺序送过所有层，得到输出。'), related: ['inference'] },
	{ id: 'inference', term: f('inference', '推理'), def: f('Using a trained model to make predictions, without changing its parameters.', '使用训练好的模型做预测，不改变其参数。'), related: ['forward-pass'] },
	{ id: 'checkpoint', term: f('checkpoint', '检查点'), def: f('A saved copy of a model’s parameters at some point during training.', '训练过程中某一时刻保存下来的模型参数副本。'), related: ['parameter'] },
	{ id: 'parity', term: f('parity (numeric check)', '一致性校验'), def: f('Checking that our in-browser computation matches the reference (PyTorch) output, within a stated tolerance.', '检验浏览器中的计算与参考实现（PyTorch）的输出在给定容差内一致。'), related: ['precision'] },

	// ---- evaluation and reliability (added by the TinyCNN page)
	{ id: 'data-augmentation', term: f('data augmentation', '数据增强'), def: f('Making extra training examples by randomly changing existing ones (shift, scale, rotate, thicker or thinner strokes) so the model learns what does not matter.', '随机改动已有样本（平移、缩放、旋转、笔画加粗或变细）来制造更多训练样本，让模型学会哪些变化无关紧要。'), related: ['regularization', 'overfitting'] },
	{ id: 'confusion-matrix', term: f('confusion matrix', '混淆矩阵'), def: f('A table of test results: each row is the true class, each column the predicted class. The diagonal counts correct answers; everything else shows which classes get mixed up.', '测试结果的表格：每行是真实类别，每列是预测类别。对角线是答对的数量，其余格子显示哪些类别被混淆。'), related: ['classifier', 'dataset-split'] },
	{ id: 'out-of-distribution', term: f('out-of-distribution input', '分布外输入'), def: f('An input unlike anything in the training data, such as a ladder shown to a model that only learned 10 other classes. Its outputs, including probabilities, can be confidently wrong.', '与训练数据都不相像的输入，例如把梯子给只学过另外 10 个类别的模型看。它的输出（包括概率）可能错得很自信。'), related: ['classifier', 'calibration'] },
	{ id: 'calibration', term: f('calibration', '校准'), def: f('How well predicted probabilities match how often the model is right: a calibrated model that says 90% is right about 9 times in 10.', '预测概率与模型实际正确频率的吻合程度：校准良好的模型说 90% 时，大约十次对九次。'), related: ['softmax', 'out-of-distribution'] },

	// ---- reinforcement learning (added by the gridworld page)
	{ id: 'agent', term: f('agent', '智能体'), def: f('The learner in reinforcement learning: it acts, sees what happens and gets rewards.', '强化学习中的学习者：它采取行动，观察结果，并得到奖励。') },
	{ id: 'reward', term: f('reward', '奖励'), def: f('A number the world gives after each action; the agent tries to collect as much as possible.', '每次行动后环境给出的一个数；智能体想尽量多拿。'), related: ['agent'] },
	{ id: 'episode', term: f('episode', '回合'), def: f('One try from the start until the task ends (a goal, a pit or a move limit).', '从起点开始直到任务结束（终点、陷阱或步数上限）的一次尝试。') },
	{ id: 'exploration', term: f('exploration vs exploitation', '探索与利用'), def: f('Trying actions to learn more about them, versus taking the action that looks best now.', '尝试一些动作以了解它们，与选择眼下看起来最好的动作之间的取舍。') },
	{ id: 'q-value', term: f('Q value, Q-table', 'Q 值、Q 表'), def: f('Q(s, a): the reward an agent expects in the long run after action a in state s; a Q-table stores one per state and action.', 'Q(s, a)：在状态 s 做动作 a 之后，长远来看期望得到的奖励；Q 表为每个状态和动作各存一个。'), related: ['reward'] },
	{ id: 'discount', term: f('discount factor γ', '折扣因子 γ'), def: f('A number between 0 and 1 that makes a reward one step later worth γ times as much.', '0 到 1 之间的数，让晚一步到手的奖励只值原来的 γ 倍。'), related: ['q-value'] },
	{ id: 'dqn', term: f('deep Q-network (DQN)', '深度 Q 网络（DQN）'), def: f('A neural network that reads a state and outputs one Q-value per action, trained from the agent’s own moves with a replay buffer and a target network.', '读入一个状态、为每个动作输出一个 Q 值的神经网络；用智能体自己走过的步来训练，并配有回放缓冲区和目标网络。'), related: ['q-value'] },
	{ id: 'epsilon-greedy', term: f('ε-greedy', 'ε-贪心'), def: f('Take the action with the highest Q-value, except with chance ε, when a random action is taken instead.', '选 Q 值最高的动作，但有 ε 的机会改成随机选一个动作。'), related: ['exploration'] },
	{ id: 'td-target', term: f('TD target', 'TD 目标'), def: f('The number a Q-value is pulled towards: the reward just received plus γ times the best Q-value of the next state (just the reward if the episode ended).', 'Q 值被拉向的那个数：刚得到的奖励，加上 γ 乘以下一个状态里最好的 Q 值（如果回合结束了，就只有奖励）。'), related: ['q-value', 'discount'] },
	{ id: 'replay-buffer', term: f('replay buffer', '回放缓冲区'), def: f('A store of the agent’s recent moves (state, action, reward, next state); training draws random batches from it.', '存放智能体最近走过的步（状态、动作、奖励、下一个状态）；训练时从里面随机抽一批。') },
	{ id: 'target-network', term: f('target network', '目标网络'), def: f('A copy of the Q-network that is refreshed only every so many updates and is used to compute the targets, so they stay still between copies.', 'Q 网络的一个副本，每隔若干次更新才刷新一次，用来计算目标值，让目标值在两次复制之间保持不动。'), related: ['dqn'] },
	{ id: 'reward-shaping', term: f('reward shaping', '奖励塑形'), def: f('Adding extra rewards or penalties to steer learning; it can change what the agent ends up doing.', '额外加一些奖励或惩罚来引导学习；它会改变智能体最后学成的样子。'), related: ['reward'] },
	{ id: 'policy', term: f('policy', '策略'), def: f('The agent’s rule for which action to take in each state.', '智能体在每个状态下选哪个动作的规则。') },
	{ id: 'regret', term: f('regret', '遗憾'), def: f('The reward a player expects to miss compared with always taking the best action.', '与一直选最好的动作相比，玩家预计少得到的奖励。') },

	// ---- recurrent networks (added by the char-rnn page)
	{ id: 'recurrent-network', term: f('recurrent network (RNN)', '循环神经网络（RNN）'), def: f('A network that reads a sequence one item at a time and passes a memory from each step to the next, using the same weights at every step.', '一次读序列中的一项、并把一份记忆从每一步传到下一步的网络；每一步用的是同一组权重。'), related: ['hidden-state', 'cell-state'] },
	{ id: 'hidden-state', term: f('hidden state (h)', '隐藏状态（h）'), def: f('The numbers a recurrent network passes to the next step and to its output after each item it reads.', '循环网络每读一项之后，传给下一步、也用来产生输出的那组数。'), related: ['recurrent-network', 'cell-state'] },
	{ id: 'cell-state', term: f('cell state (c)', '细胞状态（c）'), def: f('An LSTM’s inner memory: at each step it keeps a share of its old value (the forget gate) and adds a new one (the input gate times the candidate).', 'LSTM 的内部记忆：每一步保留旧值的一部分（遗忘门），再加上一个新值（输入门乘以候选值）。'), related: ['gate', 'hidden-state'] },
	{ id: 'gate', term: f('gate (LSTM)', '门（LSTM）'), def: f('A number between 0 and 1, made by a sigmoid, that scales how much passes: the forget gate scales the old memory, the input gate the new value, the output gate what is shown.', '由 sigmoid 算出的 0 到 1 之间的数，决定通过多少：遗忘门缩放旧记忆，输入门缩放新值，输出门缩放露出来的部分。'), related: ['cell-state', 'activation-function'] },

	// ---- graphs (added by the GCN page)
	{ id: 'graph', term: f('graph (nodes and edges)', '图（节点和边）'), def: f('A set of things (nodes) and the pairs of them that are linked (edges), such as members and their friendships.', '一组事物（节点）以及它们之间有联系的成对关系（边），例如成员和他们的朋友关系。') },
	{ id: 'adjacency-matrix', term: f('adjacency matrix', '邻接矩阵'), def: f('A table with a row and a column per node and a 1 where two nodes are linked.', '每个节点占一行一列的表，两个节点相连的地方是 1。'), related: ['graph'] },
	{ id: 'degree', term: f('degree', '度'), def: f('How many edges a node has. In a GCN, d also counts the self-loop: friends + 1.', '一个节点有多少条边。在 GCN 里，d 还把自环算进去：朋友数 + 1。'), related: ['graph'] },
	{ id: 'one-hot', term: f('one-hot vector', '独热向量'), def: f('A row of 0s with a single 1 that says which item it is.', '一行 0 里只有一个 1，用来标明它是哪一个。') },
	{ id: 'message-passing', term: f('message passing', '消息传递'), def: f('Each node sends numbers along its edges, and each node combines the numbers it receives.', '每个节点沿着边发出一些数，每个节点把收到的数合在一起。'), related: ['graph'] },
	{ id: 'semi-supervised', term: f('semi-supervised learning', '半监督学习'), def: f('Learning from a few labelled examples plus many unlabelled ones, here linked by a graph.', '从少量有标签的样本和大量没有标签的样本中学习，这里它们由一张图连在一起。'), related: ['label'] },
	{ id: 'over-smoothing', term: f('over-smoothing', '过度平滑'), def: f('After many rounds of averaging with neighbours, all nodes end up with nearly the same numbers.', '和邻居平均很多轮之后，所有节点的数都变得几乎一样。'), related: ['message-passing'] },

	// ---- generative models (added by the flow-matching page)
	{ id: 'generative-model', term: f('generative model', '生成模型'), def: f('A model that learns to produce new examples that look like its training data, instead of labelling them.', '学会造出与训练数据相似的新样本的模型，而不是给样本贴标签。') },
	{ id: 'gaussian-noise', term: f('standard normal noise', '标准正态噪声'), def: f('Random numbers from a bell curve centred on 0 with spread (standard deviation) 1; about 95% of them fall between −2 and 2.', '取自以 0 为中心、宽度（标准差）为 1 的钟形曲线的随机数；大约 95% 落在 −2 和 2 之间。') },
	{ id: 'velocity-field', term: f('velocity field', '速度场'), def: f('An arrow at every place (and time) that says which way, and how fast, a point there moves.', '每个位置（和时刻）都有一个箭头，表示那里的点往哪个方向、以多快的速度移动。') },
	{ id: 'mse', term: f('mean squared error', '均方误差'), def: f('The average of the squared differences between predictions and targets.', '预测值与目标值之差的平方的平均。'), related: ['loss'] },
	{ id: 'euler-method', term: f('Euler’s method', '欧拉法'), def: f('Following a velocity in small steps: new position = position + step size × velocity.', '沿着速度一小步一小步地走：新位置 = 位置 + 步长 × 速度。'), related: ['velocity-field'] },

	// ---- image and text (added by the TinyCLIP page)
	{ id: 'contrastive-learning', term: f('contrastive learning', '对比学习'), def: f('Learning by comparison: pull matching pairs together and push mismatched pairs apart, without class labels.', '通过比较来学习：把相配的对拉近，把不相配的对推远，不需要类别标签。'), related: ['self-supervised'] },
	{ id: 'cosine-similarity', term: f('cosine similarity', '余弦相似度'), def: f('Two vectors’ products added up, divided by both lengths: 1 when they point the same way, 0 at right angles, −1 when opposite.', '两个向量逐项相乘再相加，除以两者的长度：方向相同时为 1，垂直时为 0，相反时为 −1。'), related: ['embedding'] },
	{ id: 'infonce', term: f('InfoNCE loss', 'InfoNCE 损失'), def: f('CLIP’s loss: in each row and each column of a batch’s scores, the cross-entropy of the true pair after a softmax, averaged.', 'CLIP 的损失：在一批分数的每一行和每一列上做 softmax，取正确那一对的交叉熵，再求平均。'), related: ['cross-entropy', 'softmax'] },
	{ id: 'zero-shot', term: f('zero-shot classification', '零样本分类'), def: f('Classifying with classes the model was never trained on as classes, here by writing each class as a caption.', '用模型从没作为类别训练过的类别来分类；这里的做法是把每个类别写成一句描述。'), related: ['classifier'] },
	{ id: 'bag-of-words', term: f('bag of words', '词袋'), def: f('Text kept as the words it contains and how often, without their order.', '只保留文本里有哪些词、各出现几次，而不保留它们的顺序。'), related: ['token'] },

	// ---- time series (added by the TinyForecaster page)
	{ id: 'time-series', term: f('time series', '时间序列'), def: f('Numbers measured one after another in time, such as a temperature every hour.', '按时间先后测得的一串数，比如每小时的气温。') },
	{ id: 'forecast-window', term: f('window', '窗口'), def: f('The last few values a forecaster reads to predict the next one. Sliding it one step gives the next training example.', '预测模型为了预测下一个值而读取的最近几个值。把它滑动一步，就得到下一个训练样例。'), related: ['time-series'] },
	{ id: 'lstm', term: f('LSTM (long short-term memory)', 'LSTM（长短期记忆网络）'), def: f('A recurrent network that reads a sequence one step at a time and keeps a memory, the cell state c, which gates update.', '一种循环网络：一步一步读序列，并保留一份由各个门更新的记忆，即细胞状态 c。') },
	{ id: 'rollout', term: f('rollout', '滚动预测'), def: f('Forecasting many steps ahead by appending each forecast to the history and reading it back as input.', '把每个预测接到历史后面、再当作输入读回去，以此向前预测很多步。'), related: ['autoregressive'] },
	{ id: 'baseline', term: f('baseline', '基准'), def: f('A simple rule a model is compared with. A model is only worth using if it beats it on new data.', '拿来和模型比较的简单规则。模型只有在新数据上胜过它，才值得用。') },
	{ id: 'random-walk', term: f('random walk', '随机游走'), def: f('A series in which each value is the previous one plus a new random step.', '每个值都等于前一个值加上一个新的随机步的序列。'), related: ['time-series'] },

	// ---- physics-informed networks (added by the PINN page)
	{ id: 'derivative', term: f('derivative (u′, u″)', '导数（u′、u″）'), def: f('How fast a quantity changes: u′ is the slope of u (here the velocity), u″ the slope of u′ (the acceleration).', '一个量变化得有多快：u′ 是 u 的斜率（这里是速度），u″ 是 u′ 的斜率（加速度）。') },
	{ id: 'autodiff', term: f('automatic differentiation (autodiff)', '自动微分'), def: f('Computing exact derivatives of a program by the chain rule, step by step through its operations; backpropagation is one use of it.', '沿着程序的每一步运算用链式法则算出精确的导数；反向传播就是它的一种用法。'), related: ['backpropagation'] },
	{ id: 'chain-rule', term: f('chain rule', '链式法则'), def: f('How to get the slope of a function of a function: multiply the slopes of the steps. If u = f(g(t)), then u′ = f′(g(t)) · g′(t).', '求“函数套函数”的斜率的方法：把每一步的斜率乘起来。若 u = f(g(t))，则 u′ = f′(g(t)) · g′(t)。'), related: ['autodiff', 'backpropagation'] },
	{ id: 'finite-difference', term: f('finite difference', '有限差分'), def: f('An estimate of a slope from two nearby values: (u(t + h) − u(t − h)) / 2h for a small step h.', '用两个相近的值估计斜率：取一个小步长 h，算 (u(t + h) − u(t − h)) / 2h。'), related: ['derivative'] },
	{ id: 'residual', term: f('residual', '残差'), def: f('What is left when you put a curve into an equation that should give 0: here u″ + c·u′ + k·u. Zero means the curve obeys the law there.', '把曲线代入一个本应等于 0 的方程后剩下的部分：这里是 u″ + c·u′ + k·u。为 0 表示曲线在那里遵守定律。'), related: ['derivative'] },
	{ id: 'pinn', term: f('physics-informed neural network (PINN)', '物理信息神经网络（PINN）'), def: f('A network trained on data and, at the same time, on how badly its output breaks a known law (the residual).', '一种网络：训练时既看数据，也看它的输出违反已知定律的程度（残差）。'), related: ['residual', 'loss'] },
	{ id: 'extrapolation', term: f('extrapolation', '外推'), def: f('Predicting outside the range the data covers, such as after the last measured time.', '在数据覆盖的范围之外做预测，例如在最后一个测量时刻之后。') },

	// ---- recommendation (added by the matrix-factorization page)
	{ id: 'rating-matrix', term: f('rating matrix', '评分矩阵'), def: f('A table with one row per person and one column per item; a cell holds that person’s rating of that item, and most cells are empty.', '每个人一行、每个物品一列的表；格子里是这个人给这个物品的评分，大多数格子是空的。') },
	{ id: 'latent-factors', term: f('latent factors', '隐因子'), def: f('The few learned numbers that describe each person and each item in matrix factorization; nobody names them, training finds them.', '矩阵分解中描述每个人、每个物品的少数几个学出来的数；没有人给它们命名，是训练找出来的。'), related: ['rating-matrix'] },
	{ id: 'dot-product', term: f('dot product', '点积'), def: f('Multiply two lists of numbers entry by entry and add the products: (a, b)·(c, d) = a·c + b·d. It is large when the two point the same way.', '把两列数逐项相乘，再把乘积加起来：(a, b)·(c, d) = a·c + b·d。两个向量方向越一致，点积越大。') },
	{ id: 'cold-start', term: f('cold start', '冷启动'), def: f('A recommender that learns only from ratings knows nothing about a new user or a new item until it has some ratings for it; one that also reads features, such as a two-tower model, can score a new item at once.', '只从评分里学习的推荐系统，在新用户或新物品有一些评分之前，对它一无所知；也读特征的推荐系统（比如双塔模型）能马上给新物品打分。'), related: ['rating-matrix', 'two-tower-model'] },

	// ---- masked language models (added by the TinyBERT page)
	{ id: 'masked-language-model', term: f('masked language model (MLM)', '掩码语言模型（MLM）'), def: f('A model trained to fill in hidden tokens from the tokens around them, on both sides. BERT is one.', '一种训练任务是根据两侧的词元补出被藏起来的词元的模型。BERT 就是这样的模型。'), related: ['self-supervised', 'pre-training'] },
	{ id: 'special-token', term: f('special token', '特殊词元'), def: f('A token that stands for no text but marks something for the model, such as [CLS] (the start), [SEP] (the end of a sentence) or [MASK] (a hidden token).', '不代表任何文字、只为模型做标记的词元，例如 [CLS]（开头）、[SEP]（一句话的结尾）或 [MASK]（被藏起来的词元）。'), related: ['token', 'vocabulary'] },
	{ id: 'segment-embedding', term: f('segment embedding', '段嵌入'), def: f('A learned row added to every token that says which sentence it belongs to (A or B), so a model that reads two sentences at once can tell them apart.', '加到每个词元上的一行学来的数，表示它属于哪一句（A 或 B），让同时读两句话的模型能把它们分开。'), related: ['embedding', 'positional-encoding'] },

	// ---- word vectors (added by the word2vec page)
	{ id: 'skip-gram', term: f('skip-gram', '跳字模型（skip-gram）'), def: f('word2vec’s training task: from one word (the center word), predict each word near it (its context words).', 'word2vec 的训练任务：由一个词（中心词）预测它附近的每个词（上下文词）。'), related: ['word-window'] },
	{ id: 'word-window', term: f('window, context words', '窗口、上下文词'), def: f('The words up to w places to the left and right of a center word, in the same sentence; each forms one (center, context) pair with it.', '中心词左右各 w 个位置以内、同一句子里的词；每个都和中心词组成一个（中心词，上下文词）对。'), related: ['skip-gram'] },
	{ id: 'pca', term: f('PCA (principal component analysis)', '主成分分析（PCA）'), def: f('Finds the directions in which a cloud of points spreads the most; the top two give the flat picture that keeps the most of the spread.', '找出一团点分布最开的方向；取前两个方向，就得到保留分布最多的平面图。') },

	// ---- adversarial networks (added by the 2D GAN page)
	{ id: 'gan', term: f('GAN (generative adversarial network)', 'GAN（生成对抗网络）'), def: f('Two networks trained against each other: a generator makes samples, a discriminator tells them from real data.', '两个互相对抗着训练的网络：生成器造样本，判别器把它们和真实数据分开。') },
	{ id: 'generator', term: f('generator', '生成器'), def: f('The network that turns random noise into samples (here: points on a plane).', '把随机噪声变成样本（这里是平面上的点）的网络。'), related: ['gan'] },
	{ id: 'discriminator', term: f('discriminator', '判别器'), def: f('The network that scores a sample: its estimated probability that the sample is real.', '给样本打分的网络：它估计样本是真实数据的概率。'), related: ['gan'] },
	{ id: 'noise', term: f('noise (latent input)', '噪声（潜在输入）'), def: f('The random numbers a generator starts from; a new draw gives a new sample.', '生成器的出发点：一组随机数，每抽一次就得到一个新样本。'), related: ['generator'] },
	{ id: 'mode-collapse', term: f('mode collapse', '模式坍缩'), def: f('A GAN failure: the generator covers only some of the data’s clusters (modes) and ignores the rest.', 'GAN 的一种失败：生成器只覆盖数据里的一部分团（模式），忽略其余的。'), related: ['gan'] },

	// ---- policy gradients (added by the cart-pole page)
	{ id: 'state', term: f('state', '状态'), def: f('The numbers that describe the world at one moment; the policy decides from them.', '描述某一时刻世界的那组数；策略根据它们做决定。'), related: ['policy'] },
	{ id: 'return', term: f('return (discounted)', '（折扣）回报'), def: f('The rewards from one step to the end of the episode, each later one multiplied by one more γ: G = r + γ × (G of the next step).', '从某一步到回合结束的奖励之和，越往后的奖励多乘一次 γ：G = r + γ ×（下一步的 G）。'), related: ['reward', 'discount'] },
	{ id: 'advantage', term: f('advantage', '优势'), def: f('How much better a step’s return is than usual: here (G − mean) / standard deviation over one round of episodes.', '某一步的回报比平常好多少：这里是 (G − 平均值) / 标准差，在一轮回合里计算。'), related: ['return'] },
	{ id: 'policy-gradient', term: f('policy gradient (REINFORCE)', '策略梯度（REINFORCE）'), def: f('Training a policy by making actions followed by a high return more likely and the others less likely.', '训练策略的一种方法：让后面跟着高回报的动作更可能出现，让其余的更少出现。'), related: ['policy', 'gradient'] },

	// ---- classic machine learning (added by the classic-ML canvas)
	{ id: 'knn', term: f('k nearest neighbours (kNN)', 'k 近邻（kNN）'), def: f('A classifier that stores the training points and answers with the most common class among the k points nearest to a new point.', '一种分类器：把训练点存起来，对新点取离它最近的 k 个点中最常见的类别作为回答。'), related: ['decision-boundary'] },
	{ id: 'decision-boundary', term: f('decision boundary, decision map', '决策边界、决策图'), def: f('The map of a classifier’s answer at every place of the input space; the boundary is where the answer changes.', '分类器在输入空间每一处给出的回答画成的图；边界就是回答发生变化的地方。'), related: ['classifier'] },
	{ id: 'decision-tree', term: f('decision tree', '决策树'), def: f('A model that asks yes/no questions about one feature at a time (“is x ≤ 0.3?”); each answer leads to the next question, and the last box gives the class.', '一次只就一个特征问一个是非题（“x ≤ 0.3 吗？”）的模型；每个回答通向下一个问题，最后落到的方框给出类别。'), related: ['gini'] },
	{ id: 'gini', term: f('Gini impurity', '基尼不纯度'), def: f('1 − Σ p², where p is the share of each class in a set of points: 0 when one class only, larger when the classes are mixed.', '1 − Σ p²，其中 p 是每个类别在一组点中的占比：只有一类时为 0，类别越混越大。'), related: ['decision-tree'] },
	{ id: 'clustering', term: f('clustering', '聚类'), def: f('Grouping points by similarity without any labels (unsupervised learning).', '在没有任何标签的情况下按相似程度给点分组（无监督学习）。'), related: ['k-means'] },
	{ id: 'k-means', term: f('k-means', 'k-means（k 均值）'), def: f('A clustering method that repeats two steps: every point joins its nearest centroid, then every centroid moves to the mean of its points.', '一种聚类方法，反复做两步：每个点归到最近的质心，然后每个质心移到它那组点的平均位置。'), related: ['centroid', 'clustering'] },
	{ id: 'centroid', term: f('centroid', '质心'), def: f('The centre of a group of points: the mean of their x values and the mean of their y values.', '一组点的中心：它们 x 的平均值和 y 的平均值。'), related: ['k-means'] },

	// ---- autoencoders (added by the VAE page)
	{ id: 'latent-space', term: f('latent space', '隐空间'), def: f('The space of the code numbers a model squeezes its input into; here a plane, 2 numbers per point.', '模型把输入压缩成的编码数所在的空间；这里是一个平面，每个点 2 个数。') },
	{ id: 'encoder-decoder', term: f('encoder and decoder', '编码器与解码器'), def: f('A model in two parts: the encoder turns the input into a few numbers (a code), and the decoder turns those numbers into the output: a redrawn picture for an autoencoder, a translated sentence, one word at a time, for seq2seq.', '由两部分组成的模型：编码器把输入变成少数几个数（编码），解码器再把这些数变成输出：对自编码器是一幅重画的图，对 seq2seq 是一句一次写一个词的译文。') },
	{ id: 'reparameterization', term: f('reparameterization trick', '重参数化技巧'), def: f('Writing a random sample as z = μ + σ·ε with ε drawn from N(0, 1), so z depends smoothly on μ and σ and training can adjust them.', '把随机采样写成 z = μ + σ·ε，其中 ε 从 N(0, 1) 中抽取，这样 z 平滑地依赖 μ 和 σ，训练就能调整它们。') },
	{ id: 'kl-divergence', term: f('KL divergence', 'KL 散度'), def: f('A measure of how different one probability distribution is from another; 0 when they are the same.', '衡量一个概率分布和另一个有多不同的量；两者相同时为 0。') },
	{ id: 'bce', term: f('binary cross-entropy', '二元交叉熵'), def: f('A loss for numbers between 0 and 1: small when the predicted value is close to the target, large when it is confidently wrong.', '用于 0 到 1 之间数值的损失：预测值接近目标时小，很自信却错了时大。'), related: ['cross-entropy', 'loss'] },

	// ---- sequence to sequence (added by the Tiny seq2seq page)
	{ id: 'context-vector', term: f('context vector', '上下文向量'), def: f('In attention, the weighted mix of the encoder states that one decoder step reads: each state is multiplied by its attention weight and the results are added up.', '在注意力里，解码器某一步读到的编码器状态的加权混合：每个状态乘以它的注意力权重，再加起来。'), related: ['attention', 'encoder-decoder'] },
	{ id: 'greedy-decoding', term: f('greedy decoding', '贪心解码'), def: f('Writing an output by always taking the item with the largest chance at each step, and feeding it back in as the next step’s input.', '每一步都选概率最大的那一项来写输出，并把它作为下一步的输入送回去。'), related: ['sampling', 'autoregressive'] },

	// ---- filtering (added by the Kalman filter page)
	{ id: 'kalman-filter', term: f('Kalman filter', '卡尔曼滤波'), def: f('A method that tracks a changing quantity from noisy measurements by repeating two steps: predict with a motion model, then correct the prediction by a share (the gain) of each new measurement.', '一种从有噪声的测量中跟踪变化量的方法，反复做两步：用运动模型预测，再按一定比例（增益）用新的测量修正预测。'), related: ['kalman-gain', 'motion-model'] },
	{ id: 'motion-model', term: f('motion model', '运动模型'), def: f('The filter’s rule for how the state changes in one time step; here: the position moves on by the velocity, and the velocity stays the same.', '滤波器描述状态在一个时间步内如何变化的规则；这里是：位置按速度往前走，速度保持不变。') },
	{ id: 'variance', term: f('variance and standard deviation', '方差和标准差'), def: f('The standard deviation is the typical size of an error; the variance is its square. When the errors follow a bell curve (a normal distribution), a band of ±2 standard deviations holds the true value about 95 times in 100.', '标准差是误差的典型大小，方差是它的平方。误差服从钟形的正态分布时，±2 个标准差的范围大约 100 次里有 95 次包含真实值。') },
	{ id: 'kalman-gain', term: f('Kalman gain', '卡尔曼增益'), def: f('K = P⁻/(P⁻ + R), with P⁻ the variance of the prediction and R that of the measurement: the share of the gap between the two that the filter moves; near 0 it keeps the prediction, near 1 it takes the measurement.', 'K = P⁻/(P⁻ + R)，其中 P⁻ 是预测的方差，R 是测量的方差：滤波器在预测和测量之间挪动的比例；接近 0 时保持预测，接近 1 时采用测量。'), related: ['variance'] },
	{ id: 'process-noise', term: f('process noise Q and measurement noise R', '过程噪声 Q 和测量噪声 R'), def: f('Q: how much the true motion may change in ways the motion model does not know. R: the variance of the sensor’s error. Both are settings the filter is given.', 'Q：真实运动可能以运动模型不知道的方式改变多少。R：传感器误差的方差。两者都是交给滤波器的设定。'), related: ['kalman-gain'] },
	{ id: 'rmse', term: f('root mean square error (RMSE)', '均方根误差（RMSE）'), def: f('Square every error, average the squares, take the square root: one number for the typical size of the errors.', '把每个误差平方，求平均，再开平方根：用一个数表示误差的典型大小。') },
	{ id: 'moving-average', term: f('moving average', '移动平均'), def: f('The average of the last few measurements, recomputed at every step; it smooths noise but trails behind a moving value.', '最近几次测量的平均值，每一步重新计算；它能抹平噪声，但会落后于一个在变化的值。') },

	// ---- graph layers (added by the GAT / GraphSAGE / GIN page)
	{ id: 'neighbour', term: f('neighbour', '邻居'), def: f('A node linked to a given node by an edge: here, a member’s friends.', '和某个节点之间有一条边的节点：这里就是一个成员的朋友。'), related: ['graph'] },
	{ id: 'aggregation', term: f('aggregation', '聚合'), def: f('The rule that combines the messages a node receives into one: a weighted sum, an average, a plain sum or attention.', '把一个节点收到的消息合成一份的规则：加权和、平均、直接求和或注意力。'), related: ['message-passing'] },
	{ id: 'concatenation', term: f('concatenation', '拼接'), def: f('Placing two rows of numbers end to end to make one longer row.', '把两行数首尾相接，拼成一行更长的数。') },
	{ id: 'leaky-relu', term: f('LeakyReLU', 'LeakyReLU'), def: f('Like ReLU, but a negative number is multiplied by a small slope (here 0.2) instead of becoming 0.', '和 ReLU 类似，但负数不变成 0，而是乘以一个小斜率（这里是 0.2）。'), related: ['relu'] },

	// ---- search and self-play (added by the AlphaZero-lite page)
	{ id: 'mcts', term: f('Monte Carlo tree search (MCTS)', '蒙特卡洛树搜索（MCTS）'), def: f('Looking ahead by growing a tree of positions one simulation at a time, spending more simulations on the moves that look best so far.', '一次一次地模拟，把局面长成一棵树来往后看；看起来越好的走法，分到的模拟越多。'), related: ['policy'] },
	{ id: 'prior', term: f('prior (policy head)', '先验（策略头）'), def: f('The network’s first-impression chance for each legal move, before any search; the chances add up to 1.', '搜索之前，网络给每一步合法走法的第一印象概率；这些概率加起来等于 1。'), related: ['policy', 'softmax'] },
	{ id: 'value-head', term: f('value (value head)', '价值（价值头）'), def: f('The network’s guess of how the game will end for the player to move, from −1 (a loss) to +1 (a win).', '网络对“这盘棋对走棋一方会怎样结束”的猜测，从 −1（输）到 +1（赢）。'), related: ['head'] },
	{ id: 'puct', term: f('PUCT score', 'PUCT 分数'), def: f('U = Q + c·P·√N / (1 + n): how good a move looks so far (Q) plus a bonus for a high prior P and few visits n.', 'U = Q + c·P·√N / (1 + n)：一步棋目前看起来有多好（Q），加上一份奖励分，先验 P 越高、访问次数 n 越少，奖励越大。'), related: ['exploration', 'q-value'] },
	{ id: 'self-play', term: f('self-play', '自我对弈'), def: f('Training by letting the program play against itself; the games are the only training data.', '让程序自己和自己下棋来训练；这些对局就是全部的训练数据。'), related: ['reward'] },

	// ---- state-space models (added by the Tiny Mamba page)
	{ id: 'state-space-model', term: f('state-space model (SSM)', '状态空间模型（SSM）'), def: f('A sequence model that reads one token at a time and carries a state of fixed size from token to token, updated by a simple rule. Mamba is a selective one: the update depends on the token.', '一次读一个词元、把一份大小固定的状态从一个词元带到下一个词元的序列模型，状态按一条简单的规则更新。Mamba 是“选择性”的：更新方式取决于当前词元。'), related: ['token', 'attention'] },
	{ id: 'step-size', term: f('step size Δ (delta)', '步长 Δ（delta）'), def: f('A number of at least 0 that each token computes for each channel. Near 0 the state is kept and nothing is written; large, the old state fades and the token is written in.', '每个词元为每个通道算出的一个不小于 0 的数。接近 0 时状态被保留、什么都不写入；很大时旧状态淡去、写入这个词元。'), related: ['state-space-model'] },
	{ id: 'kv-cache', term: f('key-value cache', '键值缓存'), def: f('The keys and values a Transformer keeps for every earlier token so that new tokens can look back at them. It grows with every token.', 'Transformer 为前面每个词元保存的键和值，让新的词元能回头看它们。每多一个词元，它就变大一点。'), related: ['attention', 'query-key-value'] },

	// ---- adversarial examples (added by the adversarial examples page)
	{ id: 'adversarial-example', term: f('adversarial example', '对抗样本'), def: f('An input changed on purpose, often by very little, so that a model gives a wrong answer.', '被故意改动（常常只改一点点）、让模型给出错误答案的输入。'), related: ['fgsm', 'pgd'] },
	{ id: 'perturbation', term: f('perturbation', '扰动'), def: f('The change x′ − x added to an input; an attack keeps it small, for example at most ε per pixel.', '加到输入上的改动 x′ − x；攻击会让它保持很小，比如每个像素最多改 ε。') },
	{ id: 'fgsm', term: f('FGSM (fast gradient sign method)', 'FGSM（快速梯度符号法）'), def: f('One step x′ = x + ε·sign(∇ₓL): every pixel moves by ε in the direction that raises the loss.', '一步 x′ = x + ε·sign(∇ₓL)：每个像素都朝让损失变大的方向移动 ε。'), related: ['gradient'] },
	{ id: 'pgd', term: f('PGD (projected gradient descent)', 'PGD（投影梯度下降）'), def: f('Many small sign steps on the loss, each followed by clipping back into the allowed box around the input.', '在损失上走很多个符号小步，每步之后都裁剪回输入周围允许的范围里。'), related: ['fgsm'] },
	{ id: 'adversarial-training', term: f('adversarial training', '对抗训练'), def: f('Training on adversarial examples with their correct labels, so the model learns to resist them.', '用带正确标签的对抗样本来训练，让模型学会抵抗它们。') },
	{ id: 'white-box-attack', term: f('white-box attack', '白盒攻击'), def: f('An attack that knows the model’s weights and can compute its exact gradient.', '知道模型权重、能算出它精确梯度的攻击。') },

	// ---- segmentation (added by the tiny U-Net page)
	{ id: 'segmentation', term: f('segmentation', '分割'), def: f('Labelling every pixel of a picture with a class, so the answer is a map of classes as big as the picture.', '给图片的每个像素标上一个类别，所以答案是一张和图片一样大的类别图。'), related: ['classifier'] },
	{ id: 'upsampling', term: f('upsampling', '上采样'), def: f('Making a map larger, for example twice as wide and tall: nearest-neighbour upsampling copies each number into a 2×2 block; bilinear upsampling fills each new cell with a weighted mix of the 4 nearest old cells.', '把一张图放大，例如宽和高各放大一倍：最近邻上采样把每个数复制成一个 2×2 的块；双线性上采样让每个新格子取最近 4 个旧格子的加权混合。'), related: ['pooling', 'feature-map'] },
	{ id: 'skip-connection', term: f('skip connection', '跳跃连接'), def: f('A path that copies a layer’s maps past the layers in between and joins them to a later layer; in a U-Net it concatenates the down path’s sharp maps with the upsampled ones, so fine detail is not lost.', '把某一层的图越过中间的层复制过去、接到后面某一层的通路；在 U-Net 里，它把下行路径清晰的图和上采样的图拼接起来，这样细节不会丢失。'), related: ['u-net', 'concatenation', 'residual-connection'] },
	{ id: 'iou', term: f('IoU (intersection over union)', 'IoU（交并比）'), def: f('For one class: the pixels both the answer and the truth give that class, divided by the pixels at least one of them gives it. 1 is perfect.', '对一个类别：答案和真实标注都给了这个类别的像素数，除以至少有一方给了这个类别的像素数。1 表示完全正确。'), related: ['segmentation'] },

	// ---- diffusion (added by the tiny DDPM page)
	{ id: 'diffusion-model', term: f('diffusion model', '扩散模型'), def: f('A generative model that learns to undo a gradual noising: it makes data by starting from noise and removing it step by step.', '学会撤销逐步加噪的生成模型：它从噪声出发，一步步去掉噪声来造出数据。'), related: ['generative-model'] },
	{ id: 'forward-process', term: f('forward process', '前向过程'), def: f('Adding a little noise at each of T steps until only noise is left; x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε (x₀ scaled to −1 … 1) jumps to step t at once.', '在 T 步里每步加一点噪声，直到只剩噪声；x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε（x₀ 先缩放到 −1 … 1）一下子跳到第 t 步。'), related: ['diffusion-model'] },
	{ id: 'noise-schedule', term: f('noise schedule', '噪声调度'), def: f('How much noise each step adds (β_t); ᾱ_t, the product of 1 − β over steps 1 to t, says how much of the data is left.', '每一步加多少噪声（β_t）；ᾱ_t 是第 1 到第 t 步的 1 − β 之积，表示数据还剩多少。'), related: ['forward-process'] },
	{ id: 'denoising-step', term: f('denoising step', '去噪步'), def: f('One step of sampling: subtract part of the predicted noise from x_t, rescale, and add a little fresh noise to get x_{t−1}.', '采样的一步：从 x_t 中减去预测噪声的一部分，重新缩放，再加一点新噪声，得到 x_{t−1}。'), related: ['diffusion-model'] },
	{ id: 'u-net', term: f('U-Net', 'U-Net'), def: f('A convolutional network shaped like a U: it shrinks the picture step by step, grows it back, and copies maps across with skip connections.', '形状像 U 的卷积网络：它一步步把图缩小，再放大回来，并用跳跃连接把图横着复制过去。'), related: ['skip-connection', 'convolution'] },
	{ id: 'timestep-embedding', term: f('timestep embedding', '时间步嵌入'), def: f('The step number t turned into a list of sine and cosine waves, so the network knows how much noise to expect.', '把步数 t 变成一串正弦和余弦波，让网络知道该预期多少噪声。'), related: ['positional-encoding'] },

	// ---- score-based models (added by the score matching page)
	{ id: 'density', term: f('density', '密度'), def: f('How crowded the points are near a place: high where many points lie close together, near 0 far from all of them.', '某个位置附近点的拥挤程度：许多点挨在一起的地方高，远离所有点的地方接近 0。') },
	{ id: 'score-function', term: f('score', '分数'), def: f('The gradient of the log density, ∇ log p(x): at every place, an arrow pointing the way the density rises fastest.', '对数密度的梯度 ∇ log p(x)：在每个位置，一个指向密度上升最快方向的箭头。'), related: ['density', 'gradient'] },
	{ id: 'denoising-score-matching', term: f('denoising score matching', '去噪分数匹配'), def: f('Learning the score from samples: add noise of size σ to a data point and train a network to point back, (x − x̃) / σ².', '从样本学分数：给数据点加上大小为 σ 的噪声，训练网络指回去，即 (x − x̃) / σ²。'), related: ['score-function'] },
	{ id: 'langevin-dynamics', term: f('Langevin dynamics', '朗之万动力学'), def: f('A sampler that repeats x ← x + (α/2)·score + √α·z with fresh noise z: a climb toward dense places plus a random jiggle.', '一种采样方法，反复执行 x ← x + (α/2)·分数 + √α·z（z 每次重新抽）：朝密集处爬一点，再随机抖一下。'), related: ['score-function'] },
	{ id: 'annealing', term: f('annealing', '退火'), def: f('Sampling with a large noise level first and lowering it step by step, so the samples settle from a rough shape onto the fine one.', '先用大的噪声水平采样，再一步步降低，让样本从粗略的形状慢慢落到精细的形状上。'), related: ['langevin-dynamics'] },

	// ---- node embeddings (added by the node2vec page)
	{ id: 'graph-walk', term: f('random walk on a graph', '图上的随机游走'), def: f('Start at a node and hop to a randomly chosen neighbour, again and again; the nodes visited, in order, are the walk.', '从一个节点出发，一次又一次地随机跳到一个邻居；按顺序经过的节点就是这条游走。'), related: ['graph', 'neighbour'] },
	{ id: 'node2vec-pq', term: f('p and q (node2vec)', 'p 和 q（node2vec）'), def: f('The two knobs of a node2vec walk: going straight back weighs 1/p, moving to a node farther from the one it came from weighs 1/q, and a node next to both weighs 1.', 'node2vec 游走的两个旋钮：直接走回去的权重是 1/p，走到离来处更远的节点权重是 1/q，走到和两者都相邻的节点权重是 1。'), related: ['graph-walk'] },
	{ id: 'bfs-dfs', term: f('BFS and DFS (breadth-first, depth-first search)', 'BFS 和 DFS（广度优先、深度优先搜索）'), def: f('Two ways to explore a graph: breadth-first visits everything near the start before going farther; depth-first follows one path as far as it can.', '探索图的两种方式：广度优先先把起点附近的都走遍再往远处走；深度优先沿着一条路尽量往远处走。'), related: ['graph'] },
	{ id: 'node-embedding', term: f('node embedding', '节点嵌入'), def: f('A learned vector of numbers for each node of a graph, so that nodes that play alike parts get similar vectors.', '为图中每个节点学出的一个数字向量，使作用相似的节点得到相似的向量。'), related: ['embedding', 'graph'] },

	// ---- retrieval and ranking (added by the two-tower page)
	{ id: 'two-tower-model', term: f('two-tower model', '双塔模型'), def: f('A recommender made of two networks: one turns a user into a vector, the other turns an item into a vector, and the score is their dot product.', '由两个网络组成的推荐模型：一个把用户变成向量，另一个把物品变成向量，分数是两者的点积。'), related: ['embedding', 'dot-product'] },
	{ id: 'candidate-retrieval', term: f('retrieval (candidate generation)', '召回（候选生成）'), def: f('The first stage of a recommender: from the whole catalog, quickly pick a short list of candidates, for example the items with the highest dot products.', '推荐系统的第一阶段：从整个物品库里快速挑出一份短的候选名单，比如点积最高的那些物品。'), related: ['two-tower-model', 'ranking-stage'] },
	{ id: 'ranking-stage', term: f('ranking stage', '排序阶段'), def: f('The second stage of a recommender: a model that looks harder at the short list only and puts it in its final order.', '推荐系统的第二阶段：一个只仔细看短名单、并给出最终顺序的模型。'), related: ['candidate-retrieval'] },
	{ id: 'in-batch-negatives', term: f('in-batch negatives', '批内负样本'), def: f('A training trick: in a batch of (user, item) pairs, each user’s own item is the right answer and the other users’ items serve as wrong ones, so no extra negatives have to be drawn.', '一种训练技巧：在一批（用户，物品）对里，每个用户自己的物品是正确答案，其他用户的物品当作错误答案，所以不用另外抽负样本。'), related: ['softmax', 'cross-entropy', 'batch'] },

	// ---- anomaly detection (added by the ECG autoencoder page)
	{ id: 'ecg', term: f('electrocardiogram (ECG)', '心电图（ECG）'), def: f('A recording of the heart’s electrical activity. Each beat shows a small P wave, a tall QRS spike and a T wave.', '心脏电活动的记录。每次心跳都有一个小的 P 波、一个高的 QRS 尖峰和一个 T 波。') },
	{ id: 'autoencoder', term: f('autoencoder', '自编码器'), def: f('A network trained to rebuild its own input through a narrow middle layer, the code. It learns what typical inputs look like, so inputs unlike its training data come back changed.', '一种被训练来重建自身输入的网络，输入要穿过一个狭窄的中间层，即编码。它学到典型输入的样子，所以和训练数据不像的输入被重建后会走样。'), related: ['encoder-decoder', 'bottleneck'] },
	{ id: 'bottleneck', term: f('bottleneck (code)', '瓶颈（编码）'), def: f('The narrowest layer of an autoencoder. Everything the decoder knows about the input must pass through these few numbers.', '自编码器最窄的那一层。解码器对输入所知道的一切，都必须经过这几个数。') },
	{ id: 'anomaly-score', term: f('anomaly score', '异常分数'), def: f('One number for how unusual an input looks to a model; here the typical size of the rebuild error (the root mean square error). At or above a chosen threshold, the input is flagged.', '表示一个输入在模型看来有多反常的一个数；这里是重建误差的典型大小（均方根误差）。达到或超过选定的阈值，这个输入就被标记。'), related: ['rmse'] },
	{ id: 'decision-threshold', term: f('threshold', '阈值'), def: f('The cut-off that turns a score into yes or no: scores at or above it are flagged. Lowering it catches more cases but raises more false alarms.', '把分数变成“是”或“否”的分界线：达到或超过它的分数被标记。把它调低能抓到更多，但误报也更多。') },
	{ id: 'precision-recall', term: f('precision and recall', '精确率和召回率'), def: f('Recall: the share of the real cases that are flagged. Precision: the share of the flagged cases that are real. A lower threshold raises recall and usually lowers precision.', '召回率：真实情况中被标记出来的比例。精确率：被标记的情况中真实的比例。阈值调低会提高召回率，通常也会降低精确率。') },
	{ id: 'roc-curve', term: f('ROC curve and AUC', 'ROC 曲线和 AUC'), def: f('The catch rate (recall) plotted against the false-alarm rate, for every threshold. The area under it (AUC) is the chance that a random real case scores higher than a random normal one: 0.5 is guessing, 1 is perfect.', '对每个阈值，把检出率（召回率）对误报率画出来的曲线。曲线下面积（AUC）是随机一个真实情况比随机一个正常情况得分更高的概率：0.5 等于瞎猜，1 是完美。'), related: ['precision-recall', 'decision-threshold'] }
];

export const GLOSSARY_BY_ID: Record<string, GlossaryTerm> = Object.fromEntries(GLOSSARY.map((g) => [g.id, g]));
