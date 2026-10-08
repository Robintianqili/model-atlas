/**
 * Every model in the brief's §5 tables, all tiers. Backlog rows ("Planned backlog: …") are
 * split into one entry per model so the atlas shows each one.
 *
 * Status: only `tinycnn` and `tinygpt` are live in v0; everything else is `planned`.
 * Sizes and parameter counts are copied from the brief (approximate; "design" = a target for an
 * own model). Live pages show the measured size of their shipped files instead.
 *
 * To add a model: append an entry here (keep slugs lowercase-hyphenated), then build its page
 * under src/routes/m/<slug>/ and flip `status` to 'live'. See KIT.md "Adding a model".
 */
import type { ModelEntry, I18n } from './types.ts';

const f = (en: string, zh: string): I18n => ({ en, zh });
const n = (name: string, zh?: string): I18n => (zh ? { en: name, zh } : { en: name, zh: name });

export const MODELS: ModelEntry[] = [
	// ------------------------------------------------------------------ 1. Vision
	{
		slug: 'tinycnn',
		area: 'vision',
		name: n('TinyCNN'),
		family: f('Convolutional network (CNN)', '卷积神经网络（CNN）'),
		tier: 'P0',
		status: 'live',
		inputs: ['drawing'],
		own: true,
		runtime: 'Own JS forward pass (checked against PyTorch)',
		mode: 'live',
		hook: f(
			'Draw a doodle and watch each convolution find strokes, corners and shapes.',
			'画一幅涂鸦，看每个卷积层如何找出笔画、拐角和形状。'
		),
		shows: f(
			'Your drawing as the exact input tensor, every 3×3 kernel sliding with live arithmetic, ReLU, max-pooling, flattening and the softmax over the classes.',
			'你的画作为精确的输入张量；每个 3×3 卷积核滑动并实时计算；ReLU、最大池化、展平，以及各类别的 softmax。'
		),
		prereqs: [],
		lesson: {
			steps: 10,
			summary: f('From your drawing to the answer: pixels, kernels, ReLU, pooling, dense, softmax.', '从你的画到答案：像素、卷积核、ReLU、池化、全连接、softmax。')
		}
	},
	{
		slug: 'adversarial-tinycnn',
		area: 'vision',
		name: f('Adversarial examples (FGSM / PGD)', '对抗样本（FGSM / PGD）'),
		family: f('Attack on a CNN', '针对 CNN 的攻击'),
		tier: 'P1',
		status: 'live',
		inputs: ['drawing'],
		own: true,
		params: '6,722 (TinyCNN’s)',
		size: '27 KB of weights (TinyCNN’s, shared)',
		runtime: 'Own JS (input gradient in the autograd core, checked against PyTorch)',
		mode: 'live',
		hook: f('Nudge every pixel of your drawing a little along the gradient, and flip TinyCNN’s answer.', '让你的画的每个像素沿着梯度挪一点点，就能翻转 TinyCNN 的答案。'),
		shows: f(
			'The gradient of the loss for every pixel and its sign, an ε slider with x + ε·sign = x′, the change amplified, the class shares before and after, PGD’s steps on the step player, the smallest ε that flips your drawing, and measured rates on the test drawings with the transfer to TinyViT.',
			'损失对每个像素的梯度和它的符号、带 ε 滑块的 x + ε·sign = x′、放大显示的改动、改动前后的类别份额、逐步播放的 PGD、能让你的画翻转的最小 ε，以及在测试画上实测的比例和对 TinyViT 的转移。'
		),
		prereqs: ['tinycnn'],
		lesson: {
			steps: 10,
			summary: f(
				'Measure the loss of TinyCNN’s answer on your drawing, compute its gradient for every pixel, step along the sign (FGSM), repeat small clipped steps (PGD), find how big the change must be, and see the limits of a white-box attack.',
				'在你的画上算出 TinyCNN 答案的损失，求出它对每个像素的梯度，沿着符号走一步（FGSM），再重复裁剪过的小步（PGD），找出改动要多大，并看清白盒攻击的局限。'
			)
		}
	},
	{
		slug: 'resnet-18',
		area: 'vision',
		name: n('ResNet-18'),
		family: f('Residual CNN', '残差卷积网络'),
		tier: 'P1',
		status: 'planned',
		inputs: ['image', 'webcam'],
		own: false,
		params: '≈11.7M',
		size: '≈12 MB (q8)',
		runtime: 'ONNX Runtime Web / Transformers.js',
		mode: 'both',
		hook: f(
			'A real ImageNet classifier: residual additions, and where it looks, with a live class activation map.',
			'真实的 ImageNet 分类器：残差相加，以及它在看哪里（实时类激活图）。'
		),
		shows: f(
			'Feature maps of the 4 stages, the residual add, global average pooling and a live CAM at the last block; precomputed Grad-CAM variants for preset images.',
			'4 个阶段的特征图、残差相加、全局平均池化、最后一个块的实时 CAM；预设图片的 Grad-CAM 等变体为预计算。'
		),
		prereqs: ['tinycnn']
	},
	{
		slug: 'tinyvit',
		area: 'vision',
		name: n('TinyViT'),
		family: f('Vision Transformer', '视觉 Transformer'),
		tier: 'P1',
		status: 'live',
		inputs: ['drawing'],
		own: true,
		params: '116,938',
		size: '457 KB of weights',
		runtime: 'Own JS forward pass (checked against PyTorch)',
		mode: 'live',
		hook: f(
			'The same drawing, read as patches with attention instead of convolutions.',
			'同一幅画，用注意力把它当作图块来读，而不是用卷积。'
		),
		shows: f(
			'Patchify (a convolution whose stride equals its kernel), the [CLS] row and position embeddings, Q/K/V per head, the [CLS] row’s attention drawn over your drawing, the MLP and the 10 class shares; then TinyCNN on the same drawing, side by side.',
			'切块（步长等于卷积核大小的卷积）、[CLS] 行与位置嵌入、每个头的 Q/K/V、画在你的画上的 [CLS] 行注意力、MLP 和 10 个类别的份额；最后与 TinyCNN 在同一幅画上并排对比。'
		),
		prereqs: ['tinycnn', 'tinygpt'],
		lesson: {
			steps: 11,
			summary: f(
				'Cut your drawing into 16 patches, follow the [CLS] row as it looks at them, mixes what it sees and gives the answer, then compare with TinyCNN on the same drawing.',
				'把你的画切成 16 个图块，跟着 [CLS] 行看向它们、混合看到的内容并给出答案，再和 TinyCNN 在同一幅画上比较。'
			)
		}
	},
	{
		slug: 'dinov2',
		area: 'vision',
		name: n('DINOv2-small (with registers)'),
		family: f('Self-supervised Vision Transformer', '自监督视觉 Transformer'),
		tier: 'P1',
		status: 'planned',
		inputs: ['image'],
		own: false,
		params: '≈23M',
		size: '≈20 MB (q4)',
		runtime: 'Transformers.js worker (WebGPU, falls back to WASM)',
		mode: 'both',
		hook: f(
			'A vision model trained without labels whose attention still finds objects.',
			'一个不用标签训练的视觉模型，它的注意力依然能找到物体。'
		),
		shows: f(
			'CLS-to-patch attention per layer and head, attention rollout, a PCA image of patch features and the top-5 ImageNet classes.',
			'每层每头的 CLS→图块注意力、注意力展开（rollout）、图块特征的 PCA 图像，以及前 5 个 ImageNet 类别。'
		),
		notes: f(
			'The upstream weights license must be checked before self-hosting.',
			'自托管前需核实上游权重的许可。'
		),
		prereqs: ['tinyvit']
	},
	{
		slug: 'deit-tiny',
		area: 'vision',
		name: n('DeiT-Tiny (raw WebGPU)'),
		family: f('Vision Transformer', '视觉 Transformer'),
		tier: 'P2',
		status: 'planned',
		inputs: ['image'],
		own: false,
		size: '≈5 MB',
		runtime: 'Raw WebGPU shaders',
		mode: 'live',
		hook: f('A small ImageNet ViT running on hand-written GPU shaders.', '一个在手写 GPU 着色器上运行的小型 ImageNet ViT。'),
		shows: f('Layer and head sliders over the attention maps.', '用滑块切换各层、各头的注意力图。'),
		prereqs: ['tinyvit']
	},
	{
		slug: 'slimsam',
		area: 'vision',
		name: n('SlimSAM (segment anything)'),
		family: f('Promptable segmentation', '可提示分割'),
		tier: 'P2',
		status: 'planned',
		inputs: ['image'],
		own: false,
		params: '≈10M',
		size: '≈13.8 MB (quantized)',
		runtime: 'Transformers.js',
		mode: 'live',
		hook: f('Click a point on a photo and get a mask for that object.', '在照片上点一下，得到该物体的分割掩码。'),
		shows: f(
			'Timing per stage, the 64×64×256 image embedding as a PCA image, all 3 candidate masks with IoU scores and a threshold slider on the mask logits.',
			'各阶段耗时、64×64×256 图像嵌入的 PCA 图、3 个候选掩码及其 IoU 分数、掩码 logit 的阈值滑块。'
		),
		prereqs: ['resnet-18']
	},
	{
		slug: 'tiny-detector',
		area: 'vision',
		name: f('Tiny object detector (YOLO-style)', '小型目标检测器（YOLO 风格）'),
		family: f('Object detection', '目标检测'),
		tier: 'P2',
		status: 'planned',
		inputs: ['image', 'webcam'],
		own: true,
		params: 'YOLOS-tiny ≈6.6M, or an own model ≤5 MB',
		size: '≤5 MB own (design) / 9.66 MB YOLOS-tiny q8',
		runtime: 'ORT Web / Transformers.js',
		mode: 'live',
		hook: f(
			'See every box a detector proposes before non-maximum suppression cleans them up.',
			'看看检测器在非极大值抑制之前提出的每一个框。'
		),
		shows: f(
			'Grid cells as small classifiers, objectness per scale, all boxes before NMS and an animated NMS with IoU and confidence sliders.',
			'网格单元作为小分类器、各尺度的目标置信度、NMS 前的所有框、带 IoU 与置信度滑块的 NMS 动画。'
		),
		prereqs: ['tinycnn']
	},
	{
		slug: 'tiny-unet-seg',
		area: 'vision',
		name: f('Tiny U-Net segmentation', '小型 U-Net 分割'),
		family: f('Encoder–decoder CNN', '编码器–解码器 CNN'),
		tier: 'P2',
		status: 'live',
		inputs: ['drawing'],
		own: true,
		params: '29,644',
		size: '116 KB of weights',
		runtime: 'Own JS forward pass (checked against PyTorch)',
		mode: 'live',
		hook: f('Stamp or paint circles, squares and triangles, and watch a U-Net label every pixel.', '盖章或涂画圆、正方形和三角形，看 U-Net 给每个像素打标签。'),
		shows: f(
			'Your own scene through the U: one live map per level and how far one cell sees, one upsampled cell worked out, a twin trained without skip connections beside the U-Net, the per-pixel softmax, the mask with one pixel’s three windows, and IoU per class.',
			'你自己的场景走过整个 U 形：每一层一张实时的图，以及一个格子能看到多大一片；算出一个上采样格子；去掉跳跃连接训练的对照网络和 U-Net 并排；逐像素 softmax；掩码和一个像素的三个窗口；各类别的 IoU。'
		),
		prereqs: ['tinycnn'],
		lesson: {
			steps: 11,
			summary: f(
				'Follow a 32×32 picture down a U-Net (smaller maps that see more), through the bottleneck and back up (upsampling plus skip connections), to a softmax per pixel and a class map; score it with IoU, then find a picture it gets wrong.',
				'跟着一幅 32×32 的图走过 U-Net：向下（图变小、看得更广）、穿过瓶颈、再向上（上采样加跳跃连接），得到逐像素的 softmax 和类别图；用 IoU 打分，再找一幅它答错的图。'
			)
		}
	},
	{
		slug: 'depth-estimation',
		area: 'vision',
		name: f('Monocular depth estimation', '单目深度估计'),
		family: f('Dense prediction', '稠密预测'),
		tier: 'P2',
		status: 'planned',
		inputs: ['image'],
		own: false,
		runtime: 'ORT Web (model not chosen yet)',
		mode: 'precomputed',
		hook: f('Guess how far away each pixel is, from a single photo.', '只用一张照片，估计每个像素有多远。'),
		shows: f('A depth map with an overlay slider, and the features that signal distance.', '带叠加滑块的深度图，以及提示远近的特征。'),
		notes: f('Model choice and size are not verified yet.', '模型选择与大小尚未核实。'),
		prereqs: ['tiny-unet-seg']
	},
	{
		slug: 'crnn-ocr',
		area: 'vision',
		name: f('OCR with CRNN + CTC', 'CRNN + CTC 文字识别'),
		family: f('CNN + RNN + CTC', 'CNN + RNN + CTC'),
		tier: 'P2',
		status: 'planned',
		inputs: ['drawing', 'image'],
		own: true,
		size: '≤5 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Write a word and watch letters emerge from a row of column features.', '写一个单词，看字母如何从一列列特征中浮现。'),
		shows: f('Conv features per column, RNN states and the CTC collapse from frame labels to text.', '每列的卷积特征、RNN 状态，以及 CTC 把逐帧标签合并成文字。'),
		prereqs: ['tinycnn', 'char-rnn']
	},
	{
		slug: 'style-transfer',
		area: 'vision',
		name: f('Neural style transfer', '神经风格迁移'),
		family: f('Optimization on CNN features', '基于 CNN 特征的优化'),
		tier: 'P2',
		status: 'planned',
		inputs: ['image'],
		own: false,
		runtime: 'ORT Web / precomputed',
		mode: 'both',
		hook: f('Mix the content of one picture with the textures of another.', '把一张图的内容和另一张图的纹理混在一起。'),
		shows: f('Gram matrices as "style", content loss vs style loss, and the image changing step by step.', '把 Gram 矩阵当作“风格”，内容损失与风格损失，图像逐步变化。'),
		prereqs: ['resnet-18']
	},
	{
		slug: 'super-resolution',
		area: 'vision',
		name: f('Super-resolution', '超分辨率'),
		family: f('Upsampling CNN', '上采样 CNN'),
		tier: 'P2',
		status: 'planned',
		inputs: ['image'],
		own: true,
		size: '≤5 MB (design)',
		runtime: 'Own JS / ORT Web',
		mode: 'live',
		hook: f('Turn a small image into a larger one, and see what the network invents.', '把小图放大，看看网络“编造”了哪些细节。'),
		shows: f('Sub-pixel convolution, before and after crops, and where detail is made up.', '亚像素卷积、放大前后的局部对比，以及哪些细节是编出来的。'),
		prereqs: ['tinycnn']
	},
	{
		slug: 'pose-estimation',
		area: 'vision',
		name: f('Pose estimation', '姿态估计'),
		family: f('Keypoint heatmaps', '关键点热力图'),
		tier: 'P2',
		status: 'planned',
		inputs: ['webcam', 'image'],
		own: false,
		runtime: 'ORT Web',
		mode: 'live',
		hook: f('Find shoulders, elbows and knees as peaks in heatmaps.', '把肩、肘、膝找成热力图中的峰值。'),
		shows: f('One heatmap per keypoint, peak picking and the resulting skeleton.', '每个关键点一张热力图、取峰值、连成骨架。'),
		prereqs: ['tinycnn']
	},
	{
		slug: 'video-3dcnn',
		area: 'vision',
		name: f('Tiny 3D CNN for video', '视频用小型 3D CNN'),
		family: f('3D convolution', '三维卷积'),
		tier: 'P2',
		status: 'planned',
		inputs: ['video'],
		own: true,
		size: '≤5 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Compare a model that sees motion with one that sees single frames.', '比较能看到运动的模型和只看单帧的模型。'),
		shows: f('3D kernels sliding over time; frame-wise CNN vs 3D CNN on the same clip.', '在时间维上滑动的 3D 卷积核；同一段视频上逐帧 CNN 与 3D CNN 的对比。'),
		prereqs: ['tinycnn']
	},
	{
		slug: 'pointnet',
		area: 'vision',
		name: f('PointNet on drawn point clouds', '手绘点云上的 PointNet'),
		family: f('Point-set network', '点集网络'),
		tier: 'P2',
		status: 'planned',
		inputs: ['pointcloud', 'points2d'],
		own: true,
		size: '≤5 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Classify a shape made of loose points, given in any order.', '对一组无序散点组成的形状进行分类。'),
		shows: f('The shared per-point MLP, max-pooling as an order-independent step, and the critical points.', '逐点共享的 MLP、与顺序无关的最大池化，以及关键点。'),
		prereqs: ['mlp-playground']
	},
	{
		slug: 'nerf-toy',
		area: 'vision',
		name: f('NeRF on a line or plane (toy)', '一维/二维 NeRF（玩具）'),
		family: f('Neural field', '神经场'),
		tier: 'P2',
		status: 'planned',
		inputs: ['points2d'],
		own: true,
		size: '<1 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('A network that stores a scene as a function of position.', '一个把场景存成“位置的函数”的网络。'),
		shows: f('Positional encoding, samples along a ray and volume rendering on a 2D toy scene.', '位置编码、沿光线采样，以及二维玩具场景中的体渲染。'),
		prereqs: ['mlp-playground']
	},
	{
		slug: 'gaussian-splat-toy',
		area: 'vision',
		name: f('Gaussian splatting (toy)', '高斯泼溅（玩具）'),
		family: f('Explicit radiance field', '显式辐射场'),
		tier: 'P2',
		status: 'planned',
		inputs: ['image'],
		own: true,
		size: '<1 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Represent a picture as many soft colored blobs, and fit them.', '把一幅图表示成许多柔和的彩色斑点，并拟合它们。'),
		shows: f('Blobs moving, growing and splitting while the fit improves.', '拟合过程中斑点移动、变大和分裂。'),
		prereqs: ['nerf-toy']
	},

	// ------------------------------------------------------------------ 2. Language
	{
		slug: 'tokenizers',
		area: 'language',
		name: f('Real tokenizers side by side', '真实分词器对比'),
		family: f('Tokenizer', '分词器'),
		tier: 'P0',
		status: 'planned',
		inputs: ['text'],
		own: false,
		size: '0.67–9.1 MB per tokenizer.json (loaded on demand)',
		runtime: '@huggingface/tokenizers in a worker',
		mode: 'live',
		hook: f(
			'Type anything and see how BERT, GPT-4, Llama 3, T5 and Claude tokenizers cut it up.',
			'随便输入，看看 BERT、GPT-4、Llama 3、T5、Claude 的分词器怎么切分它。'
		),
		shows: f(
			'Normalizer, pre-tokenizer, merges and post-processor, with a UTF-8 byte view. Each tokenizer file loads only when you pick it.',
			'归一化、预分词、合并规则与后处理，并显示 UTF-8 字节。每个分词器文件只在你选中时加载。'
		),
		prereqs: ['tinygpt']
	},
	{
		slug: 'tinygpt',
		area: 'language',
		name: n('TinyGPT'),
		family: f('Transformer decoder (GPT)', 'Transformer 解码器（GPT）'),
		tier: 'P0',
		status: 'live',
		inputs: ['text'],
		own: true,
		runtime: 'Own JS forward pass (checked against PyTorch)',
		mode: 'live',
		hook: f(
			'Type a prompt and follow every number from tokens to next-token probabilities.',
			'输入一段提示，追踪从词元到“下一个词元概率”的每一个数字。'
		),
		shows: f(
			'Embedding lookup, positions, LayerNorm, Q/K/V, attention scores, the causal mask, softmax, weighted values, output projection, the residual stream, the MLP, logits and sampling, all computed live.',
			'嵌入查表、位置、LayerNorm、Q/K/V、注意力分数、因果掩码、softmax、加权求和、输出投影、残差流、MLP、logits 与采样——全部实时计算。'
		),
		prereqs: [],
		lesson: {
			steps: 12,
			summary: f(
				'From your text to the next token: tokens, embeddings, LayerNorm, attention, the MLP, the logit lens, sampling.',
				'从你的文本到下一个词元：词元、嵌入、层归一化、注意力、MLP、logit lens、采样。'
			)
		}
	},
	{
		slug: 'gpt-2',
		area: 'language',
		name: n('GPT-2 small (124M)'),
		family: f('Transformer decoder (GPT)', 'Transformer 解码器（GPT）'),
		tier: 'P1',
		tierNote: f('Precomputed data is P0; live mode is P1', '预计算数据属于 P0；实时模式属于 P1'),
		status: 'planned',
		inputs: ['text'],
		own: false,
		params: '124M',
		size: 'Precomputed traces; live q8 128 MB (opt-in)',
		runtime: 'Precomputed traces; optional Transformers.js',
		mode: 'both',
		hook: f('The TinyGPT diagram at real scale: 12 layers, 12 heads, 768 dimensions.', 'TinyGPT 图解的真实尺度版：12 层、12 个头、768 维。'),
		shows: f(
			'Curated prompts with precomputed attention and activations; an optional live mode on desktop.',
			'精选提示词的预计算注意力和激活值；桌面端可选实时模式。'
		),
		prereqs: ['tinygpt']
	},
	{
		slug: 'tinybert',
		area: 'language',
		name: n('TinyBERT (masked LM)'),
		family: f('Transformer encoder (BERT)', 'Transformer 编码器（BERT）'),
		tier: 'P1',
		status: 'live',
		inputs: ['text'],
		own: true,
		params: '364,420',
		size: '1.46 MB of weights',
		runtime: 'Own JS forward pass (checked against PyTorch)',
		mode: 'live',
		hook: f('Mask any word and watch a model that reads in both directions fill it in.', '遮住任意一个词，看一个双向阅读的模型如何把它补上。'),
		shows: f(
			'Bidirectional vs causal attention (next to TinyGPT on your text), [MASK] prediction, segment embeddings and BERT’s 15% / 80-10-10 masking, trained on the same corpus as TinyGPT.',
			'双向注意力与因果注意力对比（在你的文字上与 TinyGPT 并排）、[MASK] 预测、段嵌入和 BERT 的 15% / 80-10-10 掩码规则；与 TinyGPT 使用相同语料训练。'
		),
		prereqs: ['tinygpt'],
		lesson: {
			steps: 10,
			summary: f(
				'Hide one token of your text and follow the [MASK] row as it reads both sides of the gap, next to TinyGPT, which reads only to the left; see TinyBERT’s guesses, how training makes its own puzzles, and why it is a poor writer.',
				'藏起你文字里的一个词元，跟着 [MASK] 行读空缺的两侧，并和只读左边的 TinyGPT 对照；看 TinyBERT 的猜测、训练如何自己出题，以及它为什么不擅长写作。'
			)
		}
	},
	{
		slug: 'bert-base',
		area: 'language',
		name: n('BERT-base (110M)'),
		family: f('Transformer encoder (BERT)', 'Transformer 编码器（BERT）'),
		tier: 'P1',
		status: 'planned',
		inputs: ['text'],
		own: false,
		params: '110M',
		size: 'Precomputed traces only',
		runtime: 'Precomputed (BertViz-style layout)',
		mode: 'precomputed',
		hook: f(
			'Real attention heads on sentence pairs, including fine-tuned MRPC and RTE checkpoints.',
			'真实注意力头在句子对上的表现，包括在 MRPC、RTE 上微调过的检查点。'
		),
		shows: f(
			'A layers × heads grid, a neuron view that continues past softmax (×V, projection, residual, LayerNorm, FFN) and the [CLS] head.',
			'层×头网格、延伸到 softmax 之后的神经元视图（×V、投影、残差、LayerNorm、FFN），以及 [CLS] 分类头。'
		),
		prereqs: ['tinybert']
	},
	{
		slug: 'word2vec',
		area: 'language',
		name: n('word2vec (skip-gram)'),
		family: f('Word embeddings', '词嵌入'),
		tier: 'P1',
		status: 'live',
		inputs: ['text'],
		own: true,
		params: '2 × V × 5 (V ≤ 40 words: at most 400 numbers)',
		size: 'about 2 KB of preset text; no weights shipped',
		runtime: 'Own autograd core in a worker',
		mode: 'live',
		hook: f('Train word vectors on your own tiny corpus and watch similar words drift together.', '在你自己的小语料上训练词向量，看相似的词慢慢靠拢。'),
		shows: f(
			'The sliding window’s (center, context) pairs, one SGD step worked out on both weight tables, the softmax and loss, and a PCA map of the vectors as they train.',
			'滑动窗口给出的（中心词，上下文词）训练对、在两张权重表上算一遍的一步 SGD、softmax 和损失，以及训练中词向量的 PCA 图。'
		),
		prereqs: [],
		lesson: {
			steps: 11,
			summary: f(
				'A sliding window turns your sentences into (center, context) pairs; one training step is worked out with its numbers; then 3,000 steps pull words that share context words together, and a second run shows how noisy a tiny corpus is.',
				'滑动窗口把你的句子变成（中心词，上下文词）训练对；带着数字算一遍一步训练；然后 3,000 步训练让上下文相同的词靠在一起；再训练一次，看看很小的语料有多不稳定。'
			)
		}
	},
	{
		slug: 'char-rnn',
		area: 'language',
		name: f('Character RNN (LSTM)', '字符级 RNN（LSTM）'),
		family: f('Recurrent network', '循环神经网络'),
		tier: 'P1',
		status: 'live',
		inputs: ['text'],
		own: true,
		params: '29,626',
		size: '116 KB of weights',
		runtime: 'Own JS forward pass (checked against PyTorch)',
		mode: 'live',
		hook: f('A network with memory reads text one character at a time.', '一个有“记忆”的网络，一次读一个字符。'),
		shows: f(
			'The hidden state of all 64 units as a heatmap under your text, one unit’s gates worked out (c = f·c + i·g), the next-character chances, the network unrolled through time, a unit that counts letters, sampling vs always taking the top character, and how fast the memory fades.',
			'你文字下方 64 个单元隐藏状态的热力图、一个单元逐个门的计算（c = f·c + i·g）、下一个字符的概率、按时间展开的网络、一个数字母的单元、随机抽样与总选最可能字符的对比，以及记忆消退得有多快。'
		),
		prereqs: [],
		lesson: {
			steps: 11,
			summary: f(
				'Follow an LSTM as it reads your text one character at a time: its memory, one unit worked out gate by gate, its guess for the next character, how it writes, and how fast its memory fades.',
				'跟着 LSTM 一次读一个字符地读你的文字：它的记忆、一个单元逐个门的计算、它对下一个字符的猜测、它怎样写字，以及它的记忆消退得有多快。'
			)
		}
	},
	{
		slug: 'sentence-embeddings',
		area: 'language',
		name: f('Sentence embeddings and retrieval', '句向量与检索'),
		family: f('Bi-encoder / cross-encoder', '双编码器 / 交叉编码器'),
		tier: 'P1',
		status: 'planned',
		inputs: ['text'],
		own: false,
		size: 'MiniLM-class model (size unverified)',
		runtime: 'Transformers.js',
		mode: 'both',
		hook: f(
			'Turn sentences into vectors, find the nearest ones, and build a 3-step RAG pipeline.',
			'把句子变成向量，找最相近的句子，并搭一个三步 RAG 流程。'
		),
		shows: f(
			'Mean pooling, normalized vectors and a cosine matrix; bi-encoder vs cross-encoder; brute force vs HNSW-style search.',
			'平均池化、归一化向量与余弦相似度矩阵；双编码器与交叉编码器；暴力搜索与 HNSW 式近似搜索。'
		),
		prereqs: ['word2vec']
	},
	{
		slug: 'tinyseq2seq',
		area: 'language',
		name: f('Tiny seq2seq with attention', '带注意力的小型 seq2seq'),
		family: f('RNN encoder–decoder', 'RNN 编码器–解码器'),
		tier: 'P2',
		status: 'live',
		inputs: ['text'],
		own: true,
		params: '16,941',
		size: '66 KB of weights',
		runtime: 'Own JS forward pass (checked against PyTorch)',
		mode: 'live',
		hook: f('Convert date formats and watch the decoder look back at the input.', '转换日期格式，看解码器如何回看输入。'),
		shows: f(
			'Encoder states, attention scores, softmax weights and the context vector with real numbers, the alignment matrix filling in, and greedy decoding step by step.',
			'编码器状态、注意力分数、softmax 权重和上下文向量（真实数值）、逐行填满的对齐矩阵，以及一步一步的贪心解码。'
		),
		prereqs: ['char-rnn'],
		lesson: {
			steps: 11,
			summary: f(
				'An encoder reads your date one character at a time; a decoder writes it as 2021-03-05, and at every step attention scores your characters, turns the scores into weights and mixes the encoder states into a context vector. Watch the alignment matrix fill in, then test formats it never saw.',
				'编码器一次读你日期的一个字符；解码器把它写成 2021-03-05 的样子，每一步注意力都给你的字符打分，把分数变成权重，再把编码器状态混合成上下文向量。看对齐矩阵一行行填满，再测试它没见过的格式。'
			)
		}
	},
	{
		slug: 'tiny-encdec',
		area: 'language',
		name: f('Tiny encoder–decoder Transformer (T5-style)', '小型编码器–解码器 Transformer（T5 风格）'),
		family: f('Transformer encoder–decoder', 'Transformer 编码器–解码器'),
		tier: 'P2',
		status: 'planned',
		inputs: ['text'],
		own: true,
		params: '≈0.2M (design)',
		runtime: 'Own JS / ORT Web',
		mode: 'live',
		hook: f('The same task with cross-attention instead of an RNN.', '同一个任务，用交叉注意力代替 RNN。'),
		shows: f('Encoder self-attention, decoder causal attention and cross-attention for every output token.', '编码器自注意力、解码器因果注意力，以及每个输出词元的交叉注意力。'),
		prereqs: ['tinygpt', 'tinyseq2seq']
	},
	{
		slug: 'tinymamba',
		area: 'language',
		name: f('Tiny Mamba (state-space model)', '小型 Mamba（状态空间模型）'),
		family: f('Selective state-space model', '选择性状态空间模型'),
		tier: 'P2',
		status: 'live',
		inputs: ['tokens', 'text'],
		own: true,
		params: '2,474',
		size: '9.7 KB of weights',
		runtime: 'Own JS forward pass (checked against PyTorch)',
		mode: 'live',
		hook: f('A sequence model without attention: a scan that decides what to remember.', '没有注意力的序列模型：一个决定记住什么的扫描过程。'),
		shows: f(
			'Δ, B and C at every token of your row, the state filled in token by token, one scan step worked out with real numbers, its memory against attention’s, and where a fixed state runs out on selective copying.',
			'你这一行每个词元的 Δ、B、C，逐个词元填进去的状态，用真实数字算出的一步扫描，它和注意力的记忆对比，以及固定状态在选择性复制任务上在哪里不够用。'
		),
		prereqs: ['char-rnn', 'tinygpt'],
		lesson: {
			steps: 10,
			summary: f(
				'Give a tiny selective state-space model a row of dots and letters and follow its fixed-size state token by token: the step size Δ keeps the state at the dots and writes the letters in, B writes, C reads, and at each question mark a letter comes back; then compare its memory with attention’s and find what it cannot hold.',
				'给一个小小的选择性状态空间模型一行点和字母，逐个词元跟踪它大小固定的状态：步长 Δ 在点处保持状态、在字母处写入，B 负责写，C 负责读，每个问号处找回一个字母；再把它的记忆和注意力的比一比，看看它装不下什么。'
			)
		}
	},
	{
		slug: 'tinymoe',
		area: 'language',
		name: f('Tiny mixture of experts', '小型混合专家模型'),
		family: f('Sparse MoE Transformer', '稀疏 MoE Transformer'),
		tier: 'P2',
		status: 'planned',
		inputs: ['text'],
		own: true,
		params: '<0.2M (design)',
		runtime: 'Own JS + precomputed traces',
		mode: 'both',
		hook: f('Each token picks 2 of 4 experts. See who handles what.', '每个词元从 4 个专家中选 2 个。看看谁负责什么。'),
		shows: f('Router logits, a layer × expert grid per token and a cumulative routing heatmap.', '路由 logits、每个词元的层×专家网格，以及累计路由热力图。'),
		prereqs: ['tinygpt']
	},
	{
		slug: 'logit-lens',
		area: 'language',
		name: f('Logit lens on a small LLM', '小型大模型上的 logit lens'),
		family: f('Interpretability view', '可解释性视图'),
		tier: 'P2',
		status: 'planned',
		inputs: ['text'],
		own: false,
		size: 'Precomputed (SmolLM2-135M class)',
		runtime: 'Precomputed',
		mode: 'precomputed',
		hook: f('Decode the hidden state after every layer: when did the model decide?', '在每一层之后解码隐藏状态：模型是什么时候“拿定主意”的？'),
		shows: f('Per-layer top predictions for curated prompts.', '精选提示词在每一层的最高预测。'),
		prereqs: ['tinygpt']
	},
	{
		slug: 'bilstm-crf',
		area: 'language',
		name: f('Named entities with BiLSTM-CRF', 'BiLSTM-CRF 命名实体识别'),
		family: f('Sequence labeling', '序列标注'),
		tier: 'P2',
		status: 'planned',
		inputs: ['text'],
		own: true,
		params: '<0.2M (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Tag names, places and dates, and watch the Viterbi path pick the best tag sequence.', '标注人名、地名和日期，看维特比路径如何选出最佳标签序列。'),
		shows: f('Emission vs transition scores and the Viterbi lattice.', '发射分数与转移分数，以及维特比格。'),
		prereqs: ['char-rnn']
	},
	{
		slug: 'reasoning-agents',
		area: 'language',
		name: f('Reasoning models and tool use', '推理模型与工具调用'),
		family: f('LLM traces', '大模型轨迹'),
		tier: 'P2',
		status: 'planned',
		inputs: ['text'],
		own: false,
		runtime: 'Precomputed traces from a nanochat-scale model',
		mode: 'precomputed',
		hook: f('Step through reasoning and tool-call traces from a small chat model.', '逐步查看一个小型对话模型的推理与工具调用轨迹。'),
		shows: f('Thinking tokens, tool calls and their results, and the final answer.', '思考词元、工具调用及其结果，以及最终回答。'),
		prereqs: ['tinygpt']
	},

	// ------------------------------------------------------------------ 3. Speech / audio
	{
		slug: 'kws',
		area: 'audio',
		name: f('Keyword spotting (speech commands)', '关键词识别（语音指令）'),
		family: f('CNN on spectrograms', '频谱图上的 CNN'),
		tier: 'P0',
		tierNote: f("In the brief's MVP; deferred until after v0", '属于设计简报中的 MVP；推迟到 v0 之后'),
		status: 'planned',
		inputs: ['audio'],
		own: true,
		params: '1,468,684',
		size: '5.9 MB (fp32)',
		runtime: 'TF.js port (retrained copy)',
		mode: 'live',
		hook: f('Say "yes", "no" or "left" and watch your voice become a spectrogram and then a word.', '说出 “yes”“no” 或 “left”，看你的声音如何变成频谱图，再变成一个词。'),
		shows: f(
			'FFT, a 232×43 spectrogram, 4 conv layers, a 2000-unit dense layer and the softmax over 20 words; precomputed top clips per channel.',
			'FFT、232×43 频谱图、4 个卷积层、2000 单元全连接层、20 个词的 softmax；每个通道的高激活片段为预计算。'
		),
		prereqs: ['tinycnn']
	},
	{
		slug: 'whisper-tiny',
		area: 'audio',
		name: n('Whisper-tiny.en'),
		family: f('Speech Transformer (encoder–decoder)', '语音 Transformer（编码器–解码器）'),
		tier: 'P1',
		status: 'planned',
		inputs: ['audio'],
		own: false,
		params: '39M',
		size: '≈41 MB (q8, opt-in)',
		runtime: 'Transformers.js / ORT Web with a chunk loader',
		mode: 'both',
		hook: f('Speech to text: a log-mel spectrogram goes in, tokens come out, and cross-attention lines them up.', '语音转文字：输入对数梅尔频谱，输出词元，交叉注意力把两者对齐。'),
		shows: f(
			'30-second padding, the 80-bin log-mel, the conv stem, encoder layers, token-by-token decoding and the cross-attention alignment.',
			'30 秒填充、80 维对数梅尔频谱、卷积前端、编码器各层、逐词元解码，以及交叉注意力对齐。'
		),
		prereqs: ['kws', 'tinygpt']
	},
	{
		slug: 'wav2vec2-ctc',
		area: 'audio',
		name: n('wav2vec 2.0 + CTC'),
		family: f('Self-supervised speech encoder', '自监督语音编码器'),
		tier: 'P2',
		status: 'planned',
		inputs: ['audio'],
		own: false,
		runtime: 'Transformers.js / precomputed',
		mode: 'both',
		hook: f('Frame-by-frame letters collapse into words with CTC.', '逐帧的字母通过 CTC 合并成单词。'),
		shows: f('Frame-level outputs, an interactive CTC lattice and the masked-frame contrastive objective.', '帧级输出、可交互的 CTC 格，以及遮蔽帧对比学习目标。'),
		prereqs: ['kws']
	},
	{
		slug: 'kokoro',
		area: 'audio',
		name: n('Kokoro-82M text-to-speech'),
		family: f('Text-to-speech', '语音合成'),
		tier: 'P2',
		status: 'planned',
		inputs: ['text'],
		own: false,
		params: '82M',
		size: '92.4 MB (q8, opt-in)',
		runtime: 'kokoro-js',
		mode: 'both',
		hook: f('Type a sentence and follow it through phonemes to a waveform.', '输入一句话，追踪它从音素到波形的过程。'),
		shows: f('Text normalization, IPA phonemes, tokens, waveform and spectrogram, and voice-vector interpolation.', '文本归一化、IPA 音素、词元、波形与频谱图，以及音色向量插值。'),
		notes: f('The eSpeak NG (GPL-3.0) bundling question must be resolved first.', '需先解决 eSpeak NG（GPL-3.0）的打包许可问题。'),
		prereqs: ['tinygpt']
	},
	{
		slug: 'crepe',
		area: 'audio',
		name: n('CREPE pitch tracker'),
		family: f('CNN on raw audio', '原始音频上的 CNN'),
		tier: 'P2',
		status: 'planned',
		inputs: ['audio'],
		own: false,
		runtime: 'TF.js',
		mode: 'live',
		hook: f('Hum a tune and see the full pitch distribution over time.', '哼一段旋律，看完整的音高分布随时间变化。'),
		shows: f('The 360-bin output as a heatmap over time and the weighted-average decoding.', '360 个音高区间的输出热力图，以及加权平均解码。'),
		prereqs: ['kws']
	},
	{
		slug: 'rvq-codec',
		area: 'audio',
		name: f('Neural audio codec (RVQ)', '神经音频编解码器（RVQ）'),
		family: f('Residual vector quantization', '残差向量量化'),
		tier: 'P2',
		status: 'planned',
		inputs: ['audio'],
		own: true,
		size: '≤2 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('How audio becomes tokens: each codebook stage adds finer detail.', '音频如何变成词元：每一级码本补充更细的细节。'),
		shows: f('Codebook stages adding residual detail, and a bitrate vs quality slider.', '逐级码本补充残差细节，以及码率与音质的滑块。'),
		prereqs: ['vq-vae']
	},
	{
		slug: 'audio-classifier',
		area: 'audio',
		name: f('General audio classification (YAMNet / AST)', '通用音频分类（YAMNet / AST）'),
		family: f('Audio event classifier', '音频事件分类'),
		tier: 'P2',
		status: 'planned',
		inputs: ['audio'],
		own: false,
		runtime: 'ORT Web / precomputed',
		mode: 'both',
		hook: f('Dogs, sirens, music: label everyday sounds.', '狗叫、警笛、音乐：给日常声音打标签。'),
		shows: f('The log-mel input and class scores over time.', '对数梅尔输入与随时间变化的类别分数。'),
		notes: f('Model availability is unverified.', '模型可用性尚未核实。'),
		prereqs: ['kws']
	},
	{
		slug: 'vad',
		area: 'audio',
		name: f('Voice activity detection', '语音活动检测'),
		family: f('Frame classifier', '帧分类器'),
		tier: 'P2',
		status: 'planned',
		inputs: ['audio'],
		own: true,
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Find exactly when someone starts and stops speaking.', '精确找到有人开始和停止说话的时刻。'),
		shows: f('Frame scores, smoothing and thresholds.', '逐帧分数、平滑与阈值。'),
		prereqs: ['kws']
	},
	{
		slug: 'speaker-embeddings',
		area: 'audio',
		name: f('Speaker embeddings', '说话人嵌入'),
		family: f('Metric learning on audio', '音频上的度量学习'),
		tier: 'P2',
		status: 'planned',
		inputs: ['audio'],
		own: true,
		runtime: 'Own JS / precomputed',
		mode: 'both',
		hook: f('One vector per voice: the same speaker lands close, different speakers far apart.', '每个声音一个向量：同一个人靠近，不同的人远离。'),
		shows: f('An embedding map of recorded clips and their cosine similarities.', '录音片段的嵌入地图与余弦相似度。'),
		prereqs: ['kws', 'siamese-metric']
	},
	{
		slug: 'source-separation',
		area: 'audio',
		name: f('Source separation', '音源分离'),
		family: f('Spectrogram masking', '频谱掩码'),
		tier: 'P2',
		status: 'planned',
		inputs: ['audio'],
		own: false,
		runtime: 'Precomputed',
		mode: 'precomputed',
		hook: f('Pull the voice out of a song.', '把人声从歌曲中分离出来。'),
		shows: f('One spectrogram mask per source.', '每个音源一张频谱掩码。'),
		prereqs: ['tiny-unet-seg']
	},
	{
		slug: 'music-generation',
		area: 'audio',
		name: f('Music generation', '音乐生成'),
		family: f('Audio-token language model', '音频词元语言模型'),
		tier: 'P2',
		status: 'planned',
		inputs: ['text'],
		own: false,
		runtime: 'Precomputed',
		mode: 'precomputed',
		hook: f('A precomputed walkthrough of a model that writes music as tokens.', '预计算演示：一个用词元“写”音乐的模型。'),
		shows: f('Audio tokens generated step by step and decoded to sound.', '逐步生成的音频词元，以及解码出的声音。'),
		prereqs: ['rvq-codec', 'tinygpt']
	},

	// ------------------------------------------------------------------ 4. Time series
	{
		slug: 'scalar-lstm',
		area: 'time-series',
		name: f('Scalar LSTM (1 unit)', '单单元 LSTM'),
		family: f('Recurrent network', '循环神经网络'),
		tier: 'P1',
		status: 'planned',
		inputs: ['series'],
		own: true,
		params: '16',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('An LSTM small enough to see and edit all 16 of its weights.', '一个小到可以看见并编辑全部 16 个权重的 LSTM。'),
		shows: f('Every gate and the cell state at each time step.', '每个时间步的所有门与细胞状态。'),
		prereqs: []
	},
	{
		slug: 'tinyforecaster',
		area: 'time-series',
		name: n('TinyForecaster'),
		family: f('LSTM forecaster', 'LSTM 预测器'),
		tier: 'V1',
		status: 'live',
		inputs: ['series'],
		own: true,
		params: 'LSTM with 16 memory numbers + a linear head (1,233 parameters)',
		runtime: 'Own autograd core in a worker',
		mode: 'live',
		hook: f('Draw a curve and train a forecaster on it in your browser.', '画一条曲线，在浏览器里训练一个预测器。'),
		shows: f(
			'Sliding windows, the LSTM gates for one window, one forecast worked out, live training, a 24-step rollout against the truth, a seasonal baseline to beat, and random walks that no model can forecast.',
			'滑动窗口、一个窗口里 LSTM 的各个门、算出来的一个预测、实时训练、与真实值对比的 24 步滚动预测、一个要打败的季节性基准，以及任何模型都预测不了的随机游走。'
		),
		// LSTM gates are taught on the char-rnn page (design §7; the user may switch this back to scalar-lstm)
		prereqs: ['char-rnn'],
		lesson: {
			steps: 10,
			summary: f(
				'An LSTM learns your series in the browser: windows, gates, one forecast worked out, training, forecasting the held-out part, a baseline to beat and a series no model can forecast.',
				'一个 LSTM 在浏览器里学习你的序列：窗口、门、算出一个预测、训练、预测留出的部分、要打败的基准，以及一种任何模型都预测不了的序列。'
			)
		}
	},
	{
		slug: 'ecg-autoencoder',
		area: 'time-series',
		name: f('ECG anomaly autoencoder', '心电图异常检测自编码器'),
		family: f('Autoencoder', '自编码器'),
		tier: 'P1',
		status: 'live',
		inputs: ['series'],
		own: true,
		params: '9,424',
		size: '38 KB of weights',
		runtime: 'Own JS forward pass (checked against PyTorch)',
		mode: 'live',
		hook: f('Draw a heartbeat; the network flags the beats it cannot rebuild.', '画一段心跳；网络会标出它无法重建的心跳。'),
		shows: f(
			'Encoder, bottleneck and decoder, the rebuild overlay, the error at every point, the anomaly score, a threshold on a histogram of test scores, and the ROC curve.',
			'编码器、瓶颈层与解码器，重建叠加图，每个点的误差，异常分数，测试分数直方图上的阈值，以及 ROC 曲线。'
		),
		prereqs: ['mlp-playground'],
		lesson: {
			steps: 10,
			summary: f(
				'Follow a heartbeat through an autoencoder trained only on normal beats, see why odd beats come back changed, turn the error into a score, choose a threshold, read the ROC curve, and draw an odd beat yourself.',
				'跟着一次心跳走过只用正常心跳训练的自编码器，看清为什么异常心跳被重建后会走样，把误差变成分数，选阈值，读懂 ROC 曲线，再亲手画一个异常心跳。'
			)
		}
	},
	{
		slug: 'kalman-filter',
		area: 'time-series',
		name: f('Kalman filter', '卡尔曼滤波'),
		family: f('Linear state-space baseline', '线性状态空间基线'),
		tier: 'P1',
		status: 'live',
		inputs: ['series'],
		own: true,
		params: 'none learned: the settings Q and R, and a 2-number state with its 2 × 2 covariance',
		size: 'no files: the tracks and the sensor noise are made in your browser',
		runtime: 'Own TypeScript, live in the browser (checked against a Python run)',
		mode: 'live',
		hook: f(
			'The classic predict-and-correct loop: follow a cart through noisy sensor readings, one second at a time.',
			'经典的“预测—校正”循环：透过有噪声的传感器读数，一秒一秒地跟踪一辆小车。'
		),
		shows: f(
			'One predict and update cycle with real numbers: the prediction, the reading, the Kalman gain and the new estimate, with the uncertainty as a band and as a covariance ellipse; the whole run against the raw readings and a moving average; tuning Q and R; a sudden turn.',
			'带真实数字的一轮预测与更新：预测、读数、卡尔曼增益和新的估计，不确定程度画成误差带和协方差椭圆；整段运行与原始读数、移动平均的对比；调节 Q 和 R；突然掉头。'
		),
		prereqs: [],
		lesson: {
			steps: 10,
			summary: f(
				'Follow a cart through noisy sensor readings: predict with a motion model, weigh the reading with the Kalman gain, update, and see where the filter beats a moving average and where a sudden turn fools it.',
				'借助有噪声的传感器读数跟踪一辆小车：用运动模型预测，用卡尔曼增益权衡读数，更新估计；看看滤波器在哪里胜过移动平均，又在哪里被突然掉头骗过。'
			)
		}
	},
	{
		slug: 'tiny-tcn',
		area: 'time-series',
		name: f('Tiny TCN, Transformer and PatchTST-style forecasters', '小型 TCN、Transformer 与 PatchTST 式预测器'),
		family: f('Convolutional and attention forecasters', '卷积与注意力预测器'),
		tier: 'P2',
		status: 'planned',
		inputs: ['series'],
		own: true,
		params: '<50k (design)',
		runtime: 'Own JS (plus an own ARIMA baseline)',
		mode: 'live',
		hook: f('Three ways to read the past: dilated convolutions, patches and attention.', '读取过去的三种方式：空洞卷积、切片与注意力。'),
		shows: f('The receptive field of dilated convolutions, patching, attention over past steps and an ARIMA comparison.', '空洞卷积的感受野、切片、对过去时间步的注意力，以及与 ARIMA 的对比。'),
		prereqs: ['tinyforecaster']
	},
	{
		slug: 'ts-foundation',
		area: 'time-series',
		name: f('Forecasting foundation models (Chronos / TimesFM)', '时间序列基础模型（Chronos / TimesFM）'),
		family: f('Pre-trained forecaster', '预训练预测模型'),
		tier: 'P2',
		status: 'planned',
		inputs: ['series'],
		own: false,
		runtime: 'Precomputed',
		mode: 'precomputed',
		hook: f('Zero-shot forecasts from models pre-trained on many series.', '在大量序列上预训练的模型做零样本预测。'),
		shows: f('Values as tokens (Chronos) or patches (TimesFM), quantile fan charts, and zero-shot vs our trained forecaster.', '把数值当作词元（Chronos）或切片（TimesFM）、分位数扇形图，以及零样本与我们训练的预测器对比。'),
		notes: f('Sizes and licenses are unverified.', '大小与许可尚未核实。'),
		prereqs: ['tinyforecaster', 'tinygpt']
	},
	{
		slug: 'fourier-widget',
		area: 'time-series',
		name: f('Fourier series widget', '傅里叶级数小工具'),
		family: f('Prerequisite widget', '前置知识小工具'),
		tier: 'P1',
		tierNote: f('Prerequisite widget for the time-series and audio pages', '时间序列与音频页面的前置小工具'),
		status: 'planned',
		inputs: ['drawing', 'series'],
		own: true,
		runtime: 'JS',
		mode: 'live',
		hook: f('Draw a wave and rebuild it from sine waves.', '画一段波形，用正弦波把它重建出来。'),
		shows: f('A components slider and the partial sums.', '分量数量滑块与部分和。'),
		prereqs: []
	},

	// ------------------------------------------------------------------ 5. Recommendation
	{
		slug: 'matrix-factorization',
		area: 'recommendation',
		name: f('Matrix factorization', '矩阵分解'),
		family: f('Latent-factor recommender', '隐因子推荐'),
		tier: 'V1',
		status: 'live',
		inputs: ['ratings'],
		own: true,
		params: '(24 people + you + 20 movies) × 2 = 90 numbers',
		size: 'No files: synthetic ratings made in your browser',
		runtime: 'Own autograd SGD in a worker (checked against PyTorch)',
		mode: 'live',
		hook: f('Rate ten movies and watch your taste become a point among the movies.', '给十部电影打分，看你的口味变成电影之间的一个点。'),
		shows: f(
			'The rating matrix with its holes, your own ratings, two numbers per person and per movie on a map, a prediction as a dot product, one gradient-descent step worked out with its numbers, the map organising itself as training runs, your top recommendations, and cold start.',
			'带空格的评分矩阵、你自己的评分、地图上每个人和每部电影的两个数、作为点积的预测、带着数字算出的一步梯度下降、训练中自己变得有条理的地图、给你的前三名推荐，以及冷启动。'
		),
		prereqs: [],
		lesson: {
			steps: 9,
			summary: f(
				'A table of ratings full of holes, your own ratings, two numbers for every person and movie, a prediction as a dot product, one training step with real numbers, the map organising itself, your recommendations, and the cold-start problem.',
				'满是空格的评分表、你自己的评分、每个人和每部电影的两个数、作为点积的预测、带真实数字的一步训练、自己变得有条理的地图、给你的推荐，以及冷启动问题。'
			)
		}
	},
	{
		slug: 'two-tower',
		area: 'recommendation',
		name: f('Two-tower retrieval', '双塔召回'),
		family: f('Embedding retrieval', '嵌入召回'),
		tier: 'P2',
		status: 'live',
		inputs: ['ratings'],
		own: true,
		params: '672 learned numbers in two towers, plus a 36 × 8 item table computed from them',
		size: 'About 16 KB: spec, weights (with the item table) and facts',
		runtime: 'Own JS forward pass (checked against PyTorch)',
		mode: 'both',
		hook: f('One tower for you, one for the movies; a dot product picks your shortlist from the whole catalog.', '一座塔给你，一座塔给电影；点积从整个片库里为你挑出候选。'),
		shows: f(
			'Movies as features, the item tower and its precomputed table, your choices through the user tower, a score as a dot product, retrieval of a shortlist, a tiny ranking stage, why the split is fast, a new movie without retraining, in-batch negatives, and what the features cannot tell.',
			'作为特征的电影、物品塔和提前算好的物品表、经过用户塔的你的选择、作为点积的分数、召回候选、一个小小的排序阶段、这种拆分为什么快、无需重新训练的新电影、批内负样本，以及特征说不出的东西。'
		),
		prereqs: ['matrix-factorization'],
		lesson: {
			steps: 12,
			summary: f(
				'Movies as features, the item tower and its table computed in advance, your choices, the user tower, a score as a dot product, retrieval of a shortlist, a ranker, why it is fast, a new movie without retraining, training with in-batch negatives, and what the towers cannot know.',
				'作为特征的电影、提前算好的物品塔和物品表、你的选择、用户塔、作为点积的分数、召回候选、排序器、它为什么快、无需重新训练的新电影、用批内负样本训练，以及双塔无法知道的东西。'
			)
		}
	},
	{
		slug: 'ctr-models',
		area: 'recommendation',
		name: f('Click-through models (FM, Wide & Deep / DLRM)', '点击率模型（FM、Wide & Deep / DLRM）'),
		family: f('Feature-cross models', '特征交叉模型'),
		tier: 'P2',
		status: 'planned',
		inputs: ['tabular'],
		own: true,
		size: '<0.5 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Predict a click from crossed features.', '用交叉特征预测一次点击。'),
		shows: f('Feature crosses as pairwise dot products, and the wide vs deep parts.', '作为两两点积的特征交叉，以及 wide 与 deep 两部分。'),
		prereqs: ['matrix-factorization', 'mlp-playground']
	},
	{
		slug: 'sasrec',
		area: 'recommendation',
		name: f('Sequential recommender (tiny SASRec)', '序列推荐（小型 SASRec）'),
		family: f('Self-attentive recommender', '自注意力推荐'),
		tier: 'P2',
		status: 'planned',
		inputs: ['ratings'],
		own: true,
		size: '<0.5 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Your click history as a sequence, read with causal attention.', '把你的点击历史当作序列，用因果注意力来读。'),
		shows: f('Causal attention over your history and next-item scores.', '对历史的因果注意力，以及下一个物品的分数。'),
		prereqs: ['tinygpt', 'matrix-factorization']
	},
	{
		slug: 'item-embedding-map',
		area: 'recommendation',
		name: f('Item embedding map', '物品嵌入地图'),
		family: f('Embedding visualization', '嵌入可视化'),
		tier: 'P2',
		status: 'planned',
		inputs: ['ratings'],
		own: true,
		runtime: 'regl / WebGL',
		mode: 'precomputed',
		hook: f('Explore a map where neighbors are liked by similar people.', '探索一张地图：相邻的物品被相似的人喜欢。'),
		shows: f('A WebGL scatter of item embeddings and the neighbors of a clicked item.', 'WebGL 物品嵌入散点图，以及所点物品的近邻。'),
		prereqs: ['matrix-factorization']
	},

	// ------------------------------------------------------------------ 6. Tabular / classic ML
	{
		slug: 'mlp-playground',
		area: 'tabular',
		name: f('MLP playground', 'MLP 游乐场'),
		family: f('Multilayer perceptron', '多层感知机'),
		tier: 'V1',
		status: 'live',
		inputs: ['points2d'],
		own: true,
		params: '2 → 4 → 4 → 2 by default (42 parameters)',
		runtime: 'Own autograd core in a worker',
		mode: 'live',
		hook: f('Paint two classes of points and train a small network to separate them.', '画出两类点，训练一个小网络把它们分开。'),
		shows: f('Per-neuron heatmaps over the input plane, weights as edges, a gradient overlay and one point traced with numbers.', '每个神经元在输入平面上的热力图、作为连线的权重、梯度叠加，以及带数字追踪的单个点。'),
		prereqs: [],
		lesson: {
			steps: 11,
			summary: f(
				'One neuron is a soft straight line, a layer is several, and the next layer bends them into curved borders. Then one training step slowed down, training to the end, and two ways it fails.',
				'一个神经元是一条软直线，一层是好几条，下一层再把它们弯成曲线边界。然后放慢看一步训练、训练到底，以及两种失败的方式。'
			)
		}
	},
	{
		slug: 'classic-ml',
		area: 'tabular',
		name: f('Classic-ML canvas', '经典机器学习画布'),
		family: f('Classic machine learning', '经典机器学习'),
		tier: 'P1',
		status: 'live',
		inputs: ['points2d'],
		own: true,
		params: 'none learned by gradient: kNN stores the points, the tree its splits, k-means its centroids',
		runtime: 'Own TypeScript, live in the browser (no neural network)',
		mode: 'live',
		hook: f(
			'Draw your own dataset and watch k nearest neighbours, a decision tree and k-means work on it, step by step.',
			'画出你自己的数据集，一步步看 k 近邻、决策树和 k-means 怎样处理它。'
		),
		shows: f(
			'kNN (distances, the vote, decision maps as k changes), a CART decision tree (Gini impurity, the best split, the tree growing box by box, depth versus overfitting) and k-means (assign, update, repeat; its limits) on one shared “draw your own dataset” canvas, every number computed live.',
			'在同一张“画你自己的数据集”画布上：kNN（距离、投票、随 k 变化的决策图）、CART 决策树（基尼不纯度、最好的切分、树逐个方框地生长、深度与过拟合）和 k-means（分配、更新、重复，以及它的局限），每个数都实时算出。'
		),
		prereqs: [],
		lesson: {
			steps: 12,
			summary: f(
				'Draw a dataset, then watch k nearest neighbours vote, a decision tree pick its questions by Gini impurity and k-means find groups, every number computed live on your points.',
				'画一个数据集，然后看 k 近邻投票、决策树用基尼不纯度挑问题、k-means 自己找出分组，每个数都在你的点上实时算出。'
			)
		}
	},
	{
		slug: 'gaussian-process',
		area: 'tabular',
		name: f('Gaussian process regression', '高斯过程回归'),
		family: f('Bayesian nonparametric', '贝叶斯非参数方法'),
		tier: 'P2',
		status: 'planned',
		inputs: ['points2d'],
		own: true,
		runtime: 'Pure JS',
		mode: 'live',
		hook: f('Click points and watch the uncertainty band shrink around them.', '点击添加数据点，看不确定性区间在它们周围收窄。'),
		shows: f('The posterior mean and band, and the effect of the kernel.', '后验均值与区间，以及核函数的影响。'),
		prereqs: ['classic-ml']
	},
	{
		slug: 'hmm',
		area: 'tabular',
		name: f('Hidden Markov model', '隐马尔可夫模型'),
		family: f('Probabilistic sequence model', '概率序列模型'),
		tier: 'P2',
		status: 'planned',
		inputs: ['tokens'],
		own: true,
		runtime: 'Pure JS',
		mode: 'live',
		hook: f('Hidden states behind a sequence, decoded with the forward and Viterbi algorithms.', '序列背后的隐藏状态，用前向算法与维特比算法解码。'),
		shows: f('Forward and Viterbi lattices filling in.', '逐步填充的前向格与维特比格。'),
		prereqs: []
	},
	{
		slug: 'siamese-metric',
		area: 'tabular',
		name: f('Metric learning (siamese network, triplet loss)', '度量学习（孪生网络、三元组损失）'),
		family: f('Metric learning', '度量学习'),
		tier: 'P2',
		status: 'planned',
		inputs: ['drawing'],
		own: true,
		size: '<100 KB (design)',
		runtime: 'Own core',
		mode: 'live',
		hook: f('Learn a distance where similar drawings end up close together.', '学习一种距离，让相似的画彼此靠近。'),
		shows: f('An embedding space pulling positives together and pushing negatives apart.', '嵌入空间把正样本拉近、把负样本推远。'),
		prereqs: ['tinycnn']
	},
	{
		slug: 'gbt',
		area: 'tabular',
		name: f('Gradient-boosted trees', '梯度提升树'),
		family: f('Tree ensemble', '树集成'),
		tier: 'P2',
		status: 'planned',
		inputs: ['tabular'],
		own: false,
		size: '≈72 KB',
		runtime: 'TF-DF in the browser, or our own trees',
		mode: 'live',
		hook: f('Fill in a form and follow your row down every tree.', '填写一张表单，跟着你的这一行数据走过每一棵树。'),
		shows: f('The path through each tree, the running sum and a SHAP waterfall.', '每棵树中的路径、累加和，以及 SHAP 瀑布图。'),
		prereqs: ['classic-ml']
	},

	// ------------------------------------------------------------------ 7. RL and control
	{
		slug: 'gridworld',
		area: 'rl',
		name: f('Bandits and gridworld (tabular RL)', '多臂老虎机与网格世界（表格型强化学习）'),
		family: f('Value-based reinforcement learning', '基于价值的强化学习'),
		tier: 'V1',
		status: 'live',
		inputs: ['game'],
		own: true,
		params: 'none (a Q-table of 4 numbers per cell)',
		runtime: 'Own TypeScript, live in the browser (no neural network)',
		mode: 'live',
		hook: f('Pull slot-machine arms yourself, then build a maze and watch Q-learning solve it.', '先亲手拉老虎机，再搭一个迷宫，看 Q 学习如何解开它。'),
		shows: f(
			'Bandits: your own pulls, running averages and regret curves for greedy, ε-greedy, UCB and Thompson sampling. Gridworld: an editable maze, the Q-table as triangles, a value heatmap, policy arrows checked against value iteration, and the Q-learning update with numbers filled in.',
			'老虎机：你自己的拉杆、逐步更新的平均值，以及贪心、ε-贪心、UCB 和汤普森采样的遗憾曲线。网格世界：可编辑的迷宫、画成三角形的 Q 表、价值热力图、与价值迭代对照的策略箭头，以及填好数字的 Q 学习更新式。'
		),
		prereqs: [],
		lesson: {
			steps: 17,
			summary: f(
				'Pull slot machines yourself, see why always picking the best-looking one gets stuck, then watch Q-learning solve your own maze one update at a time.',
				'先亲手拉老虎机，看看为什么总选看起来最好的会卡住；再看 Q 学习一次更新一次地解开你自己的迷宫。'
			)
		}
	},
	{
		slug: 'cart-pole',
		area: 'rl',
		name: f('Cart-pole with REINFORCE', 'REINFORCE 倒立摆'),
		family: f('Policy gradient', '策略梯度'),
		tier: 'P1',
		status: 'live',
		inputs: ['game'],
		own: true,
		params: '4 → 16 → 1 policy (97 parameters)',
		runtime: 'Own autograd core in a worker',
		mode: 'live',
		hook: f(
			'Train a small network to balance a pole by trial and reward, and watch each step pull its push up or down.',
			'靠尝试和奖励训练一个小网络去平衡杆子，看每一步把自己的那一推往上拉或往下压。'
		),
		shows: f(
			'Your start state on a moving cart-pole, p(left) worked out from the policy’s weights, a sampled push, one physics step, discounted returns, normalized advantages, each step’s pull on the gradient, one update, the episode length over 100 rounds of training, and the same training with five seeds.',
			'你的起始状态（会动的倒立摆）、由策略权重算出的 p(左)、一次采样的推法、一步物理模拟、折扣回报、归一化优势、每一步对梯度的拉力、一次更新、100 轮训练中回合长度的变化，以及同样的训练换五个随机种子。'
		),
		prereqs: ['gridworld', 'mlp-playground'],
		lesson: {
			steps: 12,
			summary: f(
				'Four numbers describe the cart and pole; a small network turns them into a chance of pushing left. Follow one episode, credit its pushes with returns and advantages, slow one update down, then train and see how much luck matters.',
				'四个数描述小车和杆；一个小网络把它们变成向左推的概率。跟着一个回合走一遍，用回报和优势给每一推记功，放慢看一次更新，然后训练，并看看运气有多重要。'
			)
		}
	},
	{
		slug: 'snake-dqn',
		area: 'rl',
		name: f('Snake DQN', '贪吃蛇 DQN'),
		family: f('Deep Q-network', '深度 Q 网络'),
		tier: 'P2',
		status: 'live',
		inputs: ['game'],
		own: true,
		params: '171,235 (two 3 × 3 convolutions with 16 and 32 channels, then 2,592 → 64 → 3 Q-values)',
		size: '669 KB of weights, plus 669 KB for the target network',
		runtime: 'Own JS forward pass (checked against PyTorch) and DQN steps in a worker',
		mode: 'live',
		hook: f('Q-values drawn right on the snake board, then one DQN update worked out with real numbers.', '把 Q 值直接画在贪吃蛇棋盘上，再用真实数字算出一次 DQN 更新。'),
		shows: f(
			'Your board as three planes of numbers, three Q-values as arrows, ε-greedy with a real draw, the reward, the target r + γ · max Q_target and the loss for one move, a replay buffer and a batch drawn from it, the target network’s staircase, the training curve, the agent’s game next to yours, and what a step penalty does to it.',
			'把你的棋盘变成三层数、画成箭头的三个 Q 值、带真实抽签的 ε-贪心、奖励、一步的目标值 r + γ · max Q_target 和损失、回放缓冲区以及从中抽出的一批、目标网络的阶梯、训练曲线、智能体的对局与你的对局并排，以及每步扣分对它的影响。'
		),
		prereqs: ['gridworld', 'tinycnn'],
		lesson: {
			steps: 11,
			summary: f(
				'A small network reads your snake board and rates each move; one DQN step is worked out with real numbers (ε-greedy, reward, target, loss), then the replay buffer, the target network, training, and a game against the agent.',
				'一个小网络读你的贪吃蛇棋盘，给每一步打分；用真实数字算一步 DQN（ε-贪心、奖励、目标值、损失），再看回放缓冲区、目标网络、训练，并和智能体比一局。'
			)
		}
	},
	{
		slug: 'alphazero-ttt',
		area: 'rl',
		name: f('AlphaZero-lite tic-tac-toe', '简化版 AlphaZero 井字棋'),
		family: f('Policy/value network + tree search', '策略/价值网络 + 树搜索'),
		tier: 'P2',
		status: 'live',
		inputs: ['game'],
		own: true,
		params: '11,242 (9 → 96 → 96, then 9 priors and 1 value)',
		size: '44 KB of weights',
		runtime: 'Own JS forward pass (checked against PyTorch) and tree search',
		mode: 'live',
		hook: f(
			'Guess a tiny network’s favourite move, watch a tree search check it, then play against the search.',
			'先猜一个小网络最看好的一步，看树搜索怎样检查它，再和搜索下一盘。'
		),
		shows: f(
			'Your board as nine numbers and a mask, the priors and the value against perfect play, one PUCT simulation slowed down (select with live numbers, expand, back up with the sign flip), the tree growing over 50 simulations, visits versus priors, a game against the search, and the self-play training log.',
			'把你的棋盘变成九个数和一个掩码、先验和价值（与完美下法对照）、放慢的一次 PUCT 模拟（带实时数字的选择、展开、带变号的回传）、50 次模拟中长大的树、访问次数对先验、和搜索下一盘，以及自我对弈的训练记录。'
		),
		prereqs: ['gridworld'],
		lesson: {
			steps: 11,
			summary: f(
				'A tiny policy/value network looks at your tic-tac-toe board, then a tree search checks its first impression with 50 simulations; you play against it and see how self-play trained it.',
				'一个小小的策略/价值网络先看你的井字棋棋盘，再由树搜索用 50 次模拟检查它的第一印象；你和它下一盘，并看看自我对弈是怎样训练它的。'
			)
		}
	},
	{
		slug: 'ppo-reacher',
		area: 'rl',
		name: f('PPO on a 2D reacher', '二维机械臂上的 PPO'),
		family: f('Policy optimization', '策略优化'),
		tier: 'P2',
		status: 'planned',
		inputs: ['game'],
		own: true,
		size: '<0.5 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Continuous control with clipped policy updates.', '用截断的策略更新做连续控制。'),
		shows: f('The policy distribution over actions and the clipped objective.', '动作上的策略分布与截断目标。'),
		prereqs: ['cart-pole']
	},
	{
		slug: 'world-model',
		area: 'rl',
		name: f('World model (imagined rollouts)', '世界模型（想象推演）'),
		family: f('Model-based RL', '基于模型的强化学习'),
		tier: 'P2',
		status: 'planned',
		inputs: ['game'],
		own: true,
		size: '<0.5 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Learn the rules of a world, then plan inside the dream.', '先学会世界的规则，再在“梦”里做规划。'),
		shows: f('Real vs imagined rollouts and how model error grows.', '真实与想象的推演，以及模型误差如何增长。'),
		prereqs: ['gridworld']
	},
	{
		slug: 'behavior-cloning',
		area: 'rl',
		name: f('Imitation learning and offline RL', '模仿学习与离线强化学习'),
		family: f('Learning from demonstrations', '从示范中学习'),
		tier: 'P2',
		status: 'planned',
		inputs: ['game'],
		own: true,
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Teach by demonstration, and see why small mistakes compound.', '通过示范来教，看看小错误为何会越积越大。'),
		shows: f('Behavior cloning vs DAgger in the gridworld, with compounding error.', '网格世界中行为克隆与 DAgger 的对比，以及误差累积。'),
		prereqs: ['gridworld']
	},
	{
		slug: 'diffusion-policy',
		area: 'rl',
		name: f('Diffusion policy (2D toy)', '扩散策略（二维玩具）'),
		family: f('Generative robot policy', '生成式机器人策略'),
		tier: 'P2',
		status: 'planned',
		inputs: ['points2d'],
		own: true,
		runtime: 'Own JS (reuses the flow-matching engine)',
		mode: 'live',
		hook: f('A robot policy that denoises a whole action trajectory.', '一个对整段动作轨迹去噪的机器人策略。'),
		shows: f('Action trajectories refined over denoising steps.', '在去噪步骤中逐步细化的动作轨迹。'),
		prereqs: ['flow-2d', 'behavior-cloning']
	},
	{
		slug: 'vla-walkthrough',
		area: 'rl',
		name: f('Vision-language-action models', '视觉-语言-动作模型'),
		family: f('Robot foundation model', '机器人基础模型'),
		tier: 'P2',
		status: 'planned',
		inputs: ['image', 'text'],
		own: false,
		runtime: 'Precomputed walkthrough',
		mode: 'precomputed',
		hook: f('How a robot model reads a camera image and an instruction, then outputs actions.', '机器人模型如何读取相机画面和指令，然后输出动作。'),
		shows: f('Image tokens, text tokens and action tokens in one sequence.', '同一序列中的图像词元、文本词元与动作词元。'),
		prereqs: ['vlm', 'behavior-cloning']
	},

	// ------------------------------------------------------------------ 8. Graphs
	{
		slug: 'gcn',
		area: 'graphs',
		name: f('Graph convolutional network (GCN)', '图卷积网络（GCN）'),
		family: f('Message-passing GNN', '消息传递图神经网络'),
		tier: 'V1',
		status: 'live',
		inputs: ['graph'],
		own: true,
		params: '156 (34 → 4 → 2 graph layers, 2 → 2 head)',
		size: '0.6 KB of weights',
		runtime: 'Own JS forward pass (checked against PyTorch) and live retraining',
		mode: 'live',
		hook: f(
			'Edit the friendships of a karate club and watch messages pass between its members.',
			'编辑一个空手道俱乐部成员之间的朋友关系，看消息如何在成员之间传递。'
		),
		shows: f(
			'The friendships as a table and as a node-link drawing in sync, one-hot features, self-loops and Â with live degrees, every message into one member with numbers, two hops, the 2-D embedding by faction, retraining from two members you label, and over-smoothing.',
			'同步的朋友关系表与节点-连线图、独热特征、自环和带实时度数的 Â、带数字的传给一个成员的每条消息、两跳、按派别分开的二维嵌入、用你标注的两个成员重新训练，以及过度平滑。'
		),
		prereqs: ['mlp-playground'],
		lesson: {
			steps: 11,
			summary: f(
				'Follow one member of Zachary’s karate club through a graph convolutional network: friendships as a table, messages from friends with real numbers, two layers, the two factions in two numbers, then label members yourself and retrain.',
				'跟着 Zachary 空手道俱乐部的一个成员走过图卷积网络：把朋友关系写成表，带着真实数字看朋友发来的消息，再到两层、用两个数分出两派，最后自己标注成员并重新训练。'
			)
		}
	},
	{
		slug: 'gnn-variants',
		area: 'graphs',
		name: n('GAT / GraphSAGE / GIN'),
		family: f('Message-passing GNN', '消息传递图神经网络'),
		tier: 'P1',
		status: 'live',
		inputs: ['graph'],
		own: true,
		params: '808 in four networks (GCN 156, GraphSAGE 300, GIN 184, GAT 168)',
		size: '3.2 KB of weights',
		runtime: 'Own JS forward pass (checked against PyTorch)',
		mode: 'live',
		hook: f('The same graph, different ways to listen to neighbours.', '同一张图，用不同的方式“倾听”邻居。'),
		shows: f(
			'One member’s friends weighted four ways (GCN, GraphSAGE, GIN, GAT attention per edge), the four embeddings and accuracies, and why an attention weight is a clue, not an explanation.',
			'一个成员的朋友被四种方式加权（GCN、GraphSAGE、GIN、每条边的 GAT 注意力），四种嵌入和准确率，以及为什么注意力权重是线索而不是解释。'
		),
		prereqs: ['gcn'],
		lesson: {
			steps: 11,
			summary: f(
				'Follow one member of Zachary’s karate club through four kinds of graph layer: a GCN’s fixed weights, GraphSAGE’s average, GIN’s sum and GAT’s attention, with real numbers; then compare their embeddings and accuracy, and test whether attention explains anything.',
				'跟着 Zachary 空手道俱乐部的一个成员走过四种图网络层：GCN 的固定权重、GraphSAGE 的平均、GIN 的求和和 GAT 的注意力，全部带真实数字；再比较它们的嵌入和准确率，并检验注意力能不能解释什么。'
			)
		}
	},
	{
		slug: 'node2vec',
		area: 'graphs',
		name: f('Node embeddings (DeepWalk / node2vec)', '节点嵌入（DeepWalk / node2vec）'),
		family: f('Random-walk embeddings', '随机游走嵌入'),
		tier: 'P2',
		status: 'live',
		inputs: ['graph'],
		own: true,
		params: '2 × 34 × 8 = 544 (two tables of one 8-number vector per member)',
		size: 'No weights shipped; trained live (one 0.3 KB facts file)',
		runtime: 'Own autograd core in a worker',
		mode: 'live',
		hook: f('Random walks turn a graph into sentences for word2vec.', '随机游走把图变成可以喂给 word2vec 的“句子”。'),
		shows: f(
			'One biased walk step worked out from p and q, walks as sentences of member ids and their (center, context) pairs, one skip-gram step with real numbers, the PCA map of the member vectors by faction, your own p and q against the start setting, nearest members, the GCN’s result next to node2vec’s, and the limits.',
			'根据 p 和 q 算出的一步有偏游走、由成员编号组成句子的游走和它们的（中心，上下文）训练对、带真实数字的一步 skip-gram、按派别着色的成员向量 PCA 图、你自己的 p 和 q 与开始设置的对比、最近的成员、GCN 的结果和 node2vec 的对比，以及局限。'
		),
		prereqs: ['word2vec', 'gcn'],
		lesson: {
			steps: 10,
			summary: f(
				'Biased random walks on Zachary’s karate club become sentences of member ids; skip-gram learns one vector per member from them, the two sides of the club fall apart on the map, and p and q decide how far the walks wander.',
				'在 Zachary 空手道俱乐部上做有偏随机游走，得到由成员编号组成的句子；skip-gram 从中为每个成员学一个向量，俱乐部的两方在图上分开，而 p 和 q 决定游走走多远。'
			)
		}
	},
	{
		slug: 'graph-transformer',
		area: 'graphs',
		name: f('Graph transformer', '图 Transformer'),
		family: f('Attention on graphs', '图上的注意力'),
		tier: 'P2',
		status: 'planned',
		inputs: ['graph'],
		own: true,
		size: '<200 KB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Attention over all nodes, with the graph structure as a bias.', '对所有节点做注意力，并把图结构作为偏置。'),
		shows: f('Attention maps with structural encodings.', '带结构编码的注意力图。'),
		prereqs: ['gcn', 'tinygpt']
	},
	{
		slug: 'egnn',
		area: 'graphs',
		name: f('E(3)-equivariant GNN', 'E(3) 等变图神经网络'),
		family: f('Geometric GNN', '几何图神经网络'),
		tier: 'P2',
		status: 'planned',
		inputs: ['graph'],
		own: true,
		size: '<200 KB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Rotate a molecule, and the outputs rotate with it.', '旋转分子，输出也随之旋转。'),
		shows: f('An equivariance check and coordinate updates.', '等变性检验与坐标更新。'),
		prereqs: ['gcn']
	},
	{
		slug: 'transe',
		area: 'graphs',
		name: f('Knowledge-graph embeddings (TransE)', '知识图谱嵌入（TransE）'),
		family: f('Relational embeddings', '关系嵌入'),
		tier: 'P2',
		status: 'planned',
		inputs: ['graph'],
		own: true,
		size: '<200 KB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Facts as vector arithmetic: head + relation ≈ tail.', '把事实当作向量运算：头实体 + 关系 ≈ 尾实体。'),
		shows: f('Head + relation ≈ tail drawn in 2D, and training on a small knowledge graph.', '在二维中画出“头 + 关系 ≈ 尾”，并在小型知识图谱上训练。'),
		prereqs: ['word2vec']
	},

	// ------------------------------------------------------------------ 9. Generative
	{
		slug: 'flow-2d',
		area: 'generative',
		name: f('2D flow matching / diffusion', '二维流匹配 / 扩散'),
		family: f('Flow matching', '流匹配'),
		tier: 'P0',
		tierNote: f("In the brief's MVP; deferred until after v0", '属于设计简报中的 MVP；推迟到 v0 之后'),
		status: 'live',
		inputs: ['points2d'],
		own: true,
		params: '8,706',
		size: 'No weights; a 1.8 MB training tape on phones',
		runtime: 'Own autograd core in a worker',
		mode: 'live',
		hook: f('Paint a 2D shape and train a model that turns noise into it.', '画一个二维形状，训练一个把噪声变成它的模型。'),
		shows: f(
			'Training pairs on straight lines, the velocity field on a time slider, Euler sampling paths, and 4 steps against 50.',
			'直线上的训练配对、带时间滑块的速度场、欧拉采样的路径，以及 4 步与 50 步的对比。'
		),
		prereqs: ['mlp-playground'],
		lesson: {
			steps: 12,
			summary: f(
				'Pairs of noise and data joined by straight lines, a network that learns the arrow along them, then Euler steps that carry fresh noise onto your shape, and what fewer, bigger steps do.',
				'把噪声和数据用直线配成对，让网络学会沿线的箭头，再用欧拉法一步步把新的噪声带到你的形状上，最后看步数更少、步子更大时会怎样。'
			)
		}
	},
	{
		slug: 'gan-2d',
		area: 'generative',
		name: n('2D GAN'),
		family: f('Generative adversarial network', '生成对抗网络'),
		tier: 'P1',
		status: 'live',
		inputs: ['points2d'],
		own: true,
		params: 'G 2 → 16 → 16 → 2 and D 2 → 16 → 16 → 1 (691 parameters)',
		size: 'No weights; 0.3 KB of presets',
		runtime: 'Own autograd core in a worker',
		mode: 'live',
		hook: f('Watch a generator and a discriminator push against each other on a 2D plane.', '看生成器与判别器在二维平面上相互较劲。'),
		shows: f(
			'D’s heat map with its ½ line, G’s sheet (the bent noise grid), gradient arrows, one D step and one G step with their numbers, both loss curves, and mode collapse on eight blobs.',
			'判别器的热力图和 ½ 线、生成器的“布”（弯曲的噪声网格）、梯度箭头、带数字的一步判别器训练和一步生成器训练、两条损失曲线，以及八团点上的模式坍缩。'
		),
		prereqs: ['mlp-playground'],
		lesson: {
			steps: 9,
			summary: f(
				'A generator turns noise into points, a discriminator judges them against your points, and they train in turns. One step of each slowed down, the whole game, how it differs from flow matching, and its best-known failure.',
				'生成器把噪声变成点，判别器拿它们和你的点比较、打分，两者轮流训练。放慢看各自的一步，再看整场博弈、它和流匹配有什么不同，以及它最有名的失败方式。'
			)
		}
	},
	{
		slug: 'vae-2d',
		area: 'generative',
		name: f('VAE with a 2D latent space', '二维隐空间 VAE'),
		family: f('Variational autoencoder', '变分自编码器'),
		tier: 'P1',
		status: 'live',
		inputs: ['drawing'],
		own: true,
		params: '435,988',
		size: '1.7 MB of weights',
		runtime: 'Own JS forward pass (checked against PyTorch)',
		mode: 'live',
		hook: f(
			'Draw a star, a ladder, an umbrella, glasses or a T-shirt, then drag a point through the 2-D map it lives on.',
			'画一颗星、一架梯子、一把伞、一副眼镜或一件 T 恤，然后在它所在的二维地图上拖动一个点。'
		),
		shows: f(
			'The encoder’s μ and log σ², one sample z = μ + σ·ε (the reparameterization trick), the decoder’s redrawing, the loss (BCE + KL) on your drawing, and a map of 2,000 test drawings to drag through, find a border on and walk across.',
			'编码器的 μ 与 log σ²、一次采样 z = μ + σ·ε（重参数化技巧）、解码器的重画、你这幅画的损失（BCE + KL），以及一张由 2,000 幅测试画构成的地图：在上面拖动、找边界、从一类走到另一类。'
		),
		prereqs: ['tinycnn'],
		lesson: {
			steps: 10,
			summary: f(
				'Follow your drawing into a 2-number code with a spread, draw one sample, decode it, compute the loss, then explore the map: drag, find a border, walk between drawings, and see the limits.',
				'跟着你的画变成带散布的 2 个数的编码，抽一次样，解码，算出损失；然后探索地图：拖动、找边界、在画之间走一走，最后看局限。'
			)
		}
	},
	{
		slug: 'score-2d',
		area: 'generative',
		name: f('Score matching and Langevin sampling', '分数匹配与朗之万采样'),
		family: f('Score-based model', '基于分数的模型'),
		tier: 'P1',
		status: 'live',
		inputs: ['points2d'],
		own: true,
		params: '3 → 64 → 64 → 2 (4,546 parameters)',
		size: 'No files: the network trains in your browser',
		runtime: 'Own autograd core in a worker',
		mode: 'live',
		hook: f('Arrows that point toward the data, and samplers that follow them.', '指向数据的箭头，以及跟着箭头走的采样器。'),
		shows: f(
			'The density and score of your points, denoising score matching with real numbers, the learned score field at six noise levels, Langevin chains with step-size and noise sliders, and annealing from large to small noise.',
			'你的点的密度和分数、带真实数值的去噪分数匹配、六个噪声水平上学到的分数场、带步长和噪声滑块的朗之万链，以及从大噪声到小噪声的退火。'
		),
		prereqs: ['flow-2d'],
		lesson: {
			steps: 10,
			summary: f(
				'The density of your points and its score, a network that learns the score from noisy copies, Langevin chains that follow it with a little noise, annealing from large to small noise, and where the learned score fails.',
				'你的点的密度和它的分数，一个从加噪副本学分数的网络，带一点噪声跟着分数走的朗之万链，从大噪声到小噪声的退火，以及学到的分数在哪里失灵。'
			)
		}
	},
	{
		slug: 'vq-vae',
		area: 'generative',
		name: n('VQ-VAE'),
		family: f('Discrete latent autoencoder', '离散隐变量自编码器'),
		tier: 'P2',
		status: 'planned',
		inputs: ['drawing'],
		own: true,
		size: '<0.5 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Compress a drawing into a handful of codebook entries.', '把一幅画压缩成少量码本条目。'),
		shows: f('Nearest-code lookup and codebook usage.', '最近码字查找与码本使用情况。'),
		prereqs: ['vae-2d']
	},
	{
		slug: 'realnvp-2d',
		area: 'generative',
		name: f('RealNVP 2D flow', 'RealNVP 二维流模型'),
		family: f('Normalizing flow', '归一化流'),
		tier: 'P2',
		status: 'planned',
		inputs: ['points2d'],
		own: true,
		size: '<0.5 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Invertible layers that warp the plane step by step.', '可逆层一步步扭曲平面。'),
		shows: f('Coupling layers warping a grid.', '耦合层扭曲网格。'),
		prereqs: ['flow-2d']
	},
	{
		slug: 'pixelcnn-maskgit',
		area: 'generative',
		name: f('PixelCNN and MaskGIT-style generators', 'PixelCNN 与 MaskGIT 式生成器'),
		family: f('Autoregressive and masked generation', '自回归与掩码生成'),
		tier: 'P2',
		status: 'planned',
		inputs: ['drawing'],
		own: true,
		size: '<0.5 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Generate an image one pixel at a time, or many tokens at once.', '一次生成一个像素，或一次生成许多词元。'),
		shows: f('Raster vs parallel masked decoding order on 16×16 images.', '16×16 图像上逐行扫描与并行掩码解码的顺序对比。'),
		prereqs: ['tinygpt', 'vq-vae']
	},
	{
		slug: 'ddpm-unet',
		area: 'generative',
		name: f('Tiny DDPM with a U-Net', '带 U-Net 的小型 DDPM'),
		family: f('Diffusion model', '扩散模型'),
		tier: 'P2',
		status: 'live',
		inputs: ['drawing'],
		own: true,
		params: '60,001',
		size: '240 KB of weights',
		runtime: 'Own JS forward pass (checked against PyTorch); sampling in a worker',
		mode: 'both',
		hook: f('Bury your drawing in noise, then watch a small U-Net turn fresh noise into a new drawing, step by step.', '把你的画埋进噪声，再看一个小 U-Net 把新的噪声一步步变成一幅新画。'),
		shows: f(
			'The forward noising with the real ᾱ schedule, the U-Net’s noise guess on your drawing, one denoising step with real numbers, 100-step chains run live, 12 seeds, 10 steps against 100, and the U-Net’s skip connections.',
			'按真实 ᾱ 噪声调度的前向加噪、U-Net 对你的画的噪声猜测、代入真实数值的一步去噪、在浏览器里实时运行的 100 步链条、12 个种子、10 步与 100 步的对比，以及 U-Net 的跳跃连接。'
		),
		prereqs: ['flow-2d', 'tiny-unet-seg'],
		lesson: {
			steps: 11,
			summary: f(
				'Bury your drawing in noise with the real schedule, see a U-Net guess the noise, work out one denoising step, then watch drawings appear from noise: other seeds, fewer steps, the U-Net inside, and the limits.',
				'用真实的噪声调度把你的画埋进噪声，看 U-Net 猜噪声，算出一步去噪，再看画从噪声里出现：换种子、减步数、看 U-Net 内部，最后看局限。'
			)
		}
	},
	{
		slug: 'tiny-dit',
		area: 'generative',
		name: f('Tiny diffusion transformer (DiT)', '小型扩散 Transformer（DiT）'),
		family: f('Diffusion transformer', '扩散 Transformer'),
		tier: 'P2',
		status: 'planned',
		inputs: ['drawing'],
		own: true,
		size: '≤8 MB (design)',
		runtime: 'ORT Web',
		mode: 'both',
		hook: f('The same denoising task with patch tokens and attention.', '同一个去噪任务，改用图块词元与注意力。'),
		shows: f('Patch tokens, adaLN conditioning, and few-step distillation compared by step count.', '图块词元、adaLN 条件化，以及按步数比较的少步蒸馏。'),
		prereqs: ['ddpm-unet', 'tinyvit']
	},
	{
		slug: 'stable-diffusion',
		area: 'generative',
		name: f('Stable Diffusion (precomputed)', 'Stable Diffusion（预计算）'),
		family: f('Latent diffusion', '潜空间扩散'),
		tier: 'P2',
		status: 'planned',
		inputs: ['text'],
		own: false,
		runtime: 'Precomputed on a GPU',
		mode: 'precomputed',
		hook: f('Prompt pairs through a real text-to-image model, step by step.', '成对的提示词逐步通过真实的文生图模型。'),
		shows: f('Text encoder, U-Net steps, cross-attention maps, classifier-free guidance and the VAE decode.', '文本编码器、U-Net 各步、交叉注意力图、无分类器引导与 VAE 解码。'),
		prereqs: ['ddpm-unet', 'tinyclip']
	},
	{
		slug: 'acgan',
		area: 'generative',
		name: f('ACGAN generator', 'ACGAN 生成器'),
		family: f('Class-conditional GAN', '类别条件 GAN'),
		tier: 'P2',
		status: 'planned',
		inputs: ['none'],
		own: false,
		params: '2,657,897',
		size: '10.6 MB',
		runtime: 'TF.js (Apache-2.0 example)',
		mode: 'live',
		hook: f("Slide through a generator's latent space and switch the class.", '在生成器的隐空间里滑动，并切换类别。'),
		shows: f('Latent sliders, the class input and each generator layer.', '隐变量滑块、类别输入与生成器各层。'),
		prereqs: ['gan-2d']
	},
	{
		slug: 'energy-history',
		area: 'generative',
		name: f('Hopfield networks, RBMs and energy-based models', 'Hopfield 网络、RBM 与能量模型'),
		family: f('Energy-based models (history corner)', '能量模型（历史角）'),
		tier: 'P2',
		status: 'planned',
		inputs: ['drawing', 'points2d'],
		own: true,
		size: '<10 KB',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Memories as valleys in an energy landscape.', '把记忆看成能量地形中的山谷。'),
		shows: f('Energy descent and Gibbs sampling.', '能量下降与吉布斯采样。'),
		prereqs: []
	},

	// ------------------------------------------------------------------ 10. Multimodal
	{
		slug: 'tinyclip',
		area: 'multimodal',
		name: n('TinyCLIP'),
		family: f('Contrastive image–text model', '对比式图文模型'),
		tier: 'V1',
		status: 'live',
		inputs: ['drawing', 'text'],
		own: true,
		params: '15,905',
		size: '62 KB of weights',
		runtime: 'Own JS forward pass (checked against PyTorch)',
		mode: 'live',
		hook: f('Draw a shape, type captions, and see which caption your drawing matches.', '画一个形状、写几句描述，看你的画和哪句描述最匹配。'),
		shows: f(
			'Two towers (drawing and caption), vectors of length 1, cosine times a learned scale, the 6×6 score table before and after training, row and column softmax, the InfoNCE loss and its gradient, zero-shot classes from your own captions, and the word order a bag of words cannot see.',
			'两座塔（画和描述）、长度为 1 的向量、余弦乘以学到的缩放、训练前后的 6×6 分数表、按行和按列的 softmax、InfoNCE 损失及其梯度、用你自己写的描述做零样本分类，以及词袋看不到的词序。'
		),
		prereqs: ['tinycnn', 'tinygpt'],
		lesson: {
			steps: 10,
			summary: f(
				'Follow a drawing and a caption through two towers to one score, then a whole batch of scores, the loss that trains them, your own captions, and the word order the model cannot see.',
				'跟着一幅画和一句描述穿过两座塔，得到一个分数；再看一整批的分数、训练它们的损失、你自己写的描述，以及模型看不到的词序。'
			)
		}
	},
	{
		slug: 'clip',
		area: 'multimodal',
		name: n('CLIP ViT-B/32'),
		family: f('Contrastive image–text model', '对比式图文模型'),
		tier: 'P1',
		status: 'planned',
		inputs: ['image', 'text'],
		own: false,
		params: '≈151M',
		size: '≈154 MB quantized (opt-in); precomputed by default',
		runtime: 'Precomputed; optional Transformers.js',
		mode: 'both',
		hook: f('The real CLIP on curated images and captions.', '真实的 CLIP，作用在精选图片和描述上。'),
		shows: f('The similarity matrix with real images, zero-shot bars and an embedding map.', '真实图片的相似度矩阵、零样本条形图与嵌入地图。'),
		prereqs: ['tinyclip']
	},
	{
		slug: 'vlm',
		area: 'multimodal',
		name: f('Vision-language model (SmolVLM / Florence-2)', '视觉语言模型（SmolVLM / Florence-2）'),
		family: f('Vision-language model', '视觉语言模型'),
		tier: 'P2',
		status: 'planned',
		inputs: ['image', 'text'],
		own: false,
		runtime: 'Transformers.js / precomputed',
		mode: 'both',
		hook: f('Ask a question about a picture.', '就一张图片提问。'),
		shows: f('Image tokens entering the language model, and precomputed cross-attention for each generated token.', '图像词元进入语言模型，以及每个生成词元的预计算交叉注意力。'),
		notes: f('Model sizes are unverified.', '模型大小尚未核实。'),
		prereqs: ['clip', 'tinygpt']
	},
	{
		slug: 'llava-toy',
		area: 'multimodal',
		name: f('LLaVA-style connector (toy)', 'LLaVA 式连接器（玩具）'),
		family: f('Vision encoder + projector + LM', '视觉编码器 + 投影层 + 语言模型'),
		tier: 'P2',
		status: 'planned',
		inputs: ['drawing', 'text'],
		own: true,
		size: '<2 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Glue TinyViT to TinyGPT with a small projector and ask about your drawing.', '用一个小投影层把 TinyViT 接到 TinyGPT 上，然后就你的画提问。'),
		shows: f('Frozen TinyViT, the MLP projector and image tokens inside the TinyGPT sequence.', '冻结的 TinyViT、MLP 投影层，以及 TinyGPT 序列中的图像词元。'),
		prereqs: ['tinyvit', 'tinygpt']
	},
	{
		slug: 'clap-toy',
		area: 'multimodal',
		name: f('CLAP-style audio–text model (toy)', 'CLAP 式音频-文本模型（玩具）'),
		family: f('Contrastive audio–text model', '对比式音频-文本模型'),
		tier: 'P2',
		status: 'planned',
		inputs: ['audio', 'text'],
		own: true,
		size: '<1 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Speak a word and match it to captions.', '说一个词，把它和文字描述匹配起来。'),
		shows: f('The same similarity-matrix view as TinyCLIP, with audio.', '与 TinyCLIP 相同的相似度矩阵视图，换成音频。'),
		prereqs: ['tinyclip', 'kws']
	},

	// ------------------------------------------------------------------ 11. Science
	{
		slug: 'pinn',
		area: 'science',
		name: f('Physics-informed network (damped oscillator)', '物理信息神经网络（阻尼振子）'),
		family: f('Physics-informed neural network (PINN)', '物理信息神经网络（PINN）'),
		tier: 'V1',
		status: 'live',
		inputs: ['series'],
		own: true,
		params: '1 → 32 → 32 → 1 (1,153 parameters)',
		runtime: 'Own autograd core in a worker (second derivatives by autodiff)',
		mode: 'live',
		hook: f('Fit a curve from a few data points plus the laws of physics.', '只用几个数据点加上物理定律来拟合一条曲线。'),
		shows: f(
			'The law of a damped spring, u′ and u″ by autodiff vs finite differences, the residual strip, data, physics and start loss bars, a plain network vs a PINN side by side and after the data ends, the weight λ of the law, and a wrong law that still fits the points.',
			'阻尼弹簧的定律、用自动微分与有限差分求出的 u′ 和 u″、残差条带、数据 / 物理 / 起始损失条形图、普通网络与 PINN 并排对比及数据之后的走向、定律的权重 λ，以及仍能拟合数据点的错误定律。'
		),
		prereqs: ['mlp-playground'],
		lesson: {
			steps: 10,
			summary: f(
				'A plain network fits a few measured points of a damped spring and wanders off after them; a PINN, trained on the same points plus the law of the spring, carries the swing forward. Then the weight λ of the law, and what a wrong law does.',
				'普通网络能拟合阻尼弹簧的几个测量点，但在这些点之后就乱走；PINN 用同样的点再加上弹簧的定律来训练，能把振动延续下去。然后看定律的权重 λ，以及错误的定律会造成什么。'
			)
		}
	},
	{
		slug: 'esm-2',
		area: 'science',
		name: n('ESM-2 protein language model (8M)'),
		family: f('Protein language model', '蛋白质语言模型'),
		tier: 'P1',
		status: 'planned',
		inputs: ['protein'],
		own: false,
		params: '≈7.6M',
		size: '8.06 MB (q8)',
		runtime: 'Transformers.js (self-hosted q8)',
		mode: 'live',
		hook: f('Mask an amino acid and see what a protein language model expects there.', '遮住一个氨基酸，看蛋白质语言模型认为那里应该是什么。'),
		shows: f('The masked-residue distribution and attention heads; structures shown with cached fallbacks.', '被遮蔽残基的概率分布与注意力头；结构展示带缓存后备。'),
		notes: f('The ESM-2 weights license must be checked first.', '需先核实 ESM-2 权重的许可。'),
		prereqs: ['tinybert']
	},
	{
		slug: 'alphafold',
		area: 'science',
		name: f('AlphaFold walkthrough', 'AlphaFold 讲解'),
		family: f('Protein structure prediction', '蛋白质结构预测'),
		tier: 'P2',
		status: 'planned',
		inputs: ['protein'],
		own: false,
		runtime: 'Precomputed (structure database + Mol*)',
		mode: 'precomputed',
		hook: f('From sequence to 3D structure, with the confidence map linked to the model.', '从序列到三维结构，并把置信度图与结构联动。'),
		shows: f('A PAE heatmap linked to 3D, single vs pair representations and the triangle update on a small protein.', '与三维联动的 PAE 热力图、单体与成对表示，以及小蛋白上的三角更新。'),
		prereqs: ['esm-2']
	},
	{
		slug: 'fno',
		area: 'science',
		name: f('Fourier neural operator (1D)', '傅里叶神经算子（一维）'),
		family: f('Neural operator', '神经算子'),
		tier: 'P2',
		status: 'planned',
		inputs: ['series'],
		own: true,
		size: '<2 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('Learn to solve the heat equation by filtering in frequency space.', '通过在频域中滤波，学会求解热传导方程。'),
		shows: f('The Fourier layer as a filter in frequency space, and a spectral truncation slider.', '把傅里叶层看作频域滤波器，以及频谱截断滑块。'),
		prereqs: ['fourier-widget', 'mlp-playground']
	},
	{
		slug: 'weather-gnn',
		area: 'science',
		name: f('Weather GNNs (GraphCast / GenCast)', '天气图神经网络（GraphCast / GenCast）'),
		family: f('Mesh GNN', '网格图神经网络'),
		tier: 'P2',
		status: 'planned',
		inputs: ['graph'],
		own: false,
		runtime: 'Precomputed walkthrough',
		mode: 'precomputed',
		hook: f('Message passing on a mesh that wraps the globe.', '在包裹地球的网格上传递消息。'),
		shows: f('A precomputed walkthrough of message passing on the mesh.', '网格消息传递的预计算讲解。'),
		prereqs: ['gcn']
	},
	{
		slug: 'molecule-gnn',
		area: 'science',
		name: f('Molecular property prediction and generation', '分子性质预测与生成'),
		family: f('Molecular GNN', '分子图神经网络'),
		tier: 'P2',
		status: 'planned',
		inputs: ['graph'],
		own: true,
		size: '<2 MB (design)',
		runtime: 'Own JS (reuses the GCN page)',
		mode: 'live',
		hook: f('Predict and design molecules with the GCN page\'s tools.', '用 GCN 页面的工具预测和设计分子。'),
		shows: f('Property predictions as you edit a molecule, and a simple generation loop.', '编辑分子时的性质预测，以及一个简单的生成循环。'),
		prereqs: ['gcn']
	},
	{
		slug: 'dna-lm',
		area: 'science',
		name: f('DNA language model', 'DNA 语言模型'),
		family: f('Genomic language model', '基因组语言模型'),
		tier: 'P2',
		status: 'planned',
		inputs: ['tokens', 'text'],
		own: true,
		size: '<2 MB (design)',
		runtime: 'Own JS',
		mode: 'live',
		hook: f('k-mer tokens turn a genome into text a model can read.', 'k-mer 词元把基因组变成模型能读的“文本”。'),
		shows: f('k-mer tokenization (reusing the tokenizer views) and masked prediction.', 'k-mer 分词（复用分词视图）与掩码预测。'),
		prereqs: ['tinybert', 'tokenizers']
	},
	{
		slug: 'protein-design',
		area: 'science',
		name: f('Protein design (ProteinMPNN / RFdiffusion)', '蛋白质设计（ProteinMPNN / RFdiffusion）'),
		family: f('Generative protein design', '生成式蛋白质设计'),
		tier: 'P2',
		status: 'planned',
		inputs: ['protein'],
		own: false,
		runtime: 'Precomputed',
		mode: 'precomputed',
		hook: f('Design a sequence for a shape, and a shape from noise.', '为一个形状设计序列，也从噪声中生成形状。'),
		shows: f('Precomputed design traces.', '预计算的设计过程。'),
		prereqs: ['alphafold', 'ddpm-unet']
	},
	{
		slug: 'medical-segmentation',
		area: 'science',
		name: f('Medical image segmentation', '医学图像分割'),
		family: f('Encoder–decoder CNN', '编码器–解码器 CNN'),
		tier: 'P2',
		status: 'planned',
		inputs: ['image'],
		own: true,
		size: '<2 MB (design)',
		runtime: 'Own JS (reuses the U-Net page)',
		mode: 'live',
		hook: f('The U-Net page applied to medical scans.', '把 U-Net 页面用在医学影像上。'),
		shows: f('Segmentation masks on sample scans, with the same down/up path views.', '示例影像上的分割掩码，以及相同的下采样/上采样路径视图。'),
		prereqs: ['tiny-unet-seg']
	}
];

export const MODEL_BY_SLUG: Record<string, ModelEntry> = Object.fromEntries(MODELS.map((m) => [m.slug, m]));

/** Live first, then by tier (P0, V1, P1, P2), then catalog order. */
export function sortModels(list: ModelEntry[]): ModelEntry[] {
	const statusRank = { live: 0, precomputed: 1, planned: 2 } as const;
	const tierRank = { P0: 0, V1: 1, P1: 2, P2: 3 } as const;
	return list
		.map((m, i) => ({ m, i }))
		.sort(
			(a, b) =>
				statusRank[a.m.status] - statusRank[b.m.status] ||
				tierRank[a.m.tier] - tierRank[b.m.tier] ||
				a.i - b.i
		)
		.map((x) => x.m);
}
