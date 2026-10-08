// tiny-unet-seg: the package manifest (design §4.9). A tiny U-Net trained offline (pipeline/tiny-unet-seg) to label every
// pixel of a 32×32 synthetic scene as background, circle, square or triangle, traced live on the learner's scene:
//   fwd     the exported U-Net on the scene (down path, bottleneck, up path with two skip connections, per-pixel softmax)
//   noskip  the same network trained the same way without the two concatenations (static/models/tiny-unet-seg/noskip)
// Scenes are made by scenes.ts render() (the training set too: make_data.ts); measured numbers are in facts.json.
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { presetRaw } from './scenes.ts';

const I = (en: string, zh: string) => ({ en, zh });

const manifest: ExplainerManifest = {
	slug: 'tiny-unet-seg',
	type: 'model',
	kinds: ['trace'],
	quality: 'reviewed',
	assets: { dir: 'models/tiny-unet-seg', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'shapes32', start: presetRaw('three') },
	output: { view: 'tiny-unet-seg/mask', layer: 'probs', props: { mode: 'pred' } },
	sources: { fwd: { kind: 'trace' }, noskip: { kind: 'trace', assets: 'models/tiny-unet-seg/noskip' } },
	// the stack layout notes a residual line for any repeated block; this U-Net has none (notes: Kit issues)
	overview: { layout: 'pipeline' },
	page: {
		goals: [
			I('Say what segmentation asks for: one class for every pixel, 1,024 answers for a 32×32 picture.', '说出分割要做什么：给每个像素一个类别，一幅 32×32 的图就要 1,024 个答案。'),
			I('Explain why the down path makes the maps smaller, and how far one cell can see at each level.', '解释为什么下行路径把图变小，以及每一层的一个格子能看到多大一片。'),
			I('Say what a skip connection carries and why, without it, the mask’s edges come out blurry.', '说出跳跃连接带来了什么，以及为什么没有它，掩码的边缘会变模糊。'),
			I('Read a per-pixel softmax and an IoU table, and find a picture the model gets wrong.', '读懂逐像素的 softmax 和 IoU 表，并找到一幅模型答错的图。')
		],
		prequestion: {
			id: 'tiny-unet-seg-pre',
			question: I('A pixel in the middle of a filled circle looks exactly like a pixel in the middle of a filled square: the same grey. How can a network still tell which shape it belongs to?', '实心圆中间的一个像素，和实心正方形中间的一个像素看起来一模一样：同样的灰。网络怎么还能分清它属于哪个形状？'),
			options: [I('By the pixel’s own brightness', '靠这个像素自己的亮度'), I('By looking at a wide area around the pixel', '靠看这个像素周围的一大片'), I('It cannot: it guesses', '分不清：只能猜')],
			answer: 1,
			explain: I('Only the outline tells a circle from a square, and the outline can be several pixels away. So the network must combine a wide view (what shape is around here?) with a sharp one (exactly where is the edge?). The U-Net does both.', '只有轮廓能区分圆和正方形，而轮廓可能在好几个像素之外。所以网络必须把大范围的视野（这附近是什么形状？）和精细的视野（边缘到底在哪？）结合起来。U-Net 两者都做到了。')
		},
		tryThis: [
			{ id: 'giant', title: I('A giant circle', '一个巨大的圆'), setup: { input: { shapes: [{ k: 1, x: 16, y: 16, r: 14 }], seed: 22 } }, predict: I('This circle is almost twice as big as any in training. Will the model still call it a circle?', '这个圆几乎比训练中最大的还大一倍。模型还会说它是圆吗？'), look: I('Look at the IoU table: part of the circle gets another class.', '看 IoU 表：圆的一部分被判成了别的类别。'), goto: { step: 'iou' } },
			{ id: 'ring', title: I('A painted ring', '画一个圈'), setup: { input: { paint: [], seed: 23 }, preset: undefined }, predict: I('Paint a thin ring with the circle class. Will the model call the ring a circle?', '用“圆”这个类别涂一个细圈。模型会把这个圈说成圆吗？'), look: I('The model was trained on filled shapes only.', '模型只用实心形状训练过。'), goto: { step: 'task' } }
		],
		misconception: { claim: I('A segmentation network classifies each pixel by looking at that pixel.', '分割网络看着一个像素，就给这个像素分类。'), truth: I('Each answer is computed from a window of many pixels: up to the whole picture through the bottleneck, plus a sharp nearby window through the skip connections.', '每个答案都是从很多像素的一个窗口算出来的：经过瓶颈时可以是整幅图，再加上经过跳跃连接的一个清晰的近处窗口。'), step: 'skip' },
		metrics: {
			test_miou: { label: I('Mean IoU on the test scenes', '测试场景上的平均 IoU'), fmt: 'pct' },
			test_pixel_acc: { label: I('Pixels right on the test scenes', '测试场景上答对的像素'), fmt: 'pct' },
			test_edge_acc: { label: I('Border pixels right on the test scenes', '测试场景上答对的边界像素'), fmt: 'pct' },
			noskip_test_miou: { label: I('Mean IoU without skip connections', '去掉跳跃连接后的平均 IoU'), fmt: 'pct' }
		},
		sections: { intros: { training: I('Trained in PyTorch on 16,000 synthetic scenes made by the page’s own code, with the cross-entropy of every pixel. The page runs the exported weights; the numbers below were measured on 2,000 test scenes.', '用 PyTorch 训练，数据是页面自己的代码生成的 16,000 个合成场景，损失是每个像素的交叉熵。页面运行的是导出的权重；下面的数是在 2,000 个测试场景上测出来的。') } },
		lessonNote: I('Eleven steps: a picture as 1,024 pixels to label, the down path, the bottleneck, upsampling, why it is blurry alone, skip connections, the per-pixel softmax, the mask, IoU, a task and the limits.', '十一步：把图看成 1,024 个要标注的像素、下行路径、瓶颈、上采样、为什么单靠它会模糊、跳跃连接、逐像素 softmax、掩码、IoU、一个任务，以及局限。')
	},
	glossary: [
		{ id: 'segmentation', term: I('segmentation', '分割'), def: I('Labelling every pixel of a picture with a class, so the answer is a map of classes as big as the picture.', '给图片的每个像素标上一个类别，所以答案是一张和图片一样大的类别图。'), related: ['classifier'] },
		{ id: 'upsampling', term: I('upsampling', '上采样'), def: I('Making a map larger, here by filling each new cell with a weighted mix of the nearest old cells (bilinear).', '把一张图放大；这里每个新格子由最近的几个旧格子按权重混合而成（双线性）。'), related: ['pooling', 'feature-map'] },
		{ id: 'skip-connection', term: I('skip connection (U-Net)', '跳跃连接（U-Net）'), def: I('A path that copies the maps of a down-path level across to the up path and joins them (concatenates) with the upsampled maps, so fine detail is not lost.', '一条把下行路径某一层的图直接复制到上行路径、并和上采样的图拼接在一起的通路，这样细节不会丢失。'), related: ['concatenation', 'residual-connection'] },
		{ id: 'iou', term: I('IoU (intersection over union)', 'IoU（交并比）'), def: I('For one class: the pixels both the answer and the truth give that class, divided by the pixels at least one of them gives it. 1 is perfect.', '对一个类别：答案和真实标注都给了这个类别的像素数，除以至少有一方给了这个类别的像素数。1 表示完全正确。'), related: ['segmentation'] }
	],
	credits: [{ name: 'U-Net: Ronneberger, Fischer and Brox, Convolutional Networks for Biomedical Image Segmentation (MICCAI 2015)', url: 'https://arxiv.org/abs/1505.04597', license: 'Paper (arXiv); ideas only (no code, weights or data used)', usedFor: 'the down path, the bottleneck, the up path with concatenated skip connections and the per-pixel softmax, reimplemented at toy size', attribution: 'Olaf Ronneberger, Philipp Fischer and Thomas Brox' }],
	catalog: {
		hook: I('Stamp or paint circles, squares and triangles, and watch a U-Net label every pixel.', '盖章或涂画圆、正方形和三角形，看 U-Net 给每个像素打标签。'),
		shows: I('Your own scene through the U: one live map per level and how far one cell sees, one upsampled cell worked out, a twin trained without skip connections beside the U-Net, the per-pixel softmax, the mask with one pixel’s three windows, and IoU per class.', '你自己的场景走过整个 U 形：每一层一张实时的图，以及一个格子能看到多大一片；算出一个上采样格子；去掉跳跃连接训练的对照网络和 U-Net 并排；逐像素 softmax；掩码和一个像素的三个窗口；各类别的 IoU。'),
		runtime: I('Own JS forward pass (checked against PyTorch)', '自研 JS 前向计算（已与 PyTorch 核对）'),
		params: I('29,644', '29,644'),
		size: I('116 KB of weights', '权重 116 KB')
	}
};

export default manifest;
