// tiny-unet-seg: the guided lesson (11 steps). Every number is a {placeholder}: structure from the spec (dims()), live
// values from the traces of the learner's scene (seg.ts), measured ones from facts.json ({fact:...}).
//   1 a picture is 1,024 pixels to label → 2 the down path shrinks and sees more → 3 the bottleneck → 4 upsampling, one
//   cell worked out → 5 upsampling alone loses the edges (predict) → 6 skip connections bring them back → 7 a softmax per
//   pixel → 8 the mask and one pixel's windows → 9 IoU per class → 10 task: find a picture it gets wrong → 11 limits.
// Claims not computed live: claims.test.ts.
import type { Box, ExplainerRuntime, I18n, LessonDef, TmplVal } from '../../atlas/types/index.ts';
import { num, pick } from '../../atlas/lesson/say.ts';
import { intText } from '../../atlas/ops/common.ts';
import { rawOf } from './adapters/shapes32/adapter.ts';
import { CLASSES, PRESETS, S, TRAIN_R, presetRaw } from './scenes.ts';
import { PATHS, SURE, T, borders, classMap, leastSure, probe, scoreOf, unsure, windowOf } from './seg.ts';

type RT = ExplainerRuntime;
const words = (v: I18n) => v as unknown as TmplVal;
const out = (rt: RT, id: string, i: number) => Number(rt.spec?.layers.find((l) => l.id === id)?.outShape[i] ?? NaN);
/** structural numbers, read from the spec (never typed into the text) */
export function dims(rt: RT) {
	const m = rt.sources.fwd?.model ?? rt.model;
	const sees = (id: string) => {
		const h = out(rt, id, 1) / 2, b = m?.receptiveField(id, { y0: h, x0: h, y1: h, x1: h });
		return b ? b.x1 - b.x0 + 1 : NaN;
	};
	const w = PATHS.map((p) => windowOf(m, S / 2, S / 2, p)).map((b) => (b ? b.x1 - b.x0 + 1 : NaN));
	return { side: out(rt, 'enc1', 1), n: out(rt, 'enc1', 1) ** 2, k: out(rt, 'probs', 0), c1: out(rt, 'enc1', 0), c2: out(rt, 'enc2', 0), c3: out(rt, 'bott', 0),
		h2: out(rt, 'enc2', 1), h3: out(rt, 'bott', 1), block: out(rt, 'enc1', 1) / out(rt, 'bott', 1), cat1: out(rt, 'cat1', 0), d8: out(rt, 'dec1', 0),
		rf1: sees('enc1'), rf2: sees('enc2'), rf3: sees('bott'), w1: w[0], w2: w[1], w3: w[2] };
}
/** "95 circle and 40 square pixels": pixel counts of the shape classes in a class map */
function shapeList(lab: ArrayLike<number> | undefined): TmplVal {
	if (!lab) return NaN;
	const n = [1, 2, 3].map((k) => Array.prototype.filter.call(lab, (v: number) => v === k).length);
	const parts = n.map((c, j) => ({ c, k: j + 1 })).filter((q) => q.c > 0);
	if (!parts.length) return words({ en: 'no shape pixels at all', zh: '0 个形状像素' });
	const en = parts.map((q) => `${intText(q.c)} ${CLASSES[q.k].en}`);
	return words({ en: `${en.length > 1 ? `${en.slice(0, -1).join(', ')} and ${en[en.length - 1]}` : en[0]} pixels`, zh: parts.map((q) => `${intText(q.c)} 个${CLASSES[q.k].zh}像素`).join('、') });
}
const cname = (k: number) => words({ en: CLASSES[k]?.en ?? '?', zh: CLASSES[k]?.zh ?? '?' });
/** the upsampling step's cell: up2's strongest channel, row 5, column 6 (on the start picture, beside the circle) */
function upUnit(rt: RT): number[] {
	const u = T(rt, 'up2');
	if (!u) return [0, 0, 0];
	const [C, H] = u.shape;
	let b = 0, bs = -1;
	for (let c = 0; c < C; c++) {
		let s = 0;
		for (let i = 0; i < H * H; i++) s += Math.abs(u.data[c * H * H + i]);
		if (s > bs) (bs = s), (b = c);
	}
	return [b, 5, 6];
}
function softUnit(rt: RT): number[] {
	const [r, c] = leastSure(rt), m = classMap(rt);
	return [m ? m.k[r * S + c] : 0, r, c];
}

const lesson: LessonDef = {
	title: { en: 'A tiny U-Net: label every pixel', zh: '小型 U-Net：给每个像素打标签' },
	summary: {
		en: 'Follow a 32×32 picture down a U-Net (smaller maps that see more), through the bottleneck and back up (upsampling plus skip connections), to a softmax per pixel and a class map; score it with IoU, then find a picture it gets wrong.',
		zh: '跟着一幅 32×32 的图走过 U-Net：向下（图变小、看得更广）、穿过瓶颈、再向上（上采样加跳跃连接），得到逐像素的 softmax 和类别图；用 IoU 打分，再找一幅它答错的图。'
	},
	steps: [
		{
			id: 'pixels', source: 'fwd', at: { layer: 'input' }, view: { key: 'tiny-unet-seg/mask', props: () => ({ mode: 'truth' }) }, provenance: 'live',
			title: { en: 'A picture is 1,024 pixels to label', zh: '一幅图是 1,024 个要标注的像素' },
			narration: {
				en: 'The model sees {n} grey numbers, one per pixel of this {side}×{side} picture. A classifier would give one answer for the whole picture; segmentation asks for one answer per pixel: background, circle, square or triangle, {n} answers in all. Your picture has {bg} background pixels and {shapes}.',
				zh: '模型看到 {n} 个灰度数，这幅 {side}×{side} 的图每个像素一个。分类器对整幅图只给一个答案；分割却要每个像素一个答案：背景、圆、正方形或三角形，一共 {n} 个答案。你的图有 {bg} 个背景像素，以及 {shapes}。',
				vars: (rt) => {
					const y = scoreOf(rt)?.truth;
					return { ...dims(rt), bg: y ? y.filter((v) => v === 0).length : NaN, shapes: shapeList(y) };
				}
			},
			look: { en: 'The tint is the class you gave each pixel. Slide the opacity to 0 to see the grey numbers alone.', zh: '着色是你给每个像素的类别。把不透明度拉到 0，就只看到灰度数。' },
			terms: ['segmentation', 'classifier']
		},
		{
			id: 'down', source: 'fwd', at: { layer: 'enc2' }, view: { key: 'tiny-unet-seg/unet', props: () => ({ part: 'down' }) }, provenance: 'live',
			title: { en: 'The down path: smaller maps that see more', zh: '下行路径：图更小，看得更广' },
			narration: {
				en: 'Two 3×3 convolutions with ReLU turn the picture into {c1} maps of {side}×{side}; one cell of them depends on a {rf1}×{rf1} window of pixels. Max pooling halves each side, and two more convolutions give {c2} maps of {h2}×{h2}, where one cell already sees {rf2}×{rf2} pixels. Each level trades detail for a wider view.',
				zh: '两个 3×3 卷积加 ReLU 把图变成 {c1} 张 {side}×{side} 的图；其中一个格子取决于 {rf1}×{rf1} 的像素窗口。最大池化把每条边减半，再经过两个卷积，得到 {c2} 张 {h2}×{h2} 的图，这里一个格子已经能看到 {rf2}×{rf2} 个像素。每往下一层，就用细节换来更广的视野。',
				vars: (rt) => dims(rt)
			},
			look: { en: 'Under each level: how many pixels one of its cells sees. Below: all {c2} maps of down 2; tap a level to open its maps.', zh: '每一层下面写着它的一个格子能看到多少像素。下方是下行 2 的全部 {c2} 张图；点一层可以展开它的图。', vars: (rt) => dims(rt) },
			terms: ['convolution', 'pooling', 'receptive-field', 'feature-map']
		},
		{
			id: 'bottleneck', source: 'fwd', at: { layer: 'bott' }, view: { key: 'tiny-unet-seg/unet', props: () => ({ part: 'bottleneck' }) }, provenance: 'live',
			title: { en: 'The bottleneck: what is here, roughly where', zh: '瓶颈：知道是什么，只大致知道在哪' },
			narration: {
				en: 'A second pooling and two more convolutions give the bottleneck: {c3} maps of only {h3}×{h3} cells. One cell stands for a {block}×{block} block of pixels and sees {rf3}×{rf3} of them, nearly the whole picture. So the bottleneck can tell which shape is around, but it is {block} times coarser than the picture: it cannot say exactly where an edge runs.',
				zh: '再一次池化和两个卷积，得到瓶颈：{c3} 张只有 {h3}×{h3} 个格子的图。一个格子代表 {block}×{block} 的一块像素，能看到 {rf3}×{rf3} 个像素，几乎是整幅图。所以瓶颈能分辨附近是什么形状，但它比原图粗 {block} 倍：说不准边缘到底在哪里。',
				vars: (rt) => dims(rt)
			},
			look: { en: 'The {c3} bottleneck maps below: each cell is a {block}×{block} block of the picture.', zh: '下方是瓶颈的 {c3} 张图：每个格子是原图的一块 {block}×{block}。', vars: (rt) => dims(rt) },
			terms: ['encoder-decoder', 'receptive-field']
		},
		{
			id: 'upsample', source: 'fwd', at: { layer: 'up2' }, select: (rt) => ({ layer: 'up2', unit: upUnit(rt) }), view: 'tiny-unet-seg/upmix', provenance: 'live',
			title: { en: 'Upsampling: each new cell mixes 4 old ones', zh: '上采样：每个新格子混合 4 个旧格子' },
			narration: {
				en: 'The up path must grow the maps back to {side}×{side}. Upsampling doubles each side and has no weights to learn: every new cell is a mix of the 4 old cells nearest to it, with fixed weights that add up to 1, the nearest getting {wNear|n4} and the farthest {wFar|n3}. Here the bottleneck’s {h3}×{h3} maps become {h2}×{h2}: on map {ch}, the new cell at row {row}, column {col} (counting from 0) is {y}, the sum in the table.',
				zh: '上行路径要把图放大回 {side}×{side}。上采样把每条边放大一倍，没有要学习的权重：每个新格子都是离它最近的 4 个旧格子的混合，权重固定、加起来是 1，最近的分得 {wNear|n4}，最远的分得 {wFar|n3}。这里瓶颈的 {h3}×{h3} 的图变成 {h2}×{h2}：在第 {ch} 张图上，第 {row} 行第 {col} 列（从 0 数起）的新格子是 {y}，也就是表里的总和。',
				vars: (rt) => {
					const m = rt.sources.fwd?.model, tr = rt.sources.fwd?.trace, u = upUnit(rt);
					const e = m && tr ? m.explain(tr, 'up2', u) : null;
					return { ...dims(rt), wNear: e?.values.wNear ?? NaN, wFar: e?.values.wFar ?? NaN, y: e?.result ?? NaN, ch: u[0], row: u[1], col: u[2] };
				}
			},
			look: { en: 'The 2×2 window on the old map, the new cell, and the 4 products that add up to it. Hover or tap the new map to try another cell.', zh: '旧图上的 2×2 窗口、新格子，以及加起来等于它的 4 个乘积。在新图上悬停或点按，可以换一个格子。' },
			terms: ['upsampling']
		},
		{
			id: 'blurry', source: 'fwd', at: { layer: 'probs' }, view: { key: 'tiny-unet-seg/mask', props: () => ({ mode: 'compare' }) }, provenance: 'live',
			title: { en: 'Upsampling alone: predict the class map', zh: '只靠上采样：先猜猜类别图' },
			predict: {
				id: 'tiny-unet-seg-blurry',
				question: { en: 'We trained the same network without the two skip connections: its up path gets only the bottleneck’s coarse maps, upsampled. What will its class map look like?', zh: '我们把同一个网络去掉两条跳跃连接再训练：它的上行路径只拿到瓶颈的粗糙图，经上采样放大。它的类别图会是什么样？' },
				options: [{ en: 'The same as the U-Net’s', zh: '和 U-Net 的一样' }, { en: 'The right shapes, but with soft, unsure edges', zh: '形状对，但边缘模糊、没把握' }, { en: 'Only background', zh: '全是背景' }],
				answer: 1,
				explain: { en: 'The bottleneck knows roughly what is where, so the shapes are found. But the exact edge was lost in pooling, and upsampling only blends coarse cells: along the edges its shares blend two classes, and a few border pixels flip to the wrong class. Count the unsure pixels under the two maps.', zh: '瓶颈大致知道什么在哪里，所以形状都找到了。但精确的边缘在池化时丢了，而上采样只会把粗糙的格子混合起来：沿着边缘，它的份额在两个类别之间摇摆，还有几个边界像素被判错。数一数两张图下面没把握的像素。' }
			},
			narration: {
				en: 'Left: the same network, trained the same way, but its up path gets only the bottleneck’s coarse maps, upsampled. Right: the U-Net, whose up path also gets copies of the down path’s sharper maps (the skip connections of step 6). Both read your picture now.',
				zh: '左边：同样的网络、同样的训练，但它的上行路径只拿到瓶颈的粗糙图，经上采样放大。右边：U-Net，它的上行路径还拿到下行路径那些更清晰的图的副本（第 6 步讲的跳跃连接）。两者现在都在读你的图。'
			},
			look: { en: 'Answer first; then compare the edges of the two maps. Each pixel mixes the class colours by its shares.', zh: '先回答；再比较两张图的边缘。每个像素按份额混合各类别的颜色。' },
			terms: ['upsampling']
		},
		{
			id: 'skip', source: 'fwd', at: { layer: 'cat1' }, view: { key: 'tiny-unet-seg/unet', props: () => ({ part: 'skip' }) }, provenance: 'live',
			title: { en: 'Skip connections bring the edges back', zh: '跳跃连接把边缘找回来' },
			narration: {
				en: 'The U-Net copies each down level’s maps across to the up path and joins them to the upsampled ones (concatenation): the {c1} sharp maps of down 1 beside {cUp} upsampled ones, {cat1} in all. The next convolution reads both: what is here, from below, and exactly where the edge runs, from across. {edges} On the test scenes {fact:testEdgeAcc|pct} of border pixels are right with the skips, {fact:noskipEdgeAcc|pct} without.',
				zh: 'U-Net 把下行每一层的图直接复制到上行路径，和上采样的图拼接在一起：下行 1 的 {c1} 张清晰的图放在 {cUp} 张上采样的图旁边，一共 {cat1} 张。下一个卷积同时读两者：从下面得知这里是什么，从旁边得知边缘到底在哪。{edges}在测试场景上，有跳跃连接时边界像素答对 {fact:testEdgeAcc|pct}，没有时 {fact:noskipEdgeAcc|pct}。',
				vars: (rt) => {
					const s = scoreOf(rt), n = scoreOf(rt, 'noskip');
					const d = dims(rt);
					if (!s || !n) return { ...d, cUp: d.cat1 - d.c1, edges: NaN };
					const u = unsure(rt), z = unsure(rt, 'noskip'), pc = `${Math.round(SURE * 100)}%`, tail = u < z ? { en: '', zh: '' } : u + z ? { en: ': here they do not help', zh: '：在这里跳跃连接没有帮上忙' } : { en: ': both are sure of every pixel', zh: '：两者对每个像素都有把握' };
					return { ...d, cUp: d.cat1 - d.c1, edges: s.edge ? words({ en: `On your picture (step 5) ${u} pixels are less than ${pc} sure with the skips, ${z} without${tail.en}.`, zh: `在你的图上（第 5 步），有跳跃连接时 ${u} 个像素把握不到 ${pc}，没有时 ${z} 个${tail.zh}。` })
						: words({ en: 'Your picture has no border between two classes, so here the skips have no edge to fix.', zh: '你的图里没有两个类别之间的边界，所以这里跳跃连接没有边缘可修正。' }) };
				}
			},
			look: { en: 'In the tiles, ⇢ marks the maps copied across (sharp edges) and ↑ the maps that came up from below (soft).', zh: '在小图里，⇢ 标的是复制过来的图（边缘清晰），↑ 标的是从下面上来的图（柔和）。' },
			terms: ['skip-connection', 'concatenation']
		},
		{
			id: 'softmax', source: 'fwd', at: { layer: 'probs' }, select: (rt) => ({ layer: 'probs', unit: softUnit(rt) }), view: 'zoom.softmax', provenance: 'live',
			title: { en: 'A softmax for every pixel', zh: '每个像素一个 softmax' },
			narration: {
				en: 'A last 1×1 convolution (at each pixel, a weighted sum of the last {d8} maps’ values there) gives every pixel {k} scores, one per class, and a softmax turns each pixel’s {k} scores into shares that add up to 1: {n} softmaxes side by side. The least sure pixel of your picture, row {row}, column {col} (counting from 0), gives {top} {pTop|pct} and {second} {pSecond|pct}; {where}.',
				zh: '最后一个 1×1 卷积（在每个像素上，对最后 {d8} 张图在该处的值加权求和）给每个像素 {k} 个分数，每个类别一个；softmax 再把每个像素的 {k} 个分数变成加起来为 1 的份额：{n} 个 softmax 并排进行。你的图里最没把握的像素在第 {row} 行第 {col} 列（从 0 数起），它给{top} {pTop|pct}、{second} {pSecond|pct}；{where}。',
				vars: (rt) => {
					const [k, r, c] = softUnit(rt), pr = T(rt, 'probs'), y = scoreOf(rt)?.truth;
					const p = pr ? [0, 1, 2, 3].map((j) => pr.data[j * S * S + r * S + c]) : [NaN, NaN, NaN, NaN];
					const second = p.map((v, j) => ({ v, j })).filter((q) => q.j !== k).sort((a, b) => b.v - a.v)[0];
					const edge = !!y && borders(y)[r * S + c] === 1;
					return { ...dims(rt), row: r, col: c, top: cname(k), pTop: p[k], second: cname(second.j), pSecond: second.v,
						where: pick(edge, { en: 'it sits on a border between two classes', zh: '它正好在两个类别的边界上' }, { en: 'it is not on a border of your classes', zh: '它不在你给的类别的边界上' }) };
				}
			},
			look: { en: 'The 4 scores of one pixel, e to the power of each, and the shares they become.', zh: '一个像素的 4 个分数、各自的 e 次方，以及它们变成的份额。' },
			terms: ['softmax', 'logit']
		},
		{
			id: 'mask', source: 'fwd', at: { layer: 'probs' }, view: { key: 'tiny-unet-seg/mask', props: () => ({ mode: 'pred', rf: true }) }, provenance: 'live',
			title: { en: 'The mask, and what one pixel saw', zh: '掩码，以及一个像素看到了什么' },
			narration: {
				en: 'Taking the class with the largest share at every pixel gives the class map, the mask: here {shapes}. The answer for the marked pixel, row {row}, column {col} (counting from 0), is computed from three windows of the picture: {a1} pixels through skip 1, {a2} through skip 2, and {a3}{whole} through the bottleneck.',
				zh: '在每个像素上取份额最大的类别，就得到类别图，也就是掩码：这里有 {shapes}。标出的像素（第 {row} 行第 {col} 列，从 0 数起）的答案来自图上的三个窗口：经过跳跃连接 1 的 {a1} 个像素，经过跳跃连接 2 的 {a2} 个，以及经过瓶颈的 {a3} 个{whole}。',
				vars: (rt) => {
					const [r, c] = probe(rt), b = PATHS.map((p) => windowOf(rt.sources.fwd?.model ?? null, r, c, p));
					const wh = (x: Box | null) => (x ? `${x.x1 - x.x0 + 1}×${x.y1 - x.y0 + 1}` : NaN), full = wh(b[2]) === `${S}×${S}`;
					return { shapes: shapeList(classMap(rt)?.k), row: r, col: c, a1: wh(b[0]), a2: wh(b[1]), a3: wh(b[2]), whole: pick(full, { en: ', the whole picture,', zh: '，也就是整幅图' }, { en: '', zh: '' }) };
				}
			},
			look: { en: 'Drag the opacity slider. Tap a pixel: its shares and its three windows follow (cut at the picture’s border).', zh: '拖动不透明度滑块。点一个像素：它的份额和三个窗口会跟着变（在图的边上会被截断）。' },
			terms: ['receptive-field', 'segmentation']
		},
		{
			id: 'iou', source: 'fwd', at: { layer: 'probs' }, view: { key: 'tiny-unet-seg/mask', props: () => ({ mode: 'iou' }) }, provenance: 'live',
			title: { en: 'How right is it? IoU per class', zh: '答得有多对？各类别的 IoU' },
			narration: {
				en: 'To score a class map, count for each class the pixels that the answer and your classes both give it (the intersection) and the pixels that at least one of them gives it (the union). Their ratio is the IoU, intersection over union: here {ious}, so the mean IoU over these classes is {miou|pct}.{vs}',
				zh: '给类别图打分时，对每个类别数两种像素：答案和你给的类别都是它的（交集），以及至少一方是它的（并集）。两者之比就是 IoU（交并比）：这里{ious}，所以这几个类别的平均 IoU 是 {miou|pct}。{vs}',
				vars: (rt) => {
					const s = scoreOf(rt), w = scoreOf(rt, 'noskip')?.per.reduce((a, b) => (b.iou < a.iou ? b : a));
					const vs = s && !s.wrong && w && w.iou < 1 ? words({ en: ` The two counts are equal here; step 5’s network without skips would get ${CLASSES[w.k].en} ${intText(w.inter)} of ${intText(w.union)}.`, zh: `这里两个数都相等；第 5 步那个没有跳跃连接的网络会得到${CLASSES[w.k].zh} ${intText(w.union)} 个中 ${intText(w.inter)} 个。` }) : words({ en: '', zh: '' });
					const ious = s ? words({ en: s.per.map((c) => `${CLASSES[c.k].en} ${intText(c.inter)} of ${intText(c.union)}`).join(', '), zh: s.per.map((c) => `${CLASSES[c.k].zh} ${intText(c.union)} 个中 ${intText(c.inter)} 个`).join('、') }) : NaN;
					return { ious, vs, miou: s?.miou ?? NaN };
				}
			},
			look: { en: '{slashes} On the {fact:testN} test scenes the mean IoU was {fact:testMiou|pct}.', zh: '{slashes}在 {fact:testN} 个测试场景上，平均 IoU 是 {fact:testMiou|pct}。', vars: (rt) => {
				const s = scoreOf(rt), e = s ? borders(s.truth) : null;
				if (!s || !e) return { slashes: NaN };
				const w = s.map.k.filter((k, i) => k !== s.truth[i] && e[i] === 1).length;
				const one = s.wrong === 1, all = w === s.wrong, n = intText(s.wrong);
				return { slashes: s.wrong ? words(one ? { en: `A slash marks the one wrong pixel; it ${w ? 'sits' : 'does not sit'} on a border.`, zh: `斜线标出唯一答错的像素；它${w ? '' : '不'}在边界上。` } : all ? { en: `A slash marks each wrong pixel: all ${n} sit on a border.`, zh: `斜线标出每个答错的像素：这 ${n} 个全在边界上。` } : { en: `A slash marks each wrong pixel: ${intText(w)} of the ${n} ${w === 1 ? 'sits' : 'sit'} on a border.`, zh: `斜线标出每个答错的像素：${n} 个中有 ${intText(w)} 个在边界上。` }) : words({ en: 'No pixel is wrong here.', zh: '这里没有答错的像素。' }) };
			} },
			terms: ['iou']
		},
		{
			id: 'task', source: 'fwd', at: { layer: 'probs' }, view: { key: 'tiny-unet-seg/mask', props: () => ({ mode: 'iou' }) }, provenance: 'live',
			title: { en: 'Your turn: find a picture it gets wrong', zh: '轮到你了：找一幅它答错的图' },
			task: {
				prompt: { en: 'Edit the picture (stamp, paint or erase) until one shape class has an IoU below one half.', zh: '修改这幅图（盖章、涂画或擦除），直到某个形状类别的 IoU 低于一半。' },
				done: (rt) => !rawOf(rt.input).preset && (scoreOf(rt)?.worst?.iou ?? 1) < 0.5,
				hint: { en: 'Paint a thin ring with the circle class, or stamp a circle bigger than any in training.', zh: '用“圆”类别涂一个细圈，或者盖一个比训练中都大的圆。' },
				waiting: { en: 'Not yet: every shape class still has an IoU of one half or more.', zh: '还没有：每个形状类别的 IoU 都还不低于一半。' },
				doneText: { en: 'Found one: that class’s IoU is below one half.', zh: '找到了：这个类别的 IoU 低于一半。' }
			},
			narration: {
				en: 'The model does well on scenes like its training set. Now break it: change the picture until a shape class scores an IoU below one half. Try a shape bigger than any in training, a thin painted outline, or shapes on top of each other. {now}',
				zh: '模型在和训练集相似的场景上表现很好。现在来难倒它：修改这幅图，让某个形状类别的 IoU 低于一半。试试比训练中都大的形状、涂一个细细的轮廓，或者把形状叠在一起。{now}',
				vars: (rt) => {
					const s = scoreOf(rt), w = s?.worst, p = w ? num(w.iou, 'pct') : NaN;
					return { now: !s || (w && typeof p === 'number') ? NaN : !w ? words({ en: 'Right now there is no shape to score.', zh: '现在图里和答案里都还没有形状可打分。' }) : words(w.iou === 1 ? { en: `Right now every shape class scores ${(p as I18n).en}.`, zh: `现在每个形状类别的 IoU 都是 ${(p as I18n).zh}。` } : { en: `Right now the worst shape class is ${CLASSES[w.k].en} at ${(p as I18n).en}.`, zh: `现在最差的形状类别是${CLASSES[w.k].zh}，IoU 为 ${(p as I18n).zh}。` }) };
				}
			},
			look: { en: 'The editor is in the side panel (or under “Your input”). The IoU table updates with every edit.', zh: '编辑器在侧栏（或“你的输入”下面）。IoU 表会随每次修改更新。' },
			terms: ['iou', 'out-of-distribution']
		},
		{
			id: 'limits', kindOf: 'limits', source: 'fwd', at: { layer: 'probs' }, view: { key: 'tiny-unet-seg/mask', props: () => ({ mode: 'limits' }) }, provenance: 'live',
			title: { en: 'Limits: synthetic shapes, not photos', zh: '局限：合成的形状，不是照片' },
			narration: {
				en: 'This is a toy. The model saw only {fact:trainN} pictures of three kinds of filled shape, brighter than a plain noisy background, of size {r0} to {r1} (for a circle, its radius in pixels). Outside that it fails: a mean IoU of {fact:bigMiou|pct} on bigger shapes, {fact:hollowMiou|pct} on outlines and {fact:darkMiou|pct} on dark shapes on a light background, against {fact:testMiou|pct} on scenes like its training set. Real segmentation needs real labelled images, many more classes and a far bigger network.',
				zh: '这只是个玩具。模型只见过 {fact:trainN} 幅图，里面是三种实心形状，比单调的噪声背景更亮，大小从 {r0} 到 {r1}（对圆来说就是半径，以像素计）。超出这些它就不行了：更大的形状平均 IoU 只有 {fact:bigMiou|pct}，轮廓只有 {fact:hollowMiou|pct}，亮背景上的暗形状只有 {fact:darkMiou|pct}，而在和训练集相似的场景上是 {fact:testMiou|pct}。真正的分割需要真实的标注图片、多得多的类别和大得多的网络。',
				vars: () => ({ r0: TRAIN_R[0], r1: TRAIN_R[1] })
			},
			look: { en: 'Each bar is the mean IoU on scenes training never used: {fact:testN} like the training set, {fact:oodN} of each other kind.', zh: '每条横条是在训练从没用过的场景上的平均 IoU：和训练集相似的 {fact:testN} 个，其他每种各 {fact:oodN} 个。' },
			terms: ['out-of-distribution']
		}
	],
	fixtures: PRESETS.map((p) => ({ id: p.id, label: p.label, raw: presetRaw(p.id) }))
};

export default lesson;
