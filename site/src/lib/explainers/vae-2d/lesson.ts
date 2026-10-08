// vae-2d: the guided lesson (10 steps). Every number is a {placeholder}: structure from the spec (dims()), live values
// from the trace of the learner's drawing and from the decoder (vae.ts), measured ones from facts.json ({fact:...}).
//   1 pixels → 2 the encoder squeezes to μ → 3 the spread σ → 4 sample z = μ + σ·ε (predict) → 5 the decoder redraws →
//   6 the loss: BCE + KL → 7 the latent map → 8 drag through it (task: find a border) → 9 walk between two drawings →
//   10 limits: blurry, typical redrawings and a 2-number bottleneck. Claims not computed live: claims.test.ts.
import type { ExplainerRuntime, I18n, LessonDef, TmplVal } from '../../atlas/types/index.ts';
import { K, T, className, codeOf, extraOf, greyShare, latentOf, lossOf, near, atBoundary, klFromLog, zhAfter, type XY } from './vae.ts';
import { pick } from '../../atlas/lesson/say.ts';
import { targetOf } from './views/latent/meta.ts';
import { fmtPct } from '../../atlas/core/format.ts';

type RT = ExplainerRuntime;
const words = (v: I18n) => v as unknown as TmplVal;

/** structural numbers, read from the spec (never typed into the text) */
export function dims(rt: RT) {
	const out = (id: string) => Number(rt.spec?.layers.find((l) => l.id === id)?.outShape[0] ?? NaN);
	return { n: out('flat'), h1: out('enc1'), h2: out('enc2'), zdim: out('mu'), k: K, side: Number(rt.spec?.inputs?.[0]?.shape[2] ?? NaN) };
}
/** "6 star, 4 ladder": the kinds among the K nearest test drawings of a point */
export function hood(rt: RT, p: XY): TmplVal {
	const lat = latentOf(rt);
	if (!lat || !p.every(Number.isFinite)) return NaN;
	const n = near(lat, p);
	const parts = n.order.filter((c) => n.counts[c] > 0).map((c) => ({ n: n.counts[c], name: className(rt, c) }));
	return words({ en: parts.map((q) => `${q.n} ${q.name.en}`).join(', '), zh: parts.map((q) => `${q.n} 幅${zhAfter(q.name.zh)}`).join('、') });
}
const probeOf = (rt: RT): XY => extraOf(rt.input).probe ?? codeOf(rt).mu;
const ink = (t: ArrayLike<number> | undefined) => (t ? Array.from(t).filter((v) => v > 0.5).length : NaN);

const lesson: LessonDef = {
	title: { en: 'A VAE: squeeze a drawing into 2 numbers and draw it again', zh: 'VAE：把一幅画挤成 2 个数，再画出来' },
	summary: {
		en: 'Follow your drawing into a 2-number code with a spread, draw one sample, decode it, compute the loss, then explore the map: drag, find a border, walk between drawings, and see the limits.',
		zh: '跟着你的画变成带散布的 2 个数的编码，抽一次样，解码，算出损失；然后探索地图：拖动、找边界、在画之间走一走，最后看局限。'
	},
	steps: [
		{
			auto: 'input',
			override: {
				title: { en: 'Your drawing becomes 784 numbers', zh: '你的画变成 784 个数' },
				narration: {
					en: 'The model never sees your strokes: it sees {n} numbers in a {side}×{side} grid, 0 for empty paper and 1 for full ink. A variational autoencoder (VAE) has one job: squeeze these {n} numbers into just {zdim}, then draw the picture again from those {zdim}. It learned this from {fact:trainN} drawings of {fact:classesN} kinds: stars, ladders, umbrellas, glasses and T-shirts.',
					zh: '模型看不到你的笔画，它只看到 {side}×{side} 网格里的 {n} 个数：0 是空白的纸，1 是满满的墨。变分自编码器（VAE）只做一件事：把这 {n} 个数挤成仅仅 {zdim} 个，再从这 {zdim} 个数把画重新画出来。它是从 {fact:classesN} 类、共 {fact:trainN} 幅画里学会的：星星、梯子、雨伞、眼镜和 T 恤。',
					vars: (rt) => dims(rt)
				},
				look: { en: 'Each cell is one of the numbers: the more it stands out from the background, the more ink (1 is full ink). These numbers are all the model gets.', zh: '每个格子是一个数：和底色差得越多，墨越多（1 是满墨）。模型能拿到的只有这些数。' }
			}
		},
		{
			id: 'squeeze',
			source: 'fwd',
			at: { layer: 'mu' },
			select: () => ({ layer: 'mu', unit: [0] }),
			view: 'zoom.weighted-sum',
			provenance: 'live',
			title: { en: 'The encoder squeezes it into 2 numbers: μ', zh: '编码器把它挤成 2 个数：μ' },
			narration: {
				en: 'First the encoder squeezes. A layer of weighted sums followed by ReLU (keep positive numbers, turn negative ones into 0) turns the {n} pixels into {h1} numbers, a second one into {h2}. One more weighted sum of those {h2} gives μ₁ = {mu0}, and another gives μ₂ = {mu1}. This pair μ is your drawing’s place on a map with only {zdim} directions: the latent space.',
				zh: '首先，编码器开始“挤”。一层加权和再接 ReLU（正数保留，负数变成 0），把 {n} 个像素变成 {h1} 个数，第二层再变成 {h2} 个。对这 {h2} 个数再做一次加权和得到 μ₁ = {mu0}，另一次加权和得到 μ₂ = {mu1}。这一对 μ 就是你的画在一张只有 {zdim} 个方向的地图上的位置：隐空间。',
				vars: (rt) => ({ ...dims(rt), mu0: codeOf(rt).mu[0], mu1: codeOf(rt).mu[1] })
			},
			look: { en: 'The {h2} products that add up to μ₁, plus the bias b: one more number the layer adds at the end.', zh: '加起来得到 μ₁ 的 {h2} 个乘积，再加上偏置 b：这一层最后再加上的一个数。', vars: (rt) => dims(rt) },
			terms: ['dense-layer', 'relu', 'bias', 'encoder-decoder', 'latent-space']
		},
		{
			id: 'spread',
			source: 'fwd',
			at: { layer: 'z' },
			view: { key: 'vae-2d/latent', props: () => ({ mode: 'cloud' }) },
			provenance: 'live',
			title: { en: 'Plus a spread: how sure the encoder is', zh: '再加一个散布：编码器有多确定' },
			narration: {
				en: 'Two more weighted sums give log σ² = ({lv0}, {lv1}), the logarithm of a variance σ², which measures spread. Undoing the log, σ = exp(½ · log σ²) = ({s0}, {s1}). So the encoder does not name one point but a small cloud around μ, drawn as rings at 1σ and 2σ. On the test drawings the median σ is {fact:sigmaMedian|n2}, small next to the map, whose points spread about {fact:muStd} on each side of 0.',
				zh: '另外两次加权和给出 log σ² = ({lv0}, {lv1})，即方差 σ² 的对数，方差衡量散布。把对数还原，σ = exp(½ · log σ²) = ({s0}, {s1})。所以编码器给出的不是一个点，而是 μ 周围的一小团云，图中画成 1σ 和 2σ 的圈。在测试画上，σ 的中位数是 {fact:sigmaMedian|n2}，和地图比起来很小：地图上的点在 0 的两侧大约散开 {fact:muStd}。',
				vars: (rt) => {
					const c = codeOf(rt);
					return { lv0: c.logvar[0], lv1: c.logvar[1], s0: c.sigma[0], s1: c.sigma[1] };
				}
			},
			look: { en: 'A close-up around μ: the solid ring is 1σ, the dashed one 2σ.', zh: 'μ 附近的特写：实线圈是 1σ，虚线圈是 2σ。' },
			terms: ['latent-space']
		},
		{
			id: 'sample',
			source: 'fwd',
			at: { layer: 'z' },
			view: { key: 'vae-2d/latent', props: () => ({ mode: 'cloud', sample: true }) },
			provenance: 'live',
			title: { en: 'One random point: z = μ + σ·ε', zh: '随机取一个点：z = μ + σ·ε' },
			predict: {
				id: 'vae-2d-sample',
				question: { en: 'ε is usually between −2 and 2. Where will z = μ + σ·ε land?', zh: 'ε 通常在 −2 到 2 之间。z = μ + σ·ε 会落在哪里？' },
				options: [
					{ en: 'Exactly on μ', zh: '正好在 μ 上' },
					{ en: 'Close to μ: within about 2σ', zh: '靠近 μ：大约在 2σ 以内' },
					{ en: 'Anywhere on the map', zh: '地图上任何地方' }
				],
				answer: 1,
				explain: {
					en: 'Each number of z moves away from μ by σ times ε, and ε is rarely beyond ±2, so z lands close to μ: about 86 of 100 draws fall inside the dashed ring, a small step next to the whole map. The noise matters in training: the decoder must redraw a whole neighbourhood of μ well, not one exact point.',
					zh: 'z 的每个数离开 μ 的距离是 σ 乘以 ε，而 ε 很少超出 ±2，所以 z 落在 μ 附近：每 100 次抽取大约有 86 次落在虚线圈以内，和整张地图比只是很小的一步。这点噪声在训练时很重要：解码器必须把 μ 周围一整片都画好，而不只是一个精确的点。'
				}
			},
			narration: {
				en: 'To hand one point to the decoder, the model draws two random numbers ε from a standard bell curve N(0, 1), this time ε = ({e0}, {e1}), and computes z = μ + σ·ε, number by number. Why this way? All the chance sits in ε, so z is a plain sum: keep ε, move μ₁ a little, and z₁ moves by exactly as much. That lets training work out how to nudge μ and σ: the reparameterization trick.',
				zh: '为了交给解码器一个点，模型从标准钟形曲线 N(0, 1) 里抽两个随机数 ε，这次是 ε = ({e0}, {e1})，再逐个算出 z = μ + σ·ε。为什么这样算？所有的偶然性都在 ε 里，所以 z 只是一个普通的加和：ε 不变，μ₁ 挪一点，z₁ 就正好挪同样多。训练因此能算出该怎样微调 μ 和 σ：这就是重参数化技巧。',
				vars: (rt) => ({ e0: codeOf(rt).eps[0], e1: codeOf(rt).eps[1] })
			},
			look: { en: 'The magenta dot is z. Press “Draw another ε” and watch where it can go.', zh: '品红色的点就是 z。按“再抽一次 ε”，看它能跑到哪里。' },
			terms: ['reparameterization']
		},
		{
			id: 'decode',
			source: 'fwd',
			at: { layer: 'out' },
			view: { key: 'vae-2d/pair', props: () => ({ mode: 'redraw' }) },
			provenance: 'live',
			title: { en: 'The decoder draws it again', zh: '解码器把它重新画出来' },
			narration: {
				en: 'The decoder runs the other way: from the {zdim} numbers of z, weighted sums with ReLU grow {h2}, then {h1}, then {n} scores, one per pixel, and a sigmoid squashes each score into ink between 0 and 1. From z = ({z0}, {z1}) it redraws a picture with {inkOut} pixels more than half inked; your drawing has {inkIn}. All it knows about your drawing came through those {zdim} numbers.',
				zh: '解码器反过来工作：从 z 的 {zdim} 个数出发，加权和加 ReLU 先变出 {h2} 个数，再变出 {h1} 个，最后变出 {n} 个分数，每个像素一个；sigmoid 再把每个分数压成 0 到 1 之间的墨量。从 z = ({z0}, {z1}) 出发，它重画出的图里墨量超过一半的像素有 {inkOut} 个；你的画有 {inkIn} 个。它对你的画所知道的一切，都是经过这 {zdim} 个数传过来的。',
				vars: (rt) => ({ ...dims(rt), z0: codeOf(rt).z[0], z1: codeOf(rt).z[1], inkOut: ink(T(rt, 'out')?.data), inkIn: ink(T(rt, 'input')?.data) })
			},
			look: { en: 'Green frame: your drawing. Magenta frame: the redrawing.', zh: '绿框：你的画。品红框：重画。' },
			terms: ['encoder-decoder', 'activation-function']
		},
		{
			id: 'loss',
			source: 'fwd',
			at: { layer: 'out' },
			view: { key: 'vae-2d/pair', props: () => ({ mode: 'loss' }) },
			provenance: 'live',
			title: { en: 'What training asked for: the loss', zh: '训练要的是什么：损失' },
			narration: {
				en: 'Training asked for two things at once. Redraw well: binary cross-entropy (BCE) is small when each redrawn pixel’s ink is close to yours; the {n} pixel losses add up to {rec} here. Stay close to the standard bell curve N(0, 1): the KL divergence of your cloud from it grows when μ is far from 0 or σ far from 1; here it is {kl}, {why}. Over the test drawings the two parts average {fact:testRec} and {fact:testKl}.',
				zh: '训练同时要求两件事。一是画得像：每个重画像素的墨量和你的越接近，二元交叉熵（BCE）就越小；这里 {n} 个像素的损失加起来是 {rec}。二是贴近标准钟形曲线 N(0, 1)：μ 离 0 越远、σ 离 1 越远，你这团云和它之间的 KL 散度就越大；这里它是 {kl}，{why}。在测试画上，这两部分平均是 {fact:testRec} 和 {fact:testKl}。',
				vars: (rt) => {
					const L = lossOf(rt), c = codeOf(rt);
					const why = pick(klFromLog(c.mu, c.logvar), { en: 'mostly from the −log σ² term, because σ is so small', zh: '主要来自 −log σ² 这一项，因为 σ 太小' }, { en: 'mostly from the μ² term, because μ is far from 0', zh: '主要来自 μ² 这一项，因为 μ 离 0 远' });
					return { ...dims(rt), rec: L?.rec ?? NaN, kl: L?.kl ?? NaN, why };
				}
			},
			look: { en: 'The loss map: the more a pixel stands out, the more it costs. The table adds the two KL parts to the BCE.', zh: '损失图：像素越显眼，代价越大。表格把 KL 的两部分和 BCE 加在一起。' },
			terms: ['loss', 'bce', 'kl-divergence']
		},
		{
			id: 'map',
			source: 'fwd',
			at: { layer: 'z' },
			view: { key: 'vae-2d/latent', props: () => ({ mode: 'map' }) },
			provenance: 'live',
			title: { en: 'The map of all drawings', zh: '所有画的地图' },
			narration: {
				en: 'Here are the μ of {mapN} test drawings the model never trained on, coloured by kind. It was never told the kinds, yet each kind gathers in its own region: for {fact:knnAgree|pct} of the test drawings, the most common kind among their {k} nearest neighbours on the map is their own. Among the {k} test drawings nearest to your μ: {here}.',
				zh: '这里是 {mapN} 幅模型训练时没见过的测试画的 μ，按类别着色。从没有人告诉过它类别，可每一类都聚在自己的区域里：对 {fact:knnAgree|pct} 的测试画来说，地图上离它最近的 {k} 幅画里，最多的一类正是它自己那一类。离你的 μ 最近的 {k} 幅测试画里：{here}。',
				vars: (rt) => ({ ...dims(rt), mapN: latentOf(rt)?.x.length ?? NaN, here: hood(rt, codeOf(rt).mu) })
			},
			look: { en: 'The green cross is your μ. Tick the box to see what each place decodes to.', zh: '绿色十字是你的 μ。勾选方框，看每个位置解码出什么。' },
			terms: ['latent-space']
		},
		{
			id: 'drag',
			source: 'fwd',
			at: { layer: 'z' },
			view: { key: 'vae-2d/latent', props: () => ({ mode: 'drag' }) },
			provenance: 'live',
			title: { en: 'Drag through the map', zh: '在地图上拖动' },
			narration: {
				en: 'Every point of the map is a drawing: the decoder turns any z into {n} pixels, even where no test drawing landed. Drag the ring (or move it with the arrow keys) and watch the redrawing change as it leaves one kind’s region. The ring is at z = ({p0}, {p1}); among its {k} nearest test drawings: {here}.',
				zh: '地图上的每个点都是一幅画：解码器能把任何 z 变成 {n} 个像素，哪怕那里没有落下任何测试画。拖动圆圈（或用方向键移动），看它离开一类的区域时重画怎样变化。圆圈现在在 z = ({p0}, {p1})；离它最近的 {k} 幅测试画里：{here}。',
				vars: (rt) => {
					const p = probeOf(rt);
					return { ...dims(rt), p0: p[0], p1: p[1], here: hood(rt, p) };
				}
			},
			look: { en: 'The magenta ring and the picture decoded at its z.', zh: '品红色的圆圈，以及在它的 z 处解码出的画。' },
			task: {
				prompt: {
					en: 'Find a border: leave the ring where two kinds meet, so that at least three of its ten nearest test drawings are of one kind and at least three of another.',
					zh: '找一条边界：把圆圈放在两类相遇的地方，使离它最近的十幅测试画里至少三幅是一类、至少三幅是另一类。'
				},
				done: (rt) => {
					const lat = latentOf(rt), p = extraOf(rt.input).probe;
					return !!lat && !!p && atBoundary(lat, p);
				},
				waiting: { en: 'Not on a border yet: most of the ring’s neighbours are one kind.', zh: '还不在边界上：圆圈的邻居大多是同一类。' },
				hint: { en: 'Borders lie between two colours of dots. Put the ring halfway between two clusters and watch the redrawing look like a bit of both.', zh: '边界在两种颜色的点之间。把圆圈放到两团点的中间，看重画变得两类都有点像。' }
			},
			terms: ['latent-space']
		},
		{
			id: 'walk',
			source: 'fwd',
			at: { layer: 'z' },
			view: { key: 'vae-2d/latent', props: () => ({ mode: 'interp' }) },
			provenance: 'live',
			title: { en: 'Walk from one drawing to another', zh: '从一幅画走到另一幅' },
			narration: {
				en: 'Walk in a straight line from your μ to the μ of a typical {target} drawing (pick another kind with the buttons), and decode 7 evenly spaced points on the way. Every point gives a full drawing, with no blank gap between the kinds: the KL part of the loss pulled every cloud towards 0, so the regions touch. Halfway (picture 4), at z = ({m0}, {m1}), the nearest test drawings are {here}.',
				zh: '从你的 μ 沿直线走到一幅典型{target}的 μ（用按钮可以换一类），在路上等距取 7 个点并解码。每个点都给出一幅完整的画，两类之间没有空白的缺口：损失里的 KL 部分把每团云都拉向 0，所以各区域连在一起。走到一半（第 4 幅），在 z = ({m0}, {m1})，最近的测试画是 {here}。',
				vars: (rt) => {
					const lat = latentOf(rt), mu = codeOf(rt).mu;
					if (!lat || !mu.every(Number.isFinite)) return { target: NaN, m0: NaN, m1: NaN, here: NaN };
					const tg = targetOf(rt, lat, mu), mid: XY = [(mu[0] + tg.mu[0]) / 2, (mu[1] + tg.mu[1]) / 2];
					const nm = className(rt, tg.c);
					return { target: words({ en: nm.en, zh: zhAfter(nm.zh) }), m0: mid[0], m1: mid[1], here: hood(rt, mid) };
				}
			},
			look: { en: 'Pictures 1 to 7 along the line: where does the kind switch?', zh: '沿线的第 1 到第 7 幅画：类别在哪里切换？' },
			terms: ['kl-divergence', 'latent-space']
		},
		{
			id: 'limits',
			source: 'fwd',
			at: { layer: 'out' },
			view: { key: 'vae-2d/pair', props: () => ({ mode: 'limits' }) },
			provenance: 'live',
			kindOf: 'limits',
			title: { en: 'Limits: blurry, typical and only two numbers', zh: '局限：模糊、典型，而且只有两个数' },
			narration: {
				en: 'The redrawing is blurry: on the test drawings {fact:greyOut|pct} of a redrawing’s inked pixels are grey (between a fifth and four fifths ink), against {fact:greyIn|pct} in the drawings themselves, whose stroke edges are grey too ({here}). The decoder draws what the drawings near z have in common, not your strokes. More numbers help detail but not blur: a VAE with {fact:z16Dim} latent numbers, trained the same way, gets a test BCE of {fact:z16Rec} instead of {fact:testRec}, and {fact:z16GreyOut|pct} grey pixels.',
				zh: '重画是模糊的：在测试画上，重画的有墨像素中有 {fact:greyOut|pct} 是灰的（墨量在五分之一到五分之四之间），而原画只有 {fact:greyIn|pct}（原画的灰来自笔画边缘；{here}）。解码器画的是 z 附近那些画的共同之处，而不是你的笔画。更多的数能补细节，却补不了模糊：同样方法训练的、有 {fact:z16Dim} 个隐变量的 VAE，测试 BCE 是 {fact:z16Rec} 而不是 {fact:testRec}，灰色像素占 {fact:z16GreyOut|pct}。',
				vars: (rt) => {
					const gIn = greyShare(T(rt, 'input')?.data ?? []), gOut = greyShare(T(rt, 'out')?.data ?? []);
					if (!Number.isFinite(gIn)) return { here: words({ en: 'your pad is blank, so there is nothing to compare', zh: '你的画板是空的，没有可比较的' }) };
					return { here: words({ en: `yours: ${fmtPct(gIn)}, its redrawing: ${fmtPct(gOut)}`, zh: `你的画 ${fmtPct(gIn)}，它的重画 ${fmtPct(gOut)}` }) };
				}
			},
			look: { en: 'In the two maps below the pictures, the grey pixels stand out: compare them. Then try the house preset: a kind it never saw is redrawn as a faint mix of kinds it knows.', zh: '在画下面的两张图里，灰色像素最显眼：比较一下。再试试“房子”预设：它没见过的类别会被重画成它认识的几类淡淡的混合。' },
			terms: ['bce']
		}
	],
	fixtures: [
		{ id: 'border', label: { en: 'a ladder, third ε draw, ring on a border', zh: '一架梯子，第三次抽 ε，圆圈在边界上' }, raw: { strokes: [[{ x: 0.35, y: 0.1 }, { x: 0.35, y: 0.9 }], [{ x: 0.65, y: 0.1 }, { x: 0.65, y: 0.9 }], [{ x: 0.35, y: 0.3 }, { x: 0.65, y: 0.3 }], [{ x: 0.35, y: 0.5 }, { x: 0.65, y: 0.5 }], [{ x: 0.35, y: 0.7 }, { x: 0.65, y: 0.7 }]], draw: 2, probe: [0.02, 0.94], target: 'umbrella' } }
	]
};

export default lesson;
