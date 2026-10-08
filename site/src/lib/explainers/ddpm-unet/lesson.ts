// ddpm-unet: the guided lesson (11 steps). Every number is a {placeholder}: live values from the trace of the learner's
// noised drawing and from the processes' frames (process.ts), structure from the spec, measured ones from facts.json.
//   1 your drawing → 2 add a little noise → 3 after many steps only noise → 4 the network guesses the noise (predict) →
//   5 one denoising step with real numbers → 6 repeat: a drawing appears → 7 other seeds → 8 fewer steps (compare) →
//   9 the U-Net inside → 10 pick a seed (task) → 11 limits. Claims not computed live: claims.test.ts.
import type { At, ExplainerRuntime, LessonDef } from '../../atlas/types/index.ts';
import { pick } from '../../atlas/lesson/say.ts';
import { fmtNum, paren } from '../../atlas/core/format.ts';
import { FEW, FILM, N, SEED0, T, seedOf } from './ddpm.ts';
import { filmOf, galleryOf, inkShare, traceOf } from './views/pics/meta.ts';

type RT = ExplainerRuntime;
const L = (en: string, zh: string) => ({ en, zh });
const view = (mode: string) => ({ key: 'ddpm-unet/pics', props: () => ({ mode }) });
const shape = (rt: RT, id: string) => rt.spec?.layers.find((l) => l.id === id)?.outShape.map(Number) ?? [NaN, NaN];
const m = (at: At) => at.frame?.metrics ?? {};
/** a factor of a worked product: a negative number in brackets, "0.310 × (−0.670)" (the kit's paren; NaN for lint) */
const par = (v: number | undefined) => (typeof v === 'number' && Number.isFinite(v) ? paren(v, fmtNum) : NaN);
const mix = (rt: RT): Record<string, number | string> => {
	const c = traceOf({ rt });
	return c ? { t: c.t, abar: c.abar, a: c.a, b: c.b, x0: c.x0[c.i], s: par(c.s), e: par(c.eps[c.i]), xt: c.xt[c.i] } : {};
};
const end = (rt: RT, at: At) => {
	const f = filmOf({ at });
	return { K: f?.K ?? NaN, seed: seedOf(rt.input), ink: inkShare(f?.x) };
};

const lesson: LessonDef = {
	title: L('A tiny diffusion model: from noise to a drawing', '小小扩散模型：从噪声到一幅画'),
	summary: L(
		'Bury your drawing in noise with the real schedule, see a U-Net guess the noise, work out one denoising step, then watch drawings appear from noise: other seeds, fewer steps, the U-Net inside, and the limits.',
		'用真实的噪声调度把你的画埋进噪声，看 U-Net 猜噪声，算出一步去噪，再看画从噪声里出现：换种子、减步数、看 U-Net 内部，最后看局限。'
	),
	steps: [
		{
			auto: 'input',
			override: {
				title: L('Your drawing: 784 numbers', '你的画：784 个数'),
				narration: {
					en: 'A diffusion model makes new drawings out of random noise. This one learned from {fact:trainN} Quick, Draw! drawings of {fact:classesN} kinds: stars, umbrellas and houses. Like them, your drawing x₀ is {n} numbers, 0 for paper and 1 for ink. The plan: bury drawings in noise a little at a time, and train a network to undo it.',
					zh: '扩散模型从随机噪声里造出新的画。这一个是从 {fact:classesN} 类、共 {fact:trainN} 幅 Quick, Draw! 画里学出来的：星星、雨伞和房子。和它们一样，你的画 x₀ 也是 {n} 个数：0 是纸，1 是墨。思路是：把画一点点埋进噪声，再训练一个网络把这个过程倒过来。',
					vars: () => ({ n: N })
				},
				look: L('The cells that stand out from the paper are your ink (1 = full ink): these numbers are where everything starts.', '和纸的颜色差得越多的格子，墨越多（1 = 满墨）：一切都从这些数开始。'),
				terms: ['diffusion-model', 'generative-model']
			}
		},
		{
			id: 'mix',
			source: 'fwd',
			at: { layer: 'xt' },
			view: view('mix'),
			provenance: 'live',
			title: L('Add a little noise', '加一点噪声'),
			narration: {
				en: 'Noise ε is {n} random numbers from a bell curve, centred on 0, so the drawing is first stretched to match: 2x₀ − 1 runs from −1 (paper) to 1 (ink). Step t = {t} mixes them: x_t = √ᾱ_t·(2x₀ − 1) + √(1 − ᾱ_t)·ε, where ᾱ_t = {abar} is read from a fixed table, the noise schedule (next step): the smaller ᾱ_t, the less of the drawing is left. At the marked pixel, √ᾱ_t = {a} and √(1 − ᾱ_t) = {b}, so x_t = {a} × {s} + {b} × {e} = {xt}.',
				zh: '噪声 ε 是从钟形曲线里抽出的 {n} 个随机数，以 0 为中心，所以先把画拉伸成同样的范围：2x₀ − 1 从 −1（纸）到 1（墨）。第 t = {t} 步把两者混合：x_t = √ᾱ_t·(2x₀ − 1) + √(1 − ᾱ_t)·ε，其中 ᾱ_t = {abar}，是从一张固定的表（噪声调度，见下一步）里查出来的：ᾱ_t 越小，画剩下的就越少。在标出的像素上，√ᾱ_t = {a}，√(1 − ᾱ_t) = {b}，所以 x_t = {a} × {s} + {b} × {e} = {xt}。',
				vars: (rt) => ({ n: N, ...mix(rt) })
			},
			look: L('The marked pixel in all three pictures; the noise is drawn blue (below 0) and orange (above 0).', '三幅图里标出的像素；噪声用蓝色（小于 0）和橙色（大于 0）表示。'),
			terms: ['forward-process', 'gaussian-noise']
		},
		{
			id: 'forward',
			source: 'noise',
			at: 'last',
			play: { from: 'first', to: 'last', interval: 60, unit: L('step', '步') },
			view: view('noise'),
			provenance: 'live',
			title: L('After many steps, only noise', '很多步之后，只剩噪声'),
			narration: {
				en: 'Each step adds a little noise β_t, rising from {fact:beta1} at t = 1 to {fact:betaT} at t = {T}, and ᾱ_t is the product of all the (1 − β) so far, so the drawing’s weight √ᾱ_t keeps falling. At t = {T} it is {a}, while the noise’s weight is {b}: nothing of your drawing is left to see. Noising one step at a time gives exactly the same mix as the one-jump formula of step 2, so training jumps straight to any step.',
				zh: '每一步加一点噪声 β_t，从 t = 1 时的 {fact:beta1} 升到 t = {T} 时的 {fact:betaT}；ᾱ_t 是到目前为止所有 (1 − β) 的乘积，所以画的权重 √ᾱ_t 一直在降。到 t = {T} 时它只有 {a}，而噪声的权重是 {b}：你的画已经什么都看不出来了。一步一步地加噪，得到的混合和第 2 步那条一步到位的公式完全一样，所以训练时直接跳到任意一步。',
				vars: (rt, at) => ({ T, a: m(at).a, b: m(at).b })
			},
			look: L('Scrub t: the solid curve is the drawing’s weight √ᾱ_t, the dashed one the noise’s.', '拖动 t：实线是画的权重 √ᾱ_t，虚线是噪声的权重。'),
			terms: ['noise-schedule', 'forward-process']
		},
		{
			id: 'job',
			source: 'fwd',
			at: { layer: 'out' },
			view: view('predict'),
			provenance: 'live',
			title: L('The network’s job', '网络的任务'),
			predict: {
				id: 'ddpm-unet-job',
				question: L('The U-Net sees the noisy x_t and the step t. What is it trained to output?', 'U-Net 看到加噪的 x_t 和步数 t。它被训练成输出什么？'),
				options: [L('The clean drawing x₀', '干净的画 x₀'), L('The noise ε that was mixed in', '混进去的噪声 ε'), L('A label: star, umbrella or house', '一个标签：星星、雨伞或房子')],
				answer: 1,
				explain: L(
					'It outputs ε̂, a guess of the noise, one number per pixel. Training mixes noise into drawings at random steps and scores the guess by its mean squared difference from the real ε. Knowing the noise is as good as knowing the drawing: 2x₀ − 1 = (x_t − √(1 − ᾱ_t)·ε) / √ᾱ_t.',
					'它输出 ε̂，也就是对噪声的猜测，每个像素一个数。训练时在随机的步数把噪声混进画里，用猜测和真实 ε 之差的平方的平均来打分。知道了噪声就等于知道了画：2x₀ − 1 = (x_t − √(1 − ᾱ_t)·ε) / √ᾱ_t。'
				)
			},
			narration: {
				en: 'The network is a U-Net (step 9 opens it up) with {fact:params} learned numbers. It is shown only two things: the noisy picture x_t, here at t = {t}, and the number t itself. Its output is {n} numbers, one per pixel, like its input.',
				zh: '这个网络是一个 U-Net（第 9 步会打开它看），有 {fact:params} 个学来的数。它只看到两样东西：加噪的画 x_t（这里 t = {t}），以及 t 这个数本身。它输出 {n} 个数，每个像素一个，和输入一样多。',
				vars: (rt) => ({ n: N, t: mix(rt).t ?? NaN })
			},
			look: L('Compare the right picture with the middle one; then try another step t with the buttons.', '把右边的图和中间的比一比；再用按钮换一个步数 t 试试。'),
			terms: ['u-net', 'mse']
		},
		{
			id: 'step',
			source: 'chain',
			at: { phase: 'step', step: 78 },
			view: view('step'),
			provenance: 'live',
			title: L('One denoising step, with real numbers', '一步去噪，代入真实的数'),
			narration: {
				en: 'Sampling runs the other way: start from pure noise, and at each step the network guesses the noise ε̂ in x_t. The step subtracts only a small part of the guess, {coef} times it (β_t / √(1 − ᾱ_t)), scales up by 1/√α_t = {inv} (α_t = 1 − β_t), and adds a little fresh noise σ_t·z: z is {n} new random numbers, σ_t = √β_t = {sigma}. These pictures are not your drawing but a run that began as pure noise; from t = {t} to {tn}, at the marked pixel: {inv} × ({xt} − {coef} × {epsP}) + {sigma} × {zP} = {x}.',
				zh: '采样反过来走：从纯噪声出发，每一步由网络猜出 x_t 里的噪声 ε̂。这一步只减去猜测的一小部分，即它的 {coef} 倍（β_t / √(1 − ᾱ_t)），再放大到 1/√α_t = {inv} 倍（α_t = 1 − β_t），然后加上一点新的噪声 σ_t·z：z 是 {n} 个新的随机数，σ_t = √β_t = {sigma}。这些图不是你的画，而是一次从纯噪声开始的采样；从 t = {t} 到 {tn}，在标出的像素上：{inv} × ({xt} − {coef} × {epsP}) + {sigma} × {zP} = {x}。',
				vars: (rt, at) => ({ ...m(at), n: N, tn: m(at).tNext, epsP: par(m(at).eps), zP: par(m(at).z) })
			},
			look: L('Left to right: x_t, the network’s guess ε̂, and x_t after the step.', '从左到右：x_t、网络的猜测 ε̂，以及走完这一步的图。'),
			terms: ['denoising-step']
		},
		{
			id: 'chain',
			source: 'chain',
			at: 'last',
			play: { from: 'first', to: 'last', interval: 50, unit: L('step', '步') },
			view: view('chain'),
			provenance: 'live',
			title: L('Repeat: a drawing appears', '重复：一幅画出现了'),
			narration: {
				en: 'Repeat that step {K} times, from t = {T} down to 1: {K} runs of the network for one drawing, computed live in your browser. The starting noise and every z come from seed {seed}: a number that fixes all the random draws of a run. Scrub through the run and watch when shapes appear out of the noise; seed {seed} ends with {ink|pct} of its pixels more than half ink, and real drawings average {fact:inkReal|pct}.',
				zh: '把这一步重复 {K} 次，从 t = {T} 一直到 1：画一幅画要运行网络 {K} 次，都在你的浏览器里实时算出。出发时的噪声和每一个 z 都来自种子 {seed}：种子是一个数，它决定一次采样里所有的随机抽取。拖动进度，看形状在什么时候从噪声里冒出来；种子 {seed} 最后有 {ink|pct} 的像素墨量超过一半，真实的画平均是 {fact:inkReal|pct}。',
				vars: (rt, at) => ({ T, ...end(rt, at) })
			},
			look: { en: 'The big picture is the current step; the small ones show the run so far, every {gap} steps.', zh: '大图是当前这一步；小图是到目前为止每隔 {gap} 步的图。', vars: () => ({ gap: T / 5 }) },
			terms: ['denoising-step']
		},
		{
			id: 'seeds',
			source: 'seeds',
			at: 'last',
			view: view('gallery'),
			provenance: 'precomputed',
			title: L('Different seeds, different drawings', '不同的种子，不同的画'),
			narration: {
				en: 'The seed picks the starting noise and every fresh z along the way; nothing else changes. Same network, same formula, {n} seeds, {n} different drawings, computed ahead of time ({runs} network runs). TinyCNN, the site’s drawing classifier, calls {fact:judgedIn|pct} of {fact:judgedN} samples a star, an umbrella or a house.',
				zh: '种子决定出发时的噪声，以及路上每一次新加的 z；别的都不变。同一个网络、同一条公式，{n} 个种子画出 {n} 幅不同的画，是提前算好的（运行网络 {runs} 次）。本站的画分类器 TinyCNN 把 {fact:judgedN} 个样本中的 {fact:judgedIn|pct} 认作星星、雨伞或房子。',
				vars: (rt) => ({ n: galleryOf({ rt }).length, runs: galleryOf({ rt }).length * T })
			},
			look: { en: 'Press a seed to see its run at every {film}th step.', zh: '点一个种子，看它每隔 {film} 步的图。', vars: () => ({ film: FILM }) },
			terms: ['gaussian-noise']
		},
		{
			id: 'few',
			source: 'chain',
			at: 'last',
			compare: { a: 'chain', b: 'few', label: [L('All steps', '全部步数'), L('Ten big steps', '十大步')] },
			view: view('final'),
			provenance: 'live',
			title: L('Fewer steps: faster, but worse', '步数更少：更快，但更差'),
			narration: {
				en: 'Can we skip steps? The right panel starts from the same noise but takes {k} big steps, each removing ten steps’ worth of noise, instead of {K}: ten times fewer network runs. Each big step has to guess much more at once, and it shows: over {fact:judgedN} samples, TinyCNN recognizes {fact:judgedInFew|pct} of the {k}-step drawings against {fact:judgedIn|pct} with all steps.',
				zh: '能不能少走几步？右边从同样的噪声出发，只走 {k} 大步，每一步去掉十步的噪声，而不是 {K} 步：网络运行的次数只有十分之一。每一大步都得一下子猜很多，结果看得出来：在 {fact:judgedN} 个样本上，TinyCNN 只认出 {k} 步的画中的 {fact:judgedInFew|pct}，而走全部步数时是 {fact:judgedIn|pct}。',
				vars: () => ({ K: T, k: T / FEW })
			},
			look: L('The same seed in both panels: compare the strokes and the leftover specks.', '两边是同一个种子：比较笔画和残留的斑点。'),
			terms: ['denoising-step']
		},
		{
			id: 'unet',
			source: 'fwd',
			at: { layer: 'out' },
			view: view('unet'),
			provenance: 'live',
			title: L('Inside the U-Net', 'U-Net 的内部'),
			narration: {
				en: 'Convolutions read x_t at {s1}×{s1} pixels and make {c1} channels of it (maps, one per pattern); max pooling halves the maps twice, to {s3}×{s3}, while the channels grow to {c3}, so each number there sees a wide area. Upsampling climbs back up, and two skip connections copy maps from the way down to the way up, so fine detail is not lost. The step t enters as sine and cosine waves, turned into a bias added in {nt} of the blocks; the U-Net segmentation page uses the same U.',
				zh: '卷积在 {s1}×{s1} 像素上读 x_t，从中生成 {c1} 个通道（{c1} 张图，每张对应一种图案）；最大池化把图减半两次，到 {s3}×{s3}，通道增加到 {c3} 个，所以那里的每个数都能看到很大一片。上采样再一路爬回来，两条跳跃连接把下行路径的图复制到上行路径上，细节就不会丢。步数 t 变成正弦和余弦波，再变成偏置，加到其中 {nt} 个块里；U-Net 图像分割那一页用的也是同样的 U 形。',
				vars: (rt) => ({ s1: shape(rt, 'in.conv')[1], c1: shape(rt, 'in.conv')[0], s3: shape(rt, 'mid.conv1')[1], c3: shape(rt, 'mid.conv1')[0], nt: rt.spec?.layers.filter((l) => l.op === 'add').length ?? NaN })
			},
			look: L('Follow the U from the noisy picture x_t: down to the small 7×7 maps, then back up to the noise guess ε̂; the dashed skip connections jump straight across.', '从加噪的画 x_t 开始沿着这个 U 走：往下到 7×7 的小图，再往上回到噪声猜测 ε̂；虚线的跳跃连接直接跨过去。'),
			terms: ['u-net', 'convolution', 'channel', 'skip-connection', 'upsampling', 'pooling', 'timestep-embedding']
		},
		{
			id: 'pick',
			source: 'chain',
			at: 'last',
			play: { from: 'first', to: 'last', interval: 50, unit: L('step', '步') },
			view: view('pick'),
			provenance: 'live',
			title: L('Your turn: pick a seed', '轮到你了：选一个种子'),
			task: {
				prompt: L('Pick a seed other than 0 in the grid and watch its run, computed live.', '在网格里选一个不是 0 的种子，看它实时算出的采样过程。'),
				done: (rt) => seedOf(rt.input) !== SEED0,
				waiting: L('Not yet: press one of the other seeds.', '还没有：点一个别的种子。'),
				hint: L('Each tile is a seed’s final drawing. Its run takes a few seconds to compute.', '每一格是一个种子画出的最后的画。它的采样要算几秒钟。')
			},
			narration: {
				en: 'Pick a seed you like. Its whole run is then computed in your browser, {K} network runs, each one the step you worked out above. Seed {seed} ends with {ink|pct} of its pixels more than half ink: it matches the precomputed tile, because the same seed gives the same noise.',
				zh: '选一个你喜欢的种子。它的整个采样过程会在你的浏览器里算出来：运行网络 {K} 次，每一次都是你在上面算过的那一步。种子 {seed} 最后有 {ink|pct} 的像素墨量超过一半：它和提前算好的那一格一样，因为同一个种子给出同样的噪声。',
				vars: (rt, at) => end(rt, at)
			},
			look: L('Each tile is a seed’s last picture, made ahead of time; the big picture is the live run of the seed you picked.', '每一格是一个种子提前算好的最后一幅图；大图是你选的种子实时算出的采样。'),
			terms: ['denoising-step']
		},
		{
			id: 'limits',
			source: 'seeds',
			at: 'last',
			view: view('limits'),
			provenance: 'precomputed',
			kindOf: 'limits',
			title: L('Limits: tiny, slow, three kinds', '局限：小、慢、只有三类'),
			narration: {
				en: 'This is a toy: {fact:params} numbers that know {fact:classesN} kinds. TinyCNN recognizes {fact:judgedIn|pct} of its samples against {fact:judgedReal|pct} of real drawings: many samples are scribbles, and it cannot draw a cat or follow a prompt. It is slow too: every drawing needs {T} network runs{speed}. Large models such as Stable Diffusion denoise a compressed picture, follow text, and use samplers with far fewer steps.',
				zh: '这只是个玩具：{fact:params} 个数，只认识 {fact:classesN} 类。TinyCNN 认出它的样本中的 {fact:judgedIn|pct}，而真实的画是 {fact:judgedReal|pct}：很多样本只是乱涂，它也画不出猫，听不懂提示词。它还很慢：每幅画都要运行网络 {T} 次{speed}。Stable Diffusion 这样的大模型在压缩过的图上去噪、听从文字，还用步数少得多的采样方法。',
				vars: (rt) => {
					const ms = rt.sources.fwd?.trace?.meta?.ms;
					return { T, speed: pick(typeof ms === 'number' && ms > 0, { en: `; one run took ${Math.round(ms ?? 0)} ms in your browser`, zh: `；在你的浏览器里运行一次用了 ${Math.round(ms ?? 0)} 毫秒` }, { en: '', zh: '' }) };
				}
			},
			look: L('Compare the twelve samples with your own drawing (green frame): which of them would you call a star, an umbrella or a house?', '把这十二个样本和你自己的画（绿框）比一比：其中哪些你会叫作星星、雨伞或房子？'),
			terms: ['diffusion-model']
		}
	],
	fixtures: [{ id: 'seed5', label: L('a star, step 50, seed 5', '一颗星星，第 50 步，种子 5'), raw: { strokes: [[{ x: 0.5, y: 0.1 }, { x: 0.62, y: 0.9 }, { x: 0.1, y: 0.4 }, { x: 0.9, y: 0.4 }, { x: 0.38, y: 0.9 }, { x: 0.5, y: 0.1 }]], t: 50, seed: 5 } }]
};

export default lesson;
