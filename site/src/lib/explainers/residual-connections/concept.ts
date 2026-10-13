// residual-connections: the concept page around the lesson (K2 concept packages; atlas/ui/concept/panels.ts ConceptDef).
// hook (text), toy, real and compare are StepDefs on the lesson's own runtime and sources, shown as free-standing stages.
//   toy      the learner's setup: the gradient bars of both switch settings (source bars)
//   real     the learner's setup trained live (source train)
//   compare  20 layers without and with skips, trained side by side (source race)
import type { At, I18n, StepDef, TmplVal, TrainState } from '../../atlas/types/index.ts';
import { num } from '../../atlas/lesson/say.ts';
import { PLAIN, SKIPS, metricsOf, onOff, rate, times } from './data.ts';

const words = (v: I18n) => v as unknown as TmplVal;

/** How far layer 0's weights have moved from the first frame of a training tape to this frame (Frobenius norm). */
function moved(at: At): number {
	const tape = at.tape;
	if (!tape || !tape.length) return NaN;
	const w0 = (tape.state(0) as Partial<TrainState>).params?.['net.0.weight']?.data;
	const w = (at.state as Partial<TrainState> | undefined)?.params?.['net.0.weight']?.data;
	if (!w0 || !w) return NaN;
	let s = 0;
	for (let i = 0; i < w.length; i++) s += (w[i] - w0[i]) ** 2;
	return Math.sqrt(s);
}

export const hook: I18n = {
	en: 'Stacking more layers should make a network more capable. Yet the 20-layer network on this page learns nothing that a 3-layer one learns easily, until each of its layers adds its result to its input instead of replacing it. That one addition, a skip connection, sits in every ResNet block and in every Transformer block, TinyGPT\'s included.',
	zh: '把更多的层叠起来，网络理应更强。可是本页这个 20 层的网络，连 3 层网络轻松学会的东西都学不会；直到让它的每一层把结果加到自己的输入上，而不是替换掉输入。这一个加法就是跳跃连接，每个 ResNet 块、每个 Transformer 块里都有它，TinyGPT 也不例外。'
};

export const toy: StepDef = {
	id: 'toy',
	title: { en: 'Flip the switch, change the depth', zh: '拨动开关，改变深度' },
	source: 'bars',
	at: { phase: 'backward' },
	view: { key: 'residual-connections/bars', props: () => ({ mode: 'grads', both: true, mark: 'ends' }) },
	provenance: 'live',
	narration: {
		en: 'Your {depth}-layer network, skips {onoff}, at its starting weights: layer 0 gets a gradient of {g0} and the last layer {gLast}, a ratio of {ratio}. The other bar of each row is the same network with the switch flipped, where the ratio is {oRatio}.',
		zh: '你的 {depth} 层网络，跳跃连接{onoff}，处在初始权重：第 0 层的梯度是 {g0}，最后一层是 {gLast}，相差 {ratio} 倍。每一行的另一根条是同一个网络把开关反过来的结果，那时相差 {oRatio} 倍。',
		vars: (_rt, at) => {
			const m = metricsOf(at);
			return { depth: m.depth, onoff: words(onOff(m.skip)), g0: m.g0, gLast: m.gLast, ratio: times(m.ratio), oRatio: times(m.oRatio) };
		}
	},
	look: { en: 'The top rows are the first layers: the ones that starve without skips.', zh: '最上面几行是最前面的几层：没有跳跃连接时，它们几乎得不到梯度。' }
};

export const real: StepDef = {
	id: 'real',
	title: { en: 'Your network, trained', zh: '训练你的网络' },
	source: 'train',
	at: 'last',
	view: { key: 'loss-curve', props: () => ({ log: true }) },
	provenance: 'live',
	terms: ['gradient-descent'],
	narration: {
		en: 'Your {depth}-layer network, skips {onoff}, trained with {n} updates of gradient descent at learning rate {lr}: the loss went from {start} to {loss}, and {acc} of the points are right. Layer 0\'s weights moved {move} in all.',
		zh: '你的 {depth} 层网络，跳跃连接{onoff}，用学习率 {lr} 做了 {n} 次梯度下降更新：损失从 {start} 变成 {loss}，分对了 {acc} 的点。第 0 层的权重一共移动了 {move}。',
		vars: (rt, at) => {
			const m = metricsOf(at);
			const s = (rt.input ?? {}) as { depth?: number; skip?: number };
			return {
				depth: s.depth ?? NaN,
				onoff: words(onOff(s.skip ?? 0)),
				n: m.step,
				lr: rate(m.lr),
				start: at.tape?.frame(1)?.metrics?.loss ?? NaN,
				loss: m.loss,
				acc: num(m.accuracy, 'pct'),
				move: moved(at)
			};
		}
	}
};

export const compare: StepDef = {
	id: 'compare',
	title: { en: 'Twenty layers: without and with skips', zh: '二十层：无跳跃连接与有跳跃连接' },
	source: 'race',
	at: 'last',
	view: { key: 'loss-curve', props: () => ({ metrics: ['loss', 'lossPlain'], log: true, labels: { loss: SKIPS, lossPlain: PLAIN } }) },
	provenance: 'live',
	narration: {
		en: 'The same starting weights, the same {n} updates. Without skips the loss stays at {lossPlain}; with skips it falls to {loss}, and {acc} of the points are right against {accPlain}.',
		zh: '同样的初始权重，同样的 {n} 次更新。没有跳跃连接时，损失停在 {lossPlain}；有跳跃连接时，损失降到 {loss}，分对 {acc} 的点，而没有时是 {accPlain}。',
		vars: (_rt, at) => {
			const m = metricsOf(at);
			return { n: m.step, lossPlain: m.lossPlain, loss: m.loss, acc: num(m.accuracy, 'pct'), accPlain: num(m.accPlain, 'pct') };
		}
	}
};

export const pitfalls = [
	{
		claim: { en: 'A vanishing gradient is a gradient of exactly zero.', zh: '梯度消失就是梯度恰好等于零。' },
		truth: {
			en: 'It is tiny, not zero: layer 0 of the plain 20-layer network still gets a gradient, only many thousands of times smaller than the last layer\'s, too small for its weights to move in a few hundred updates.',
			zh: '它是极小，而不是零：普通 20 层网络的第 0 层仍然有梯度，只是比最后一层小好几万倍，小到在几百次更新里它的权重几乎移动不了。'
		},
		step: 'vanish'
	},
	{
		claim: { en: 'Every skip connection is a residual connection.', zh: '每一种跳跃连接都是残差连接。' },
		truth: {
			en: 'A residual connection adds a layer\'s input to its output, h + f(h). A U-Net\'s skip connections copy maps across the U and concatenate them with the upsampled ones: they carry fine detail to the decoder rather than adding back what came in.',
			zh: '残差连接是把一层的输入加到它的输出上：h + f(h)。U-Net 的跳跃连接则是把图横跨 U 形复制过去，和上采样的图拼接起来：它们把细节送给解码器，而不是把输入加回去。'
		},
		step: 'skip'
	},
	{
		claim: { en: 'With skip connections you can stack as many layers as you like, for free.', zh: '有了跳跃连接，层想叠多少就叠多少，没有代价。' },
		truth: {
			en: 'Every block adds to the signal that passes through, so it grows with depth. Residual networks usually pair the skips with normalization, such as the LayerNorm in front of every attention and MLP block of TinyGPT.',
			zh: '每个块都往穿过的信号上加东西，所以信号随深度变大。残差网络通常会把跳跃连接和归一化搭配使用，比如 TinyGPT 里每个注意力块和 MLP 块前面的 LayerNorm。'
		},
		step: 'limits'
	}
];

export const further = [
	{ label: { en: 'TinyGPT on this site: read its residual stream with the logit lens', zh: '本站的 TinyGPT：用 logit lens 读取它的残差流' }, href: '/m/tinygpt/#lens' },
	{ label: { en: 'The tiny U-Net on this site: skip connections that bring the edges back', zh: '本站的小型 U-Net：把边缘找回来的跳跃连接' }, href: '/m/tiny-unet-seg/#lesson-skip' },
	{ label: { en: 'Backpropagation on this site: the chain rule one edge at a time', zh: '本站的反向传播页面：一次一条边地走链式法则' }, href: '/train/backprop/' },
	{ label: { en: 'He, Zhang, Ren and Sun (2015): Deep Residual Learning for Image Recognition (the ResNet paper)', zh: 'He、Zhang、Ren、Sun（2015）：《用于图像识别的深度残差学习》（ResNet 论文）' }, href: 'https://arxiv.org/abs/1512.03385' },
	{ label: { en: 'Dive into Deep Learning: residual networks (free online book)', zh: '《动手学深度学习》：残差网络（免费在线书）' }, href: 'https://d2l.ai/chapter_convolutional-modern/resnet.html' }
];
