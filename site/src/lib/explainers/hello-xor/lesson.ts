// hello-xor: the guided lesson (design §4.6). Nine steps over all three kinds:
//   1-3 auto steps of the trained network (A): your point, the hidden weighted sums, tanh
//   4   the answer, with a predict-first question (A)
//   5-6 one gradient-descent step slowed down (C): a forward pass and the update, with live numbers
//   7   training live in the browser (B)
//   8   the live-trained weights on your point (A on B), with a task
//   9   the honest limit: a start that gets stuck (B, kindOf 'limits'; backed by claims.test.ts)
// Every number in the narration is a {placeholder} bound to a value that was just computed.
//   npm run atlas -- lesson hello-xor lint      every step x every fixture x en/zh
//   npm run atlas -- lesson hello-xor print     the narration tables
import type { At, ExplainerRuntime, LessonDef, Tensor, Trace } from '../../atlas/types/index.ts';
import { toPoint } from './adapters/xor-point/adapter.ts';

const point = (rt: ExplainerRuntime) => toPoint(rt.input);
const traceOf = (rt: ExplainerRuntime, source: string): Trace | null => rt.sources[source]?.trace ?? null;
const out = (t: Trace | null, id: string): number => {
	const x: Tensor | undefined = t?.t[id];
	return x && x.data.length ? x.data[0] : NaN;
};
const metric = (at: At, k: string): number => at.frame?.metrics?.[k] ?? NaN;

const lesson: LessonDef = {
	title: { en: 'Hello XOR: a network you can follow number by number', zh: '你好 XOR：一个能逐个数字看懂的网络' },
	summary: {
		en: 'Follow your point through a trained 2-2-1 network, slow down one training step, then train it live and see where it can get stuck.',
		zh: '让你的点走过训练好的 2-2-1 网络，放慢看一步训练，再实时训练它，看看它会在哪里卡住。'
	},
	steps: [
		{ auto: 'input', override: { title: { en: 'Your point is two numbers', zh: '你的点就是两个数' } } },
		{ auto: 'op:fc1', override: { title: { en: 'Two weighted sums', zh: '两个加权和' } } },
		{ auto: 'op:act1', override: { title: { en: 'Squash each sum with tanh', zh: '用 tanh 压缩每个和' } } },
		{
			id: 'answer',
			source: 'fwd',
			at: { layer: 'fc2' },
			select: () => ({ layer: 'fc2', unit: [0] }),
			view: 'layer.tensor',
			provenance: 'live',
			title: { en: 'The answer for your point', zh: '你的点得到的回答' },
			predict: {
				id: 'hello-xor-answer',
				// The narration is visible while the stage waits for this answer, so the question is about a point the
				// narration does not answer: (1, 1), while the lesson starts at (1, 0). The narration tells the learner how
				// to check the guess (set the point to (1, 1); the narration and the stage follow it live).
				question: { en: 'XOR(1, 1) is 0. What should the trained network answer for the point (1, 1)?', zh: 'XOR(1, 1) 是 0。训练好的网络对点 (1, 1) 应该回答多少？' },
				options: [
					{ en: 'Close to 0', zh: '接近 0' },
					{ en: 'Close to 1', zh: '接近 1' },
					{ en: 'Exactly 0.5', zh: '正好 0.5' }
				],
				answer: 0,
				explain: {
					en: 'It was trained so that points where XOR is false come out near 0 and points where it is true come out near 1.',
					zh: '它被训练成：XOR 为假的点输出接近 0，为真的点输出接近 1。'
				}
			},
			narration: {
				en: 'For your point ({x1}, {x2}) the trained network answers {y|n3}. Read it as a vote: near 1 means XOR is true, near 0 means false. To check your guess for (1, 1), set your point to (1, 1) in the input panel and watch this number change.',
				zh: '对你的点 ({x1}, {x2})，训练好的网络回答 {y|n3}。把它当成一次投票：接近 1 表示 XOR 为真，接近 0 表示为假。想验证你对 (1, 1) 的猜测，就在输入面板里把点改成 (1, 1)，看这个数怎么变。',
				vars: (rt) => ({ x1: point(rt)[0], x2: point(rt)[1], y: out(traceOf(rt, 'fwd'), 'fc2') })
			},
			look: { en: 'The one output number: is it near 1 or near 0?', zh: '那唯一的输出数：它接近 1 还是接近 0？' }
		},
		{
			id: 'sgd-forward',
			source: 'step1',
			at: { phase: 'forward', sub: 3 },
			view: 'zoom.formula',
			provenance: 'live',
			title: { en: 'Training starts from random weights', zh: '训练从随机权重开始' },
			narration: {
				en: 'Now we train a fresh copy of the network, one step slowed down. It runs all four XOR points forward with random starting weights; the last one, ({x1}, {x2}), comes out as {yhat}, but the right answer is {target}.',
				zh: '现在从头训练这个网络的一个新副本，把一步放慢来看。它用随机的起始权重把四个 XOR 点都算一遍；最后一个点 ({x1}, {x2}) 得到 {yhat}，可正确答案是 {target}。',
				vars: (_rt, at) => ({ x1: metric(at, 'x1'), x2: metric(at, 'x2'), yhat: metric(at, 'yhat'), target: metric(at, 'target') })
			}
		},
		{
			id: 'sgd-update',
			source: 'step1',
			at: { phase: 'update' },
			view: 'zoom.formula',
			provenance: 'live',
			title: { en: 'One gradient-descent step', zh: '一步梯度下降' },
			narration: {
				en: 'The loss, the average squared error over the four points, is {loss}. Backpropagation gives every weight a gradient: how fast the loss grows when that weight grows. Each weight then steps against its gradient, scaled by the learning rate {lr}: the first output weight goes from {old} to {old} − {lr} × ({g}) = {new}, and the loss falls to {lossAfter}.',
				zh: '损失是四个点平方误差的平均值：{loss}。反向传播给每个权重一个梯度：这个权重变大时，损失增长得有多快。然后每个权重朝梯度的反方向走一步，步长按学习率 {lr} 缩放：第一个输出权重从 {old} 变成 {old} − {lr} × ({g}) = {new}，损失降到 {lossAfter}。',
				vars: (_rt, at) => ({ old: metric(at, 'old'), lr: metric(at, 'lr'), g: metric(at, 'g'), new: metric(at, 'new'), loss: metric(at, 'loss'), lossAfter: metric(at, 'lossAfter') })
			}
		},
		{
			id: 'train',
			source: 'train',
			at: 'last',
			view: 'loss-curve',
			provenance: 'live',
			title: { en: 'Repeat it a few hundred times', zh: '把它重复几百次' },
			narration: {
				en: 'Real training repeats steps like this {steps|int} times. This run uses the Adam optimizer, which also adapts the size of each weight’s step. The loss ends at {loss}, and {correct|int} of the 4 points come out right. The network you met first was trained the same way in PyTorch for {fact:trainSteps|int} steps and ended at a loss of {fact:trainLoss}.',
				zh: '真正的训练把这样的一步重复 {steps|int} 次。这次运行用的是 Adam 优化器，它还会调整每个权重的步长。损失最后是 {loss}，4 个点里有 {correct|int} 个答对。你最先见到的那个网络也是这样在 PyTorch 里训练的，训练了 {fact:trainSteps|int} 步，最后损失是 {fact:trainLoss}。',
				vars: (_rt, at) => ({ steps: at.frame?.step ?? NaN, loss: metric(at, 'loss'), correct: metric(at, 'correct') })
			}
		},
		{
			id: 'live',
			source: 'live',
			at: { layer: 'net.2' },
			select: () => ({ layer: 'net.2', unit: [0] }),
			view: 'layer.tensor',
			provenance: 'live',
			title: { en: 'Your trained network, on your point', zh: '你训练的网络，用在你的点上' },
			narration: {
				en: 'These are the weights just trained in your browser, applied to your point ({x1}, {x2}). The live network answers {y|n3}.',
				zh: '这是刚在你的浏览器里训练出的权重，用在你的点 ({x1}, {x2}) 上。这个实时网络回答 {y|n3}。',
				vars: (rt) => ({ x1: point(rt)[0], x2: point(rt)[1], y: out(traceOf(rt, 'live'), 'net.2') })
			},
			task: {
				prompt: { en: 'Set your point to (1, 1). Does the live network answer close to 0?', zh: '把你的点改成 (1, 1)。实时网络的回答接近 0 吗？' },
				done: (rt) => point(rt)[0] === 1 && point(rt)[1] === 1,
				hint: { en: 'Use the 1 buttons for x1 and x2 in the input panel.', zh: '在输入面板里点 x1 和 x2 的 1 按钮。' }
			}
		},
		{
			id: 'stuck',
			source: 'stuck',
			at: 'last',
			view: 'loss-curve',
			provenance: 'live',
			kindOf: 'limits',
			title: { en: 'Where training gets stuck', zh: '训练会卡在哪里' },
			narration: {
				en: 'The same network from other random starting weights (seed {seed|int}) stops at a loss of {loss} with only {correct|int} of 4 points right, and more steps do not help. Two hidden units leave little room: with a third unit, this start learns XOR too.',
				zh: '同一个网络换一组随机起始权重（随机种子 {seed|int}），停在损失 {loss}，4 个点只答对 {correct|int} 个，再多训练也没用。两个隐藏单元的余地太小：加第三个单元，这个起点也能学会 XOR。',
				vars: (_rt, at) => ({ seed: at.tape?.seed ?? NaN, loss: metric(at, 'loss'), correct: metric(at, 'correct') })
			}
		}
	]
};

export default lesson;
