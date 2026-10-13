// residual-connections: the manifest of a CONCEPT package (K2): routed at /train/residual-connections/ by the concept
// route, which renders AutoConceptPage (goals, hook, the lesson, toy, real, compare, pitfalls, checkpoint, links).
// A 20-layer MLP (8 units per hidden layer, tanh) on 120 two-moons points, with and without skip connections, on the
// kit's autograd: the gradient that reaches each layer, the backward pass through one layer with and without its skip,
// and training runs (net.ts, process.ts). Sources:
//   bars    the learner's setup ({ depth, skip }): one forward and backward pass at the starting weights   (process 'grads')
//   plain   the same for 20 layers without skips (input ignored)                                         (process 'grads')
//   res     the same for 20 layers with skips (input ignored)                                            (process 'grads')
//   train   the learner's setup trained with plain gradient descent                                      (process 'train')
//   race    20 layers plain and with skips, trained in lockstep (input ignored)                          (process 'race')
//   depths  the same training at 3, 6, 10 and 20 layers, plain and with skips (input ignored)            (process 'depths')
// facts.json holds TinyGPT's block counts, read from static/models/tinygpt/spec.json (claims.test.ts checks them).
// The page around the lesson is concept.ts.
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { MAX_DEPTH, START } from './data.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it). */
type ConceptManifest = Omit<ExplainerManifest, 'type'> & { type: 'concept' };

const fixed = (skip: number) => () => ({ depth: MAX_DEPTH, skip });

const manifest: ConceptManifest = {
	slug: 'residual-connections',
	type: 'concept',
	kinds: ['process'],
	quality: 'reviewed',
	assets: { dir: 'models/residual-connections', auto: ['facts.json'] },
	input: { adapter: 'setup', start: { ...START } },
	sources: {
		// every source runs the kit's autograd (tested against PyTorch fp64 in the kit's own tests; claims.test.ts checks
		// the layer-0 gradient against central differences); no Python reference of their own, so the badges say "Live"
		bars: { kind: 'process', process: 'grads', reference: false },
		plain: { kind: 'process', process: 'grads', input: fixed(0), reference: false },
		res: { kind: 'process', process: 'grads', input: fixed(1), reference: false },
		train: { kind: 'process', process: 'train', reference: false },
		race: { kind: 'process', process: 'race', input: () => null, reference: false },
		depths: { kind: 'process', process: 'depths', input: () => null, reference: false }
	},
	page: {
		goals: [
			{
				en: 'Explain why the gradient that reaches the first layers of a deep plain network can be thousands of times smaller than the last layer\'s: backpropagation multiplies one factor per layer.',
				zh: '解释为什么在一个很深的普通网络里，传到最前面几层的梯度会比最后一层小几千倍：反向传播每经过一层就乘上一个因子。'
			},
			{
				en: 'Write a residual block as h + f(h) and work out, with real numbers, how its skip passes the gradient back unchanged next to the block\'s share.',
				zh: '把残差块写成 h + f(h)，并用真实的数值算出：跳跃连接如何把梯度原样传回，与块传回的那一份相加。'
			},
			{
				en: 'Compare plain and residual networks of several depths on live training runs, and name what skips do not fix.',
				zh: '用实时训练比较不同深度的普通网络和残差网络，并说出跳跃连接解决不了的问题。'
			}
		],
		prequestion: {
			id: 'residual-connections-pre',
			question: {
				en: 'A network that learns a task well with 3 layers gets 17 more layers, trained the same way. What happens?',
				zh: '一个用 3 层就能学好某个任务的网络，再加 17 层，用同样的方法训练。会发生什么？'
			},
			options: [
				{ en: 'It learns the task better: more layers, more power', zh: '学得更好：层越多，能力越强' },
				{ en: 'It can stop learning almost entirely', zh: '它可能几乎完全学不动' },
				{ en: 'Nothing changes: the extra layers just copy their input', zh: '没有变化：多出来的层只会照抄输入' }
			],
			answer: 1,
			explain: {
				en: 'More layers can represent more, but they must be trained, and the training signal reaches the early layers only through all the later ones. In a plain deep network it can fade to almost nothing on the way. A skip connection gives it a direct path.',
				zh: '层越多，能表示的东西确实越多，但这些层必须被训练，而训练信号只能穿过后面所有的层才能到达前面几层。在很深的普通网络里，它可能在路上衰减到几乎为零。跳跃连接给它开了一条直通的路。'
			}
		},
		quiz: [
			{
				q: {
					en: 'Going back through each of 18 layers multiplies the gradient by about 0.5. Roughly how much of it reaches the first of them?',
					zh: '反向穿过 18 层中的每一层，梯度都大约乘以 0.5。到达最前面那一层时大约还剩多少？'
				},
				options: [
					{ en: 'About 0.5 × 18 = 9 times as much', zh: '大约 0.5 × 18 = 9 倍' },
					{ en: 'About 0.5¹⁸, roughly four millionths', zh: '大约 0.5¹⁸，约百万分之四' },
					{ en: 'About half', zh: '大约一半' }
				],
				answer: 1,
				explain: {
					en: 'The factors multiply: 0.5¹⁸ = 1/262,144 ≈ 0.0000038. That is why the gradient vanishes exponentially with depth.',
					zh: '这些因子是相乘的：0.5¹⁸ = 1/262,144 ≈ 0.0000038。所以梯度会随深度按指数衰减。'
				}
			},
			{
				q: {
					en: 'A layer computes h + f(h). The gradient arriving at its output is 0.2, and its block f sends back −0.05. What leaves towards its input?',
					zh: '某层计算 h + f(h)。到达它输出的梯度是 0.2，它的块 f 传回 −0.05。传向它输入的梯度是多少？'
				},
				options: [
					{ en: '0.15: the skip passes 0.2 back unchanged and the block adds −0.05', zh: '0.15：跳跃连接原样传回 0.2，块再加上 −0.05' },
					{ en: '−0.05: only the block counts', zh: '−0.05：只有块算数' },
					{ en: '−0.01: the two are multiplied', zh: '−0.01：两者相乘' }
				],
				answer: 0,
				explain: {
					en: 'The derivative of "add h" is 1, so the skip copies the arriving gradient; the chain rule adds the two paths: 0.2 + (−0.05) = 0.15.',
					zh: '“加上 h”这一步的导数是 1，所以跳跃连接把到达的梯度原样复制回去；链式法则把两条路相加：0.2 + (−0.05) = 0.15。'
				}
			}
		],
		misconception: {
			claim: { en: 'A skip connection makes the layer it skips unnecessary.', zh: '有了跳跃连接，被跳过的那一层就没用了。' },
			truth: {
				en: 'The skip does not replace the block: the layer still computes h + f(h), and training still changes f. The skip only guarantees that the gradient, and the input, always have a path that no block can shrink.',
				zh: '跳跃连接并不取代块：这一层算的仍然是 h + f(h)，训练也仍然在改变 f。跳跃连接只是保证梯度和输入始终有一条不会被任何块削弱的通路。'
			},
			step: 'skip'
		},
		lessonNote: {
			en: 'Ten steps: watch a 20-layer network fail to learn, find where its gradient vanishes, add skip connections and follow the gradient through one of them with real numbers, then train both and change the depth.',
			zh: '十步：看一个 20 层的网络学不动，找到它的梯度在哪里消失；加上跳跃连接，用真实的数值跟着梯度穿过其中一个；然后训练两个网络，再改变深度。'
		}
	},
	glossary: [
		{
			id: 'gradient-norm',
			term: { en: 'gradient norm', zh: '梯度范数' },
			related: ['gradient', 'vanishing-gradient'],
			def: {
				en: 'One number for the size of a whole gradient: the square root of the sum of its squared entries. A layer whose gradient norm is tiny barely changes in an update.',
				zh: '用一个数表示整个梯度的大小：把它所有元素的平方加起来再开平方。梯度范数极小的层，在一次更新中几乎不变。'
			}
		},
		{
			id: 'residual-stream',
			term: { en: 'residual stream', zh: '残差流' },
			related: ['residual-connection', 'transformer'],
			def: {
				en: 'In a Transformer, the vector of each token that runs from the embedding to the output; every attention and MLP block reads it and adds its result back to it through a residual connection.',
				zh: 'Transformer 里每个词元的向量从嵌入一直流到输出；每个注意力块和 MLP 块都读取它，再通过残差连接把自己的结果加回去。'
			}
		}
	],
	// No third-party data, model or code: the points come from the kit's seeded generator, the networks are written here
	// and the kit's autograd runs them; TinyGPT's numbers come from this site's own model folder.
	credits: []
};

export default manifest;
