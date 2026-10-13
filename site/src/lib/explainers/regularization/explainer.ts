// regularization: the manifest of a CONCEPT package (K2), routed at /train/regularization/ (AutoConceptPage: goals,
// hook, the lesson, toy, real, compare, pitfalls, checkpoint, links). A 2 -> width -> width -> 2 network learns to sort
// the points of two noisy moons in the browser (the kit's autograd); some points train it, the rest test it. It
// overfits, and the page fights that with the real thing each time: the L2 penalty term in the loss, dropout masks,
// early stopping, more data and augmented data. Sources (process.ts):
//   run      the learner's knobs (noise, train share, width, λ, p), 600 Adam updates, a check every 10   (process 'fit')
//   methods  the starting knobs: plain, L2, dropout, augmentation, too much L2 and a small network in lockstep
//   stop     the starting knobs, the tape ending where early stopping stops                             (process 'fit')
//   drop     the starting knobs with dropout p = 0.5, one frame per update for 24 updates               (process 'fit')
//   cnn      TinyCNN's recorded training (precomputed, cnn.ts)                                            (process 'cnn')
// The page around the lesson is concept.ts.
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { EVERY, PATIENCE, START } from './data.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it). */
type ConceptManifest = Omit<ExplainerManifest, 'type'> & { type: 'concept' };

const fixed = () => null;
const L = (en: string, zh: string) => ({ en, zh });

const manifest: ConceptManifest = {
	slug: 'regularization',
	type: 'concept',
	kinds: ['process'],
	quality: 'reviewed',
	input: { adapter: 'knobs', start: { ...START } },
	sources: {
		// the kit's autograd and Adam (tested against PyTorch fp64 in the kit's own tests); no Python reference of the
		// package's own (dropout masks and nudges come from the kit's rng), so the badges say "Live"
		run: { kind: 'process', process: 'fit', reference: false },
		methods: { kind: 'process', process: 'methods', input: fixed, reference: false },
		stop: { kind: 'process', process: 'fit', input: fixed, opt: { stopEarly: 1 }, reference: false },
		drop: { kind: 'process', process: 'fit', input: fixed, opt: { every: 1, steps: 24, dropout: 0.5 }, reference: false },
		cnn: { kind: 'process', process: 'cnn', input: fixed, reference: false }
	},
	page: {
		goals: [
			L(
				'Read a training curve and a test curve together, and say when a network starts to overfit: the test loss turns up while the training loss keeps falling.',
				'把训练曲线和测试曲线放在一起读，并说出网络从什么时候开始过拟合：训练损失还在下降，测试损失却开始上升。'
			),
			L(
				'Explain what the L2 penalty adds to the loss (λ·Σw²), what dropout does at each update, and why it multiplies the kept units by 1/(1 − p).',
				'说明 L2 惩罚给损失加了什么（λ·Σw²）、dropout 在每次更新时做了什么，以及为什么要把保留下来的单元乘以 1/(1 − p)。'
			),
			L(
				'Predict how network size, early stopping, more data and data augmentation change the test loss, and check it on live runs.',
				'预测网络大小、早停、更多数据和数据增强会怎样改变测试损失，并用实时训练来验证。'
			)
		],
		prequestion: {
			id: 'regularization-pre',
			question: L(
				'A network sorts every one of its training points correctly. What does that tell you about new points?',
				'一个网络把它的训练点全部分对了。这能说明它在新的点上表现如何吗？'
			),
			options: [
				L('It will sort new points correctly too', '它在新的点上也会分对'),
				L('Not much: it may have memorized the training points, noise and all', '说明不了太多：它可能连同噪声一起把训练点背了下来'),
				L('It will get new points wrong', '它在新的点上一定会分错')
			],
			answer: 1,
			explain: L(
				'A perfect score on the training points is easy for a big enough network: it can bend its border around every point, including the ones noise pushed to the wrong place. Only points it never trained on, the test points, show how well it does on new data.',
				'对足够大的网络来说，在训练点上拿满分很容易：它可以把分界线绕过每一个点，包括被噪声推到错误位置的点。只有它从没训练过的点，也就是测试点，才能说明它在新数据上的表现。'
			)
		},
		quiz: [
			{
				q: L('Dropout with p = 0.25. During training, what happens to the output of a unit that is kept?', 'Dropout 的 p = 0.25。训练时，一个被保留的单元的输出会怎样？'),
				options: [L('It is multiplied by 1/(1 − 0.25), about 1.33', '乘以 1/(1 − 0.25)，约等于 1.33'), L('It is multiplied by 0.25', '乘以 0.25'), L('Nothing happens to it', '什么也不变')],
				answer: 0,
				explain: L(
					'A quarter of the units are off, so the next layer would get about three quarters of its usual total. Multiplying the kept ones by 1/(1 − p) puts the total back where it is when nothing is dropped, at test time.',
					'有四分之一的单元被关掉，下一层收到的总和大约只剩平时的四分之三。把保留的单元乘以 1/(1 − p)，总和就回到了不关掉任何单元时（也就是测试时）的水平。'
				)
			},
			{
				q: L(
					`The training loss is still falling, but the test loss has not reached a new low for the last ${PATIENCE * EVERY} updates. What does early stopping do?`,
					`训练损失还在下降，但测试损失已经 ${PATIENCE * EVERY} 次更新没有创新低了。早停会怎么做？`
				),
				options: [L('Stops and keeps the weights from where the test loss was lowest', '停止训练，保留测试损失最低时的权重'), L('Keeps training, because the training loss still falls', '继续训练，因为训练损失还在下降'), L('Starts again from new random weights', '从新的随机权重重新开始')],
				answer: 0,
				explain: L(
					'The falling training loss only says the network fits its training points better. The test loss says it does worse on new points, so early stopping goes back to the best weights it has seen.',
					'训练损失下降只说明网络对训练点拟合得更好了。测试损失说明它在新的点上变差了，所以早停会退回到它见过的最好的那组权重。'
				)
			}
		],
		misconception: {
			claim: L('A lower training loss always means a better network.', '训练损失越低，网络就一定越好。'),
			truth: L(
				'Only on the training points. Here the training loss falls almost to 0 while the test loss climbs: the network is learning the noise of its training points.',
				'只在训练点上成立。在这里，训练损失几乎降到 0，测试损失却在上升：网络学到的是训练点里的噪声。'
			),
			step: 'curves'
		},
		lessonNote: L(
			'Ten steps: watch a big network overfit few noisy points, predict whether a smaller one does better, then fight overfitting with early stopping, an L2 penalty, dropout and augmented data, close the gap yourself, and overdo it.',
			'十步：看一个大网络在少量有噪声的点上过拟合，预测小网络会不会更好，然后用早停、L2 惩罚、dropout 和数据增强来对付过拟合，自己动手缩小差距，最后看看用力过猛会怎样。'
		)
	},
	glossary: [
		{
			id: 'weight-decay',
			term: L('weight decay (L2 regularization)', '权重衰减（L2 正则化）'),
			def: L(
				'Shrinking every weight a little toward 0 at every update. An L2 penalty λ·Σw² in the loss does it through its gradient 2λw; AdamW does it directly, outside the gradient step.',
				'每次更新时把每个权重往 0 缩小一点。在损失里加 L2 惩罚 λ·Σw²，是通过它的梯度 2λw 做到这一点；AdamW 则在梯度那一步之外直接缩小权重。'
			),
			related: ['regularization', 'overfitting']
		},
		{
			id: 'early-stopping',
			term: L('early stopping', '早停'),
			def: L(
				'Stopping training when the loss on held-out data has not improved for a while, and keeping the weights from its lowest point.',
				'当留出数据上的损失一段时间没有改善时停止训练，并保留它最低点时的权重。'
			),
			related: ['overfitting', 'dataset-split']
		},
		{
			id: 'underfitting',
			term: L('underfitting', '欠拟合'),
			def: L(
				'When a model is too simple, or held back too hard, to fit even its training data well.',
				'模型太简单，或者被约束得太狠，连训练数据都拟合不好。'
			),
			related: ['overfitting', 'regularization']
		}
	],
	// The catalog entry's interaction, rewritten to promise only what the page has (promote prints it into the patch;
	// the concept catalog fields are not in the frozen CatalogFields type, as in the other concept packages)
	catalog: {
		interaction: L(
			'Watch the training and test curves split apart as a big network overfits a few noisy points of two moons, predict whether a smaller network does better, then fight overfitting with early stopping, an L2 penalty (its real term in the loss), dropout (units blinking per update) and nudged training points. Turn five knobs until the test loss drops, and see how TinyCNN was trained (augmented drawings, weight decay, the kept epoch).',
			'看一个大网络在两个月牙的少量带噪声的点上过拟合，训练曲线和测试曲线逐渐分开；预测更小的网络会不会更好，再用早停、L2 惩罚（损失里真实的那一项）、dropout（单元每次更新随机熄灭）和挪动训练点来对付过拟合。调节五个旋钮让测试损失降下来，并看看 TinyCNN 是怎么训练的（增强过的手绘图、权重衰减、保留的那一轮）。'
		)
	} as unknown as ExplainerManifest['catalog'],
	// The moons, the network and every run are written here and run on the kit's autograd. The knobs follow the idea of
	// TensorFlow Playground (noise and train-ratio sliders; no code copied). The TinyCNN numbers are this site's own
	// training record of a model trained on Quick, Draw! drawings.
	credits: [
		{
			name: 'TensorFlow Playground',
			url: 'https://github.com/tensorflow/playground',
			license: 'Apache-2.0',
			usedFor: 'The idea of noise and train-ratio sliders next to a live network; reimplemented on the Model Atlas autograd core',
			attribution: 'Daniel Smilkov and Shan Carter, Google (TensorFlow Playground)'
		},
		{
			name: 'Quick, Draw! dataset',
			url: 'https://github.com/googlecreativelab/quickdraw-dataset',
			license: 'CC BY 4.0',
			usedFor: "TinyCNN's recorded training curve (the drawings it was trained on)",
			attribution: 'Quick, Draw! dataset by Google Creative Lab'
		}
	]
};

export default manifest;
