// loss-gradient-descent: the manifest of a CONCEPT package (K2): routed at /train/loss-gradient-descent/ by the concept
// route, which renders AutoConceptPage (goals, hook, the lesson, toy, real, compare, pitfalls, checkpoint, links).
// One neuron, y = w·x + b, two dials and an error score: "beat the error score" by hand, then let gradient descent turn
// the dials with the real numbers of the kit's autograd. Sources:
//   gd       fine steps from the learner's dials (score, gradient, update), η = 0.1        (process 'gd', fine)
//   run      30 updates from the dials, one frame per update                              (process 'gd', coarse)
//   rates    the same 30 updates at η = 0.1, 0.01 (too small) and 0.7 (above 2/3, the largest rate at which this
//            bowl settles), in lockstep                                                     (process 'rates')
//   real     a 2-8-8-2 network trained with plain gradient descent on 120 two-moons points, and the same network
//            from other random starting weights, in lockstep                                (process 'realPair')
//   realStep one update of that network, slowed down (fine)                                (process 'realNet', fine)
// The page around the lesson is concept.ts.
import type { ExplainerManifest } from '../../atlas/types/index.ts';

/** A concept package's manifest: the frozen ExplainerManifest with type 'concept' (until a types revision admits it). */
type ConceptManifest = Omit<ExplainerManifest, 'type'> & { type: 'concept' };

const fixed = () => null;

const manifest: ConceptManifest = {
	slug: 'loss-gradient-descent',
	type: 'concept',
	kinds: ['process'],
	quality: 'draft',
	input: { adapter: 'dials', start: { w: 0, b: 0 } },
	sources: {
		gd: { kind: 'process', process: 'gd', detail: 'fine', opt: { lr: 0.1, steps: 20 } },
		run: { kind: 'process', process: 'gd', opt: { lr: 0.1, steps: 30 } },
		rates: { kind: 'process', process: 'rates', opt: { lr: 0.1, lrSlow: 0.01, lrBig: 0.7, steps: 30 } },
		// the network runs are trainingProcess frames (the kit's autograd and SGD, tested against PyTorch fp64 in the kit's
		// own tests); no Python reference of their own, so their badge says "Live"
		real: { kind: 'process', process: 'realPair', input: fixed, opt: { steps: 300 }, reference: false },
		realStep: { kind: 'process', process: 'realNet', input: fixed, detail: 'fine', opt: { steps: 1 }, reference: false }
	},
	page: {
		goals: [
			{
				en: 'Compute an error score (the mean squared error) for a line through eight points, and say why squaring the errors helps.',
				zh: '为穿过八个点的直线算出误差分数（均方误差），并说明为什么要把误差平方。'
			},
			{
				en: 'Read the gradient as the slope of the score under each dial, and do one gradient-descent update by hand: new = old − η × slope.',
				zh: '把梯度读作分数在每个旋钮方向上的斜率，并亲手做一次梯度下降更新：新值 = 旧值 − η × 斜率。'
			},
			{
				en: 'Predict what a too-small and a too-large learning rate do, and check it on live runs.',
				zh: '预测学习率太小和太大会怎样，并用实时运行来验证。'
			}
		],
		prequestion: {
			id: 'loss-gradient-descent-pre',
			question: {
				en: 'A computer has to find the two numbers of the line that fits eight points best. What does it need to know at each try?',
				zh: '计算机要找出最贴合八个点的直线的两个数。每试一次，它需要知道什么？'
			},
			options: [{ en: 'Only the score: how wrong the line is', zh: '只需要分数：这条线错得有多厉害' }, { en: 'The score and which way each number should move to lower it', zh: '分数，以及每个数往哪边动能让分数变低' }, { en: 'The right answer, given in advance', zh: '事先给出的正确答案' }],
			answer: 1,
			explain: {
				en: 'A score alone says how bad a try is, not how to improve it. The gradient adds, for every number, the direction and steepness of the change that lowers the score. Gradient descent follows it in small steps.',
				zh: '只有分数，只能知道这一次试得有多差，不知道怎么改进。梯度还给出每个数往哪个方向、以多大的坡度变化能让分数下降。梯度下降就是沿着它小步前进。'
			}
		},
		quiz: [
			{
				q: { en: 'The slope of the score under w is −2 and the learning rate η is 0.1. What does one update do to w?', zh: '分数在 w 方向的斜率是 −2，学习率 η 是 0.1。一次更新会让 w 怎样变化？' },
				options: [{ en: 'w grows by 0.2', zh: 'w 增加 0.2' }, { en: 'w shrinks by 0.2', zh: 'w 减少 0.2' }, { en: 'w is multiplied by 0.1', zh: 'w 乘以 0.1' }],
				answer: 0,
				explain: { en: 'new w = w − η × slope = w − 0.1 × (−2) = w + 0.2: a negative slope means the score falls as w grows, so w grows.', zh: '新的 w = w − η × 斜率 = w − 0.1 × (−2) = w + 0.2：斜率为负，说明 w 变大时分数下降，所以 w 变大。' }
			},
			{
				q: { en: 'Why do the errors get squared before they are averaged?', zh: '为什么要先把误差平方再取平均？' },
				options: [{ en: 'So that misses above and below the line cannot cancel out, and big misses count most', zh: '这样线上方和下方的偏差不会互相抵消，而且大的偏差最重要' }, { en: 'To make the score smaller', zh: '为了让分数变小' }, { en: 'Because the points are in two dimensions', zh: '因为点是二维的' }],
				answer: 0,
				explain: { en: 'A miss of +1 and a miss of −1 would add up to 0 without the square. Squared, both count 1, and a miss of 2 counts 4.', zh: '如果不平方，+1 和 −1 的偏差加起来是 0。平方之后两者都算 1，而偏差 2 算 4。' }
			}
		],
		misconception: {
			claim: { en: 'The gradient points to the best answer.', zh: '梯度指向最佳答案。' },
			truth: {
				en: 'It only says which way is downhill right where you stand. That is why gradient descent takes a small step and then looks again.',
				zh: '它只说明在你所站的位置哪边是下坡。所以梯度下降每次只走一小步，然后重新看一次。'
			},
			step: 'gradient'
		},
		lessonNote: {
			en: 'Nine steps: score your line, beat the score by hand, then watch gradient descent do it, and break it with a step that is too big.',
			zh: '九步：给你的直线打分，亲手打败这个分数，再看梯度下降怎么做到，最后用过大的步长把它弄坏。'
		}
	},
	// No third-party data, model or code: the eight points and both networks are written here; the kit's autograd runs them.
	credits: []
};

export default manifest;
