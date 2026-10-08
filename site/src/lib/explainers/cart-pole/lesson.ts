// cart-pole: the guided lesson, one idea per step: the state, the policy's p(left), sampling, one physics step, the end of
// an episode, discounted returns, advantages, the pulls that make the gradient, one update, training (predict first),
// the trained policy on your start (task), and how much the seed matters (limits). Every number is a {placeholder}
// computed from frames that ran, and the stage shows it too (the 'worked' view puts the worked formulas under a view);
// the other factual sentences are tested in claims.test.ts.
import type { At, ExplainerRuntime, I18n, LessonDef, Tensor, TmplVal, TrainState } from '../../atlas/types/index.ts';
import { num, pick } from '../../atlas/lesson/say.ts';
import { updateOf } from '../../atlas/train/helpers.ts';
import { fmt3, minus } from '../../atlas/core/format.ts';
import { deg, lenText, meanStd, policy, startOf, TAU, TH_LIMIT, X_LIMIT, type Params } from './sim.ts';
import { reinforce } from './trainer.ts';
import { episode } from './process.ts';
import { policyEpisode } from './views/scene/meta.ts';
import { PRESETS } from './adapters/cart-pole/adapter.ts';
import { cellsOf } from './views/round/meta.ts';

const L = (en: string, zh: string): I18n => ({ en, zh });
const D = reinforce.defaults;
const m = (at: At, k: string) => at.frame?.metrics?.[k] ?? NaN;
type St = Partial<TrainState> & { prev?: Params };
const st = (at: At) => (at.state ?? {}) as St;
const batch = (at: At) => (st(at).batch ?? {}) as Record<string, Tensor>;
/** a setting as written (0.95, 0.5), not rounded to 3 digits */
const set = (v: number) => (Number.isFinite(v) ? String(v) : NaN);
const LIMITS = { lim: deg(TH_LIMIT), xl: set(X_LIMIT), hz: 1 / TAU };
const dir = (left: number) => pick(left === 1, L('left', '左'), L('right', '右'));
/** the metric of every round of a coarse train tape (or the seeds tape), in round order */
function rounds(at: At, k: string): number[] {
	const out: number[] = [];
	for (let i = 0; at.tape && i < at.tape.length; i++) {
		const f = at.tape.frame(i);
		if (f.phase !== 'init' && typeof f.metrics?.[k] === 'number') out.push(f.metrics[k]);
	}
	return out;
}
const firstFull = (v: number[]) => v.findIndex((x) => x >= D.maxLen);
const scene = (props: Record<string, unknown> = {}) => ({ key: 'cart-pole/scene', props: () => props });
const round = (show: string) => ({ key: 'cart-pole/round', props: () => ({ show }) });
/** the weights at frame i (null while a new input retrains the source and frame i is not recorded yet) */
const paramsAt = (at: At, i: number) => ((at.tape && i < at.tape.length ? (at.tape.state(i) as St).params : undefined) ?? null) as Params | null;
/** the fresh policy on the learner's start: the output neuron's score z and p(left) (the p0 trace) */
const p0Of = (rt: ExplainerRuntime) => ({ z: rt.sources.p0?.trace?.t?.['net.2']?.data[0] ?? NaN, p: rt.sources.p0?.trace?.t?.['net.3']?.data[0] ?? NaN });
/** the slowed-down update (plain SGD; updateOf throws otherwise): the output bias, p(left) at the start before and after */
function upd(rt: ExplainerRuntime, at: At) {
	const s = st(at);
	if (!s.prev || !s.params) return null;
	const start = startOf(rt.input);
	return { ...updateOf(at, 'net.2.bias', [0]), p0: policy(s.prev, start).p, p1: policy(s.params, start).p, count: Object.values(s.params).reduce((a, t) => a + t.data.length, 0) };
}
const int = (v: number) => minus(String(v));

const lesson: LessonDef = {
	title: L('Cart-pole: a network learns to balance by trial and reward', '倒立摆：网络靠尝试和奖励学会平衡'),
	summary: L(
		'Four numbers describe the cart and pole; a small network turns them into a chance of pushing left. Follow one episode, credit its pushes with returns and advantages, slow one update down, then train and see how much luck matters.',
		'四个数描述小车和杆；一个小网络把它们变成向左推的概率。跟着一个回合走一遍，用回报和优势给每一推记功，放慢看一次更新，然后训练，并看看运气有多重要。'
	),
	steps: [
		{
			id: 'state',
			source: 'episode',
			at: 'first',
			view: scene(),
			provenance: 'live',
			title: L('Four numbers describe the cart-pole', '四个数描述倒立摆'),
			narration: {
				en: 'A cart rolls on a track with a pole standing on it, free to fall left or right. The state, all the policy (the rule that picks each push) will know, is four numbers: the cart’s position x = {x} m and velocity ẋ = {xd} m/s, the pole’s angle θ = {th} rad ({d}°; + means leaning right) and how fast that angle changes, θ̇ = {thd} rad/s. The task: push the cart left or right {hz} times a second, so the pole stays within {lim}° of upright and the cart within {xl} m of the middle.',
				zh: '一辆小车在轨道上滚动，车上立着一根杆，可以向左或向右倒。状态是策略（决定每一推的规则）能知道的全部信息，一共四个数：小车的位置 x = {x} m 和速度 ẋ = {xd} m/s，杆的角度 θ = {th} rad（{d}°；正数表示向右倾斜），以及这个角度变化的快慢 θ̇ = {thd} rad/s。任务是：每秒 {hz} 次把小车往左或往右推，让杆偏离竖直不超过 {lim}°，小车离轨道中点不超过 {xl} m。',
				vars: (_rt, at) => ({ x: m(at, 'x'), xd: m(at, 'xd'), th: m(at, 'th'), d: deg(m(at, 'th')), thd: m(at, 'thd'), ...LIMITS })
			},
			look: L('The four numbers under the picture. You can set them with the sliders in “Your input”.', '图下面的四个数。你可以用“你的输入”里的滑块设置它们。'),
			terms: ['state', 'policy']
		},
		{
			id: 'policy',
			source: 'p0',
			at: { layer: 'net.2' },
			select: () => ({ layer: 'net.2', unit: [0] }),
			view: {
				key: 'cart-pole/worked',
				props: (rt) => {
					const { z, p } = p0Of(rt);
					return { view: 'zoom.weighted-sum', tex: 'p = \\frac{1}{1 + e^{-z}} = \\frac{1}{1 + e^{ {{mz}} }} = {{p}}', values: { mz: -z, p },
						where: [{ tex: 'z', text: L('the score (y in the table)', '分数（表里的 y）') }, { tex: 'p', text: L('p(left), the chance of pushing left', 'p(左)，向左推的概率') }],
						say: L(`The output neuron adds the ${D.hidden} hidden values, each times its weight, plus the bias: the score z = ${fmt3(z)}. Then p(left) = 1 / (1 + e^(−z)) = ${fmt3(p)}.`, `输出神经元把 ${D.hidden} 个隐藏值各乘上权重相加，再加偏置：分数 z = ${fmt3(z)}。然后 p(左) = 1 / (1 + e^(−z)) = ${fmt3(p)}。`) };
				}
			},
			provenance: 'live',
			title: L('The policy network gives p(left)', '策略网络给出 p(左)'),
			narration: {
				en: 'The policy is a small network, 4 → {h} → 1. Each of its {h} hidden neurons takes a weighted sum of the four numbers and squashes it with tanh into the range −1 to 1; the output neuron adds the {h} results, each times its weight, plus a bias: the score z = {z}. A sigmoid turns the score into a chance, p(left) = 1 / (1 + e^(−z)) = {p}. The weights are still random, so this is close to a coin flip.',
				zh: '策略是一个小网络，4 → {h} → 1。{h} 个隐藏神经元各自对四个数做加权和，再用 tanh 压到 −1 到 1 之间；输出神经元把这 {h} 个结果各乘上自己的权重相加，再加一个偏置，得到分数 z = {z}。sigmoid 把分数变成概率：p(左) = 1 / (1 + e^(−z)) = {p}。权重还是随机的，所以这和抛硬币差不多。',
				vars: (rt): Record<string, TmplVal> => ({ h: D.hidden, ...p0Of(rt) })
			},
			look: L('The table: each hidden value (x) times its weight (w), the products added up (Σ), plus the bias b; the result y is the score z. Under it, the sigmoid.', '看表：每个隐藏值（x）乘以它的权重（w），乘积相加（Σ），再加偏置 b；结果 y 就是分数 z。下面是 sigmoid。'),
			terms: ['policy', 'neuron', 'logit']
		},
		{
			id: 'sample',
			source: 'episode',
			at: { phase: 'act', step: 0 },
			view: scene(),
			provenance: 'live',
			title: L('Sample a push', '采样一次推力'),
			narration: {
				en: 'The policy does not simply take the likelier push. It draws a random number u between 0 and 1 and pushes left when u is below p(left). Here p(left) = {p} and u = {u}, so the cart is pushed {dir}. Drawing like this is called sampling: it makes the policy try both pushes, and the pushes it tries are all it can learn from.',
				zh: '策略并不是直接选更可能的那一推。它抽一个 0 到 1 之间的随机数 u，u 小于 p(左) 就向左推。这里 p(左) = {p}，u = {u}，所以小车被向{dir}推。这样抽取叫作采样：它让策略两种推法都试一试，而它试过的推法就是它能学习的全部材料。',
				vars: (_rt, at) => ({ p: m(at, 'p'), u: m(at, 'u'), dir: dir(m(at, 'left')) })
			},
			look: L('The bar: its blue part is p(left), and the thin mark is the draw u.', '看那根条：蓝色部分是 p(左)，细竖线是抽到的 u。'),
			terms: ['exploration']
		},
		{
			id: 'physics',
			source: 'episode',
			at: { phase: 'move', step: 0 },
			view: {
				key: 'cart-pole/worked',
				props: (_rt, at) => {
					const f = at.frame?.metrics ?? {};
					const acc = L(`Push F = ${int(f.force)} N → cart ẍ = ${fmt3(f.xAcc)} m/s², pole θ̈ = ${fmt3(f.thAcc)} rad/s²`, `推力 F = ${int(f.force)} N → 小车 ẍ = ${fmt3(f.xAcc)} m/s²，杆 θ̈ = ${fmt3(f.thAcc)} rad/s²`);
					return { view: 'cart-pole/scene', note: acc, tex: episode.phases.move.formula!.tex, values: { ...f, tau: set(TAU) },
						where: [{ tex: "\\theta',\\ \\dot\\theta'", text: L('the angle and its speed after the step', '这一步之后的角度和角速度') }, { tex: '\\tau', text: L(`one time step, ${set(TAU)} s`, `一个时间步，${set(TAU)} s`) }],
						say: L(`${acc.en}. After ${set(TAU)} s: θ goes from ${fmt3(f.th)} to ${fmt3(f.th2)} rad and θ̇ from ${fmt3(f.thd)} to ${fmt3(f.thd2)} rad/s.`, `${acc.zh}。${set(TAU)} s 之后：θ 从 ${fmt3(f.th)} 变成 ${fmt3(f.th2)} rad，θ̇ 从 ${fmt3(f.thd)} 变成 ${fmt3(f.thd2)} rad/s。`) };
				}
			},
			provenance: 'live',
			title: L('One physics step', '物理模拟一步'),
			narration: {
				en: 'The push is a force F = {f} N on the cart. The cart-pole equations (Newton’s laws for a cart carrying a pole) turn it into accelerations: ẍ = {xa} m/s² for the cart and θ̈ = {ta} rad/s² for the pole’s angle. {why} Then time moves on by τ = {tau} s: θ changes by τ × θ̇ ({th} → {th2} rad) and θ̇ by τ × θ̈ ({thd} → {thd2} rad/s); x and ẋ change the same way.',
				zh: '这一推是作用在小车上的力 F = {f} N。倒立摆方程（小车载着一根杆时的牛顿定律）把它变成加速度：小车的 ẍ = {xa} m/s²，杆的角加速度 θ̈ = {ta} rad/s²。{why}然后时间前进 τ = {tau} s：θ 变化 τ × θ̇（{th} → {th2} rad），θ̇ 变化 τ × θ̈（{thd} → {thd2} rad/s）；x 和 ẋ 也同样更新。',
				vars: (_rt, at) => ({
					f: m(at, 'force'), xa: m(at, 'xAcc'), ta: m(at, 'thAcc'), tau: set(TAU), th: m(at, 'th'), th2: m(at, 'th2'), thd: m(at, 'thd'), thd2: m(at, 'thd2'),
					why: pick(m(at, 'force') > 0, L('Pushing the cart right tips the pole to the left (θ̈ < 0), like moving your hand under a stick.', '把小车往右推，杆就往左倒（θ̈ < 0），就像在手上立一根棍子时移动手一样。'), L('Pushing the cart left tips the pole to the right (θ̈ > 0), like moving your hand under a stick.', '把小车往左推，杆就往右倒（θ̈ > 0），就像在手上立一根棍子时移动手一样。'))
				})
			},
			look: L('The formula under the cart: the new angle uses the old angular speed, the new angular speed the acceleration.', '看小车下面的公式：新角度用的是旧的角速度，新角速度用的是角加速度。')
		},
		{
			id: 'episode-end',
			source: 'episode',
			at: { phase: 'end' },
			view: scene(),
			play: { from: 'first', to: { phase: 'end' }, phases: ['start', 'move', 'end'], interval: 60 },
			provenance: 'live',
			title: L('Until the pole falls: one episode', '直到杆倒下：一个回合'),
			narration: {
				en: 'Sample, push, step, again and again: every step earns a reward of +1, and the episode ends when the pole leans past {lim}° or the cart leaves the track. This one lasted T = {len} {sw} ({secs} s) and ended because {why}. That number of steps is the only feedback the policy gets.',
				zh: '采样、推、模拟一步，一遍又一遍：每一步得到 +1 的奖励，杆偏过 {lim}° 或小车离开轨道时，回合结束。这个回合持续了 T = {len} {sw}（{secs} s），结束的原因是{why}。这个步数就是策略得到的唯一反馈。',
				vars: (_rt, at) => {
					const e = m(at, 'end');
					return { ...LIMITS, len: m(at, 'len'), sw: pick(m(at, 'len') === 1, L('step', '步'), L('steps', '步')), secs: set(+(m(at, 'len') * TAU).toFixed(2)), why: e === 1 ? L('the pole fell', '杆倒了') : e === 2 ? L('the cart left the track', '小车离开了轨道') : L('it reached the step limit', '到了步数上限') };
				}
			},
			look: { en: 'The curve under the cart: the pole’s angle at every step, between the ±{lim}° lines. The slider replays the episode.', zh: '小车下面的曲线：每一步杆的角度，夹在 ±{lim}° 两条线之间。用滑块可以重放这个回合。', vars: () => LIMITS },
			terms: ['reward', 'episode']
		},
		{
			id: 'returns',
			source: 'episode',
			at: 'last',
			view: round('G'),
			play: { from: { phase: 'end' }, to: 'last', phases: ['return'], interval: 120 },
			provenance: 'live',
			title: L('Credit for every push: the return', '给每一推记功：回报'),
			narration: {
				en: 'A push matters for all the rewards after it, so each step gets a return G: its own reward plus every later one, each later reward multiplied by one more γ = {g} (the discount: far-off rewards count less). Work backwards from the last step: {chain}. Early steps, far from the fall, get large returns.',
				zh: '一推影响它之后的所有奖励，所以每一步都有一个回报 G：它自己的奖励，加上之后的每一个奖励，越往后的奖励多乘一次 γ = {g}（折扣：越远的奖励算得越少）。从最后一步倒着算：{chain}。离倒下越远的早期步骤，回报越大。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const G = Array.from(((at.state as { G?: Tensor }).G?.data ?? []) as ArrayLike<number>), T = G.length, g = m(at, 'gamma');
					if (!T) return { g: NaN, chain: NaN };
					const n = (v: number) => num(v) as I18n;
					if (T === 1) return { g: set(g), chain: L('this episode had one step, so its only return is G = 1', '这个回合只有一步，所以它唯一的回报是 G = 1') };
					const a = n(G[T - 2]), b = T > 2 ? n(G[T - 3]) : null, last = n(G[0]), gg = { en: String(g), zh: String(g) };
					return {
						g: set(g),
						chain: L(
							`G = 1 at the last step, 1 + ${gg.en} × 1 = ${a.en} one step earlier${b ? `, 1 + ${gg.en} × ${a.en} = ${b.en} before that, and so on up to G = ${last.en} for the first step` : ' at the first step'}`,
							`最后一步 G = 1，前一步 1 + ${gg.zh} × 1 = ${a.zh}${b ? `，再前一步 1 + ${gg.zh} × ${a.zh} = ${b.zh}，依此类推，直到第一步的 G = ${last.zh}` : '，也就是第一步'}`
						)
					};
				}
			},
			look: L('The row fills from the right as the returns are worked out backwards; darker means a larger return.', '回报倒着算，这一行就从右往左填满；颜色越深，回报越大。'),
			terms: ['return', 'discount']
		},
		{
			id: 'advantage',
			source: 'step1',
			at: { phase: 'forward', step: 0 },
			view: round('A'),
			provenance: 'live',
			title: L('Better or worse than usual: the advantage', '比平常好还是差：优势'),
			predict: {
				id: 'cart-pole-advantage',
				question: L('Which steps of the round will get a negative advantage (blue)?', '这一轮里，哪些步骤的优势会是负的（蓝色）？'),
				options: [L('The last steps before each episode ended, and most steps of short episodes', '每个回合结束前的最后几步，以及短回合里的大多数步'), L('The first steps of every episode', '每个回合的头几步'), L('The steps that pushed left', '向左推的那些步')],
				answer: 0,
				explain: L(
					'A return counts the rewards still to come. Near the end of an episode few are left, so G is small and falls below the round’s mean: A < 0. The early steps of long episodes have the largest returns. Which way a step pushed does not enter A at all.',
					'回报算的是还没拿到的奖励。回合快结束时剩下的奖励很少，G 就小，低于这一轮的平均值：A < 0。长回合的早期步骤回报最大。至于这一步往哪边推，A 根本不管。'
				)
			},
			narration: {
				en: 'One episode is thin evidence, so training plays a round of {n} episodes with the same policy: yours, then {more} from random starts near upright, lasting {lens} steps. Is a return large? Compare it with the round: the advantage is A = (G − mean) / standard deviation over all {N} steps, here mean {mean} and standard deviation {sd} (the typical distance of a return from the mean). For the first step of your episode: A = ({G} − {mean}) / {sd} = {A}.',
				zh: '一个回合的证据太单薄，所以训练用同一个策略玩一轮 {n} 个回合：先是你的那个，再从竖直附近的随机起点玩 {more} 个，分别持续 {lens} 步。一个回报算不算大？要和这一轮比：优势 A = (G − 平均值) / 标准差，在全部 {N} 步上计算，这里平均值是 {mean}，标准差是 {sd}（回报离平均值的典型距离）。你那个回合的第一步：A = ({G} − {mean}) / {sd} = {A}。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const b = batch(at), G = Array.from(b.ret?.data ?? []) as number[], lens = Array.from(b.lens?.data ?? []) as number[];
					if (!G.length) return { n: NaN };
					const { mean, std } = meanStd(G);
					return { n: lens.length, more: lens.length - 1, lens: lens.join(', '), N: G.length, mean, sd: std, G: G[0], A: b.adv.data[0] };
				}
			},
			look: L('Each row is one episode of the round, time running to the right. Where is the blue?', '每一行是这一轮的一个回合，时间从左往右。蓝色在哪里？'),
			terms: ['advantage', 'batch']
		},
		{
			id: 'pulls',
			source: 'step1',
			at: { phase: 'backward', step: 0 },
			view: round('pull'),
			provenance: 'live',
			title: L('Every step pulls on the network', '每一步都在拉网络'),
			narration: {
				en: 'Training lowers the loss mean(A × −log π), where π is the policy’s chance of the push it made. Each step pulls on the network: toward its push if A > 0 (orange), away if A < 0 (blue), harder when that push was unlikely. On the score z a step pulls A × (p(left) − y) / N, with y = 1 for a left push: for the ringed step {A} × ({p} − {y}) / {N} = {c}. Backpropagation adds all {N} pulls into the output bias’s gradient, {g}: {so}.',
				zh: '训练要降低损失 mean(A × −log π)，π 是策略给它实际做出的那一推的概率。于是每一步都在拉网络：A > 0（橙色）就往它自己那一推的方向拉，A < 0（蓝色）就往反方向推，那一推越不可能，拉得越狠。对分数 z，一步的拉力是 A × (p(左) − y) / N，向左推时 y = 1：圈出的那一步是 {A} × ({p} − {y}) / {N} = {c}。反向传播把全部 {N} 个拉力加成输出偏置的梯度 {g}：{so}。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const c = cellsOf(batch(at))[0]?.[0], g = st(at).grads?.['net.2.bias']?.data[0] ?? NaN;
					const so = pick(g < 0, L('negative: the update raises the bias, which by itself favors left', '它是负的，所以更新会把偏置调高，单看这一点会让向左更可能'), L('positive: the update lowers the bias, which by itself favors right', '它是正的，所以更新会把偏置调低，单看这一点会让向右更可能'));
					return c ? { A: c.A, p: c.p, y: c.left, N: batch(at).adv.data.length, c: c.pull, g, so } : { A: NaN };
				}
			},
			look: L('The color of each cell is the sign of its advantage; its strength is the size of its pull. The sum is under the picture.', '每个格子的颜色是它的优势的正负；颜色深浅是拉力的大小。总和在图的下面。'),
			terms: ['policy-gradient', 'gradient', 'backpropagation']
		},
		{
			id: 'update',
			source: 'step1',
			at: { phase: 'update', step: 0 },
			view: {
				key: 'cart-pole/worked',
				props: (rt, at) => {
					const u = upd(rt, at), net = { color: 'update', maxNodes: 16, inputs: ['x', 'ẋ', 'θ', 'θ̇'], outputs: [L('p(left)', 'p(左)')] };
					if (!u) return { view: 'network', inner: net };
					const pl = L(`p(left) at your start: ${fmt3(u.p0, 4)} → ${fmt3(u.p1, 4)}`, `你的起点的 p(左)：${fmt3(u.p0, 4)} → ${fmt3(u.p1, 4)}`);
					return { view: 'network', inner: net,
						note: L(`Circles, left to right: x, ẋ, θ, θ̇ · the ${D.hidden} hidden neurons · the score z. ${pl.en}`, `圆圈从左到右：x、ẋ、θ、θ̇ · ${D.hidden} 个隐藏神经元 · 分数 z。${pl.zh}`),
						tex: "b' = b - \\text{lr} \\times g = {{old}} - {{lr}} \\times ({{g}}) = {{new}}", values: { old: u.old, lr: set(u.lr), g: u.g, new: u.new },
						where: [{ tex: 'b', text: L('the output bias', '输出偏置') }, { tex: 'g', text: L('its gradient: all the pulls added up (step 8)', '它的梯度：所有拉力之和（第 8 步）') }],
						say: L(`Every weight moved against its gradient. The output bias: ${fmt3(u.old)} − ${set(u.lr)} × (${fmt3(u.g)}) = ${fmt3(u.new)}. ${pl.en}.`, `每个权重都沿梯度的反方向移动。输出偏置：${fmt3(u.old)} − ${set(u.lr)} × (${fmt3(u.g)}) = ${fmt3(u.new)}。${pl.zh}。`) };
				}
			},
			provenance: 'live',
			title: L('One update', '一次更新'),
			narration: {
				en: 'Gradient descent moves every weight a small step against its gradient. With learning rate {lr} the output bias goes from {old} to {old} − {lr} × ({g}) = {new}, and all {count} weights and biases move at once, each by its own gradient. One update changes little: at your start state p(left) goes from {p0|n4} to {p1|n4}{also}.',
				zh: '梯度下降让每个权重沿梯度的反方向走一小步。学习率为 {lr} 时，输出偏置从 {old} 变成 {old} − {lr} × ({g}) = {new}；全部 {count} 个权重和偏置同时移动，各自按自己的梯度。一次更新改变不大：在你的起始状态，p(左) 从 {p0|n4} 变成 {p1|n4}{also}。',
				vars: (rt, at): Record<string, TmplVal> => {
					const u = upd(rt, at);
					if (!u) return { lr: NaN };
					// the bias alone moves p(left) the way it moved; when p(left) went the other way, the other weights outweighed it
					const also = pick((u.new - u.old) * (u.p1 - u.p0) < 0, L(', the other way from what the bias alone would do: the other weights moved too', '，和偏置单独的作用方向相反：其余的权重也动了'), L('', ''));
					return { lr: set(u.lr), old: u.old, g: u.g, new: u.new, count: u.count, p0: u.p0, p1: u.p1, also };
				}
			},
			look: L('Every edge is one weight, colored by how it moved in this update: orange up, blue down, thicker for a bigger move. Under it, the bias’s update.', '每条连线是一个权重，颜色表示它在这次更新中怎么动：橙色变大，蓝色变小，越粗动得越多。下面是偏置的更新。'),
			terms: ['gradient-descent', 'learning-rate']
		},
		{
			id: 'train',
			source: 'train',
			at: 'last',
			view: scene({ curve: true }),
			play: { from: 'first', to: 'last', phases: ['init', 'snapshot'], interval: 200, unit: L('round', '轮') },
			provenance: 'live',
			title: L('Train, round after round', '一轮接一轮地训练'),
			predict: {
				id: 'cart-pole-train',
				question: L(
					'Training now repeats the round 100 times: play, returns, advantages, update. What happens to the average episode length?',
					'训练现在把这一轮重复 100 次：玩、算回报、算优势、更新。平均回合长度会怎样？'
				),
				options: [L('It stays about the same: “how long it lasted” is too weak a signal', '基本不变：“立了多久”这个信号太弱了'), L('It roughly doubles, then stops improving', '大约翻一倍，然后就不再进步'), L('It climbs close to the step limit, though not smoothly', '一路爬到接近步数上限，但并不平稳')],
				answer: 2,
				explain: L(
					'Round after round, the pushes of long episodes become more likely, and longer episodes give more to learn from. The curve is noisy and can dip, because every round is a new random sample of episodes.',
					'一轮又一轮，长回合里的推法越来越常出现，而更长的回合又提供更多可学的东西。曲线有噪声，也会下跌，因为每一轮都是一批新的随机回合。'
				)
			},
			narration: {
				en: 'Each round is one training step. This run uses the Adam optimizer, which adapts each weight’s step size, with learning rate {lr}, {n} episodes per round and at most {max} steps per episode. Before training, an episode lasts about {l0} steps on average. Make your guess, then play the training.',
				zh: '每一轮就是一步训练。这次训练用 Adam 优化器（它会给每个权重调整步长），学习率 {lr}，每轮 {n} 个回合，每个回合最多 {max} 步。训练之前，一个回合平均大约持续 {l0} 步。先猜一猜，再播放训练过程。',
				vars: (_rt, at) => ({ lr: set(m(at, 'lr')), n: D.batch, max: D.maxLen, l0: lenText(rounds(at, 'meanLen')[0] ?? NaN) })
			},
			look: L('The top curve grows round by round: the average episode length so far. Below it, that round’s policy plays one episode from your start.', '上面的曲线一轮一轮地长出来：到目前为止的平均回合长度。下面是那一轮的策略从你的起点玩的一个回合。'),
			terms: ['optimizer']
		},
		{
			id: 'trained',
			source: 'train',
			at: 'last',
			view: scene(),
			provenance: 'live',
			title: L('The trained policy on your start', '训练好的策略，从你的起点出发'),
			narration: {
				en: 'Training took the average episode length from {l0} steps in round 0 to {l1} in round {last}; {full}. Here the trained policy plays from your start state with the same random draws as the episode you followed: p(left) at the start is now {p1} (it was {p0}), {first}, and {result}.',
				zh: '训练让平均回合长度从第 0 轮的 {l0} 步变成第 {last} 轮的 {l1} 步；{full}。这里训练好的策略从你的起始状态出发，用的随机数和你跟过的那个回合完全一样：起点的 p(左) 现在是 {p1}（原来是 {p0}），{first}，{result}。',
				vars: (rt, at): Record<string, TmplVal> => {
					const v = rounds(at, 'meanLen'), f = firstFull(v), P1 = paramsAt(at, at.index), P0 = paramsAt(at, 0);
					if (!P1 || !P0 || !v.length) return { l0: NaN };
					const s = startOf(rt.input), e1 = policyEpisode(P1, rt.input), len1 = e1.len, len0 = policyEpisode(P0, rt.input).len;
					const why = e1.end === 2 ? L('the cart left the track', '小车离开了轨道') : L('the pole fell', '杆倒了');
					return { l0: lenText(v[0]), l1: lenText(v[v.length - 1]), last: v.length - 1, p1: policy(P1, s).p, p0: policy(P0, s).p,
						full: f >= 0 ? L(`every episode lasted the full ${D.maxLen} steps for the first time in round ${f}`, `第 ${f} 轮第一次做到每个回合都撑满 ${D.maxLen} 步`) : L(`no round had all its episodes reach the ${D.maxLen}-step limit`, `没有哪一轮的所有回合都撑到 ${D.maxLen} 步上限`),
						first: pick(e1.left[0] === 1, L('its first push goes left', '第一推向左'), L('its first push goes right', '第一推向右')),
						result: len1 >= D.maxLen
							? L(`the pole stays up for the full ${len1} steps instead of ${len0}`, `杆立满了 ${len1} 步，而不是 ${len0} 步`)
							: len1 === len0
								? L(`the episode again ends after ${len1} step${len1 === 1 ? '' : 's'}, because ${why.en}`, `回合又在 ${len1} 步后结束，因为${why.zh}`)
								: L(`the episode ends after ${len1} steps instead of ${len0}, because ${why.en}`, `回合在 ${len1} 步后结束（原来是 ${len0} 步），因为${why.zh}`)
					};
				}
			},
			task: {
				prompt: L('Make it harder: in “Your input”, start the pole already falling, with θ̇ at 0.5 rad/s or more either way (or pick “Leaning left and falling”). Watch p(left) and the first push, and whether the pole is saved.', '加大难度：在“你的输入”里让杆一开始就在倒，把 θ̇ 设到 0.5 rad/s 或更大（往哪边都行，也可以选“向左倾斜并在倒下”）。看 p(左) 和第一推，以及杆能不能被救回来。'),
				done: (rt) => Math.abs(startOf(rt.input)[3]) >= 0.5,
				hint: L('θ̇ is the fourth slider; a new start trains the network again first.', 'θ̇ 是第四个滑块；换了起点，网络会先重新训练一遍。'),
				waiting: L('Not yet: θ̇ at the start is still below 0.5 rad/s either way.', '还没有：起始的 θ̇ 还不到 ±0.5 rad/s。')
			},
			look: { en: 'The first push under the numbers, then the angle curve: does it stay between the ±{lim}° lines?', zh: '先看数字下面的第一推，再看角度曲线：它是否一直在 ±{lim}° 两条线之间？', vars: () => LIMITS },
			terms: ['policy']
		},
		{
			id: 'seeds',
			source: 'seeds',
			at: 'last',
			view: { key: 'loss-curve', props: () => ({ metrics: ['len1', 'len2', 'len3', 'len4', 'len5'], smooth: 0, labels: Object.fromEntries([1, 2, 3, 4, 5].map((s) => [`len${s}`, L(`seed ${s}${s === 1 ? ' (the run above)' : ''}`, `种子 ${s}${s === 1 ? '（上面那次）' : ''}`)])) }) },
			provenance: 'live',
			kindOf: 'limits',
			title: L('How much was luck?', '有多少是运气？'),
			narration: {
				en: 'Here is the same training up to round {last} with four other seeds, which change only the random first weights and draws. At round {last} the five runs average {lo} to {hi} steps per episode; {reach}. REINFORCE learns from only {n} episodes per round, so luck in a few episodes swings each update. Practical methods use more episodes, or compare each return with a learned estimate (a critic) instead of the round’s mean.',
				zh: '这里是同样的训练，到第 {last} 轮为止，换了四个随机种子，变的只是随机的初始权重和随机抽取的数。在第 {last} 轮，五次训练的平均回合长度从 {lo} 步到 {hi} 步不等；{reach}。REINFORCE 每轮只从 {n} 个回合里学习，几个回合的运气就能让每次更新摇摆。实际使用的方法会用更多回合，或者拿每个回报和一个学出来的估计值（评论家，critic）比较，而不是和这一轮的平均值比较。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const runs = [1, 2, 3, 4, 5].map((s) => rounds(at, `len${s}`)), last = runs[0].length - 1;
					if (last < 0) return { last: NaN };
					const end = runs.map((v) => v[last]), f = runs.map(firstFull), ok = f.filter((x) => x >= 0);
					const reach = ok.length === 5
						? L(`every run had a round with all episodes at ${D.maxLen} steps, the first in round ${Math.min(...ok)} and the last in round ${Math.max(...ok)}`, `每次训练都有一轮做到所有回合撑满 ${D.maxLen} 步，最早在第 ${Math.min(...ok)} 轮，最晚在第 ${Math.max(...ok)} 轮`)
						: ok.length === 0
							? L(`none of the five has had a round yet with every episode at the ${D.maxLen}-step limit`, `五次训练都还没有哪一轮所有回合都撑满 ${D.maxLen} 步`)
							: L(`${5 - ok.length} of the five have not had a round yet with every episode at the ${D.maxLen}-step limit`, `五次里有 ${5 - ok.length} 次还没有哪一轮所有回合都撑满 ${D.maxLen} 步`);
					return { last, lo: Math.min(...end), hi: Math.max(...end), reach, n: D.batch };
				}
			},
			look: L('Five curves with the same recipe: how far apart are they? Move along the chart to read any round (the chart counts rounds as training steps).', '同一个配方的五条曲线：它们相差多远？在图上移动可以读出任意一轮（图上把轮叫作训练步）。'),
			terms: ['policy-gradient']
		}
	],
	// the presets: inputs every step must survive, besides the adapter's edge fixtures (steepest start, cart at the left end)
	fixtures: PRESETS.map((p) => ({ id: p.id, label: p.label, raw: p.raw }))
};

export default lesson;
