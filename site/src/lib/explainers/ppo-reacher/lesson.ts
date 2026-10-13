// ppo-reacher: the guided lesson, one idea per step: the arm and its reward, the Gaussian policy, one episode, the critic
// and the TD error, GAE, the batch, the probability ratio (predict first), the clipped objective, training, changing ε
// (predict and task), and targets the policy never practised (limits). Every number is a {placeholder} computed from
// frames that ran, and the stage shows it too; the other factual sentences are tested in claims.test.ts.
import type { At, I18n, LessonDef, Tensor, TmplVal } from '../../atlas/types/index.ts';
import { pick } from '../../atlas/lesson/say.ts';
import { minus } from '../../atlas/core/format.ts';
import { CTRL, DEFAULT_EPS, EVAL_UP, epsOf, L1, meanEpisode, TAU, targetOf, type Params } from './sim.ts';
import { ppo } from './trainer.ts';
import { evalTargets } from './views/arm/meta.ts';
import { mbOf, workedOf } from './views/clip/meta.ts';
import { epsText } from './views/eps/meta.ts';

const L = (en: string, zh: string): I18n => ({ en, zh });
const D = ppo.defaults, U = D.epochs * D.minibatches;
const m = (at: At, k: string) => at.frame?.metrics?.[k] ?? NaN;
/** a setting or a coordinate as written (0.05, −0.3), not rounded to 3 digits */
const set = (v: number) => (Number.isFinite(v) ? minus(String(v)) : NaN);
const frames = (at: At) => Array.from({ length: at.tape ? Math.min(at.tape.length, at.index + 1) : 0 }, (_, i) => at.tape!.frame(i));
const batchOf = (at: At, i = at.index) => (at.tape && i >= 0 && i < at.tape.length ? (at.tape.state(i) as { batch?: Record<string, Tensor> }).batch : undefined);
/** a metric at the snapshot frame of every PPO iteration up to the cursor (its batch, played by that iteration's policy) */
const snaps = (at: At, k: string) => frames(at).filter((f) => f.phase === 'snapshot' && (f.step ?? 0) % U === 0).map((f) => f.metrics?.[k] ?? NaN);
const paramsAt = (at: At) => ((at.tape && at.index >= 0 && at.index < at.tape.length ? (at.tape.state(at.index) as { params?: Params }).params : undefined) ?? null);
const mean = (v: number[]) => (v.length ? v.reduce((s, x) => s + x, 0) / v.length : NaN);
/** the forward frame of update j on a fine tape */
const fwd = (at: At, j: number) => frames({ ...at, index: (at.tape?.length ?? 0) - 1 }).findIndex((f) => f.phase === 'forward' && f.step === j);
const ratioRange = (at: At, j: number) => {
	const r = Array.from(batchOf(at, fwd(at, j))?.ratio?.data ?? []);
	return r.length ? { lo: Math.min(...r), hi: Math.max(...r) } : { lo: NaN, hi: NaN };
};
const epsName = (e: number) => (e >= 100 ? L('off, so nothing is clipped', '关闭，什么都不截断') : L(String(e), String(e)));
const arm = (props: Record<string, unknown> = {}) => ({ key: 'ppo-reacher/arm', props: () => props });

const lesson: LessonDef = {
	title: L('PPO: an arm learns to reach, with clipped updates', 'PPO：机械臂靠截断的更新学会够目标'),
	summary: L(
		'A Gaussian policy picks two torques for a two-link arm. Follow one episode, judge its steps with a critic and GAE, see how PPO reuses a batch and what its clip stops, then train, change ε and find where the policy fails.',
		'高斯策略为两节机械臂选出两个力矩。跟着一个回合走一遍，用评论家和 GAE 评判每一步，看 PPO 怎样反复使用一批数据、它的截断拦住了什么，然后训练、改变 ε，并找出策略失败的地方。'
	),
	steps: [
		{ id: 'arm', source: 'episode', at: 'first', view: arm(), provenance: 'live',
			title: L('An arm, a target and a reward', '机械臂、目标和奖励'),
			narration: {
				en: 'A robot arm with two links of {l} m each must bring its fingertip to a target: yours, at ({tx}, {ty}) m, {d} m from where the fingertip starts. The action is two torques, one per joint motor: any numbers, not a choice from a list. Every {tau} s the arm moves and earns a reward: minus the distance to the target, minus {c} × the squared torques. The policy sees six numbers: the gap to the target (dx, dy), the two joint angles and their speeds.',
				zh: '一条机械臂有两节，每节 {l} m，要把指尖移到目标上：你的目标在 ({tx}, {ty}) m，离指尖的起点 {d} m。动作是两个力矩，每个关节的电机一个：可以取任意数，而不是从清单里选。每过 {tau} s，机械臂动一下并得到奖励：指尖到目标距离的相反数，再减去 {c} × 力矩的平方和。策略看到六个数：到目标的差 (dx, dy)、两个关节角和它们的角速度。',
				vars: (_rt, at) => ({ l: set(L1), tx: set(m(at, 'tx')), ty: set(m(at, 'ty')), d: m(at, 'd'), tau: set(TAU), c: set(CTRL) })
			}, look: L('The shaded half disc is where targets go; the ring with a cross is yours, the dot the fingertip.', '阴影半圆是目标所在的区域；带十字的圆圈是你的目标，圆点是指尖。'), terms: ['state', 'reward', 'policy']
		},
		{ id: 'policy', source: 'episode', at: { phase: 'act', step: 0 }, view: arm(), provenance: 'live',
			title: L('A Gaussian policy picks two torques', '高斯策略选出两个力矩'),
			narration: {
				en: 'A network with {h} tanh neurons in one hidden layer turns the six numbers into the means of the two torques: μ = ({mu1}, {mu2}). Two more learned numbers, log σ = {ls}, set the width of each bell curve: σ = {s}. The policy draws z = ({z1}, {z2}) from a standard bell curve and acts a = μ + σ·z = ({a1}, {a2}). It also records log π(a | s) = {logp}, how likely that action was: PPO needs it later.',
				zh: '一个隐藏层有 {h} 个 tanh 神经元的网络，把这六个数变成两个力矩的均值：μ = ({mu1}, {mu2})。另外两个学出来的数 log σ = {ls}，决定每条钟形曲线有多宽：σ = {s}。策略从标准钟形曲线里抽出 z = ({z1}, {z2})，做出动作 a = μ + σ·z = ({a1}, {a2})。它还记下 log π(a | s) = {logp}，也就是这个动作有多大可能：PPO 之后要用到它。',
				vars: (_rt, at) => ({ h: D.hidden, mu1: m(at, 'mu1'), mu2: m(at, 'mu2'), ls: set(m(at, 'ls1')), s: m(at, 's1'), z1: m(at, 'z1'), z2: m(at, 'z2'), a1: m(at, 'a1'), a2: m(at, 'a2'), logp: m(at, 'logp') })
			}, look: L('The two bell curves: the solid line is the mean μ, the dashed line the torque actually sampled.', '看两条钟形曲线：实线是均值 μ，虚线是实际采样到的力矩。'), terms: ['gaussian-policy', 'gaussian-noise']
		},
		{ id: 'episode', source: 'episode', at: { phase: 'end' }, view: arm(), play: { from: 'first', to: { phase: 'end' }, phases: ['start', 'move', 'end'], interval: 60 }, provenance: 'live',
			title: L('One episode', '一个回合'),
			narration: {
				en: 'Sample, move, repeat: one episode is {T} steps ({secs} s). The fresh network has learned nothing yet, so the fingertip wanders: it started {d0} m from the target and ends {d} m away, and the rewards add up to {ret}. Every step is kept, with its state, torques, log π and reward: that record is all PPO learns from.',
				zh: '采样、移动、再来一遍：一个回合有 {T} 步（{secs} s）。新网络还什么都没学，所以指尖到处乱逛：它开始时离目标 {d0} m，最后离目标 {d} m，奖励加起来是 {ret}。每一步都被记下来：状态、力矩、log π 和奖励。PPO 能学习的就只有这份记录。',
				vars: (_rt, at) => ({ T: m(at, 'T'), secs: set(m(at, 'T') * TAU), d0: at.tape?.frame(0).metrics?.d ?? NaN, d: m(at, 'd'), ret: m(at, 'ret') })
			}, look: L('The fingertip’s path, and under it the distance to the target at every step.', '看指尖的轨迹，以及下面每一步离目标的距离。'), terms: ['episode']
		},
		{ id: 'value', source: 'episode', at: { phase: 'gae' }, view: { key: 'ppo-reacher/cells', props: () => ({ delta: true }) }, provenance: 'live',
			title: L('A critic guesses what is still to come', '评论家估计还会得到多少'),
			narration: {
				en: 'A second network, the critic V(s), guesses the discounted return still to come from a state (each later reward multiplied by one more γ = {g}). It is untrained too: at the start it guessed {v0}, but the rewards that followed, discounted, add up to {G0}. The TD error δ = r + γ·V(next state) − V(state) says how much better one step went than the critic expected. At the last step, t = {t} (counting from 0): δ = {r} + {g} × ({v1}) − ({v}) = {delta}.',
				zh: '第二个网络是评论家 V(s)，它估计从某个状态起还会得到的折扣回报（越往后的奖励多乘一次 γ = {g}）。它也没训练过：起点处它估计 {v0}，可是之后的奖励折扣后加起来是 {G0}。TD 误差 δ = r + γ·V(下一个状态) − V(这个状态)，表示这一步比评论家预期的好多少。最后一步 t = {t}（从 0 数起）：δ = {r} + {g} × ({v1}) − ({v}) = {delta}。',
				vars: (_rt, at) => {
					const fs = frames(at), g = m(at, 'gamma'), r = fs.filter((f) => f.phase === 'move').map((f) => f.metrics!.r);
					return { g: set(g), v0: fs.find((f) => f.phase === 'end')?.metrics?.v0 ?? NaN, G0: r.reduce((s, x, t) => s + x * g ** t, 0), t: m(at, 't'), r: m(at, 'r'), v1: m(at, 'v1'), v: m(at, 'v'), delta: m(at, 'delta') };
				}
			}, look: L('The rows r and V(s) are filled for every step; δ and A appear from the right as they are worked out.', 'r 和 V(s) 两行每一步都有；δ 和 A 从右边开始，算出一个填一个。'), terms: ['value-network', 'td-error', 'discount']
		},
		{ id: 'gae', source: 'episode', at: 'last', view: 'ppo-reacher/cells', play: { from: { phase: 'gae' }, to: 'last', phases: ['gae'], interval: 80 }, provenance: 'live',
			title: L('The advantage, by GAE', '用 GAE 算优势'),
			narration: {
				en: 'The advantage A says how much better a step went than the critic expected, counting what followed. GAE adds up the TD errors still to come, each later one multiplied by one more γλ = {gl} (λ = {lam}; a smaller λ trusts the critic more): A_t = δ_t + γλ·A_(t+1). Second to last step: A = {d1} + {gl} × ({a2}) = {a1}; first step: A = {a0}. {neg}',
				zh: '优势 A 表示：算上之后发生的事，这一步比评论家预期的好多少。GAE 把之后各步的 TD 误差加起来，越往后的多乘一次 γλ = {gl}（λ = {lam}；λ 越小，越相信评论家）：A_t = δ_t + γλ·A_(t+1)。倒数第二步：A = {d1} + {gl} × ({a2}) = {a1}；第一步：A = {a0}。{neg}',
				vars: (_rt, at): Record<string, TmplVal> => {
					const g = frames(at).filter((f) => f.phase === 'gae').map((f) => f.metrics!), T = g.length;
					if (T < 2) return { gl: NaN };
					const A = g.map((x) => x.A), below = A.filter((a) => a < 0).length;
					return {
						gl: set(g[0].gl), lam: set(g[0].lambda), d1: g[1].delta, a2: g[0].A, a1: g[1].A, a0: g[T - 1].A,
						neg: below === T ? L('Every A of this episode is below 0: the arm earned less than even the untrained critic expected.', '这个回合的每个 A 都小于 0：机械臂得到的比没训练过的评论家预期的还少。') : L(`${below} of the ${T} advantages are below 0.`, `${T} 个优势里有 ${below} 个小于 0。`)
					};
				}
			}, look: L('The A row fills from the right; under it, the formula of the step just worked out.', 'A 这一行从右往左填满；下面是刚算出的那一步的公式。'), terms: ['gae']
		},
		{ id: 'batch', source: 'iter', at: { phase: 'forward', step: 0 }, view: 'ppo-reacher/cells', provenance: 'live',
			title: L('A batch of episodes', '一批回合'),
			narration: {
				en: 'One episode is thin evidence, so a PPO iteration plays {E} episodes with the same policy: yours, then {more} to random targets above the shoulder line, {N} steps in all. The advantages are normalized over the batch, (A − mean) / standard deviation, with mean {mean} and standard deviation {sd}: {pos} of the {N} steps now count as better than usual. Then PPO cuts the batch into {M} minibatches of {n} and updates on each, {K} times over: {U} updates from one batch.',
				zh: '一个回合的证据太单薄，所以 PPO 的一次迭代用同一个策略玩 {E} 个回合：先是你的那个，再朝肩膀那条线以上的随机目标玩 {more} 个，一共 {N} 步。优势在这一批上做归一化：(A − 平均值) / 标准差，这里平均值是 {mean}，标准差是 {sd}；于是 {N} 步里有 {pos} 步算作比平常好。然后 PPO 把这一批切成 {M} 个各 {n} 步的小批量，逐个更新，一共来 {K} 遍：一批数据换来 {U} 次更新。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const b = batchOf(at);
					if (!b?.allAdv) return { E: NaN };
					const N = b.allAdv.data.length, E = b.epRet.data.length;
					return { E, more: E - 1, N, mean: b.advMean.data[0], sd: b.advSd.data[0], pos: Array.from(b.allAdv.data).filter((a) => a > 0).length, M: D.minibatches, n: N / D.minibatches, K: D.epochs, U };
				}
			}, look: L('Row 0 is your episode. Where are the orange steps, the ones better than usual?', '编号 0 的那一行是你的回合。橙色的格子（比平常好的步）在哪里？'), terms: ['batch', 'epoch']
		},
		{ id: 'ratio', source: 'iter', at: { phase: 'forward', step: U - 1 }, view: { key: 'ppo-reacher/clip', props: () => ({ clip: false }) }, provenance: 'live',
			title: L('How far has the policy moved?', '策略挪动了多远？'),
			predict: {
				id: 'ppo-reacher-clip',
				question: L('PPO keeps learning from this same batch for every update of the iteration. What does its clip prevent?', 'PPO 在这次迭代的每一次更新里都用同一批数据学习。它的截断防止的是什么？'),
				options: [L('A sample whose ratio has already passed 1 ± ε, the way its advantage wants, pulling the policy even further', '一个比率已经朝它的优势想要的方向越过 1 ± ε 的样本，继续把策略往更远处拉'), L('Advantages growing too large', '优势变得太大'), L('The policy trying new actions', '策略去尝试新的动作')],
				answer: 0,
				explain: L(
					'The objective min(r·A, clip(r, 1 − ε, 1 + ε)·A) goes flat once r passes 1 + ε for a sample with A > 0 (or 1 − ε for A < 0): that sample’s gradient is 0. The policy stops moving further on evidence an older policy collected; samples on the other side of 1 still pull it back.',
					'目标 min(r·A, clip(r, 1 − ε, 1 + ε)·A) 在 A > 0 的样本的 r 越过 1 + ε（A < 0 时是低于 1 − ε）之后就变平了：这个样本的梯度是 0。策略不再依据旧策略收集的证据继续往前挪；在 1 另一侧的样本仍然会把它往回拉。'
				)
			},
			narration: {
				en: 'Each update changes the policy, but the batch was played by the old one, π_old. The probability ratio r = π(a | s) / π_old(a | s) = e^(log π − log π_old) says how much more likely the current policy makes each recorded action. At update 0 every r is {r0}; one Adam step later they range from {lo1} to {hi1}, and at update {j} (counting from 0) from {lo} to {hi}. PPO compares each r with a band 1 ± ε, here ε = {eps}.',
				zh: '每次更新都会改变策略，可这批数据是旧策略 π_old 玩出来的。概率比率 r = π(a | s) / π_old(a | s) = e^(log π − log π_old) 表示：对每个记录下的动作，现在的策略让它的可能性变成了原来的多少倍。第 0 次更新时，每个 r 都是 {r0}；Adam 走一步之后，它们已经分布在 {lo1} 到 {hi1} 之间；到第 {j} 次更新（从 0 数起），分布在 {lo} 到 {hi} 之间。PPO 拿每个 r 和区间 1 ± ε 比较，这里 ε = {eps}。',
				vars: (_rt, at) => {
					const r0 = ratioRange(at, 0), r1 = ratioRange(at, 1), b = mbOf(batchOf(at));
					return { r0: r0.hi, lo1: r1.lo, hi1: r1.hi, j: b?.j ?? NaN, lo: b ? Math.min(...b.r) : NaN, hi: b ? Math.max(...b.r) : NaN, eps: b ? epsName(b.eps) : NaN };
				}
			}, look: L('Every dot is one sample of this minibatch: across, its ratio r; up, its advantage A.', '每个点是这个小批量里的一个样本：横向是它的比率 r，纵向是它的优势 A。'), terms: ['probability-ratio']
		},
		{ id: 'clip', source: 'iter', at: { phase: 'forward', step: U - 1 }, view: 'ppo-reacher/clip', provenance: 'live',
			title: L('The clipped objective', '截断目标'),
			narration: {
				en: 'PPO maximizes the mean of min(r·A, clip(r, 1 − ε, 1 + ε)·A); here ε = {eps}. While r stays within 1 ± ε both terms are equal and the sample pulls; once r passes the edge its advantage pushes toward, the clipped term is smaller, the min takes it and it stops changing: no pull. The ringed sample: r = {r}, A = {A}, so r·A = {s1}, clip(r)·A = {s2} and the min is {obj}: {verdict}. In this update {k} of {n} samples are clipped.',
				zh: 'PPO 要让 min(r·A, clip(r, 1 − ε, 1 + ε)·A) 的平均值尽量大；这里 ε = {eps}。只要 r 还在 1 ± ε 之内，两项相等，这个样本就在拉；一旦 r 朝它的优势推动的方向越过边界，截断的那一项更小，min 取到它，而它不再变化：不再拉动。圈出的样本：r = {r}，A = {A}，所以 r·A = {s1}，clip(r)·A = {s2}，min 是 {obj}：{verdict}。这次更新里 {n} 个样本有 {k} 个被截断。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const b = mbOf(batchOf(at));
					if (!b) return { eps: NaN };
					const w = workedOf(b);
					return { eps: epsName(b.eps), r: w.r, A: w.A, s1: w.s1, s2: w.s2, obj: w.obj, verdict: pick(w.clipped, L('clipped, its gradient is 0', '被截断，梯度为 0'), L('not clipped, it still pulls', '没有截断，仍在拉动')), k: b.clipped.reduce((s, x) => s + x, 0), n: b.n };
				}
			}, look: L('Shaded: the two regions where a sample stops pulling. Hollow dots are the clipped samples.', '阴影：样本不再拉动的两个区域。空心点是被截断的样本。'), terms: ['ppo-clip']
		},
		{ id: 'train', source: 'train', at: 'last', view: arm({ curve: true }), play: { from: 'first', to: 'last', phases: ['init', 'snapshot'], interval: 120, unit: L('iteration', '次迭代') }, provenance: 'live',
			title: L('Train, iteration after iteration', '一次又一次地迭代训练'),
			narration: {
				en: 'Training repeats that {last} times: play a batch with the current policy, work out GAE, then {U} clipped updates of both networks (the critic is pulled toward R = A + V). The average episode return goes from {r0} with the fresh policy to {r1} with the trained one, and σ {sig} from {s0} to {s1}. Without its random part (a = μ), the trained policy brings the fingertip to {d} m from your target. Cart-pole’s REINFORCE, by contrast, uses each batch once and has no critic.',
				zh: '训练把这个过程重复 {last} 次：用当前的策略玩一批回合，算出 GAE，再对两个网络做 {U} 次截断更新（评论家被拉向 R = A + V）。平均回合回报从新策略的 {r0} 变成训练好的策略的 {r1}，σ 从 {s0} {sig}到 {s1}。去掉随机的部分（a = μ），训练好的策略把指尖带到离你的目标 {d} m 的地方。相比之下，倒立摆的 REINFORCE 每批数据只用一次，也没有评论家。',
				vars: (rt, at): Record<string, TmplVal> => {
					const r = snaps(at, 'epRet'), s = snaps(at, 'sigma'), p = paramsAt(at);
					if (!r.length || !p) return { last: NaN };
					const s1 = s[s.length - 1];
					return { last: r.length - 1, U, r0: r[0], r1: r[r.length - 1], s0: s[0], s1, sig: pick(s1 < s[0], L('shrinks', '缩小'), L('grows', '增大')), d: meanEpisode(p, targetOf(rt.input), D.T).final };
				}
			}, look: L('The curve grows iteration by iteration; under it, that iteration’s policy reaches for your target without its random part.', '曲线一次迭代一次迭代地长出来；下面是那次迭代的策略去掉随机部分后伸向你的目标。'), terms: ['optimizer']
		},
		{ id: 'epsilon', source: 'pair', at: 'last', view: 'ppo-reacher/eps', provenance: 'live',
			title: L('Change ε, watch the stability', '改变 ε，观察稳定性'),
			predict: {
				id: 'ppo-reacher-eps',
				question: L('If you switch the clip off, what will the return curve do?', '如果把截断关掉，回报曲线会怎样？'),
				options: [L('Climb faster: nothing holds the updates back', '爬得更快：没有东西拦着更新了'), L('Swing up and down, and end below the ε = 0.2 run', '上下摇摆，最后低于 ε = 0.2 的那次训练'), L('Stay the same: ε only matters for a few samples', '不变：ε 只影响少数样本')],
				answer: 1,
				explain: L(
					'Without the clip every sample keeps pulling for all the updates of a batch, so one batch can move the policy a long way: in this training the average KL per iteration is ten times larger or more, and the return lurches instead of climbing.',
					'没有截断时，每个样本在一批数据的所有更新里都一直在拉，一批数据就能把策略挪得很远：在这次训练里，每次迭代的平均 KL 大了十倍以上，回报跌跌撞撞，而不是稳步上升。'
				)
			},
			narration: {
				en: 'ε sets how far the policy may move on a sample before that sample stops pulling. {runs} Measured as a KL divergence, one iteration moved your policy by {kl} on average ({kl0} with ε = {e0}); over the last {w} iterations the return averages {ret} ({ret0} with ε = {e0}).',
				zh: 'ε 决定：策略在一个样本上挪动多远之后，这个样本就不再拉动。{runs}用 KL 散度来量，你的策略平均每次迭代挪动 {kl}（ε = {e0} 时是 {kl0}）；最后 {w} 次迭代的平均回报是 {ret}（ε = {e0} 时是 {ret0}）。',
				vars: (rt, at): Record<string, TmplVal> => {
					const fs = frames(at).filter((f) => f.phase === 'step').map((f) => f.metrics!);
					if (!fs.length) return { eps: NaN };
					const w = Math.min(10, fs.length), tail = fs.slice(-w);
					const col = (v: Record<string, number>[], k: string) => v.map((x) => x[k]);
					const e = epsText(epsOf(rt.input)), same = epsOf(rt.input) === DEFAULT_EPS;
					const runs = same ? L(`For now both runs use ε = ${e.en}, so their curves lie on top of each other.`, `现在两次训练都用 ε = ${e.zh}，所以两条曲线重合在一起。`) : L(`Your run uses ε = ${e.en}; the other curve is the same training with ε = ${DEFAULT_EPS}.`, `你这次训练用 ε = ${e.zh}；另一条曲线是同样的训练，但 ε = ${DEFAULT_EPS}。`);
					return { runs, e0: set(DEFAULT_EPS), kl: mean(col(fs, 'kl')), kl0: mean(col(fs, 'kl0')), w, ret: mean(col(tail, 'epRet')), ret0: mean(col(tail, 'epRet0')) };
				}
			},
			task: {
				prompt: L('Change ε with the buttons in the stage: try a large one (1 or off) and a small one (0.05), and compare both charts each time.', '用舞台里的按钮改变 ε：试一个大的（1 或关闭），再试一个小的（0.05），每次都比较两张图。'),
				done: (rt) => epsOf(rt.input) !== DEFAULT_EPS,
				hint: L('Each choice trains both runs again from the start, which takes a few seconds.', '每次选择都会让两次训练从头再来，要等几秒钟。'),
				waiting: L('Not yet: ε is still 0.2.', '还没有：ε 还是 0.2。')
			}, look: L('The two return curves, then the KL: how far each iteration moved the policy (log scale; the charts count iterations as steps).', '先看两条回报曲线，再看 KL：每次迭代把策略挪了多远（对数坐标；图上把迭代记作“步”）。'), terms: ['kl-divergence']
		},
		{ id: 'limits', source: 'train', at: 'last', view: arm({ eval: true }), provenance: 'live', kindOf: 'limits',
			title: L('Targets it never practised', '它从没练过的目标'),
			narration: {
				en: 'Training only ever put targets above the shoulder line. Sent to {n} test targets there, the trained arm misses by {up} m on average; sent to their mirror images below the line, by {down} m. {lesson} Getting this far also took {steps} steps of practice, about {mins} minutes of arm time.',
				zh: '训练时目标只放在肩膀那条线以上。把训练好的机械臂派往那里的 {n} 个测试目标，平均还差 {up} m；派往它们在线下的镜像位置，平均差 {down} m。{lesson}练到这一步还用了 {steps} 步，大约是机械臂 {mins} 分钟的运动时间。',
				vars: (_rt, at): Record<string, TmplVal> => {
					const p = paramsAt(at), it = snaps(at, 'epRet').length - 1;
					if (!p) return { n: NaN };
					const e = evalTargets(p), avg = (up: boolean) => mean(e.filter((x) => x.up === up).map((x) => x.d)), steps = it * D.episodes * D.T;
					const lesson = pick(avg(false) > avg(true), L('A policy only knows the situations its episodes visited.', '策略只认识它的回合到过的情形。'), L('This policy has not even learned the targets it practised yet: train it to the end first.', '这个策略连练过的目标都还没学会：先把它训练到底。'));
					return { n: EVAL_UP.length, up: avg(true), down: avg(false), lesson, steps, mins: (steps * TAU) / 60 };
				}
			}, look: L('Solid rings are above the line, dashed rings below it; each line from a ring is a miss.', '实线圆圈在线以上，虚线圆圈在线以下；从圆圈引出的每条线，就是指尖差了多远。'), terms: ['out-of-distribution']
		}
	]
};

export default lesson;
