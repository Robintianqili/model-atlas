// egnn: the guided lesson (10 steps). Every number is a {placeholder}: structure from the spec, live values from the
// trace of the learner's molecule, from the same model on turned, stretched and mirrored copies (look.ts variant) and
// from the 'spin' process; measured ones from facts.json ({fact:...}).
//   1 atoms → 2 squared distances → 3 one message → 4 the summed messages → 5 the pushes on one atom → 6 energy and
//   forces → 7 a full turn (predict) → 8 turn it yourself (task) → 9 a GCN cannot see shape (stretch) → 10 the mirror
//   image (limits).
import type { ExplainerRuntime, I18n, LessonDef, TmplVal } from '../../atlas/types/index.ts';
import { pick, plural } from '../../atlas/lesson/say.ts';
import { fmt3, fmtSetting, paren } from '../../atlas/core/format.ts';
import { focus, mol, pushes, STRETCH, T, variant, wrap } from './look.ts';
import { angleOf, energy, NMAX, NMIN, r0, type MolRaw } from './mol.ts';
import { STEP_DEG } from './process.ts';

type RT = ExplainerRuntime;
const I = (en: string, zh: string) => ({ en, zh });
const QUARTER = 90;
const NONE: Record<string, TmplVal> = {};
const ATOMS = { one: '{n} atom', many: '{n} atoms', zh: '{n} 个原子' };
const dim = (rt: RT, id: string, axis = -1) => {
	const s = rt.spec?.layers.find((l) => l.id === id)?.outShape;
	return s ? Number(s.at(axis)) : NaN;
};
const vec = (x: number, y: number) => `(${fmt3(x)}, ${fmt3(y)})`;
/** the atom nearest to the followed atom (the pair the distance and message steps work out) */
function partner(rt: RT) {
	const m = mol(rt), i = focus(rt);
	if (!m || m.n < 2) return i;
	let best = i === 0 ? 1 : 0;
	const d = (j: number) => Math.hypot(m.pos[2 * i] - m.pos[2 * j], m.pos[2 * i + 1] - m.pos[2 * j + 1]);
	for (let j = 0; j < m.n; j++) if (j !== i && d(j) < d(best)) best = j;
	return best;
}
/** the largest output (by size) of a layer's row for pair (i, j) or atom i */
function biggest(rt: RT, id: string, cell: number) {
	const t = T(rt, id);
	if (!t) return 0;
	const C = t.shape[t.shape.length - 1];
	let o = 0;
	for (let k = 1; k < C; k++) if (Math.abs(t.data[cell * C + k]) > Math.abs(t.data[cell * C + o])) o = k;
	return o;
}
/** the training molecules kept every pair at least CLOSE bond lengths apart (claim-tested on the training set) */
export const CLOSE = 0.8;
/** how the molecule differs from every training molecule: its atom count, or its closest pair (in bond lengths) */
export function unlike(m: { n: number; pos: ArrayLike<number>; kind: ArrayLike<number> }) {
	let q = Infinity, a = 0, b = 0;
	const at = (i: number, j: number) => Math.hypot(m.pos[2 * i] - m.pos[2 * j], m.pos[2 * i + 1] - m.pos[2 * j + 1]) / r0(m.kind[i], m.kind[j]);
	for (let i = 0; i < m.n; i++) for (let j = i + 1; j < m.n; j++) if (at(i, j) < q) (q = at(i, j)), (a = i), (b = j);
	return { count: m.n < NMIN || m.n > NMAX, close: q < CLOSE, q, a, b };
}
const ang = (rt: RT) => {
	const m = mol(rt), i = focus(rt);
	return m ? angleOf(m.f[2 * i], m.f[2 * i + 1]) : NaN;
};

const lesson: LessonDef = {
	title: I('EGNN: turn a molecule and the forces turn with it', 'EGNN：分子转动，力也跟着转'),
	summary: I(
		'Your atoms become positions and kinds, every pair exchanges a message built from its squared distance, each atom is pushed along the lines to the others, and the energy stays the same while the force arrows turn with the molecule.',
		'你的原子变成位置和种类，每一对原子交换一条由距离平方算出的消息，每个原子沿着与其他原子的连线被推动；分子转动时，能量保持不变，力的箭头跟着一起转。'
	),
	steps: [
		{
			id: 'atoms', source: 'fwd', at: 'first', view: { key: 'egnn/mol', props: () => ({ mode: 'atoms' }) }, provenance: 'live',
			title: I('Your molecule: positions and kinds', '你的分子：位置和种类'),
			narration: {
				en: 'Your molecule has {atoms}; each has one of three kinds (its colour) and a position x, y. The network reads two tables: the positions, {pos}, and the kinds as one-hot rows, {kind}. It treats the molecule as a graph in which every atom is joined to every other: {pairs}.',
				zh: '你的分子有 {atoms}；每个原子属于三类之一（看颜色），并有一个位置 x, y。网络读两张表：位置表 {pos}，以及用独热编码写成的种类表 {kind}。它把分子当作一张图，每个原子都和其他所有原子相连，一共 {pairs}。',
				vars: (rt) => {
					const m = mol(rt);
					if (!m) return NONE;
					const p = m.n * (m.n - 1) / 2;
					return { atoms: plural(m.n, ATOMS), pos: `${m.n} × 2`, kind: `${m.n} × 3`, pairs: plural(p, { one: '{n} pair', many: '{n} pairs', zh: '{n} 对' }) };
				}
			},
			look: I('The faint lines are the pairs: every atom talks to every other atom.', '浅色的线就是原子对：每个原子都和其他每个原子交流。'),
			terms: ['graph', 'one-hot']
		},
		{
			id: 'distances', source: 'fwd', at: { layer: 'L1.d2' },
			select: (rt) => ({ layer: 'L1.d2', unit: [focus(rt), partner(rt)] }),
			view: 'layer.tensor', provenance: 'live',
			title: I('What a layer sees: squared distances', '每一层看到的：距离平方'),
			narration: {
				en: 'Each layer first computes the squared distance d² of every pair; turning or shifting the whole molecule leaves every d² the same. {pairSay} The GCN on the gcn page read only who is joined to whom; this network also reads how far apart the atoms are, but never which way.',
				zh: '每一层先算出每一对原子的距离平方 d²；把整个分子转动或平移，每个 d² 都不变。{pairSay}GCN 页面里的网络只读谁和谁相连；这个网络还读原子之间相距多远，但从来不读朝哪个方向。',
				vars: (rt) => {
					const m = mol(rt), i = focus(rt), j = partner(rt), d = T(rt, 'L1.d2');
					if (!m || !d) return NONE;
					const dx = m.pos[2 * i] - m.pos[2 * j], dy = m.pos[2 * i + 1] - m.pos[2 * j + 1], d2 = fmt3(d.data[i * m.n + j]);
					return {
						pairSay: pick(m.n > 1, I(`For atoms ${i} and ${j} (counting from 0): (${fmt3(dx)})² + (${fmt3(dy)})² = ${d2}.`, `对原子 ${i} 和 ${j}（从 0 数起）：(${fmt3(dx)})² + (${fmt3(dy)})² = ${d2}。`), I('With one atom there is no pair: the table is a single 0.', '只有一个原子时没有原子对：这张表只有一个 0。'))
					};
				}
			},
			look: I('The table is symmetric and 0 on its diagonal; the outlined cell is the pair worked out in the text.', '这张表左右对称，对角线上是 0；带框的格子就是正文里算的那一对。'),
			terms: ['invariance']
		},
		{
			id: 'message', source: 'fwd', at: { layer: 'L1.e1' },
			select: (rt) => {
				const m = mol(rt), i = focus(rt), j = partner(rt);
				return { layer: 'L1.e1', unit: [i, j, m ? biggest(rt, 'L1.e1', i * m.n + j) : 0] };
			},
			view: 'zoom.weighted-sum', provenance: 'live',
			title: I('One message, from h and d²', '一条消息：来自 h 和 d²'),
			narration: {
				en: 'First each atom’s kind became {F} numbers, h (an embedding). The message network reads {C} numbers per pair: {pairOf}. In the stage, its output {o} (counting from 0) is a weighted sum of them plus a bias, {y}. Only one input carries geometry: d² × its weight = {d2} × {w} = {p}.',
				zh: '在这之前，每个原子的种类已经变成 {F} 个数，记作 h（嵌入）。消息网络对每一对读入 {C} 个数：{pairOf}。画面里是它的 {o} 号输出（从 0 数起）：这些数的加权和再加上偏置，等于 {y}。只有一个输入带着几何信息：d² × 它的权重 = {d2} × {w} = {p}。',
				vars: (rt) => {
					const m = mol(rt), u = rt.sel.unit ?? [], W = rt.model?.weights['L1.e1.weight']?.data, e = T(rt, 'L1.e1'), d = T(rt, 'L1.d2');
					if (!m || !W || !e || !d) return NONE;
					const [i, j, o] = u, C = dim(rt, 'L1.edge'), M = dim(rt, 'L1.e1');
					const d2 = d.data[i * m.n + j], w = W[o * C + C - 1];
					const pairOf = pick(m.n > 1, I(`h of atom ${i}, h of atom ${j} and their d²`, `原子 ${i} 的 h、原子 ${j} 的 h，以及它们的 d²`), I(`with one atom, the only pair is atom ${i} with itself, so its h twice and d² = 0`, `只有一个原子时，唯一的一对是原子 ${i} 和它自己，所以是两份它的 h 和 d² = 0`));
					return { F: dim(rt, 'embed'), C, i, j, o, y: e.data[(i * m.n + j) * M + o], d2, w: paren(w), p: d2 * w || 0, pairOf };
				}
			},
			look: I('Scroll the list to its last row: d² times its weight.', '把列表滚到最后一行：d² 乘以它的权重。'),
			terms: ['message-passing', 'embedding', 'mlp']
		},
		{
			id: 'sum', source: 'fwd', at: { layer: 'L1.agg' },
			select: (rt) => {
				const i = focus(rt);
				return { layer: 'L1.agg', unit: [i, biggest(rt, 'L1.agg', i)] };
			},
			view: 'zoom.reduce', provenance: 'live',
			title: I('Each atom adds up its messages', '每个原子把收到的消息加起来'),
			narration: {
				en: 'The message network has two layers, each followed by the activation function SiLU, and every message is {M} numbers. Atom {i} adds up the messages from the {others}: number {o} of the sum (counting from 0) is {y}. A node network then reads h and this sum, and adds its output to h, giving the atom new numbers. A sum ignores the order of the atoms, and none of it depends on direction.',
				zh: '消息网络有两层，每层后面接激活函数 SiLU，每条消息是 {M} 个数。原子 {i} 把来自{others}的消息加起来：和的 {o} 号分量（从 0 数起）是 {y}。接着，节点网络读入 h 和这个和，把它的输出加回 h，原子就有了新的数。求和不管原子的顺序，整个过程也不依赖方向。',
				vars: (rt) => {
					const m = mol(rt), u = rt.sel.unit ?? [], a = T(rt, 'L1.agg');
					if (!m || !a) return NONE;
					const M = dim(rt, 'L1.agg'), [i, o] = u;
					return { M, i, o, y: a.data[i * M + o], others: plural(m.n - 1, { one: 'other {n} atom', many: 'other {n} atoms', zh: '其他 {n} 个原子' }) };
				}
			},
			look: I('Each line of the sum is one message from another atom.', '和里的每一行都是来自另一个原子的一条消息。'),
			terms: ['activation-function', 'aggregation', 'residual-connection']
		},
		{
			id: 'push', source: 'fwd', at: { layer: 'L1.push' }, view: { key: 'egnn/mol', props: () => ({ mode: 'push' }) }, provenance: 'live',
			title: I('The coordinate update: pushes along the pairs', '坐标更新：沿着原子对推动'),
			narration: {
				en: 'Each message also gives one number w, and atom i = {i} is pushed along the line to every other atom j by (xᵢ − xⱼ) × w / (n − 1), where n = {n} is the number of atoms. {strong} Built from differences of positions, these arrows turn when the molecule turns.',
				zh: '每条消息还给出一个数 w。原子 i = {i} 沿着它与每个其他原子 j 的连线被推一下，推动量是 (xᵢ − xⱼ) × w / (n − 1)，其中 n = {n} 是原子个数。{strong}这些箭头由位置之差拼成，所以分子转动时，它们也跟着转。',
				vars: (rt) => {
					const i = focus(rt), p = pushes(rt, i);
					if (!p) return NONE;
					if (!p.terms.length) return { i, n: p.n, strong: I('With one atom there is nothing to push against, so the push is 0.', '只有一个原子时，没有可以推它的原子，所以推动是 0。') };
					const s = p.terms.reduce((b, q) => (Math.hypot(q.x, q.y) > Math.hypot(b.x, b.y) ? q : b));
					const away = s.w > 0;
					if (p.terms.length === 1) return { i, n: p.n, strong: I(`The only push comes from atom ${s.j}, ${away ? 'away from it' : 'toward it'}: ${vec(s.x, s.y)}, the arrow that moves atom ${i} in this layer.`, `唯一的推动来自原子 ${s.j}，方向${away ? '远离' : '朝向'}它：${vec(s.x, s.y)}，也就是这一层里移动原子 ${i} 的箭头。`) };
					return { i, n: p.n, strong: I(`The strongest push comes from atom ${s.j}, ${away ? 'away from it' : 'toward it'}: ${vec(s.x, s.y)}. All the pushes add up to ${vec(p.x, p.y)}, the arrow that moves atom ${i} in this layer.`, `最强的推动来自原子 ${s.j}，方向${away ? '远离' : '朝向'}它：${vec(s.x, s.y)}。所有推动加起来是 ${vec(p.x, p.y)}，也就是这一层里移动原子 ${i} 的箭头。`) };
				}
			},
			look: I('Thin dashed arrows: one push per other atom; the thick arrow is their sum.', '细虚线箭头：每个其他原子一个推动；粗箭头是它们的和。'),
			terms: ['coordinate-update']
		},
		{
			id: 'output', source: 'fwd', at: { layer: 'energy' }, view: { key: 'egnn/mol', props: () => ({ mode: 'forces' }) }, provenance: 'live',
			title: I('Energy and forces', '能量和力'),
			narration: {
				en: 'Layer 2 repeats this on the moved positions; each atom’s force is its whole move over both layers, trained to match a made-up pair formula. The energy, one share per atom added up, is {E}; the formula gives {trueE}. On {fact:testN} test molecules the energy was off by {fact:energyMAE} on average (guessing the mean: {fact:energyMAD}), each force number by {fact:forceMAE} (guessing 0: {fact:forceMAD}).{warn}',
				zh: '第 2 层在移动后的位置上把这一切再做一遍；每个原子受的力就是它在两层里的总移动量，训练时要它去对上一个虚构的原子对公式。能量由每个原子贡献一份，加起来是 {E}；公式给出的是 {trueE}。在 {fact:testN} 个测试分子上，能量平均差 {fact:energyMAE}（总猜平均值则差 {fact:energyMAD}），力的每个分量平均差 {fact:forceMAE}（总猜 0 则差 {fact:forceMAD}）。{warn}',
				vars: (rt) => {
					const m = mol(rt);
					if (!m) return NONE;
					// one reason, the closest pair first: what makes this molecule unlike every training molecule (claim-tested)
					const u = unlike(m);
					const warn = u.close ? I(` Careful: atoms ${u.a} and ${u.b} sit ${fmt3(u.q)} bond lengths apart, closer than any training pair (${CLOSE} or more): this answer is a guess.`, `小心：原子 ${u.a} 和 ${u.b} 只相距 ${fmt3(u.q)} 个键长，比任何训练分子里的原子对都近（训练时至少 ${CLOSE} 个）；这个答案只是猜测。`)
						: u.count ? I(` Careful: your molecule has ${m.n === 1 ? '1 atom' : `${m.n} atoms`}, and training molecules had ${NMIN} to ${NMAX}: this answer is a guess.`, `小心：你的分子有 ${m.n} 个原子，而训练分子只有 ${NMIN} 到 ${NMAX} 个；这个答案只是猜测。`) : I('', '');
					return { E: m.E, trueE: m.trueE, warn };
				}
			},
			look: I('Solid arrows are the network’s forces, dashed ones the formula’s.', '实线箭头是网络给出的力，虚线箭头是公式给出的力。'),
			terms: ['potential-energy']
		},
		{
			id: 'spin', source: 'spin', at: 'last', play: { from: 'first', to: 'last', interval: 700, unit: I('turn', '次转动') }, view: { key: 'egnn/mol', props: () => ({ mode: 'spin' }) },
			provenance: 'live',
			title: I('A full turn', '转一整圈'),
			predict: {
				id: 'egnn-spin',
				question: I('As the molecule turns all the way round, what will the energy and the force on the ringed atom do?', '分子转一整圈时，能量和带圈原子受的力会怎样？'),
				options: [I('The energy stays the same; the arrow turns with the molecule', '能量不变；箭头跟着分子转'), I('Both stay exactly the same', '两者都完全不变'), I('The energy wobbles as the coordinates change', '坐标在变，能量也随之起伏')],
				answer: 0,
				explain: I('Every distance is the same at every turn, so every message, every w and the energy are the same. Each push (xᵢ − xⱼ) × w turns with the line it lies on, so their sum, the force, turns by exactly the same angle as the molecule.', '每次转动后，每个距离都不变，所以每条消息、每个 w 以及能量都不变。每个推动 (xᵢ − xⱼ) × w 都随它所在的连线一起转，所以它们的和，也就是力，转过的角度和分子完全一样。')
			},
			narration: {
				en: 'The page turns your molecule by {step}° at a time, a full circle, and runs the same network on each turned copy: {turns}. It records the energy and the direction of the force on atom {i}, the ringed one.',
				zh: '页面每次把你的分子转 {step}°，转满一整圈，并在每个转过的副本上运行同一个网络：一共 {turns}。它记下能量，以及带圈的原子 {i} 受的力的方向。',
				vars: (rt) => ({ step: STEP_DEG, turns: plural(360 / STEP_DEG, { one: '{n} run', many: '{n} runs', zh: '{n} 次' }), i: focus(rt) })
			},
			look: I('Read down the table: one column stays put, the other climbs with the turn.', '顺着表往下看：一列保持不动，另一列随着转角增加。'),
			terms: ['equivariance', 'invariance']
		},
		{
			id: 'turn', source: 'fwd', at: { layer: 'force' }, view: { key: 'egnn/mol', props: () => ({ mode: 'turn' }) }, provenance: 'live',
			title: I('Turn it yourself', '自己转一转'),
			narration: {
				en: 'Turned a quarter circle, {eSay}, and {fSay}. Turning the input turns every output arrow; that is equivariance. Right now the force on atom {i} points at {ang}°.',
				zh: '转四分之一圈后，{eSay}；{fSay}。输入转动，每个输出箭头也跟着转：这就是等变性。现在，原子 {i} 受的力指向 {ang}°。',
				vars: (rt) => {
					const v = variant(rt, QUARTER);
					if (!v) return NONE;
					const q = QUARTER;
					return {
						eSay: v.dE === 0 ? I('the energy does not change at all', '能量完全不变') : I(`the energy changes by ${fmt3(v.dE)}, which is float32 rounding`, `能量只变了 ${fmt3(v.dE)}，这只是 float32 的舍入误差`),
						fSay: v.df === 0 ? I(`the forces are exactly your forces turned by ${q}°`, `力恰好等于你原来的力转 ${q}°`) : I(`the forces match your forces turned by ${q}° to within ${fmt3(v.df)}`, `力和你原来的力转 ${q}° 之后相比，差别不超过 ${fmt3(v.df)}`),
						i: focus(rt),
						ang: ang(rt)
					};
				}
			},
			task: {
				prompt: I('Use the slider in the stage to turn the molecule until the force on the ringed atom points straight up, within 10°.', '用画面里的滑块转动分子，直到带圈原子受的力指向正上方，误差在 10° 以内。'),
				done: (rt) => {
					const a = ang(rt);
					return Number.isFinite(a) && Math.abs(wrap(a - 90)) <= 10 && !!(rt.input as MolRaw | null)?.turn;
				},
				waiting: I('Not yet: keep turning.', '还没有：继续转。'),
				doneText: I('It points up, and the energy has not moved.', '它指向上方了，而能量一点没变。')
			},
			look: I('The dotted arrow marks straight up; watch the energy line under the drawing.', '点线箭头标出正上方；留意图下方那行能量。'),
			terms: ['equivariance']
		},
		{
			id: 'gcn', source: 'fwd', at: { layer: 'energy' }, view: { key: 'egnn/mol', props: () => ({ mode: 'stretch' }) }, provenance: 'live',
			title: I('Why not a GCN?', '为什么不用 GCN？'),
			narration: {
				en: 'In the stage, your molecule is stretched so that every distance grows by a factor of {s}. The formula’s energy goes from {E0} to {E1}; {follows} {P0} to {P1}. A GCN on the same full graph gets the same input for both shapes, the same {atoms}, all joined, so it must give the same answer: it cannot see shape at all.',
				zh: '画面里，你的分子被拉伸，每个距离都变成原来的 {s} 倍。公式算出的能量从 {E0} 变成 {E1}；{follows}从 {P0} 到 {P1}。在同一张全连接图上，GCN 对这两种形状得到的输入完全相同：同样的 {atoms}，两两相连，所以它只能给出同样的答案；它根本看不到形状。',
				vars: (rt) => {
					const m = mol(rt), v = variant(rt, 'stretch');
					if (!m || !v) return NONE;
					const E1 = energy(v.pos, m.kind).E;
					const same = Math.sign(E1 - m.trueE) === Math.sign(v.E - m.E);
					return { s: fmtSetting(STRETCH), E0: m.trueE, E1, P0: m.E, P1: v.E, atoms: plural(m.n, ATOMS), follows: pick(same, I('this network moves the same way:', '这个网络也朝同一方向变：'), I('this network moves the other way:', '这个网络却朝相反方向变：')) };
				}
			},
			look: I('Compare the two energy lines under the drawings.', '比较两幅图下面的两行能量。'),
			terms: ['graph', 'adjacency-matrix']
		},
		{
			id: 'mirror', source: 'fwd', at: { layer: 'energy' }, view: { key: 'egnn/mol', props: () => ({ mode: 'mirror' }) }, provenance: 'live', kindOf: 'limits',
			title: I('The mirror image', '镜像'),
			narration: {
				en: '{can} A mirror keeps every distance, so the network gives {Em} against {E}, a difference of {dE}, and its forces simply flip (largest mismatch {df}). It can never give a molecule and its mirror image different energies; that would need an input that changes sign in a mirror.{formula}',
				zh: '{can}镜像保持每个距离不变，所以网络给出 {Em}，原来是 {E}，差 {dE}；它的力也只是跟着翻转（最大差 {df}）。它永远不会给一个分子和它的镜像不同的能量；要做到这一点，需要一个在镜像下会变号的输入。{formula}',
				vars: (rt) => {
					const m = mol(rt), v = variant(rt, 'mirror');
					if (!m || !v) return NONE;
					// a mirror image the page cannot reach by turning: true unless the molecule is itself symmetric
					const chiral = mirrorDiffers(m.pos, m.kind);
					return {
						can: pick(chiral, I('Flipped left to right, your molecule cannot be turned back onto itself in the plane.', '把你的分子左右翻转之后，在平面内怎么转都转不回原来的样子。'), I('Flipped left to right, your molecule could also be reached by a turn, because it is symmetric.', '把你的分子左右翻转之后，因为它是对称的，转一下也能得到同样的样子。')),
						Em: v.E, E: m.E, dE: v.dE, df: v.df, formula: pick(Math.abs(energy(v.pos, m.kind).E - m.trueE) <= 1e-6 * Math.max(1, Math.abs(m.trueE)), I(` The made-up formula reads only distances too and gives ${fmt3(m.trueE)} for both, so on this toy task the limit does no harm.`, `虚构的公式也只读距离，两者都给出 ${fmt3(m.trueE)}，所以在这个玩具任务里，这个局限没有坏处。`), I('', ''))
					};
				}
			},
			look: I('Compare the energies under the two drawings, and the arrows.', '比较两幅图下面的能量和箭头。'),
			terms: ['equivariance']
		}
	]
};

/** true when no turn about the centre maps the mirrored atoms onto atoms of the same kind (1° steps, within 0.05 units) */
export function mirrorDiffers(pos: ArrayLike<number>, kind: ArrayLike<number>): boolean {
	const n = pos.length / 2;
	if (n < 3) return false;
	let cx = 0, cy = 0;
	for (let i = 0; i < n; i++) (cx += pos[2 * i] / n), (cy += pos[2 * i + 1] / n);
	const P = Array.from({ length: n }, (_, i) => [pos[2 * i] - cx, pos[2 * i + 1] - cy]);
	const Q = P.map(([x, y]) => [-x, y]);
	for (let d = 0; d < 360; d++) {
		const c = Math.cos((d * Math.PI) / 180), s = Math.sin((d * Math.PI) / 180);
		if (Q.every(([x, y], a) => P.some(([u, v], b) => kind[a] === kind[b] && Math.hypot(c * x - s * y - u, s * x + c * y - v) < 0.05))) return false;
	}
	return true;
}

export default lesson;
