// egnn: the package manifest. A small E(n)-equivariant graph network (pipeline/egnn: 2 EGNN layers, 3,619 parameters)
// on the learner's own toy molecule, kinds A + C:
//   fwd    the exported EGNN traced on the atoms (positions and one-hot kinds; the local adapter, mol.ts molOf)
//   spin   the same network on the molecule turned by 0, 30, …, 330 degrees, one frame per turn (process.ts; main thread,
//          it holds the model): the energy stays put and the force on the followed atom turns with the molecule
// The lesson also runs the model (look.ts variant) on a turned and a mirrored copy. Measured numbers (test errors, the
// largest rotation and mirror differences over the test set) are in facts.json (pipeline/egnn/stats.py).
import type { ExplainerManifest } from '../../atlas/types/index.ts';
import { spinInput } from './process.ts';

const I = (en: string, zh: string) => ({ en, zh });

const manifest: ExplainerManifest = {
	slug: 'egnn',
	type: 'model',
	kinds: ['trace', 'process'],
	quality: 'reviewed',
	assets: { dir: 'models/egnn', auto: ['spec.json', 'weights.bin', 'facts.json'] },
	input: { adapter: 'points2d', presetsFromAdapter: true, options: { classes: 3 } },
	output: { view: 'egnn/mol', props: { mode: 'forces' } },
	sources: {
		fwd: { kind: 'trace' },
		spin: { kind: 'process', process: 'spin', input: spinInput, run: 'main', reference: false }
	},
	overview: { layout: 'stack' },
	page: {
		goals: [
			I('Say what an EGNN reads about a pair of atoms (their numbers and their squared distance) and what it never reads (where they sit, or which way they point).', '说出 EGNN 从一对原子那里读到什么（它们的数和距离平方），又从来读不到什么（它们在哪里、朝哪个方向）。'),
			I('Build one atom’s coordinate update from its pushes, (xᵢ − xⱼ) wᵢⱼ / (n − 1), one per other atom.', '用一个原子收到的推动 (xᵢ − xⱼ) wᵢⱼ / (n − 1)（每个其他原子一个）拼出它的坐标更新。'),
			I('Predict, and check with numbers, what turning the molecule does to the energy (nothing) and to the forces (they turn with it).', '预测并用数字验证：转动分子时，能量不变，力跟着一起转。'),
			I('Say why a GCN cannot see the shape of a molecule, and what an EGNN still cannot tell apart: a molecule and its mirror image.', '说出为什么 GCN 看不到分子的形状，以及 EGNN 仍然分不清的东西：一个分子和它的镜像。')
		],
		prequestion: {
			id: 'egnn-pre',
			question: I('A network predicts the energy of a molecule and a force arrow on every atom. You turn the whole molecule by 90°. What should happen?', '一个网络预测分子的能量，并给每个原子画出一个力的箭头。你把整个分子转 90°。应该发生什么？'),
			options: [I('Energy and arrows both stay exactly as they were', '能量和箭头都完全不变'), I('The energy stays; every arrow turns by 90° with the molecule', '能量不变；每个箭头跟着分子转 90°'), I('Both change: the network sees new coordinates', '两者都会变：网络看到的是新坐标')],
			answer: 1,
			explain: I('Turning a molecule does not change its physics, so the energy must stay; but the forces are arrows attached to the atoms, so they must turn with them. An EGNN gets both right by construction, not by training.', '转动分子不改变它的物理性质，所以能量应该不变；但力是附着在原子上的箭头，所以要跟着原子一起转。EGNN 靠结构本身做到这两点，而不是靠训练学来的。')
		},
		tryThis: [
			{
				id: 'ring',
				title: I('A ring of six', '六元环'),
				setup: { preset: 'ring' },
				predict: I('In the ring one atom sits a little too far out. Which way will its force arrow point?', '环里有一个原子离得稍远了些。它的力箭头会指向哪边？'),
				look: I('The arrow of that atom, and whether the dashed formula arrow agrees.', '那个原子的箭头，以及虚线的公式箭头是否同意。'),
				goto: { step: 'output' }
			},
			{
				id: 'pair',
				title: I('Two atoms', '两个原子'),
				setup: { preset: 'pair' },
				predict: I('With only two atoms, how many pushes does each atom get?', '只有两个原子时，每个原子收到几个推动？'),
				look: I('The push arrows of the followed atom.', '被跟踪原子的推动箭头。'),
				goto: { step: 'push' }
			}
		],
		quiz: [
			{
				q: I('Which of these does the message between atoms i and j depend on?', '原子 i 和 j 之间的消息依赖于下面哪一项？'),
				options: [I('The x coordinate of atom i', '原子 i 的 x 坐标'), I('The squared distance between i and j', 'i 和 j 之间的距离平方'), I('The angle of the line from i to j', '从 i 到 j 的连线的角度')],
				answer: 1,
				explain: I('The message network reads hᵢ, hⱼ and d²ᵢⱼ only; coordinates enter only as differences, in the pushes.', '消息网络只读 hᵢ、hⱼ 和 d²ᵢⱼ；坐标只以差的形式出现在推动里。')
			},
			{
				q: I('You mirror a molecule (flip it left to right). What does the EGNN’s energy do?', '你把一个分子左右翻转成镜像。EGNN 的能量会怎样？'),
				options: [I('It stays the same', '保持不变'), I('It changes sign', '变号'), I('It changes in an unpredictable way', '不可预测地改变')],
				answer: 0,
				explain: I('A mirror keeps every distance, so every message and the energy stay the same: the network cannot tell a molecule from its mirror image.', '镜像保持每个距离不变，所以每条消息和能量都不变：网络分不清一个分子和它的镜像。')
			}
		],
		misconception: {
			claim: I('To make a network rotation-proof you show it every molecule at many angles during training.', '想让网络不怕旋转，就得在训练时把每个分子转成许多角度给它看。'),
			truth: I('An EGNN needs no turned copies: it reads only distances and builds its arrows from differences of positions, so turning cannot change its energy, and its forces turn exactly with the molecule.', 'EGNN 不需要转过的副本：它只读距离，用位置之差拼出箭头，所以转动改变不了它的能量，它的力也恰好跟着分子一起转。'),
			step: 'spin'
		},
		lessonNote: I(
			'Ten steps: your atoms, the distances, one message, the summed messages, the pushes on one atom, the energy and forces, a full turn, turn it yourself, the comparison with a GCN, and the mirror image.',
			'十步：你的原子，距离，一条消息，消息之和，一个原子受到的推动，能量和力，转一整圈，自己转一转，与 GCN 的比较，以及镜像。'
		)
	},
	glossary: [
		{ id: 'equivariance', term: I('equivariance', '等变性'), def: I('An output changes in step with the input: turn the input and the output turns the same way.', '输出随输入同步变化：把输入转一下，输出也以同样的方式转动。'), related: ['invariance'] },
		{ id: 'invariance', term: I('invariance', '不变性'), def: I('An output stays the same when the input is changed in a certain way, such as turned or shifted.', '输入以某种方式改变（比如转动或平移）时，输出保持不变。'), related: ['equivariance'] },
		{ id: 'coordinate-update', term: I('coordinate update', '坐标更新'), def: I('In an EGNN, each atom moves by a sum of pushes along the lines to the other atoms, each scaled by a learned number.', '在 EGNN 中，每个原子沿着它与其他原子的连线被推动，每个推动乘上一个学来的数，再加起来就是它的移动量。'), related: ['message-passing'] },
		{ id: 'potential-energy', term: I('potential energy', '势能'), def: I('One number for a whole arrangement of atoms; each atom’s force points the way that lowers it fastest.', '描述一整组原子排布的一个数；每个原子受的力指向使它下降最快的方向。') }
	],
	credits: [
		{
			name: 'E(n) Equivariant Graph Neural Networks (Satorras, Hoogeboom and Welling, ICML 2021)',
			url: 'https://arxiv.org/abs/2102.09844',
			license: 'idea only (no code or data used)',
			usedFor: 'the EGNN layer: messages from atom features and squared distances, coordinate updates along relative positions, the node update',
			attribution: 'V. G. Satorras, E. Hoogeboom, M. Welling, “E(n) Equivariant Graph Neural Networks”'
		},
		{
			name: 'Morse potential (P. M. Morse, Physical Review 1929)',
			url: 'https://doi.org/10.1103/PhysRev.34.57',
			license: 'idea only (a textbook formula)',
			usedFor: 'the shape of the toy pair energy that scores the training molecules (made-up parameters, not real chemistry)',
			attribution: 'P. M. Morse, “Diatomic Molecules According to the Wave Mechanics. II. Vibrational Levels”'
		}
	],
	catalog: {
		hook: I('Turn a molecule you drew: the energy stays the same and every force arrow turns with it.', '转动你画的分子：能量保持不变，每个力箭头都跟着一起转。'),
		shows: I('Your atoms as positions and one-hot kinds on a full graph, the table of squared distances, one message worked out with its numbers, the summed messages and the node update, the pushes that make up one atom’s coordinate update, the energy and forces against a made-up pair formula with test errors, a full turn in 30° steps and a turn slider, a stretch that a GCN cannot see, and the mirror image an EGNN cannot tell apart.', '把你的原子写成全连接图上的位置和独热编码的种类、距离平方表、带数字算出的一条消息、消息之和与节点更新、拼成一个原子坐标更新的各个推动、与一个虚构的原子对公式对照的能量和力（附测试误差）、每次 30° 转一整圈和一个转动滑块、GCN 看不出的拉伸，以及 EGNN 分不清的镜像。'),
		runtime: I('Own JS forward pass (checked against PyTorch)', '自研 JS 前向计算（已与 PyTorch 核对）'),
		params: I('3,619 (3 → 16 kind embedding, two EGNN layers of 1,633, 16 → 16 → 1 energy head)', '3,619（3 → 16 的种类嵌入，两个各 1,633 个参数的 EGNN 层，16 → 16 → 1 的能量输出头）'),
		size: I('14.1 KB of weights', '权重 14.1 KB'),
		prereqs: ['gcn']
	}
};

export default manifest;
