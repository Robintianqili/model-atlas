// gnn-variants view 'agg': the member in focus and how each graph layer combines its friends (props.mode):
//   club     the whole club; the member in focus and its friendships highlighted (fwd trace)
//   gcn      GCN's fixed weights Â on the member's edges and its layer-1 row, worked out (fwd trace)
//   sage     GraphSAGE: the own half of W, the friends' average, + b, tanh (fwd trace)
//   gin      GIN: (1 + ε) × own + the friends' columns, the MLP, tanh; and the "same start" sum vs average (fwd trace)
//   gat      GAT layer 1 for the member (a gat frame): scores, softmax weights α, weighted messages, + b, tanh
//   side     the four rules' weights on the same edges and their rows after layer 1 (fwd trace)
//   embed    each network's 2-D embedding of the club, coloured by faction, with its head's dividing line (fwd trace)
//   acc      how many unlabelled members each network gets right, with the averages over random starts (facts)
//   drop     remove one friendship at a time: α against how far the member's score gap moves (a drop frame)
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { adjacency, KARATE, N, NAMES, VARIANTS } from '../../graph.ts';
import { focusOf, rightOf, weightOf } from '../../rules.ts';

type P = Omit<StageProps, 'ctx'>;
const both = (en: string, zh: string) => ({ en, zh });

export function describeAgg(p: P): I18n {
	const { i, nb, g } = focusOf(p.rt);
	const mode = String(p.props.mode ?? 'club');
	const head = both(`Member ${i} of the club (${N} members, ${g.edges.length} friendships) has ${nb.length} friends: ${nb.join(', ') || 'none'}.`, `俱乐部（${N} 个成员，${g.edges.length} 段朋友关系）里的成员 ${i} 有 ${nb.length} 个朋友：${nb.join('、') || '无'}。`);
	const m = p.at.frame?.metrics ?? {};
	const w = (v: (typeof VARIANTS)[number]) => [i, ...nb].map((j) => `${j}: ${weightOf(p.rt, v, j) === null ? 'own' : fmt3(weightOf(p.rt, v, j))}`).join(', ');
	let more = both('', '');
	if (['gcn', 'sage', 'gin'].includes(mode)) more = both(` ${NAMES[mode as 'gcn']} weights per sender: ${w(mode as 'gcn')}.`, ` ${NAMES[mode as 'gcn']} 给每个发送者的权重：${w(mode as 'gcn')}。`);
	if (mode === 'side') more = both(` Weights per sender: ${VARIANTS.map((v) => `${NAMES[v]} ${w(v)}`).join('; ')}.`, ` 每个发送者的权重：${VARIANTS.map((v) => `${NAMES[v]} ${w(v)}`).join('；')}。`);
	if (mode === 'gat' && p.at.frame) more = both(` GAT step “${p.at.frame.phase}”${m.j !== undefined ? ` for sender ${m.j}` : ''}${m.e !== undefined ? `: score ${fmt3(m.e)}` : ''}${m.a !== undefined ? `: weight ${fmt3(m.a)}` : ''}.`, ` GAT 步骤“${p.at.frame.phase}”${m.j !== undefined ? `，发送者 ${m.j}` : ''}${m.e !== undefined ? `：分数 ${fmt3(m.e)}` : ''}${m.a !== undefined ? `：权重 ${fmt3(m.a)}` : ''}。`);
	if (mode === 'embed' || mode === 'acc') more = both(` Unlabelled members on the right side: ${VARIANTS.map((v) => `${NAMES[v]} ${rightOf(p.rt, v).right} of ${rightOf(p.rt, v).rest}`).join(', ')}.`, ` 分到正确一侧的未标注成员：${VARIANTS.map((v) => `${NAMES[v]} ${rightOf(p.rt, v).rest} 人中 ${rightOf(p.rt, v).right} 人`).join('，')}。`);
	if (mode === 'drop' && p.at.frame) more = m.j !== undefined ? both(` Without the friendship with member ${m.j} (attention ${fmt3(m.a)} in layer 1, ${fmt3(m.a2)} in layer 2), the score gap moves by ${fmt3(m.d)}.`, ` 去掉和成员 ${m.j} 的朋友关系（注意力：第 1 层 ${fmt3(m.a)}，第 2 层 ${fmt3(m.a2)}）后，分数差变化了 ${fmt3(m.d)}。`) : m.topA !== undefined ? both(` Score gap ${fmt3(m.gap)}. Largest α in layer 1: member ${m.topA}; in layer 2: member ${m.topA2}; largest move: member ${m.topE} (${fmt3(m.dTopE)}).`, ` 分数差 ${fmt3(m.gap)}。第 1 层 α 最大：成员 ${m.topA}；第 2 层：成员 ${m.topA2}；变化最大：成员 ${m.topE}（${fmt3(m.dTopE)}）。`) : both(` Score gap ${fmt3(m.gap)}.`, ` 分数差 ${fmt3(m.gap)}。`);
	return { en: head.en + more.en, zh: head.zh + more.zh };
}

const meta = {
	key: 'gnn-variants/agg',
	title: { en: 'One member’s friends, combined four ways', zh: '一个成员的朋友，四种合并方式' },
	kinds: ['trace', 'process'],
	describe: describeAgg,
	async demo() {
		const adj = { shape: [N, N], data: Float32Array.from(adjacency(KARATE)) };
		const rt = fakeRuntime({ slug: 'gnn-variants', input: KARATE, trace: { order: ['adj'], t: { adj } } });
		return { rt, source: 'fwd', at: atFor(rt, 'fwd', 0), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'club' } };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
