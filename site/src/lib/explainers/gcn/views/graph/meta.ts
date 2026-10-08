// gcn view 'graph': the club as a node-link drawing, in six modes (props.mode); the drawing is View.svelte.
//   adj     the drawing linked to the adjacency table A (fwd trace); click a member to focus, a square to edit a friendship
//   mp      message passing into the member in focus (an mp frame): self-loop, Â, X Wᵀ, the messages, the sum, tanh
//   hops    who the member in focus hears from after two layers (fwd trace)
//   embed   every member at its 2-D embedding (act2), coloured by faction, with the head's dividing line (fwd trace)
//   pick    the network retrained live from the labelled members (a train frame); click members to label them
//   smooth  over-smoothing: each member's lean after k rounds of averaging (a smooth frame)
import type { I18n, StageProps, Tape, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { adjacency, forwardGcn, graphOf, hops, KARATE, LABELS, N, neighbours, normalize, sorted, type Params } from '../../graph.ts';
import { mp } from '../../process.ts';

type P = Omit<StageProps, 'ctx'>;
/** Members whose two layer-2 numbers are both below −0.8 or both above 0.8 (they pile up near the embed square's corners). */
export const cornerCount = (emb: ArrayLike<number>) =>
	Array.from({ length: N }, (_, i) => [emb[i * 2], emb[i * 2 + 1]]).filter(([a, b]) => (a < -0.8 && b < -0.8) || (a > 0.8 && b > 0.8)).length;
/** The weights of the latest snapshot at or before frame i of a train tape. */
export function paramsAt(tape: Tape | null | undefined, i: number): { params: Params; step: number } | null {
	for (let k = Math.min(i, (tape?.length ?? 0) - 1); k >= 0; k--) {
		const s = tape!.state(k) as { params?: Params };
		if (s.params) return { params: s.params, step: tape!.frame(k).step ?? 0 };
	}
	return null;
}
/** The retrained network on the learner's graph: scores, embedding and how many unlabelled members it sorts right. */
export function retrained(p: Pick<P, 'rt' | 'at'>) {
	const w = paramsAt(p.at.tape, p.at.index);
	if (!w) return null;
	const g = graphOf(p.rt.input);
	const f = forwardGcn(w.params, normalize(adjacency(g), N).ahat);
	return { ...f, ...sorted(f.z, g.labelled), step: w.step, g };
}
const list = (v: number[]) => (v.length ? v.join(', ') : '–');

export function describeGraph(p: P): I18n {
	const g = graphOf(p.rt.input);
	const mode = String(p.props.mode ?? 'adj');
	const nb = neighbours(g, g.focus);
	const head = { en: `The club: ${N} members, ${g.edges.length} friendships. Member ${g.focus} is in focus (friends: ${list(nb)}).`, zh: `俱乐部：${N} 个成员，${g.edges.length} 段朋友关系。关注成员 ${g.focus}（朋友：${list(nb)}）。` };
	const m = p.at.frame?.metrics ?? {};
	const T = p.rt.sources.fwd?.trace?.t;
	let more: I18n = { en: '', zh: '' };
	const ph = mode === 'mp' && p.at.frame ? (mp.phases[p.at.frame.phase ?? '']?.label ?? { en: String(p.at.frame.phase), zh: String(p.at.frame.phase) }) : null;
	if (ph) more = { en: ` Step “${ph.en}”${m.j !== undefined ? `: the message from member ${m.j} (Â = ${fmt3(m.a)})` : ''}.`, zh: ` 步骤“${ph.zh ?? ph.en}”${m.j !== undefined ? `：来自成员 ${m.j} 的消息（Â = ${fmt3(m.a)}）` : ''}。` };
	if (mode === 'hops') {
		const d = hops(g, g.focus);
		const v = T?.act2 ? `(${fmt3(T.act2.data[g.focus * 2])}, ${fmt3(T.act2.data[g.focus * 2 + 1])})` : '…';
		more = { en: ` ${d.filter((x) => x === 1 || x === 2).length} members are one or two steps away. Layer 2 gives it ${v}.`, zh: ` 有 ${d.filter((x) => x === 1 || x === 2).length} 个成员在一到两步之内。第 2 层给它的数是 ${v}。` };
	}
	if (mode === 'embed' && T?.head) {
		const s = sorted(T.head.data, KARATE.labelled);
		const c = T.act2 ? cornerCount(T.act2.data) : 0;
		more = { en: ` In the 2-D embedding, ${s.right} of ${s.rest} unlabelled members are on their own faction's side (wrong: ${list(s.wrong)}); ${c} members sit near a corner.`, zh: ` 在二维嵌入里，${s.rest} 个未标注成员中有 ${s.right} 个在自己派别的一侧（错的：${list(s.wrong)}）；有 ${c} 个成员挤在角落附近。` };
	}
	if (mode === 'pick') {
		const r = retrained(p);
		more = r ? { en: ` Labelled: ${list(g.labelled)}. After ${r.step} steps the retrained network sorts ${r.right} of ${r.rest} others right.`, zh: ` 已标注：${list(g.labelled)}。训练 ${r.step} 步后，重新训练的网络把其余 ${r.rest} 人中的 ${r.right} 人分对了。` } : { en: ' Training…', zh: ' 训练中…' };
	}
	if (mode === 'smooth' && p.at.frame) more = { en: ` After ${m.k} rounds of averaging, the cosine similarity of Mr. Hi's and the Officer's rows is ${fmt3(m.leaders)}.`, zh: ` 平均 ${m.k} 轮之后，Mr. Hi 和干事的行的余弦相似度是 ${fmt3(m.leaders)}。` };
	return { en: head.en + more.en, zh: head.zh + more.zh };
}

const meta = {
	key: 'gcn/graph',
	title: { en: 'The club as a graph', zh: '把俱乐部画成图' },
	kinds: ['trace', 'train', 'process'],
	describe: describeGraph,
	async demo() {
		const adj = { shape: [N, N], data: Float32Array.from(adjacency(KARATE)) };
		const rt = fakeRuntime({ slug: 'gcn', input: KARATE, trace: { order: ['adj'], t: { adj } } });
		return { rt, source: 'fwd', at: atFor(rt, 'fwd', 0), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'adj', faction: LABELS } };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
