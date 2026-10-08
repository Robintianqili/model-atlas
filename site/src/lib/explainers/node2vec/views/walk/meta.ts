// node2vec view 'walk': the club and its walks (props.mode):
//   club   the club, every member coloured by the side it joined after the split (never shown to training)
//   step   one move of a walk (a 'walks' frame): the walk so far, every friend of v with its kind, weight and chance,
//          and how the draw picks the next member
//   walks  the walk of the frame, built move by move on the club, and every walk from the same member as a sentence
//   pairs  the example walk (walks.ts exampleWalk) with the window around its member WIN, its pairs, and the members
//          that most often stand in the window of that member over all the walks
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { exampleWalk, FACTIONS, inputOf, moveOf, N, WALKS, walksOf, WIN } from '../../walks.ts';

type P = Omit<StageProps, 'ctx'>;
/** a candidate's kind as text, naming the member t the walk came from (a walk's first move has no t: every friend weighs 1) */
export const kindText = (t: number, k: number): I18n =>
	t < 0 ? { en: 'first move', zh: '第一步' } : [{ en: `back to ${t}`, zh: `回到 ${t}` }, { en: `also a friend of ${t}`, zh: `也是 ${t} 的朋友` }, { en: `farther from ${t}`, zh: `离 ${t} 更远` }][k];

export function walkPic(p: P) {
	const g = inputOf(p.rt.input), W = walksOf(g), m = p.at.frame?.phase === 'move' ? p.at.frame.metrics : undefined;
	const mode = String(p.props.mode ?? 'club');
	const step = m && mode === 'step' ? { ...moveOf(W.nb, m.t, m.v, g.p, g.q), t: m.t, v: m.v, xi: m.i, u: m.u, ux: m.ux } : null;
	const cum = step ? step.a.map((_, i) => step.a.slice(0, i + 1).reduce((s, x) => s + x, 0)) : [];
	const ex = exampleWalk(g), w = mode === 'pairs' ? ex : (m?.w ?? -1);
	const path = w >= 0 ? W.walks[w].slice(0, mode === 'pairs' ? undefined : (m?.k ?? 0) + 2) : [];
	const c = mode === 'pairs' && w >= 0 ? W.walks[w][WIN] : -1;
	const top = c >= 0 ? Array.from({ length: N }, (_, j) => ({ j, n: W.cooc[c * N + j] })).filter((x) => x.n > 0 && x.j !== c).sort((a, b) => b.n - a.n || a.j - b.j).slice(0, 5) : [];
	const same = w >= 0 ? W.walks.flatMap((ids, k) => (ids[0] === W.walks[w][0] ? [{ k, ids }] : [])) : [];
	return { g, W, mode, m, step, cum, w, path, c, top, same };
}

export function describeWalk(p: P): I18n {
	const s = walkPic(p), side = [0, 1].map((f) => FACTIONS.filter((x) => x === f).length);
	const head = { en: `The club: ${N} members, ${s.g.edges.length} friendships; ${side[0]} joined Mr. Hi, ${side[1]} the Officer. p = ${s.g.p}, q = ${s.g.q}.`, zh: `俱乐部：${N} 个成员，${s.g.edges.length} 段朋友关系；${side[0]} 人跟了 Mr. Hi，${side[1]} 人跟了干事。p = ${s.g.p}，q = ${s.g.q}。` };
	let more: I18n = { en: '', zh: '' };
	if (s.step) {
		const st = s.step, row = (x: number, i: number, l: 'en' | 'zh') => `${x}${l === 'en' ? ': ' : '：'}${kindText(st.t, st.kind[i])[l]}, ${fmt3(st.a[i])}, ${fmtPct(st.prob[i])}`;
		const rows = st.x.map((x, i) => row(x, i, 'en')).join('; '), rz = st.x.map((x, i) => row(x, i, 'zh')).join('；');
		more = { en: ` At member ${s.step.v}${s.step.t >= 0 ? `, coming from ${s.step.t}` : ''}. Friends (kind, weight, chance): ${rows}. The draw u = ${fmt3(s.step.u)} picks member ${s.step.x[s.step.xi]}.`, zh: ` 在成员 ${s.step.v}${s.step.t >= 0 ? `，从 ${s.step.t} 走来` : ''}。朋友（类别、权重、概率）：${rz}。抽到 u = ${fmt3(s.step.u)}，选中成员 ${s.step.x[s.step.xi]}。` };
	} else if (s.path.length) {
		const ids = s.path.join(' ');
		more = s.mode === 'pairs'
			? { en: ` A walk: ${ids}. The member at place ${WIN} (counting from 0) is ${s.c}; its pairs are with the ${2 * WIN} members around it.`, zh: ` 一条游走：${ids}。第 ${WIN} 个位置（从 0 数起）上的是成员 ${s.c}；它和周围 ${2 * WIN} 个成员组成训练对。` }
			: { en: ` Walk ${s.w} so far: ${ids}. ${WALKS} walks start from member ${s.path[0]}.`, zh: ` 第 ${s.w} 条游走到目前为止：${ids}。从成员 ${s.path[0]} 出发的游走有 ${WALKS} 条。` };
	}
	return { en: head.en + more.en, zh: head.zh + more.zh };
}

const meta = {
	key: 'node2vec/walk',
	title: { en: 'The club and its random walks', zh: '俱乐部和它的随机游走' },
	kinds: ['process'],
	describe: describeWalk,
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { walks }, { START, WALK_SEED }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../process.ts'), import('../../walks.ts')]);
		const tape = record(walks, START, {}, { seed: WALK_SEED, maxFrames: 100000 });
		const rt = fakeRuntime({ slug: 'node2vec', input: START, tapes: { walks: tape } });
		return { rt, source: 'walks', at: atFor(rt, 'walks', 73), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'step' } };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
