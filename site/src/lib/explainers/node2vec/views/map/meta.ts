// node2vec view 'map': the member vectors (rows of W_in) of a training tape, flattened to 2-D by PCA, every member
// coloured by the side it joined (training never sees it). The axes are the PCA of the tape's last frame, so the dots
// move on fixed axes while the snapshots replay (the word2vec package's map, adapted). Props:
//   focus   a member to ring, with lines to its 3 nearest members (cosine similarity of the full vectors)
//   list    the nearest members of the focus: cosine, hops in the graph, side
//   leaders mark (✕) the members that sit closer to the other side's leader (member 0 or 33) than to their own's
//   group   members to ring together (the limits step's members with the same friends), with their mean cosine
//   stats   how far this run's walks end from their start and how far (hops) each member's 3 nearest members are
//   cut     the vertical line that best splits the two sides (vectors.ts cutOf)
//   animate replay the snapshots up to the step's frame (a step with play: 'view')
import type { I18n, StageProps, Tape, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { N, runInput, walksOf } from '../../walks.ts';
import { cosine, cutOf, factionStats, lastVecs, pca2, project, vecsAt } from '../../vectors.ts';

type P = Omit<StageProps, 'ctx'>;
export const finalPca = (tape: Tape | undefined) => { const vs = lastVecs(tape); return vs.length > 1 ? pca2(vs) : null; };

export function mapPic(p: P) {
	const vs = vecsAt(p.at.state), pca = finalPca(p.at.tape), st = factionStats(p.rt.input, vs);
	if (!pca || !st) return null;
	const xy = project(vs, pca), focus = typeof p.props.focus === 'number' ? (p.props.focus as number) : -1;
	const group = (p.props.group as number[] | undefined) ?? [];
	/** mean cosine similarity over the pairs of a set of members */
	const meanCos = (ids: number[]) => { let s = 0, n = 0; for (let a = 0; a < ids.length; a++) for (let b = a + 1; b < ids.length; b++) (s += cosine(vs[ids[a]], vs[ids[b]])), n++; return s / n; };
	const twin = group.length > 1 ? { g: meanCos(group), all: meanCos([...Array(N).keys()]) } : null;
	return { xy, final: project(lastVecs(p.at.tape), pca), pca, st, focus, group, near: focus >= 0 ? st.nn[focus].slice(0, 5) : [], cut: cutOf(xy), showCut: !!p.props.cut, step: p.at.frame?.step ?? 0, loss: p.at.frame?.metrics?.avgLoss ?? NaN, leaders: !!p.props.leaders, list: !!p.props.list, g: runInput(p.rt.input, p.source), twin, far: p.props.stats ? walksOf(runInput(p.rt.input, p.source)).far : NaN };
}

export function describeMap(p: P): I18n {
	const m = mapPic(p);
	if (!m) return { en: 'The member map: nothing to show yet.', zh: '成员向量图：还没有可显示的内容。' };
	const near = m.near.map((x) => `${x.j} (${fmt3(x.cos)})`).join(', ');
	return {
		en: `34 member vectors after ${m.step} training steps (p = ${m.g.p}, q = ${m.g.q}), drawn by PCA (keeps ${fmtPct(m.pca.share)} of their spread). One vertical cut puts ${m.cut.ok} of 34 members on their own side; ${m.st.same} members' nearest member joined their side.${m.focus >= 0 ? ` Nearest to member ${m.focus}: ${near}.` : ''}${m.leaders ? ` Closer to their own leader: ${m.st.right} of 32.` : ''}`,
		zh: `训练 ${m.step} 步后的 34 个成员向量（p = ${m.g.p}，q = ${m.g.q}），用 PCA 画成平面图（保留了 ${fmtPct(m.pca.share)} 的分布）。竖着切一刀，34 人中有 ${m.cut.ok} 人落在自己一方；有 ${m.st.same} 个成员最近的成员和自己同一方。${m.focus >= 0 ? `离成员 ${m.focus} 最近的：${near}。` : ''}${m.leaders ? `离自己一方的领头人更近的：32 人中有 ${m.st.right} 人。` : ''}`
	};
}

const meta = {
	key: 'node2vec/map',
	title: { en: 'The member vectors on a map', zh: '成员向量图' },
	kinds: ['train'],
	describe: describeMap,
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { trainingProcess }, { sg }, { START }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../../../atlas/train/trainer.ts'), import('../../trainer.ts'), import('../../walks.ts')]);
		const tape = record(trainingProcess(sg), START, { steps: 2000 }, { seed: 1, maxFrames: 100000 });
		const rt = fakeRuntime({ slug: 'node2vec', input: START, tapes: { run: tape } });
		return { rt, source: 'run', at: atFor(rt, 'run', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { focus: 8, list: true } };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
