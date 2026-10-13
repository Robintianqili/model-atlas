// transe view 'space': the entity points of a training tape flattened to 2-D by PCA, coloured by type (the type named
// next to its colour), on the axes of the tape's last frame so the points move on fixed axes while snapshots replay.
// A relation arrow r maps to the plane as it is (no mean), so the map of h + r is the map of h plus the map of r. Props:
//   rel     draw relation rel's arrow from the head of each of its facts
//   query   'input' (the learner's question) or [h, r]: h ringed, its arrow to where h + r lands (✕), the 5 nearest
//           entities (the real distances, in all 8 numbers) and any true tail ranked lower in a table, the 3 nearest joined to the landing point
//   kind    rank only entities of this type in the table (the four continents for “located in”)
//   ring    entities to ring
//   animate replay the snapshots up to the step's frame (a step with play: 'view')
import type { I18n, StageProps, Tape, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { ENT, inputOf, NE, REL } from '../../kg.ts';
import { arrowOnMap, lastVecs, onMap, pca2, ranked, tailsOf, vecsAt, type Pca } from '../../space.ts';
import { runFacts } from '../../trainer.ts';

type P = Omit<StageProps, 'ctx'>;
export const finalPca = (tape: Tape | undefined): Pca | null => { const v = lastVecs(tape); return v.E.length ? pca2(v.E) : null; };

export function spacePic(p: P) {
	const v = vecsAt(p.at.state), pca = finalPca(p.at.tape);
	if (!pca || v.E.length < NE) return null;
	const facts = runFacts(p.rt.input, p.source === 'base' ? { start: true } : {});
	const q = p.props.query === 'input' ? inputOf(p.rt.input).query : (p.props.query as [number, number] | undefined);
	const rel = typeof p.props.rel === 'number' ? (p.props.rel as number) : -1;
	const xy = v.E.map((e) => onMap(pca, e)), final = lastVecs(p.at.tape).E.map((e) => onMap(pca, e));
	const arrow = (r: number) => arrowOnMap(pca, v.R[r]);
	const kind = typeof p.props.kind === 'number' ? (p.props.kind as number) : -1;
	const truth = q ? tailsOf(facts, q[0], q[1]) : [];
	const near = q ? ranked(v, q[0], q[1]).filter((c) => kind < 0 || ENT[c.x].type === kind).map((c, k) => ({ ...c, rank: k + 1 })).filter((c) => c.rank <= 5 || truth.includes(c.x)) : [];
	const land = q ? [xy[q[0]][0] + arrow(q[1])[0], xy[q[0]][1] + arrow(q[1])[1]] : null;
	const arrows = rel >= 0 ? facts.filter((f) => f[1] === rel).map((f) => ({ h: f[0], t: f[2], to: [xy[f[0]][0] + arrow(rel)[0], xy[f[0]][1] + arrow(rel)[1]] })) : [];
	return { xy, final, pca, q, kind, near, truth, land, arrows, rel, ring: (p.props.ring as number[] | undefined) ?? [], step: p.at.frame?.step ?? 0, loss: p.at.frame?.metrics?.avgLoss ?? NaN, nFacts: facts.length };
}

export function describeSpace(p: P): I18n {
	const m = spacePic(p);
	if (!m) return { en: 'The entity map: nothing to show yet.', zh: '实体图：还没有可显示的内容。' };
	const nm = (i: number, l: 'en' | 'zh') => ENT[i].name[l];
	const list = (l: 'en' | 'zh') => m.near.map((c) => `${nm(c.x, l)} ${fmt3(c.d)}`).join(l === 'en' ? ', ' : '，');
	return {
		en: `34 entity points after ${m.step} training steps, drawn by PCA (keeps ${fmtPct(m.pca.share)} of their spread).${m.rel >= 0 ? ` Arrows: “${REL[m.rel].en}” from each of its ${m.arrows.length} heads.` : ''}${m.q ? ` Question: ${nm(m.q[0], 'en')} + “${REL[m.q[1]].en}”; nearest: ${list('en')}.` : ''}`,
		zh: `训练 ${m.step} 步后的 34 个实体点，用 PCA 画成平面图（保留了 ${fmtPct(m.pca.share)} 的分布）。${m.rel >= 0 ? `箭头：从“${REL[m.rel].zh}”的 ${m.arrows.length} 个头实体各画一支。` : ''}${m.q ? `问题：${nm(m.q[0], 'zh')} + “${REL[m.q[1]].zh}”；最近的：${list('zh')}。` : ''}`
	};
}

const meta = {
	key: 'transe/space',
	title: { en: 'Entities as points, relations as arrows', zh: '实体是点，关系是箭头' },
	kinds: ['train'],
	describe: describeSpace,
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { trainingProcess }, { transe }, { START }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../../../atlas/train/trainer.ts'), import('../../trainer.ts'), import('../../kg.ts')]);
		const tape = record(trainingProcess(transe), START, {}, { seed: 3, maxFrames: 100000 });
		const rt = fakeRuntime({ slug: 'transe', input: START, tapes: { run: tape } });
		return { rt, source: 'run', at: atFor(rt, 'run', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { rel: 0, query: 'input' } };
	},
	minHeight: 380
} satisfies ViewMeta;

export default meta;
