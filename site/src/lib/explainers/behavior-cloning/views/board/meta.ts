// behavior-cloning view 'board': the windy field with your demonstrations, the dataset's cells, the policy's move on
// every cell (from a train source's last weights), the episodes of a roll tape up to the step's frame, and the forward
// pass at the probed cell. Pure helpers here (shared with View.svelte and the lesson); the drawing is View.svelte.
//   props.show   comma list: demos, data, policy, wrong, roll, probe, edit
//   props.policy the train source whose weights are drawn: 'train' (default) or 'dagger'
//   props.roll   the roll tape whose episodes are drawn: 'roll' (default) or 'droll'
//   props.first  draw the weights of the first frame (the fresh network); props.only: only the episode at the cursor
import type { ExplainerRuntime, I18n, StageProps, Tape, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { trainingProcess } from '../../../../atlas/train/trainer.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { DEFAULTS, distances, featuresOf, OPEN, policyAt, policyMap, step, toRaw, worldOf, type Params, type World } from '../../world.ts';
import { bc, DAGGER, datasetOf } from '../../trainer.ts';

export const lastParams = (t: Tape | null | undefined): Params | null => (t?.length ? ((t.state(t.length - 1) as { params?: Params }).params ?? null) : null);
/** Everything the board draws, computed from the runtime (the learner's input and the sources that have run). */
export function boardOf(rt: ExplainerRuntime, props: Record<string, unknown>, upTo?: { tape: Tape | null | undefined; index: number }) {
	const raw = toRaw(rt.input), W = worldOf(raw.map), d = distances(W), src = props.policy === 'dagger' ? 'dagger' : 'train';
	const tape = rt.sources[src]?.tape as Tape | null | undefined, P = props.first ? ((tape?.length ? (tape.state(0) as { params?: Params }).params : null) ?? null) : lastParams(tape);
	const D = datasetOf(raw, tape, src === 'dagger' ? DAGGER : {});
	const moves = P ? policyMap(W, P) : null;
	const wrong = moves ? W.cell.map((c, s) => (c === OPEN && d[s] > 0 && d[step(W, s, moves[s])] !== d[s] - 1 ? s : -1)).filter((s) => s >= 0) : [];
	const rollTape = upTo?.tape ?? (rt.sources[props.roll === 'droll' ? 'droll' : 'roll']?.tape as Tape | null | undefined), last = upTo ? upTo.index : (rollTape?.length ?? 0) - 1;
	const eps: { cells: number[]; outcome: number | null }[] = [];
	for (let i = 0; rollTape && i <= last && i < rollTape.length; i++) {
		const f = rollTape.frame(i), m = f.metrics ?? {};
		if (f.phase === 'move') (m.t === 0 && eps.push({ cells: [m.s], outcome: null }), eps[eps.length - 1]?.cells.push(m.s2));
		else if (f.phase === 'end' && eps.length) eps[eps.length - 1].outcome = m.outcome;
	}
	if (props.only && eps.length > 1) eps.splice(0, eps.length - 1);
	const probe = P ? policyAt(P, featuresOf(W, raw.probe)) : null;
	return { raw, W, d, D, P, moves, wrong, eps, probe, dataCells: new Set(D.s), added: new Set(D.s.filter((_, i) => D.from[i] > 0)) };
}
export type Board = ReturnType<typeof boardOf>;
const cr = (W: World, s: number) => `column ${s % W.w}, row ${Math.floor(s / W.w)}`;
const crZh = (W: World, s: number) => `第 ${s % W.w} 列第 ${Math.floor(s / W.w)} 行`;

export function describeBoard(p: Omit<StageProps, 'ctx'>): I18n {
	const b = boardOf(p.rt, p.props, p.source === 'roll' || p.source === 'droll' ? { tape: p.at.tape, index: p.at.index } : undefined);
	const head = { en: `A field of ${b.W.w} by ${b.W.h} cells; the start is at ${cr(b.W, b.W.start)} (counting from 0), the goal at ${cr(b.W, b.W.cell.indexOf(2))}, pits along the bottom row. ${b.raw.demos.length} demonstrations give ${b.D.s.length} (cell, move) pairs.`, zh: `一片 ${b.W.w}×${b.W.h} 的空地；起点在${crZh(b.W, b.W.start)}（从 0 数起），终点在${crZh(b.W, b.W.cell.indexOf(2))}，最下面一行是陷阱。${b.raw.demos.length} 条示范给出 ${b.D.s.length} 个（格子，动作）对。` };
	const pol = b.moves ? { en: ` The policy's move does not lead closer to the goal on ${b.wrong.length} cells.`, zh: `策略的动作在 ${b.wrong.length} 个格子上不能让它更接近终点。` } : { en: '', zh: '' };
	const done = b.eps.filter((e) => e.outcome !== null), goals = done.filter((e) => e.outcome === 1).length;
	const roll = b.eps.length ? { en: ` ${done.length} episodes shown: ${goals} reached the goal.`, zh: `显示了 ${done.length} 个回合：${goals} 个到达终点。` } : { en: '', zh: '' };
	return { en: head.en + pol.en + roll.en, zh: head.zh + pol.zh + roll.zh };
}

const meta = {
	key: 'behavior-cloning/board',
	title: { en: 'The windy field and the policy', zh: '刮风的空地和策略' },
	kinds: ['train', 'process'],
	describe: describeBoard,
	async demo() {
		const tape = record(trainingProcess(bc), DEFAULTS, {}, { seed: 1 });
		const rt = fakeRuntime({ slug: 'behavior-cloning', input: DEFAULTS, tapes: { train: tape } });
		return { rt, source: 'train', at: atFor(rt, 'train', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { show: 'demos,data,policy,wrong' } };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
