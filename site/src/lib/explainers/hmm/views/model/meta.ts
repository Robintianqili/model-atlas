// hmm view 'model': props.show 'story' (the two weather states with the transition arrows and each state's emission bars,
// and the learner's days with the weather hidden) or 'sample' (the sampling tape: the made-up weather and activity of
// each day so far, and the current day's two draws as points on a 0-to-1 line cut into the row's intervals).
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { record } from '../../../../atlas/process/record.ts';
import { atFor } from '../../../../atlas/runtime/sources.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { ACTS, dataOf, I, obsOf, percentOf, START, STATES } from '../../hmm.ts';
import { sample } from '../../process.ts';

const arr = (x: unknown): number[] => Array.from(((x as { data?: ArrayLike<number> } | undefined)?.data ?? []) as ArrayLike<number>);
/** the made-up days so far: weather and activity (−1 not drawn yet) */
export function drawn(p: Pick<StageProps, 'at'>): { w: number[]; a: number[] } {
	const tape = p.at.tape, s = (tape && tape.length ? tape.state(Math.max(0, p.at.index)) : {}) as Record<string, unknown>;
	return { w: arr(s.weather), a: arr(s.acts) };
}

export function describeModel(p: Omit<StageProps, 'ctx'>): I18n {
	if (p.props?.show === 'sample') {
		const f = p.at.frame, m = f?.metrics ?? {}, d = drawn(p);
		if (!f) return I('Made-up days: nothing drawn yet.', '编出来的日子：还没有抽。');
		const n = d.w.filter((s) => s >= 0).length, wt = d.w.filter((s) => s >= 0).map((s) => STATES[s]), ac = d.a.filter((s) => s >= 0).map((s) => ACTS[s]);
		const last = f.phase === 'day' ? I(` Day ${m.t}: the draw ${fmt3(m.u)} picks ${STATES[m.s].en}, the draw ${fmt3(m.v)} picks ${ACTS[m.act].en}.`, `第 ${m.t} 天：抽到 ${fmt3(m.u)}，选中${STATES[m.s].zh}；抽到 ${fmt3(m.v)}，选中${ACTS[m.act].zh}。`) : I('', '');
		return I(`${n} made-up days. Weather: ${wt.map((x) => x.en).join(', ')}. Activities: ${ac.map((x) => x.en).join(', ')}.${last.en}`, `编出了 ${n} 天。天气：${wt.map((x) => x.zh).join('、')}。活动：${ac.map((x) => x.zh).join('、')}。${last.zh}`);
	}
	const tb = percentOf(p.rt.input), o = obsOf(p.rt.input), q = (v: number) => String(v / 100);
	return I(
		`Two hidden weather states. Sunny stays sunny with ${q(tb.A[0][0])}, rainy stays rainy with ${q(tb.A[1][1])}. On a sunny day she walks, shops or cleans with ${tb.B[0].map(q).join(', ')}; on a rainy day with ${tb.B[1].map(q).join(', ')}. Her ${o.length} days: ${o.map((x) => ACTS[x].en).join(', ')}, weather unknown.`,
		`两种隐藏的天气。晴天保持晴天的概率是 ${q(tb.A[0][0])}，雨天保持雨天的概率是 ${q(tb.A[1][1])}。晴天她散步、购物、打扫的概率分别是 ${tb.B[0].map(q).join('、')}；雨天分别是 ${tb.B[1].map(q).join('、')}。她的 ${o.length} 天：${o.map((x) => ACTS[x].zh).join('、')}，天气未知。`
	);
}

const meta = {
	key: 'hmm/model',
	title: I('The model: two hidden states and two tables', '模型：两种隐藏状态和两张表'),
	kinds: ['process'],
	describe: describeModel,
	async demo() {
		const { pi, A, B } = dataOf(START);
		const tape = record(sample, { pi, A, B }, { days: 12 }, { seed: 1 });
		const rt = fakeRuntime({ slug: 'hmm', input: START, tapes: { sample: tape } });
		return { rt, source: 'sample', at: atFor(rt, 'sample', 3), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { show: 'sample' } };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
