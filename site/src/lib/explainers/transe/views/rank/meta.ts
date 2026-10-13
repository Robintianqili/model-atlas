// transe view 'rank': process 'rank' up to the step's frame. The candidate being scored and its distance, then every
// candidate scored so far, nearest first (the 6 nearest, bars to scale from 0), the facts' answers marked; at the
// 'result' frame the final ranking.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { ENT } from '../../kg.ts';

export function rankPic(p: Omit<StageProps, 'ctx'>) {
	const tp = p.at.tape;
	if (!tp?.length) return null;
	const seen: { x: number; d: number; truth: boolean }[] = [];
	for (let i = 0; i <= p.at.index; i++) {
		const f = tp.frame(i);
		if (f.phase === 'score' && f.metrics) seen.push({ x: f.metrics.x, d: f.metrics.d, truth: f.metrics.truth === 1 });
	}
	const cur = p.at.frame?.phase === 'score' ? seen[seen.length - 1] : null;
	return { cur, n: seen.length, done: p.at.frame?.phase === 'result', best: [...seen].sort((a, b) => a.d - b.d || a.x - b.x).slice(0, 6), max: Math.max(...seen.map((s) => s.d), 1e-9) };
}

export function describeRank(p: Omit<StageProps, 'ctx'>): I18n {
	const m = rankPic(p);
	if (!m) return { en: 'The ranking: nothing scored yet.', zh: '排名：还没有给任何候选打分。' };
	const list = (l: 'en' | 'zh') => m.best.map((c) => `${ENT[c.x].name[l]} ${fmt3(c.d)}`).join(l === 'en' ? ', ' : '，');
	return { en: `${m.n} of 34 candidates scored; nearest so far: ${list('en')}.`, zh: `34 个候选已打分 ${m.n} 个；目前最近的：${list('zh')}。` };
}

const meta = {
	key: 'transe/rank',
	title: { en: 'Every candidate scored and ranked', zh: '给每个候选打分并排名' },
	kinds: ['process'],
	describe: describeRank,
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { trainingProcess }, { transe }, { rank, rankInput }, { START }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../../../atlas/train/trainer.ts'), import('../../trainer.ts'), import('../../process.ts'), import('../../kg.ts')]);
		const run = record(trainingProcess(transe), START, {}, { seed: 3, maxFrames: 100000 });
		const tape = record(rank, rankInput(run, 11, 0, [0]), {}, { seed: 1 });
		const rt = fakeRuntime({ slug: 'transe', input: START, tapes: { rank: tape } });
		return { rt, source: 'rank', at: atFor(rt, 'rank', 20), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
