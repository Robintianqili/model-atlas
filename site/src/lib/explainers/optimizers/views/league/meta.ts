// View 'optimizers/league': the loss of each racer on each of the three landscapes after the frame's number of updates
// (process 'league'), as a table; the lowest loss of each row is marked in words, not only by its outline.
import type { I18n, Tensor } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtInt } from '../../../../atlas/core/format.ts';
import { RACERS, RACE_SURFACES, surfaceOf, type SurfaceId } from '../../data.ts';

export interface LeaguePic {
	step: number;
	rows: { id: SurfaceId; name: I18n; losses: number[]; best: number }[];
}

export function leaguePic(p: Pick<StageProps, 'at'>): LeaguePic | null {
	const l = (p.at?.state as { losses?: Tensor } | undefined)?.losses;
	if (!l || l.data.length !== 12) return null;
	const rows = RACE_SURFACES.map((id, i) => {
		const losses = Array.from(l.data.slice(i * 4, i * 4 + 4));
		let best = 0;
		for (let j = 1; j < 4; j++) if (losses[j] < losses[best]) best = j;
		return { id, name: surfaceOf(id).name, losses, best };
	});
	return { step: p.at.frame?.metrics?.step ?? NaN, rows };
}

export function describeLeague(pic: LeaguePic | null): I18n {
	if (!pic) return { en: 'The three races have not run yet.', zh: '三场比赛还没有运行。' };
	const row = (r: LeaguePic['rows'][number]) => ({
		en: ` On the ${r.name.en}: ${RACERS.map((q, j) => `${q.name.en} ${fmt3(r.losses[j])}`).join(', ')}; lowest: ${RACERS[r.best].name.en}.`,
		zh: `${r.name.zh}：${RACERS.map((q, j) => `${q.name.zh} ${fmt3(r.losses[j])}`).join('，')}；最低的是 ${RACERS[r.best].name.zh}。`
	});
	const rows = pic.rows.map(row);
	return {
		en: `The loss of each optimizer after ${fmtInt(pic.step)} updates on each landscape.${rows.map((r) => r.en).join('')}`,
		zh: `每个优化器在每个地形上更新 ${fmtInt(pic.step)} 次之后的损失。${rows.map((r) => r.zh).join('')}`
	};
}

const meta = {
	key: 'optimizers/league',
	title: { en: 'Which optimizer ends lowest, landscape by landscape', zh: '在每个地形上，哪个优化器最后最低' },
	kinds: ['process'],
	describe: (p: StageProps) => describeLeague(leaguePic(p)),
	demo: async () => {
		const [{ record }, { league }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const tape = record(league, null, { steps: 80 });
		const rt = fakeRuntime({ slug: 'optimizers', input: null, tapes: { league: tape } });
		const i = tape.length - 1;
		return { rt, source: 'league', at: { index: i, frame: tape.frame(i), state: tape.state(i), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 200
} satisfies ViewMeta;

export default meta;
