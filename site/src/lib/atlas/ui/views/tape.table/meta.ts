// tape.table (W1-D2): the frames of a process or training tape as a table around the stage's frame.
import type { ViewMeta } from '../../../types/view.ts';
import { fmt3, type Props } from '../kit.parts/stage.ts';
import { processStage } from '../kit.parts/demo.ts';
import { tableOf } from './rows.ts';

export function describeTable(p: Props) {
	const d = tableOf(p);
	if (!d) return { en: 'A table of frames: nothing recorded yet.', zh: '帧表：还没有记录。' };
	const r = d.rows.find((x) => x.i === d.at);
	const cells = r
		? d.cols
				.filter((c) => r.cells[c.key] !== undefined)
				.map((c) => `${c.label} = ${typeof r.cells[c.key] === 'number' ? fmt3(r.cells[c.key] as number) : r.cells[c.key]}`)
				.join(', ')
		: '';
	const ph = r?.phaseLabel ?? { en: '', zh: '' };
	return {
		en: `Frame ${d.at + 1} of ${d.tape.length}: ${ph.en}${r?.step !== undefined ? `, step ${r.step}` : ''}${cells ? `; ${cells}` : ''}.`,
		zh: `第 ${d.at + 1} / ${d.tape.length} 帧：${ph.zh ?? ph.en}${r?.step !== undefined ? `，第 ${r.step} 步` : ''}${cells ? `；${cells}` : ''}。`
	};
}

const meta = {
	key: 'tape.table',
	title: { en: 'Every frame of a run, as a table', zh: '把一次运行的每一帧列成表' },
	kinds: ['train', 'process'],
	accepts: () => false,
	describe: describeTable,
	demo: async () => processStage(14),
	minHeight: 360
} satisfies ViewMeta;

export default meta;
