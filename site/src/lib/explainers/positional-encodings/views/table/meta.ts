// View 'positional-encodings/table': a position table as a heat map, one row per position and one column per dimension
// (process 'pe', phases 'learned' and 'sin'), with row p outlined and row p + k dashed when the table has them. In the
// 'learned' phase a strip under it shows the cosine similarity of row p with every row.
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { arr, demoStage, stateOf } from '../demo.ts';

export function tableOf(p: Pick<StageProps, 'at'>) {
	const s = stateOf(p);
	const t = s.table;
	const m = p.at?.frame?.metrics ?? {};
	const rows = t?.shape[0] ?? 0;
	const cols = t?.shape[1] ?? 0;
	const data = arr(t);
	const amax = Math.max(1e-9, ...data.map(Math.abs));
	return { learned: p.at?.frame?.phase === 'learned', data, rows, cols, amax, p: m.p ?? 0, k: m.k ?? 0, sim: arr(s.sim) };
}

export function describeTable(o: ReturnType<typeof tableOf>): I18n {
	if (!o.rows) return { en: 'No table yet.', zh: '还没有位置表。' };
	const what: I18n = o.learned ? { en: 'TinyGPT’s learned position table', zh: 'TinyGPT 学到的位置表' } : { en: 'The sinusoidal position table', zh: '正弦位置表' };
	const row: I18n =
		o.p < o.rows
			? { en: ` Row ${o.p} is outlined; its first numbers are ${o.data.slice(o.p * o.cols, o.p * o.cols + 4).map((v) => fmt3(v)).join(', ')}.`, zh: ` 第 ${o.p} 行带框；它的前几个数是 ${o.data.slice(o.p * o.cols, o.p * o.cols + 4).map((v) => fmt3(v)).join('、')}。` }
			: { en: ` There is no row ${o.p}.`, zh: ` 没有第 ${o.p} 行。` };
	return {
		en: `${what.en}: ${o.rows} rows (positions 0 to ${o.rows - 1}) of ${o.cols} numbers, from ${fmt3(-o.amax)} to ${fmt3(o.amax)} on a diverging colour scale.${row.en}`,
		zh: `${what.zh}：${o.rows} 行（位置 0 到 ${o.rows - 1}），每行 ${o.cols} 个数，用发散色标表示，范围 ${fmt3(-o.amax)} 到 ${fmt3(o.amax)}。${row.zh}`
	};
}

const meta = {
	key: 'positional-encodings/table',
	title: { en: 'A position table', zh: '位置表' },
	kinds: ['process'],
	describe: (p: StageProps) => describeTable(tableOf(p)),
	demo: () => demoStage('sin'),
	minHeight: 320
} satisfies ViewMeta;

export default meta;
