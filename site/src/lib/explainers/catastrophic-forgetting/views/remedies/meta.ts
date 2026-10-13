// View 'catastrophic-forgetting/remedies': the 'remedies' process at one frame as a table, one row per way of training on
// task B (process.ts REMEDIES), with two bars per row: the share of task A and of task B the network gets right. Bars
// start at 0 and end at 100%, so their lengths compare across rows. Props:
//   only?: string[]    the remedy ids to show (default: all, in REMEDIES order)
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmtPct } from '../../../../atlas/core/format.ts';
import { REMEDIES } from '../../process.ts';
import { remedyName } from '../../data.ts';

export interface RemedyRow {
	id: string;
	name: I18n;
	accA: number;
	accB: number;
}

/** The rows the view draws (pure; also the text form). Empty until the frame has the remedies' metrics. */
export function rowsOf(p: Pick<StageProps, 'at' | 'props'>): RemedyRow[] {
	const m = p.at?.frame?.metrics ?? {};
	const only = Array.isArray(p.props?.only) ? (p.props.only as string[]) : null;
	return REMEDIES.filter((r) => (only ? only.includes(r.id) : true) && Number.isFinite(m[`accA_${r.id}`])).map((r) => ({ id: r.id, name: remedyName(r.set), accA: m[`accA_${r.id}`], accB: m[`accB_${r.id}`] }));
}

export function describeRows(rows: RemedyRow[], step: number | undefined): I18n {
	if (!rows.length) return { en: 'The comparison has not run yet.', zh: '对比还没有运行。' };
	const n = Number.isFinite(step) ? String(step) : '…';
	return {
		en: `After ${n} updates on task B, the share of each task the network gets right: ${rows.map((r) => `${r.name.en}: task A ${fmtPct(r.accA)}, task B ${fmtPct(r.accB)}`).join('; ')}.`,
		zh: `在任务 B 上更新 ${n} 次之后，网络在每个任务上答对的比例：${rows.map((r) => `${r.name.zh}：任务 A ${fmtPct(r.accA)}，任务 B ${fmtPct(r.accB)}`).join('；')}。`
	};
}

const meta = {
	key: 'catastrophic-forgetting/remedies',
	title: { en: 'Ways to train on task B', zh: '在任务 B 上训练的几种做法' },
	kinds: ['process'],
	describe: (p: StageProps) => describeRows(rowsOf(p), p.at?.frame?.step),
	demo: async () => {
		const [{ record }, { remedies }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const tape = record(remedies, null, { stepsA: 100, stepsB: 40 }, { detail: 'coarse' });
		const rt = fakeRuntime({ slug: 'catastrophic-forgetting', input: null, tapes: { remedies: tape } });
		const i = tape.length - 1;
		return { rt, source: 'remedies', at: { index: i, frame: tape.frame(i), state: tape.state(i), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 220
} satisfies ViewMeta;

export default meta;
