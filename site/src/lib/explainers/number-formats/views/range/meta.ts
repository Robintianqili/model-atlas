// View 'number-formats/range': the range and precision of each format on one log axis of sizes |v| (one tick per power of
// 2): a bar from the smallest positive value to the largest, its subnormal part dashed. On a 'formats' frame x is marked;
// on a 'sweep' frame the histogram of TinyGPT's weight sizes (state 'hist', one bin per power of 2) is drawn above the
// bars. Props:
//   formats?: FmtId[]   the bars (default: the six floats)
//   table?: boolean     the trade-off table under the chart: bits, largest, smallest normal, smallest, gap at 1, values
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtInt } from '../../../../atlas/core/format.ts';
import { FLOAT_IDS, finiteCount, FMTS, gapAtOne, largest, nbits, smallest, smallestNormal, type FmtId } from '../../fmt.ts';
import { NAMES, splitText } from '../bits/meta.ts';
import { HIST_LO } from '../../process.ts';

export interface RangeRow {
	id: FmtId;
	bits: number;
	max: number;
	minNormal: number;
	min: number;
	gap1: number;
	count: number;
	float: boolean;
}
export const rangeRow = (id: FmtId): RangeRow => ({ id, bits: nbits(FMTS[id]), max: largest(id), minNormal: smallestNormal(id), min: smallest(id), gap1: gapAtOne(id), count: finiteCount(id), float: FMTS[id].kind === 'float' });

export function rangeData(p: Pick<StageProps, 'at' | 'props'>) {
	const ids: FmtId[] = Array.isArray(p.props?.formats) ? (p.props.formats as FmtId[]) : FLOAT_IDS;
	const x = p.at?.frame?.metrics?.x;
	const h = (p.at?.state as { hist?: { data: ArrayLike<number> } } | undefined)?.hist?.data;
	return { rows: ids.map(rangeRow), x: typeof x === 'number' ? x : null, hist: h ? Array.from(h) : null, histLo: HIST_LO };
}

export const big = (v: number) => (!Number.isFinite(v) ? '—' : v >= 1e5 || (v > 0 && v < 1e-3) ? fmt3(v) : String(Number(v.toPrecision(4))));

export function describeRange(d: ReturnType<typeof rangeData>): I18n {
	const en = d.rows.map((r) => `${NAMES[r.id].en} (${r.bits} bits): from ${big(r.min)} to ${big(r.max)}, smallest normal ${big(r.minNormal)}, gap at 1 ${big(r.gap1)}`).join('; ');
	const zh = d.rows.map((r) => `${NAMES[r.id].zh}（${r.bits} 位）：从 ${big(r.min)} 到 ${big(r.max)}，最小正规数 ${big(r.minNormal)}，1 处的间隔 ${big(r.gap1)}`).join('；');
	const hist = d.hist ? { en: ` Histogram of ${fmtInt(d.hist.reduce((a, b) => a + b, 0))} weight sizes.`, zh: ` 共 ${fmtInt(d.hist.reduce((a, b) => a + b, 0))} 个权重大小的直方图。` } : { en: '', zh: '' };
	const x = d.x !== null ? { en: ` x = ${fmt3(d.x)}.`, zh: ` x = ${fmt3(d.x)}。` } : { en: '', zh: '' };
	return { en: `Range of each format on a log axis: ${en}.${hist.en}${x.en}`, zh: `各格式在对数轴上的范围：${zh}。${hist.zh}${x.zh}` };
}

export { splitText };

const meta = {
	key: 'number-formats/range',
	title: { en: 'Range and precision of each format', zh: '各格式的范围与精度' },
	kinds: ['process'],
	describe: (p: StageProps) => describeRange(rangeData(p)),
	demo: async () => {
		const [{ record }, { formats }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const tape = record(formats, 0.1, {});
		const rt = fakeRuntime({ slug: 'number-formats', input: { x: 0.1 }, tapes: { formats: tape } });
		return { rt, source: 'formats', at: { index: 0, frame: tape.frame(0), state: tape.state(0), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: { table: true } };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
