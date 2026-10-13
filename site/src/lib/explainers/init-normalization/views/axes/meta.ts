// View 'init-normalization/axes': BatchNorm and LayerNorm on the same tensor, rows = examples (or tokens), columns =
// units (or dimensions). Props:
//   mode: 'batch'   the 4 × 6 corner of layer 1's weighted sums (state.t of a process 'mlp' frame): every column's mean
//                   and spread over the 4 examples, the tensor after BatchNorm, the chosen cell worked out
//         'layer'   the same tensor: every row's mean and spread over the 6 units, the tensor after LayerNorm
//         'one'     one example alone (row metrics.cr): BatchNorm over a batch of one, against LayerNorm of that row
//         'gpt'     TinyGPT's block-1 output (a process 'gpt' frame: state x, ln, bn): a heatmap of tokens × 76
//                   dimensions with the followed token's row and the chosen dimension's column outlined, and both
//                   normalizations of their shared cell
// The chosen cell is the frame's (metrics cr, cc: the largest number in size; for 'gpt' metrics row, col).
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { EPS, NB, NF, PROBE_TOKENS, batchNorm, layerNorm, meanStd } from '../../data.ts';

export type Mode = 'batch' | 'layer' | 'one' | 'gpt';
export interface AxesPic {
	mode: Mode;
	rows: number;
	cols: number;
	x: number[];
	/** BatchNorm and LayerNorm of x (for 'one': of the single row) */
	bn: number[];
	ln: number[];
	/** per column (over the rows) and per row (over the columns) */
	colStats: { mean: number; std: number }[];
	rowStats: { mean: number; std: number }[];
	r: number;
	c: number;
}

const arr = (t: unknown) => Array.from(((t as { data?: ArrayLike<number> } | undefined)?.data ?? []) as ArrayLike<number>);

export function axesOf(p: Pick<StageProps, 'at' | 'props'>): AxesPic | null {
	const mode = (['batch', 'layer', 'one', 'gpt'].includes(p.props?.mode as string) ? p.props.mode : 'batch') as Mode;
	const s = (p.at?.state ?? {}) as Record<string, unknown>;
	const m = p.at?.frame?.metrics;
	if (!m) return null;
	if (mode === 'gpt') {
		const x = arr(s.x);
		if (!x.length) return null;
		const rows = m.tokens;
		const cols = m.dims;
		return {
			mode, rows, cols, x, bn: arr(s.bn), ln: arr(s.ln), r: m.row, c: m.col,
			colStats: Array.from({ length: cols }, (_, j) => meanStd(x, j, rows, cols)),
			rowStats: Array.from({ length: rows }, (_, i) => meanStd(x, i * cols, cols))
		};
	}
	const all = arr(s.t);
	if (all.length !== NB * NF) return null;
	if (mode === 'one') {
		const r = m.cr;
		const x = all.slice(r * NF, (r + 1) * NF);
		return {
			mode, rows: 1, cols: NF, x, bn: Array.from(batchNorm(x, 1, NF, EPS)), ln: Array.from(layerNorm(x, 1, NF, EPS)), r: 0, c: m.cc,
			colStats: Array.from({ length: NF }, (_, j) => meanStd(x, j, 1, NF)),
			rowStats: [meanStd(x, 0, NF)]
		};
	}
	return {
		mode, rows: NB, cols: NF, x: all, bn: Array.from(batchNorm(all, NB, NF, EPS)), ln: Array.from(layerNorm(all, NB, NF, EPS)), r: m.cr, c: m.cc,
		colStats: Array.from({ length: NF }, (_, j) => meanStd(all, j, NB, NF)),
		rowStats: Array.from({ length: NB }, (_, i) => meanStd(all, i * NF, NF))
	};
}

/** The worked numbers of the chosen cell: its value, the column's and the row's statistics, both results. */
export function cellOf(pic: AxesPic) {
	const k = pic.r * pic.cols + pic.c;
	return { x: pic.x[k], col: pic.colStats[pic.c], row: pic.rowStats[pic.r], bn: pic.bn[k], ln: pic.ln[k] };
}

export function describeAxes(pic: AxesPic | null): I18n {
	if (!pic) return { en: 'The tensor is not ready yet.', zh: '张量还没有准备好。' };
	const c = cellOf(pic);
	const cell = `row ${pic.r}, column ${pic.c}`;
	const cellZh = `第 ${pic.r} 行、第 ${pic.c} 列`;
	const rowsTxt = (a: number[]) => Array.from({ length: pic.rows }, (_, i) => a.slice(i * pic.cols, (i + 1) * pic.cols).map(fmt3).join(', ')).join(' | ');
	const head =
		pic.mode === 'gpt'
			? { en: `TinyGPT's block-1 output for ${pic.rows} tokens × ${pic.cols} dimensions (rows: ${PROBE_TOKENS.join('|')}).`, zh: `TinyGPT 第 1 块对 ${pic.rows} 个词元 × ${pic.cols} 个维度的输出。` }
			: { en: `A ${pic.rows} × ${pic.cols} tensor (rows: examples, columns: units): ${rowsTxt(pic.x)}.`, zh: `一个 ${pic.rows} × ${pic.cols} 的张量（行：样例，列：单元）：${rowsTxt(pic.x)}。` };
	return {
		en: `${head.en} Cell ${cell} (counting from 0) holds ${fmt3(c.x)}. Its column has mean ${fmt3(c.col.mean)} and spread ${fmt3(c.col.std)}, so BatchNorm gives ${fmt3(c.bn)}; its row has mean ${fmt3(c.row.mean)} and spread ${fmt3(c.row.std)}, so LayerNorm gives ${fmt3(c.ln)}.`,
		zh: `${head.zh}${cellZh}（从 0 开始数）的值是 ${fmt3(c.x)}。它所在的列均值为 ${fmt3(c.col.mean)}、标准差为 ${fmt3(c.col.std)}，所以 BatchNorm 得到 ${fmt3(c.bn)}；它所在的行均值为 ${fmt3(c.row.mean)}、标准差为 ${fmt3(c.row.std)}，所以 LayerNorm 得到 ${fmt3(c.ln)}。`
	};
}

const meta = {
	key: 'init-normalization/axes',
	title: { en: 'BatchNorm and LayerNorm on the same tensor', zh: '同一个张量上的 BatchNorm 和 LayerNorm' },
	kinds: ['process'],
	describe: (p: StageProps) => describeAxes(axesOf(p)),
	demo: async () => {
		const [{ record }, { mlp }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const input = { act: 'tanh', std: 1, norm: 'none' };
		const tape = record(mlp, input, {});
		const rt = fakeRuntime({ slug: 'init-normalization', input, tapes: { large: tape } });
		return { rt, source: 'large', at: { index: 0, frame: tape.frame(0), state: tape.state(0), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: { mode: 'batch' } };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
