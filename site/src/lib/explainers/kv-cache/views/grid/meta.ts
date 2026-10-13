// View 'kv-cache/grid': one pass of TinyGPT's generation (process 'gen') as an attention grid: a row per query (the token
// that looks back), a column per key, lower triangle only (no token looks ahead). For the frame's pass k it marks every
// cell as computed for the first time (colored by its attention weight, averaged over the heads), computed again with the
// same numbers (no cache), or not computed at all because an earlier pass did it and its keys and values sit in the cache;
// under the grid, a key and value slot per token. Props:
//   mode?: 'plain' | 'full' | 'cache'   every row read colored by attention; without the cache; with it (default 'cache')
//   controls?: boolean          buttons for the mode and the layer and a slider over the passes (the concept page's toy)
//   check?: boolean             instead of the grid: both runs' tokens pass by pass and the largest logit difference
//   layer?: number              the layer whose attention colors the cells (default 0)
import type { I18n } from '../../../../atlas/types/core.ts';
import type { Frame } from '../../../../atlas/types/process.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmtInt } from '../../../../atlas/core/format.ts';
import { VOCAB } from '../../vocab.ts';

/** A token's text for display (spaces and line breaks made visible). */
export const tokText = (id: number) => (VOCAB[id] ?? `#${id}`).replace(/\n/g, '↵').replace(/ /g, '␣');

/** Cells of pass k: computed for the first time, computed again, and not computed (rows the cache made unnecessary). */
export function cellsOf(m: Record<string, number> | undefined, mode: 'full' | 'cache') {
	if (!m) return { n: 0, fresh: 0, redo: 0, skip: 0 };
	const n = m.n;
	const tri = (r: number) => (r * (r + 1)) / 2; // cells in rows 0..r−1
	const first = m.k === 0 ? 0 : n - 1; // the first row this pass computes for the first time
	const fresh = tri(n) - tri(first);
	return mode === 'full' ? { n, fresh, redo: tri(first), skip: 0 } : { n, fresh, redo: 0, skip: tri(first) };
}

export function describeGrid(f: Frame | undefined, mode: 'plain' | 'full' | 'cache', check: boolean): I18n {
	const m = f?.metrics;
	if (!m) return { en: 'TinyGPT has not generated anything yet.', zh: 'TinyGPT 还没有生成任何内容。' };
	if (check)
		return {
			en: `Without and with the cache, TinyGPT chose the same token in ${m.sameAll ?? m.same} of ${m.maxNew} passes; the largest logit difference is ${m.dLogitMax ?? m.dLogit}.`,
			zh: `不用缓存和用缓存时，TinyGPT 在 ${m.maxNew} 遍中有 ${m.sameAll ?? m.same} 遍选出了相同的词元；logit 的最大差异是 ${m.dLogitMax ?? m.dLogit}。`
		};
	if (mode === 'plain') return { en: `Pass ${m.k} reads ${m.n} tokens: one row per token, each looking back at the tokens up to it.`, zh: `第 ${m.k} 遍读入 ${m.n} 个词元：每个词元一行，各自回头查看它及之前的词元。` };
	const c = cellsOf(m, mode);
	const macs = mode === 'full' ? m.macsFull : m.macsCache;
	return mode === 'full'
		? { en: `Pass ${m.k} without a cache reads ${m.n} tokens: ${fmtInt(c.fresh)} attention cells are new, ${fmtInt(c.redo)} belong to earlier tokens and are computed again; ${fmtInt(macs)} multiply-adds.`, zh: `不用缓存时，第 ${m.k} 遍读入 ${m.n} 个词元：${fmtInt(c.fresh)} 个注意力单元是新的，${fmtInt(c.redo)} 个属于前面的词元，又算了一遍；共 ${fmtInt(macs)} 次乘加。` }
		: { en: `Pass ${m.k} with the cache reads ${m.n} tokens: ${fmtInt(c.fresh)} attention cells are new, ${fmtInt(c.skip)} are not computed (their keys and values come from the cache); ${fmtInt(macs)} multiply-adds.`, zh: `用缓存时，第 ${m.k} 遍读入 ${m.n} 个词元：${fmtInt(c.fresh)} 个注意力单元是新的，${fmtInt(c.skip)} 个不用计算（它们的键和值来自缓存）；共 ${fmtInt(macs)} 次乘加。` };
}

const meta = {
	key: 'kv-cache/grid',
	title: { en: 'Attention cells: new, recomputed or cached', zh: '注意力单元：新算、重算还是缓存' },
	kinds: ['process'],
	describe: (p: StageProps) => describeGrid(p.at?.frame, p.props?.mode === 'full' || p.props?.mode === 'plain' ? p.props.mode : 'cache', !!p.props?.check),
	demo: async () => {
		const { fakeRuntime } = await import('../../../../atlas/runtime/fake.ts');
		const { TapeStore } = await import('../../../../atlas/process/tape.ts');
		const { gen } = await import('../../process.ts');
		const tape = new TapeStore({ process: 'gen', schema: gen.schema, phases: gen.phases, provenance: 'illustrative' });
		const ids = new Float64Array(64).fill(-1);
		[428, 437, 258, 397, 44, 400, 282, 258].forEach((v, i) => (ids[i] = v));
		const att = new Float64Array(4 * 64 * 64);
		for (let i = 0; i < 8; i++) for (let j = 0; j <= i; j++) att[i * 64 + j] = 1 / (i + 1);
		const st = { ids: { shape: [64], data: ids }, att: { shape: [4, 64, 64], data: att } };
		tape.push({ phase: 'decode', step: 2, state: st, metrics: { k: 2, P: 6, n: 8, maxNew: 3, macsFull: 2.3e6, macsCache: 3.6e5, L: 4 } } as never);
		const rt = fakeRuntime({ slug: 'kv-cache', input: {}, tapes: { gen: tape } });
		return { rt, source: 'gen', at: { index: 0, frame: tape.frame(0), state: tape.state(0), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: { mode: 'full' } };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
