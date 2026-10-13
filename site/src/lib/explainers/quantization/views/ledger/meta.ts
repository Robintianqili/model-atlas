// View 'quantization/ledger': the rounding ledger of eight numbers. A number line per setting (the levels the codes stand
// for as ticks, each number as a dot joined to the level it rounds to) and a table: i (counting from 0), x, x / s + z,
// the code q, the value it stands for and the error. The numbers are the frame's state.x, rounded with the run's
// options (process 'ledger': bits, scheme, clip, block, alt). Props:
//   alt?: boolean        also the second setting of the run (opt.alt): its number line and its errors
//   controls?: boolean   the toy: buttons for bits, scheme, clipping and block size, applied to the learner's own values
//                        (rt.input), computed here with quant.ts ledger()
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { ledger, LEDGER_DEFAULTS, type Ledger, type LedgerOpt } from '../../quant.ts';

/** The run's ledger options (the defaults, then the source's opt). */
export function optOf(p: Pick<StageProps, 'at'>): LedgerOpt & { alt: Partial<LedgerOpt> | null } {
	const o = (p.at?.tape?.opt ?? {}) as Partial<LedgerOpt> & { alt?: Partial<LedgerOpt> | null };
	return { ...LEDGER_DEFAULTS, ...o, alt: o.alt ?? null };
}

/** The frame's numbers, or null before the run. */
export function valsAt(p: Pick<StageProps, 'at'>): number[] | null {
	const x = (p.at?.state as { x?: { data: ArrayLike<number> } } | undefined)?.x;
	return x ? Array.from(x.data) : null;
}

/** The setting in words; `short` for the ruler labels ("clipped" without where, so the label fits the ruler). */
export function settingText(o: LedgerOpt, short = false): I18n {
	const clip = o.scheme === 'absmax' && o.clip === 'second';
	const blocks = o.block < 8;
	return {
		en: `int${o.bits}, ${o.scheme}${clip ? (short ? ', clipped' : ', clipped at the second-largest |x|') : ''}${blocks ? `, blocks of ${o.block}` : ''}`,
		zh: `int${o.bits}，${o.scheme === 'absmax' ? 'absmax' : '仿射'}${clip ? (short ? '，截断' : '，在第二大的 |x| 处截断') : ''}${blocks ? `，每 ${o.block} 个值一块` : ''}`
	};
}

export function describeLedger(L: Ledger | null, alt: Ledger | null): I18n {
	if (!L) return { en: 'No numbers yet.', zh: '还没有数。' };
	const one = (A: Ledger) => A.x.map((v, i) => `${fmt3(v)} → ${A.q[i]} → ${fmt3(A.xh[i])}`).join('; ');
	const s = settingText(L.opt);
	const en = `Rounding ledger, ${s.en}: ${one(L)}. Largest error ${fmt3(L.maxErr)}, ${fmt3(L.bitsPerValue)} bits per value.`;
	const zh = `舍入账本，${s.zh}：${one(L)}。最大误差 ${fmt3(L.maxErr)}，每个值 ${fmt3(L.bitsPerValue)} 位。`;
	if (!alt) return { en, zh };
	const a = settingText(alt.opt);
	return { en: `${en} Compared with ${a.en}: largest error ${fmt3(alt.maxErr)}.`, zh: `${zh} 对比 ${a.zh}：最大误差 ${fmt3(alt.maxErr)}。` };
}

export function ledgersOf(p: Pick<StageProps, 'at' | 'props'>): { L: Ledger; alt: Ledger | null } | null {
	const vals = valsAt(p);
	if (!vals) return null;
	const o = optOf(p);
	return { L: ledger(vals, o), alt: p.props?.alt && o.alt ? ledger(vals, { ...o, ...o.alt }) : null };
}

const meta = {
	key: 'quantization/ledger',
	title: { en: 'The rounding ledger', zh: '舍入账本' },
	kinds: ['process'],
	describe: (p: StageProps) => {
		const r = ledgersOf(p);
		return describeLedger(r?.L ?? null, r?.alt ?? null);
	},
	demo: async () => {
		const [{ record }, { ledger: proc }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const input = [0.31, -0.12, 0.05, -0.47, 0.22, 0.09, -0.26, 0.4];
		const tape = record(proc as never, input, { bits: 4 });
		const rt = fakeRuntime({ slug: 'quantization', input: { x: 0.1, vals: input }, tapes: { int4: tape } });
		return { rt, source: 'int4', at: { index: 1, frame: tape.frame(1), state: tape.state(1), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
