// View 'quantization/formats': one number x in five formats, bit by bit: fp32 (1 sign, 8 exponent, 23 fraction bits),
// fp16 (1, 5, 10), bf16 (1, 8, 7), and int8 / int4 codes on the fixed range −1..1 (two's complement). Each row shows the
// value the format stores and its error. x is the frame's metric x (process 'formats'). Props:
//   flip?: boolean   the fp32 bits are buttons: a press flips that bit of x (needs setInput)
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { formatsOf, type FormatRow } from '../../quant.ts';

export const NAMES: Record<string, I18n> = {
	fp32: { en: 'fp32 (32-bit float)', zh: 'fp32（32 位浮点）' },
	fp16: { en: 'fp16 (16-bit float)', zh: 'fp16（16 位浮点）' },
	bf16: { en: 'bf16 (bfloat16)', zh: 'bf16（bfloat16）' },
	int8: { en: 'int8 on −1..1', zh: 'int8（范围 −1..1）' },
	int4: { en: 'int4 on −1..1', zh: 'int4（范围 −1..1）' }
};

/** A number as text, also when it is not finite. */
export function show(v: number, digits = 6): string {
	if (Number.isNaN(v)) return 'NaN';
	if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
	if (v === 0) return '0';
	const a = Math.abs(v);
	if (a >= 1e-4 && a < 1e7) return String(Number(v.toPrecision(digits))).replace('-', '−');
	return fmt3(v, digits);
}

export function rowsOf(p: Pick<StageProps, 'at'>): { x: number; rows: FormatRow[] } | null {
	const x = p.at?.frame?.metrics?.x;
	if (typeof x !== 'number') return null;
	return { x, rows: formatsOf(x) };
}

export function describeFormats(r: ReturnType<typeof rowsOf>): I18n {
	if (!r) return { en: 'No number yet.', zh: '还没有数。' };
	const en = r.rows.map((f) => `${f.id}: ${f.total} bits ${f.groups.map((g) => g.bits).join(' ')}, stores ${show(f.value)}, error ${show(f.err)}`).join('; ');
	const zh = r.rows.map((f) => `${f.id}：${f.total} 位 ${f.groups.map((g) => g.bits).join(' ')}，存下 ${show(f.value)}，误差 ${show(f.err)}`).join('；');
	return { en: `The number ${show(r.x)} in five formats. ${en}.`, zh: `数 ${show(r.x)} 的五种格式。${zh}。` };
}

const meta = {
	key: 'quantization/formats',
	title: { en: 'One number, five formats, bit by bit', zh: '一个数，五种格式，逐位查看' },
	kinds: ['process'],
	describe: (p: StageProps) => describeFormats(rowsOf(p)),
	demo: async () => {
		const [{ record }, { formats }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const tape = record(formats, 0.1, {});
		const rt = fakeRuntime({ slug: 'quantization', input: { x: 0.1 }, tapes: { formats: tape } });
		return { rt, source: 'formats', at: { index: 0, frame: tape.frame(0), state: tape.state(0), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 260
} satisfies ViewMeta;

export default meta;
