// tinyseq2seq view 'encoder': the learner's characters as columns, and under each one a column of numbers: its
// character vector (props.mode 'chars', layer emb) or the encoder state after reading it (mode 'read': a player reveals
// the columns left to right; mode 'last': the last column outlined, the only thing the decoder gets without attention).
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { chr, look, tr } from '../../look.ts';
import { demoProps } from '../../demo.ts';

type P = Omit<StageProps, 'ctx'>;

export function encoderOf(p: P) {
	const mode = String(p.props.mode ?? 'read');
	const L = look(p.rt);
	const x = tr(p.rt, mode === 'chars' ? 'emb' : 'enc');
	const D = x?.shape[1] ?? 0;
	const m = x ? Math.max(1e-6, ...Array.from(x.data, Math.abs)) : 1;
	const ids = Array.from(p.rt.pre?.tensors.input?.data ?? []);
	const last = x && L.T ? Array.from(x.data.subarray((L.T - 1) * D, L.T * D)) : [];
	return { mode, chars: L.chars.map(chr), ids, T: L.T, D, x, m, last };
}

const meta = {
	key: 'tinyseq2seq/encoder',
	title: { en: 'The encoder reads the date', zh: '编码器读日期' },
	kinds: ['trace'],
	describe: (p: P): I18n => {
		const E = encoderOf(p);
		if (!E.x) return { en: 'The encoder: nothing to show yet.', zh: '编码器：还没有可显示的内容。' };
		const what = E.mode === 'chars' ? { en: `its vector of ${E.D} learned numbers`, zh: `它的 ${E.D} 个学来的数组成的向量` } : { en: `the encoder state of ${E.D} numbers after reading it`, zh: `读完它之后由 ${E.D} 个数组成的编码器状态` };
		return {
			en: `${E.T} characters (${E.chars.join(' ')}); under each one, ${what.en}. The last column starts with ${E.last.slice(0, 3).map((v) => fmt3(v)).join(', ')}.`,
			zh: `${E.T} 个字符（${E.chars.join(' ')}）；每个字符下面是${what.zh}。最后一列的前几个数是 ${E.last.slice(0, 3).map((v) => fmt3(v)).join('、')}。`
		};
	},
	demo: async () => demoProps({ mode: 'read' }),
	minHeight: 300
} satisfies ViewMeta;

export default meta;
