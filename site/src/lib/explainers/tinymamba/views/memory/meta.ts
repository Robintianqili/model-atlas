// tinymamba view 'memory': what the model keeps versus what attention keeps.
//   matrix  the scan unrolled for one channel ch of block b: M[t][s] = Σ_n C_t[n] · exp(A[ch,n]·(Δ_{s+1}+…+Δ_t)) · Δ_s · B_s[n]
//           is how much token s's input u_s counts in the output at token t (y_t − D·u_t = Σ_s M[t][s]·u_s). Like an
//           attention matrix, but never stored: the scan only keeps its state.
//   bars    numbers kept: the model's state (blocks × channels × slots, the same for any length) and the keys and
//           values a Transformer of the same width d would keep (2 · d per token per layer), for this row and for 1,000 tokens.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { dims, idsOf, pickCh, T, tok } from '../../tm.ts';
import { demoProps } from '../../demo.ts';

type P = Omit<StageProps, 'ctx'>;

/** The unrolled scan of channel ch, block b: rows t (outputs), columns s ≤ t (inputs); null without the trace. */
export function unrolled(rt: P['rt'], b: number, ch: number): number[][] | null {
	const { D, N } = dims(rt);
	const dl = T(rt, `B${b}.scan#delta`);
	const Bm = T(rt, `B${b}.B`);
	const Cm = T(rt, `B${b}.C`);
	const Al = rt.model?.weights[`B${b}.scan.A_log`];
	if (!dl || !Bm || !Cm || !Al) return null;
	const n = dl.shape[0];
	const S: number[] = [];
	for (let t = 0, s = 0; t < n; t++) S.push((s += dl.data[t * D + ch]));
	return Array.from({ length: n }, (_, t) =>
		Array.from({ length: n }, (_, s) => {
			if (s > t) return 0;
			let m = 0;
			for (let k = 0; k < N; k++) m += Cm.data[t * N + k] * Math.exp(-Math.exp(Al.data[ch * N + k]) * (S[t] - S[s])) * dl.data[s * D + ch] * Bm.data[s * N + k];
			return m;
		})
	);
}

/** Numbers kept: the state of every block, and a same-width Transformer's keys and values for n tokens. */
export function memory(rt: P['rt'], n = idsOf(rt).length) {
	const { d, D, N, L } = dims(rt);
	return { state: L * D * N, kv: 2 * d * n * L, kv1000: 2 * d * 1000 * L, perToken: 2 * d * L, n, L, d, D, N };
}

export function memoryOf(p: P) {
	const b = Number(p.props.block ?? 1);
	const ch = typeof p.props.ch === 'number' ? p.props.ch : pickCh(p.rt, b);
	return { b, ch, M: unrolled(p.rt, b, ch), toks: idsOf(p.rt).map(tok), ...memory(p.rt) };
}

const meta = {
	key: 'tinymamba/memory',
	title: { en: 'A fixed state versus looking back', zh: '固定大小的状态与回头看' },
	kinds: ['trace'],
	describe: (p: P): I18n => {
		const m = memoryOf(p);
		const last = m.M?.[m.M.length - 1]?.map((v) => fmt3(v)).join(' ') ?? '';
		return {
			en: `The model keeps ${m.state} numbers for any length; attention of the same width would keep ${m.kv} for these ${m.n} tokens and ${m.kv1000} for 1,000. Channel ${m.ch} of block ${m.b}, unrolled: the last token's output counts each earlier input by ${last || 'nothing yet'}.`,
			zh: `模型无论序列多长都只保留 ${m.state} 个数；同样宽度的注意力对这 ${m.n} 个词元要保留 ${m.kv} 个数，对 1,000 个词元要 ${m.kv1000} 个。第 ${m.b} 块通道 ${m.ch} 展开后：最后一个词元的输出对前面每个输入的权重是 ${last || '暂无'}。`
		};
	},
	demo: async () => demoProps({ ch: 0 }),
	minHeight: 300
} satisfies ViewMeta;

export default meta;
