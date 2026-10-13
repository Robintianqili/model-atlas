// View 'attention-variants/attn': TinyGPT's attention in block 2 (process 'share') for the last token of the probe prompt,
// the query that predicts the next token: a heat table of the four query heads' weights over the probe's positions as
// trained (MHA, the tape's frame 0) and, beside it, at the step's frame (key/value heads shared). Below, one head's
// computation: for every position its score q·k/√d and its weight, as trained and shared. Props:
//   head?: number   the head whose computation is listed (default 0)
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtPct } from '../../../../atlas/core/format.ts';
import { PROBE_IDS, VOCAB } from '../../evalset.ts';

/** A token's text for display (a leading space shown as ␣). */
export const tokText = (id: number) => (VOCAB[id] ?? `#${id}`).replace(/\n/g, '↵').replace(/^ /, '␣');
export const TOKENS = PROBE_IDS.map(tokText);
/** The probe's last token, the query whose weights the view shows. */
export const LAST = (VOCAB[PROBE_IDS[PROBE_IDS.length - 1]] ?? '').trim();

const rows = (t: unknown, H = 4) => {
	const d = (t as { data?: ArrayLike<number> } | undefined)?.data;
	if (!d) return [] as number[][];
	const T = d.length / H;
	return Array.from({ length: H }, (_, h) => Array.from({ length: T }, (_, j) => d[h * T + j]));
};
/** The weights and scores [head][position] at the step's frame and at frame 0 (as trained). */
export function attnOf(p: Pick<StageProps, 'at'>) {
	const s = (p.at?.state ?? {}) as Record<string, unknown>;
	const s0 = (p.at?.tape?.state(0) ?? {}) as Record<string, unknown>;
	return { w: rows(s.w), s: rows(s.s), w0: rows(s0.w), s0: rows(s0.s), kv: p.at?.frame?.metrics?.kv ?? 4, layer: p.at?.frame?.metrics?.layer ?? NaN, shared: (p.at?.index ?? 0) > 0 };
}

export function describeAttn(p: Pick<StageProps, 'at' | 'props'>): I18n {
	const a = attnOf(p);
	if (!a.w.length) return { en: 'TinyGPT is not ready yet.', zh: 'TinyGPT 还没有准备好。' };
	const line = (w: number[][]) => w.map((r, h) => `head ${h}: ` + r.map((v, j) => `${TOKENS[j]} ${fmtPct(v)}`).join(', ')).join('; ');
	const lineZh = (w: number[][]) => w.map((r, h) => `头 ${h}：` + r.map((v, j) => `${TOKENS[j]} ${fmtPct(v)}`).join('，')).join('；');
	const h = Number(p.props?.head ?? 0);
	const comp = a.s[h]?.map((v, j) => `${TOKENS[j]} ${fmt3(v)}`).join(', ') ?? '';
	return a.shared
		? { en: `Attention weights of the last token in block 2 (counting from 0) as trained: ${line(a.w0)}. With ${a.kv} key/value heads: ${line(a.w)}. Head ${h}'s scores: ${comp}.`, zh: `编号 2 的块（从 0 数起）中最后一个词元的注意力权重，训练好的：${lineZh(a.w0)}。只有 ${a.kv} 个键值头时：${lineZh(a.w)}。头 ${h} 的分数：${comp}。` }
		: { en: `Attention weights of the last token in block 2 (counting from 0):${line(a.w)}. Head ${h}'s scores: ${comp}.`, zh: `编号 2 的块（从 0 数起）中最后一个词元的注意力权重：${lineZh(a.w)}。头 ${h} 的分数：${comp}。` };
}

const meta = {
	key: 'attention-variants/attn',
	title: { en: 'TinyGPT’s attention with shared keys and values', zh: '共用键和值时 TinyGPT 的注意力' },
	kinds: ['process'],
	describe: (p: StageProps) => describeAttn(p),
	demo: async () => {
		const { fakeRuntime } = await import('../../../../atlas/runtime/fake.ts');
		const { TapeStore } = await import('../../../../atlas/process/tape.ts');
		const { share } = await import('../../process.ts');
		const T = PROBE_IDS.length;
		const tape = new TapeStore({ process: 'share', schema: share.schema, phases: share.phases, provenance: 'illustrative' });
		const t = (f: (h: number, j: number) => number) => ({ shape: [4, T], data: Float64Array.from({ length: 4 * T }, (_, i) => f(Math.floor(i / T), i % T)) });
		tape.push({ phase: 'mha', step: 0, state: { w: t((h, j) => (j === T - 1 - h ? 0.6 : 0.4 / (T - 1))), s: t((h, j) => (j === T - 1 - h ? 3 : 0)) }, metrics: { kv: 4 } } as never);
		tape.push({ phase: 'shared', step: 0, state: { w: t(() => 1 / T), s: t(() => 0) }, metrics: { kv: 2 } } as never);
		const rt = fakeRuntime({ slug: 'attention-variants', input: {}, tapes: { share: tape } });
		return { rt, source: 'share', at: { index: 1, frame: tape.frame(1), state: tape.state(1), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: { head: 0 } };
	},
	minHeight: 420
} satisfies ViewMeta;

export default meta;
