// self-supervised-objectives view 'guesses' (adapted from tinybert's 'guesses', credit: the tinybert package): the two
// live models' guesses for the hidden token, the token the learner hid drawn in the accent colour.
//   props.mode 'clm':  TinyGPT only, from the tokens before the gap (its prediction after the token before the gap), with
//                      the 'chips' row in 'clm' mode above it (what TinyGPT may read).
//   props.mode 'mlm':  TinyBERT (both sides of the gap) next to TinyGPT, with the clickable 'chips' row above them.
//   props.k:           how many bars (default 6).
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmtPct } from '../../../../atlas/core/format.ts';
import { vocab } from '../../adapters/masked-text/adapter.ts';
import { bertGuess, gap, gptGuess, say } from '../../text.ts';
import { demoProps } from '../../demo.ts';

type P = Omit<StageProps, 'ctx'>;
export type Bar = { label: string; p: number; id: number; color?: string };

export function guessesOf(p: P) {
	const g = gap(p.rt);
	const mlm = p.props.mode === 'mlm';
	const bars = (q: ArrayLike<number> | null): Bar[] | null => (q ? Array.from(q, (v, id) => ({ label: vocab(id), p: v, id, color: id === g.hidden ? 'var(--accent)' : undefined })) : null);
	const b = mlm ? bertGuess(p.rt) : null;
	const gp = gptGuess(p.rt);
	const line = (who: string, x: ReturnType<typeof gptGuess>): I18n => ({ en: `${who}: top “${say(x.top)}” ${fmtPct(x.pTop)}; “${say(g.hidden)}” ${fmtPct(x.pHidden)} (rank ${x.rank}).`, zh: `${who}：第一名“${say(x.top)}” ${fmtPct(x.pTop)}；“${say(g.hidden)}” ${fmtPct(x.pHidden)}（第 ${x.rank} 名）。` });
	const lines = [b?.p ? line('TinyBERT', b) : null, gp.p ? line('TinyGPT', gp) : null].filter((x): x is I18n => !!x);
	return { mlm, bert: bars(b?.p ?? null), gpt: bars(gp.p), first: g.k === 0, k: typeof p.props.k === 'number' ? p.props.k : 6, hidden: g.k >= 0 ? say(g.hidden) : '', lines };
}

const meta = {
	key: 'self-supervised-objectives/guesses',
	title: { en: 'Guesses for the hidden token', zh: '对被藏词元的猜测' },
	kinds: ['trace'],
	describe: (p: P): I18n => {
		const G = guessesOf(p);
		if (!G.hidden) return { en: 'Guesses for the hidden token: nothing to show yet.', zh: '对被藏词元的猜测：还没有可显示的内容。' };
		return { en: `The hidden token is “${G.hidden}”. ${G.lines.map((l) => l.en).join(' ')}`, zh: `被藏的词元是“${G.hidden}”。${G.lines.map((l) => l.zh).join('')}` };
	},
	demo: async () => demoProps({ mode: 'mlm' }),
	minHeight: 300
} satisfies ViewMeta;

export default meta;
