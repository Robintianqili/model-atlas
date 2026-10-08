// tinybert view 'guesses': TinyBERT's top guesses for the hidden token (the [MASK] row of its output), the token the
// learner hid drawn in the accent colour. props.compare: TinyGPT's guess for the same token next to it, made from the
// tokens before the gap only (its prediction after the token before the gap). props.k: how many bars (default 8).
// props.pick: the clickable token row (view 'chips') above the bars, so a task can be done in the stage.
// When the same word with or without a space in front ("␣Lily" / "Lily") ranks above the hidden token, a line says so.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmtPct, visibleToken } from '../../../../atlas/core/format.ts';
import { bertGuess, gap, gptGuess, twinOf } from '../../bert.ts';
import { demoProps } from '../../demo.ts';

type P = Omit<StageProps, 'ctx'>;
export type Bar = { label: string; p: number; id: number; color?: string };

export function guessesOf(p: P) {
	const g = gap(p.rt);
	const bars = (q: ArrayLike<number> | null): Bar[] | null => (q ? Array.from(q, (v, id) => ({ label: g.word(id), p: v, id, color: id === g.hidden ? 'var(--accent)' : undefined })) : null);
	const b = bertGuess(p.rt);
	const gp = p.props.compare ? gptGuess(p.rt) : null;
	const say = (who: string, x: ReturnType<typeof bertGuess>, zh: string): I18n => ({ en: `${who}: top “${g.say(x.top)}” ${fmtPct(x.pTop)}; “${g.say(g.hidden)}” ${fmtPct(x.pHidden)} (rank ${x.rank}).`, zh: `${zh}：第一名“${g.say(x.top)}” ${fmtPct(x.pTop)}；“${g.say(g.hidden)}” ${fmtPct(x.pHidden)}（第 ${x.rank} 名）。` });
	const tw = twinOf(p.rt, g.hidden);
	const twRank = tw >= 0 ? b.order.indexOf(tw) + 1 : 0;
	const vis = (id: number) => visibleToken(g.word(id));
	const twin: I18n | null =
		b.p && twRank > 0 && twRank < b.rank
			? { en: `“${vis(tw)}” (${fmtPct(b.p[tw])}, rank ${twRank}) is the same word ${g.word(tw).startsWith(' ') ? 'with' : 'without'} a space in front (␣): for the model it is a different token from “${vis(g.hidden)}”.`, zh: `“${vis(tw)}”（${fmtPct(b.p[tw])}，第 ${twRank} 名）是同一个词${g.word(tw).startsWith(' ') ? '前面多了' : '前面少了'}一个空格（␣）：对模型来说，它和“${vis(g.hidden)}”是两个不同的词元。` }
			: null;
	const lines = [b.p ? say('TinyBERT', b, 'TinyBERT') : null, twin, gp?.p ? say('TinyGPT', gp, 'TinyGPT') : null].filter((x): x is I18n => !!x);
	return { bert: bars(b.p), gpt: bars(gp?.p ?? null), compare: !!p.props.compare, pick: !!p.props.pick, first: g.k === 0, k: typeof p.props.k === 'number' ? p.props.k : 8, hidden: g.say(g.hidden), lines };
}

const meta = {
	key: 'tinybert/guesses',
	title: { en: 'Guesses for the hidden token', zh: '对被藏词元的猜测' },
	kinds: ['trace'],
	describe: (p: P): I18n => {
		const G = guessesOf(p);
		if (!G.bert) return { en: 'Guesses for the hidden token: nothing to show yet.', zh: '对被藏词元的猜测：还没有可显示的内容。' };
		return { en: `You hid “${G.hidden}”. ${G.lines.map((l) => l.en).join(' ')}`, zh: `你藏起了“${G.hidden}”。${G.lines.map((l) => l.zh).join('')}` };
	},
	demo: async () => demoProps({ compare: true }),
	minHeight: 300
} satisfies ViewMeta;

export default meta;
