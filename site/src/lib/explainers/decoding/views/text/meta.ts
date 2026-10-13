// View 'decoding/text': a decode run as text, token by token, up to the step's frame: the prompt, then each new token
// shaded by the chance TinyGPT gave it at temperature 1 (the purple ramp), the newest outlined; under it, the ten most
// likely tokens of the distribution the newest token was drawn from, with the random number u. Props:
//   loop?: boolean   underline every copy of the longest phrase that comes back (decode.ts longestLoop)
//   rare?: number    a dashed box around every new token the model gave less than this chance (temperature 1)
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import type { Tape } from '../../../../atlas/types/process.ts';
import { fmtPct } from '../../../../atlas/core/format.ts';
import { label, longestLoop, textOf, tokOf, type Loop } from '../../decode.ts';

type P = Pick<StageProps, 'rt' | 'at' | 'props'>;
export interface TextPic {
	prompt: { id: number; text: string }[]; made: { id: number; text: string; p: number; pModel: number }[];
	top: { id: number; text: string; p: number }[]; token: number; u: number; greedy: boolean; loop: (Loop & { phrase: string }) | null;
}

/** The generated tokens of a decode tape up to frame i: ids, and per token its chances (p, pModel). */
export function runOf(tape: Tape | undefined, i: number) {
	if (!tape || tape.length === 0 || i < 0) return null;
	const promptLen = tape.frame(0).metrics?.length ?? 0;
	const st = tape.state(Math.min(i, tape.length - 1)) as { ids?: number[] };
	const ids = st.ids ?? [];
	const made: { id: number; p: number; pModel: number; u: number }[] = [];
	for (let j = 1; j <= Math.min(i, tape.length - 1); j++) {
		const m = tape.frame(j).metrics ?? {};
		made.push({ id: ids[promptLen + j - 1], p: m.p, pModel: m.pModel, u: m.u });
	}
	return { promptIds: ids.slice(0, promptLen), made };
}

export function textPicOf(p: P): TextPic | null {
	const tape = p.at?.tape;
	const r = runOf(tape, p.at?.index ?? -1);
	if (!r) return null;
	const tok = tokOf(p.rt);
	const st = (p.at?.state ?? {}) as { top?: { id: number; p: number }[]; token?: number; u?: number };
	const madeIds = r.made.map((m) => m.id);
	const L = p.props?.loop ? longestLoop(madeIds) : null;
	const last = r.made[r.made.length - 1];
	return {
		prompt: r.promptIds.map((id) => ({ id, text: tok ? tok.decode([id]) : `#${id}` })),
		made: r.made.map((m) => ({ id: m.id, text: tok ? (tok.isSpecial(m.id) ? '⟨end⟩' : tok.decode([m.id])) : `#${m.id}`, p: m.p, pModel: m.pModel })),
		top: (st.top ?? []).filter((e) => e.p > 0).map((e) => ({ id: e.id, text: label(tok, e.id), p: e.p })), token: typeof st.token === 'number' ? st.token : -1,
		u: last ? last.u : NaN, greedy: !!last && !Number.isFinite(last.u),
		loop: L ? { ...L, phrase: textOf(tok, madeIds.slice(L.start, L.start + L.len)) } : null
	};
}

export function describeText(pic: TextPic | null, rt: StageProps['rt'] | undefined): I18n {
	if (!pic) return { en: 'TinyGPT has not written anything yet: there is no prompt.', zh: 'TinyGPT 还没写出任何东西：还没有提示。' };
	const tok = tokOf(rt);
	const made = textOf(tok, pic.made.map((m) => m.id));
	const prompt = textOf(tok, pic.prompt.map((m) => m.id));
	const top = pic.top.slice(0, 5).map((e) => `“${e.text}” ${fmtPct(e.p)}`).join(', ');
	const loop = pic.loop ? ` The phrase “${pic.loop.phrase}” comes back ${pic.loop.at.length} times.` : '';
	const loopZh = pic.loop ? `短语“${pic.loop.phrase}”出现了 ${pic.loop.at.length} 次。` : '';
	return {
		en: `After the prompt “${prompt}”, TinyGPT wrote ${pic.made.length} tokens: “${made}”.${loop}${top ? ` The newest token was drawn from: ${top}.` : ''}`,
		zh: `在提示“${prompt}”之后，TinyGPT 写了 ${pic.made.length} 个词元：“${made}”。${loopZh}${top ? `最新的词元是从这些候选里抽出的：${top}。` : ''}`
	};
}

const meta = {
	key: 'decoding/text',
	title: { en: 'The text, token by token', zh: '逐个词元写出的文字' },
	kinds: ['process'],
	describe: (p: StageProps) => describeText(textPicOf(p), p.rt),
	demo: async () => {
		const { fakeRuntime } = await import('../../../../atlas/runtime/fake.ts');
		// made-up frames, written by hand (no tokenizer: ids show as #i)
		const ids = [1, 2, 3];
		const frames = [
			{ phase: 'prompt', state: { ids }, metrics: { length: 3 } },
			{ phase: 'sample', step: 0, state: { ids: [...ids, 4], token: 4, top: [{ id: 4, p: 0.6 }, { id: 5, p: 0.4 }] }, metrics: { p: 0.6, pModel: 0.5, u: 0.3 } },
			{ phase: 'sample', step: 1, state: { ids: [...ids, 4, 5], token: 5, top: [{ id: 5, p: 0.7 }, { id: 6, p: 0.3 }] }, metrics: { p: 0.7, pModel: 0.6, u: 0.2 } }
		];
		const tape = {
			process: 'decode', seed: 7, opt: {}, provenance: 'illustrative', version: 'demo', length: frames.length,
			frame: (i: number) => frames[i], state: (i: number) => frames[i].state, find: () => 0, segments: () => [], metric: () => new Float64Array(0)
		} as unknown as Tape;
		const rt = fakeRuntime({ slug: 'decoding', tapes: { sample: tape } });
		return { rt, source: 'sample', at: { index: 2, frame: tape.frame(2), state: tape.state(2), tape }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
