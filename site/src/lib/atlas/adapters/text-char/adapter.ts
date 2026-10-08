// text-char (design §4.9; catalog InputKind 'text'): text -> one id per character, as an i32 tensor [T] (char-rnn).
// The vocabulary comes from options.vocab, else the spec's `vocab` / `chars` / `alphabet` extra key (a string or a list of
// strings, the index is the id; the char-seq template writes `alphabet`). Characters outside it are mapped to options.unk / spec.unk when the model has an unknown id, otherwise
// dropped, and the stage says which. An input longer than spec.dims.T.max keeps its LAST characters.
// K1: typographic characters a phone keyboard types (curly quotes, dashes, the ellipsis, a no-break space) become their
// plain forms when the alphabet lacks them and has the plain ones (options.normalize: false turns it off), and presets()
// reads presets.json: { texts: [{ id, en, zh, text }] } (or presets: [...]).
import type { I18n } from '../../../i18n/types.ts';
import type { IntTensor, Tensor } from '../../types/core.ts';
import type { InputAdapter } from '../../types/explainer.ts';
import type { Preprocessed } from '../../types/runtime.ts';

export type TextCharRaw = null | undefined | string | { text?: string };

/** The id of every character, from options.vocab or the spec's vocab / chars / alphabet key. */
export function charVocab(spec: unknown, opts: Record<string, unknown> = {}): { chars: string[]; index: Map<string, number>; unk: number | null } {
	const s = (spec ?? {}) as Record<string, unknown>;
	const v = opts.vocab ?? s.vocab ?? s.chars ?? s.alphabet;
	const chars = typeof v === 'string' ? [...v] : Array.isArray(v) ? v.map(String) : [];
	const index = new Map<string, number>();
	chars.forEach((c, i) => {
		if (!index.has(c)) index.set(c, i);
	});
	const u = opts.unk ?? s.unk;
	const unk = typeof u === 'number' && u >= 0 && u < chars.length ? u : typeof u === 'string' && index.has(u) ? index.get(u)! : null;
	return { chars, index, unk };
}

function contextOf(spec: unknown): number {
	const m = (spec as { dims?: Record<string, { max?: number }> } | null)?.dims?.T?.max;
	return typeof m === 'number' && m > 0 ? m : Infinity;
}

const show = (c: string) => (c === ' ' ? '␣' : c === '\n' ? '↵' : c === '\t' ? '⇥' : c);

/** Plain forms of typographic characters (K1): used only when the alphabet lacks the character and has its plain form. */
export const PLAIN: Record<string, string> = { '“': '"', '”': '"', '„': '"', '‘': "'", '’': "'", '‚': "'", '–': '-', '—': '-', '−': '-', '…': '...', ' ': ' ', ' ': ' ', '\r': '' };

export const textChar: InputAdapter<TextCharRaw> = {
	id: 'text-char',
	kind: 'text',
	preprocess(raw, spec, opts) {
		const text = raw === null || raw === undefined ? '' : typeof raw === 'string' ? raw : (raw.text ?? '');
		const { chars, index, unk } = charVocab(spec, opts ?? {});
		if (!chars.length) throw new Error('text-char: no vocabulary (give options.vocab or a `vocab` / `chars` / `alphabet` key in spec.json)');
		const all: number[] = [];
		const unknown = new Set<string>();
		const plain = (opts ?? {}).normalize !== false;
		for (const ch of text) {
			const id = index.get(ch);
			if (id !== undefined) all.push(id);
			else if (plain && PLAIN[ch] !== undefined && [...PLAIN[ch]].every((c) => index.has(c))) for (const c of PLAIN[ch]) all.push(index.get(c)!);
			else {
				unknown.add(ch);
				if (unk !== null) all.push(unk);
			}
		}
		const C = contextOf(spec);
		const ids = all.length > C ? all.slice(all.length - C) : all;
		const input: IntTensor = { shape: [ids.length], data: Int32Array.from(ids), dtype: 'i32' };
		const shown: Tensor = { shape: [ids.length], data: Float32Array.from(ids) };
		const notes: I18n[] = [];
		if (unknown.size) {
			const list = [...unknown].slice(0, 8).map(show).join(' ');
			notes.push(
				unk !== null
					? { en: `These characters are not in the model’s alphabet and became the “unknown” id: ${list}`, zh: `这些字符不在模型的字母表里，被换成了“未知”id：${list}` }
					: { en: `These characters are not in the model’s alphabet and were left out: ${list}`, zh: `这些字符不在模型的字母表里，被去掉了：${list}` }
			);
		}
		const dropped = all.length - ids.length;
		if (dropped > 0)
			notes.push({ en: `Only the last ${ids.length} characters are used; the first ${dropped} were dropped.`, zh: `只用最后 ${ids.length} 个字符；前面 ${dropped} 个被去掉了。` });
		if (!ids.length) notes.push({ en: 'There is no text yet. Type something and every step fills with your characters.', zh: '还没有文字。输入一些内容，每一步都会填上你的字符。' });
		const out: Preprocessed = {
			tensors: { input },
			steps: [{ id: 'ids', label: { en: 'Character ids', zh: '字符 id' }, tensor: shown, note: { en: 'One number per character: its place in the alphabet.', zh: '每个字符一个数：它在字母表里的位置。' } }],
			empty: ids.length === 0
		};
		if (notes.length) out.notes = notes;
		// token text for the views and {top.label} (types revision 1b: Preprocessed.tokens / vocab)
		out.tokens = ids.map((id) => chars[id] ?? '');
		out.vocab = (id: number) => chars[id] ?? `#${id}`;
		return out;
	},
	presets(assets) {
		const f = (assets?.['presets.json'] ?? assets?.presets) as { texts?: unknown; presets?: unknown } | undefined;
		const list = (Array.isArray(f?.texts) ? f.texts : Array.isArray(f?.presets) ? f.presets : []) as { id?: string; en?: string; zh?: string; text?: string }[];
		return list.filter((p) => typeof p.text === 'string').map((p, i) => ({ id: p.id ?? `text-${i}`, label: { en: p.en ?? p.text!.slice(0, 24), zh: p.zh ?? p.en ?? p.text!.slice(0, 24) }, raw: p.text! }));
	},
	edgeFixtures() {
		return [
			{ id: 'empty', label: { en: 'no text', zh: '没有文字' }, raw: '' },
			{ id: 'one', label: { en: 'one character', zh: '一个字符' }, raw: 'e' },
			{ id: 'long', label: { en: 'a long text (272 characters)', zh: '一段长文字（272 个字符）' }, raw: 'once upon a time '.repeat(16) }
		];
	},
	layer0: {
		view: 'kit.tokens',
		narration: {
			what: {
				en: 'The model reads your text one character at a time. Each of the {n|int} characters becomes one id number.',
				zh: '模型一次读一个字符。你的 {n|int} 个字符每个都变成一个 id 数字。'
			},
			look: { en: 'that spaces and punctuation are characters too.', zh: '空格和标点也是字符。' }
		}
	}
};

export default textChar;
