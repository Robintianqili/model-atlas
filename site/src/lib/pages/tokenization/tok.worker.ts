/**
 * Web Worker for the tokenization page: trains the toy BPE and runs the real tokenizers, so the main
 * thread stays responsive while files are parsed (100k–150k entries) and long texts are processed.
 *
 * Messages in:  { req, type: 'train', text, maxMerges, pattern, tinyMerges }
 *               { req, type: 'load', id, url }
 *               { req, type: 'encode', texts: string[], ids: RealId[], counts?: boolean }
 * Messages out: { req, ok: true, ... } or { req, ok: false, error }
 */
import { trainTrace } from './train.ts';
import { ByteBPE, WordPiece, parseTokb, gpt2Pattern, cl100kPattern, qwenPattern, type AnyTokenizer, type RealResult } from './realtok.ts';
import { REAL_BY_ID, type RealId } from './tokenizers.ts';

const loaded = new Map<RealId, AnyTokenizer>();
const loading = new Map<RealId, Promise<AnyTokenizer>>();

async function load(id: RealId, url: string): Promise<AnyTokenizer> {
	const spec = REAL_BY_ID[id];
	const res = await fetch(url);
	if (!res.ok) throw new Error(`${spec.file}: HTTP ${res.status}`);
	if (spec.algo === 'wordpiece') return WordPiece.fromText(await res.text(), { lowercase: !!spec.lowercase });
	const vocab = parseTokb(await res.arrayBuffer());
	const re = spec.pattern === 'gpt2' ? gpt2Pattern() : spec.pattern === 'cl100k' ? cl100kPattern() : qwenPattern();
	return new ByteBPE(vocab, re, { nfc: spec.nfc, extra: spec.vocab - vocab.length });
}

type Msg =
	| { req: number; type: 'train'; text: string; maxMerges: number; pattern: string; tinyMerges: [number, number][] }
	| { req: number; type: 'load'; id: RealId; url: string }
	| { req: number; type: 'encode'; texts: string[]; ids: RealId[]; counts?: boolean };

const ctx = self as unknown as { postMessage(m: unknown): void; onmessage: ((e: MessageEvent<Msg>) => void) | null };

ctx.onmessage = async (e) => {
	const m = e.data;
	try {
		if (m.type === 'train') {
			const t0 = performance.now();
			const trace = trainTrace(m.text, m.maxMerges, m.pattern, { tinyMerges: m.tinyMerges });
			ctx.postMessage({ req: m.req, ok: true, trace, ms: performance.now() - t0 });
		} else if (m.type === 'load') {
			const t0 = performance.now();
			let p = loading.get(m.id);
			if (!p) {
				p = load(m.id, m.url);
				loading.set(m.id, p);
				p.catch(() => loading.delete(m.id));
			}
			const tok = await p;
			loaded.set(m.id, tok);
			ctx.postMessage({ req: m.req, ok: true, id: m.id, ms: performance.now() - t0 });
		} else if (m.type === 'encode') {
			// results[textIndex][id] = full result (or only [count, unk] when counts is set)
			const results: Record<string, RealResult | [number, number]>[] = [];
			const ms: Record<string, number> = {};
			for (const text of m.texts) {
				const row: Record<string, RealResult | [number, number]> = {};
				for (const id of m.ids) {
					const tok = loaded.get(id);
					if (!tok) continue;
					const t0 = performance.now();
					const r = tok.encode(text);
					ms[id] = (ms[id] ?? 0) + performance.now() - t0;
					row[id] = m.counts ? [r.count, r.unk] : r;
				}
				results.push(row);
			}
			ctx.postMessage({ req: m.req, ok: true, results, ms });
		}
	} catch (err) {
		ctx.postMessage({ req: m.req, ok: false, error: err instanceof Error ? err.message : String(err) });
	}
};
