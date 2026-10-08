/**
 * Parity: the tokenization page's JS tokenizers vs the Python references.
 *   PY=".../python.exe"; $PY pipeline/tokenizers/make_parity_cases.py
 *   node pipeline/tokenizers/parity.ts [siteRoot]      (default: site/)
 * Exits 1 on any mismatch. Also checks that byte offsets tile the text for byte-level BPE.
 */
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const siteRoot = process.argv[2] ?? join(here, '..', '..', 'site');
const rt = await import(join(siteRoot, 'src/lib/pages/tokenization/realtok.ts').replace(/\\/g, '/').replace(/^([A-Za-z]):/, 'file:///$1:'));
const { ByteBPE, WordPiece, parseTokb, gpt2Pattern, cl100kPattern, qwenPattern } = rt;

const tokDir = join(siteRoot, 'static/pages/tokenization/tok');
const ab = (f: string) => {
	const b = readFileSync(join(tokDir, f));
	return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
};
const t0 = performance.now();
const toks: Record<string, any> = {
	gpt2: new ByteBPE(parseTokb(ab('gpt2.tokb')), gpt2Pattern()),
	cl100k_base: new ByteBPE(parseTokb(ab('cl100k_base.tokb')), cl100kPattern()),
	'qwen2.5': new ByteBPE(parseTokb(ab('qwen2.5.tokb')), qwenPattern(), { nfc: true }),
	'bert-base-uncased': WordPiece.fromText(readFileSync(join(tokDir, 'bert-base-uncased.txt'), 'utf8'), { lowercase: true })
};
const loadMs = performance.now() - t0;

const data = JSON.parse(readFileSync(join(here, 'parity_cases.json'), 'utf8'));
const enc = new TextEncoder();
let fail = 0;
const perTok: Record<string, { ok: number; n: number }> = {};
const t1 = performance.now();
for (const c of data.cases) {
	for (const name of Object.keys(toks)) {
		const ref: number[] = c[name];
		const res = toks[name].encode(c.text);
		const got: number[] = res.tokens.map((t: { id: number }) => t.id);
		perTok[name] ??= { ok: 0, n: 0 };
		perTok[name].n++;
		const same = got.length === ref.length && got.every((v, i) => v === ref[i]);
		if (same) perTok[name].ok++;
		else {
			fail++;
			if (fail <= 12) console.log(`MISMATCH ${name} ${JSON.stringify(c.text)}\n  ref ${JSON.stringify(ref)}\n  got ${JSON.stringify(got)}`);
		}
		// offsets: byte-level tokens tile the (normalized) UTF-8 exactly
		if (name !== 'bert-base-uncased') {
			const norm = name === 'qwen2.5' ? c.text.normalize('NFC') : c.text;
			const total = enc.encode(norm).length;
			let o = 0;
			for (const t of res.tokens) {
				if (t.start !== o) throw new Error(`offset gap in ${name} for ${JSON.stringify(c.text)}`);
				o = t.end;
			}
			if (o !== total) throw new Error(`offsets do not cover ${name} ${JSON.stringify(c.text)}`);
		} else {
			const total = enc.encode(c.text).length;
			let last = 0;
			for (const t of res.tokens) {
				if (t.start < last || t.end < t.start || t.end > total) throw new Error(`bad BERT offsets ${JSON.stringify(c.text)}`);
				last = t.start;
			}
		}
	}
}
const encMs = performance.now() - t1;
console.log(`cases: ${data.cases.length}; references: tiktoken ${data.versions.tiktoken}, tokenizers ${data.versions.tokenizers}`);
for (const [k, v] of Object.entries(perTok)) console.log(`  ${k.padEnd(18)} ${v.ok}/${v.n} identical`);
console.log(`gpt2: tiktoken vs HF reference disagreements: ${data.gpt2_tiktoken_vs_hf_disagree.length}`);
console.log(`load ${loadMs.toFixed(0)} ms, encode all ${encMs.toFixed(0)} ms`);
console.log(fail ? `FAIL (${fail} mismatches)` : 'PASS');
process.exit(fail ? 1 : 0);
