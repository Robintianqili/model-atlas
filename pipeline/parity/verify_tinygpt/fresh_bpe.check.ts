// Verification: TypeScript tokenizer (bpe.ts) vs Python (bpe.py) on fresh strings written by the
// verifier (make_fresh_bpe.py): ids, decode round trip, encodeWithSteps.
// Run: node pipeline/parity/verify_tinygpt/fresh_bpe.check.ts
import { readFileSync } from 'node:fs';
import { BPETokenizer, trainBPE } from '../../../site/src/lib/models/bpe.ts';

const root = new URL('../../../', import.meta.url);
const read = (p: string) => JSON.parse(readFileSync(new URL(p, root), 'utf8'));
const tok = new BPETokenizer(read('site/static/models/tinygpt/tokenizer.json'));
const data = read('pipeline/parity/verify_tinygpt/fresh_bpe_cases.json');
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

let idFails = 0, rtFails = 0, decFails = 0, nTok = 0, longest = 0;
const shown: string[] = [];
for (const c of data.cases) {
  const ids = tok.encode(c.text);
  nTok += ids.length;
  longest = Math.max(longest, ids.length);
  if (!same(ids, c.ids)) {
    idFails++;
    if (shown.length < 8) {
      const cps = Array.from(c.text as string).map((ch) => 'U+' + ch.codePointAt(0)!.toString(16).toUpperCase());
      shown.push(`ids differ: ${JSON.stringify(c.text).slice(0, 80)} cps=${cps.slice(0, 12).join(' ')}\n   ts ${JSON.stringify(ids).slice(0, 200)}\n   py ${JSON.stringify(c.ids).slice(0, 200)}\n   ts pretokens ${JSON.stringify(tok.pretokenize(c.text)).slice(0, 200)}`);
    }
  }
  if (tok.decode(ids) !== c.text) rtFails++;
  if (tok.decode(c.ids) !== c.decoded) decFails++;
}
let stepFails = 0;
for (const s of data.encodeWithSteps) if (!same(tok.encodeWithSteps(s.text), s.result)) stepFails++;
let trainFails = 0;
for (const t of data.train) {
  const r = trainBPE(t.text, t.numMerges, tok.pattern);
  if (!same(r.merges, t.merges) || !same(r.steps.map((x) => x.count), t.counts)) trainFails++;
}
let cutFails = 0;
const cutToks = new Map<number, BPETokenizer>();
for (const c of data.cut) {
  if (!cutToks.has(c.k)) cutToks.set(c.k, BPETokenizer.fromMerges(tok.merges.slice(0, c.k), tok.pattern, ['<|endoftext|>']));
  if (!same(cutToks.get(c.k)!.encode(c.text), c.ids)) cutFails++;
}
let decFuzzFails = 0;
for (const d of data.decode) if (tok.decode(d.ids) !== d.text) decFuzzFails++;
console.log(`trainBPE fresh texts: ${data.train.length - trainFails}/${data.train.length} identical; tokenizer cut to k merges: ${data.cut.length - cutFails}/${data.cut.length}; decode fuzz (partial UTF-8): ${data.decode.length - decFuzzFails}/${data.decode.length}`);
console.log(`fresh strings: ${data.cases.length}, ${nTok} tokens, longest ${longest} tokens`);
console.log(`ids differ: ${idFails}; ts round-trip failures: ${rtFails}; decode(py ids) differs: ${decFails}; encodeWithSteps differ: ${stepFails}/${data.encodeWithSteps.length}`);
for (const s of shown) console.log(s);
const zh = tok.encode('我爱自然语言处理');
console.log('我爱自然语言处理 ->', zh.length, 'tokens; displays:', zh.slice(0, 6).map((i) => tok.tokenToDisplay(i)).join(' '));
process.exit(idFails + rtFails + decFails + stepFails + trainFails + cutFails + decFuzzFails === 0 ? 0 : 1);
