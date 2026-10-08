// Parity test: the TypeScript BPE tokenizer (site/src/lib/models/bpe.ts) must give exactly the
// same ids as the Python one (pipeline/tinygpt/bpe.py) on 300 test strings, plus the same token
// displays, decode results, merge steps and trained merges.
// Run: node pipeline/parity/bpe.parity.ts   (cases come from make_bpe_cases.py)
import { readFileSync } from 'node:fs';
import { BPETokenizer, trainBPE } from '../../site/src/lib/models/bpe.ts';

const root = new URL('../../', import.meta.url);
const read = (p: string) => JSON.parse(readFileSync(new URL(p, root), 'utf8'));
const tok = new BPETokenizer(read('site/static/models/tinygpt/tokenizer.json'));
const cases = read('pipeline/parity/bpe_cases.json');

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
let fails = 0;
const fail = (msg: string) => {
  fails++;
  if (fails <= 15) console.log('FAIL', msg);
};

let nTok = 0;
const t0 = performance.now();
for (const c of cases.cases) {
  const ids = tok.encode(c.text);
  nTok += ids.length;
  if (!same(ids, c.ids)) {
    const pts: string[] = [];
    for (const p of tok.splitSpecial(c.text)) pts.push(...(p.special ? [p.text] : tok.pretokenize(p.text)));
    fail(`encode ${JSON.stringify(c.text)}\n  ts ${JSON.stringify(ids)}\n  py ${JSON.stringify(c.ids)}\n  ts pretokens ${JSON.stringify(pts)}\n  py pretokens ${JSON.stringify(c.pretokens)}`);
  }
  if (tok.decode(ids) !== c.text) fail(`round trip ${JSON.stringify(c.text)}`);
}
const tEnc = performance.now() - t0;
let dispFails = 0;
cases.display.forEach((d: string, i: number) => {
  if (tok.tokenToDisplay(i) !== d) {
    dispFails++;
    fail(`display ${i}: ts ${JSON.stringify(tok.tokenToDisplay(i))} py ${JSON.stringify(d)}`);
  }
});
const steps = tok.encodeWithSteps('Once upon a time, 我爱 ' + String.fromCodePoint(0x1f600) + ' cats!<|endoftext|>');
if (!same(steps, cases.encodeWithSteps)) fail('encodeWithSteps differs');
let decFails = 0;
for (const d of cases.decode) {
  if (tok.decode(d.ids) !== d.text) {
    decFails++;
    fail(`decode ${JSON.stringify(d.ids)}: ts ${JSON.stringify(tok.decode(d.ids))} py ${JSON.stringify(d.text)}`);
  }
}
let trainFails = 0;
for (const t of cases.train) {
  const r = trainBPE(t.text, t.numMerges, tok.pattern);
  if (!same(r.merges, t.merges) || !same(r.steps.map((s) => s.count), t.counts)) {
    trainFails++;
    fail(`trainBPE on ${JSON.stringify(t.text.slice(0, 40))}...`);
  }
}

console.log(`strings: ${cases.cases.length} (${nTok} tokens, encode ${tEnc.toFixed(0)} ms)`);
console.log(`token displays: ${cases.display.length} checked, ${dispFails} differ`);
console.log(`decode cases: ${cases.decode.length} checked, ${decFails} differ`);
console.log(`trainBPE cases: ${cases.train.length} checked, ${trainFails} differ`);
console.log(fails === 0 ? 'BPE PARITY OK' : `BPE PARITY FAILED (${fails} problems)`);
process.exit(fails === 0 ? 0 : 1);
