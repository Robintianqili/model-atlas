// Usage example for page builders (and a timing smoke test): tokenizer + TinyGPT end to end.
// Run: node pipeline/parity/tinygpt.example.ts
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { loadTokenizer } from '../../site/src/lib/models/bpe.ts';
import { load, type FetchLike } from '../../site/src/lib/models/tinygpt.ts';

const base = fileURLToPath(new URL('../../site/static/models/tinygpt/', import.meta.url)).split('\\').join('/');
// In the browser this is just `fetch` and base is `${base_path}/models/tinygpt`.
const fetchFile: FetchLike = async (url) => {
  const buf = readFileSync(url);
  return {
    ok: true,
    json: async () => JSON.parse(buf.toString('utf8')),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer,
  };
};

const tok = await loadTokenizer(fetchFile, base);
const gpt = await load(fetchFile, base); // final fp32 weights; { checkpoint: 400 } loads a training snapshot

const prompt = 'Once upon a time, there was a little';
const ids = tok.encode(prompt);
console.log('tokens:', ids.map((i) => JSON.stringify(tok.tokenToDisplay(i))).join(' '));

let t0 = performance.now();
const trace = gpt.forward(ids);
console.log(`forward T=${ids.length}: ${trace.order.length} traced tensors in ${(performance.now() - t0).toFixed(1)} ms`);
const T = ids.length;
const probs = trace.t.probs.data.subarray((T - 1) * gpt.vocabSize, T * gpt.vocabSize);
const top = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, 5);
console.log('next token:', top.map((i) => `${JSON.stringify(tok.tokenToDisplay(i))} ${(probs[i] * 100).toFixed(1)}%`).join(', '));
console.log('L0 head 0 attention of the last token:', Array.from(trace.t['L0.attn.weights'].data.subarray((T - 1) * T, T * T)).map((v) => v.toFixed(2)).join(' '));

const long = tok.encode('The sun was hot. '.repeat(10));
t0 = performance.now();
gpt.forward(long.slice(0, 64));
console.log(`forward T=64: ${(performance.now() - t0).toFixed(1)} ms`);

t0 = performance.now();
const g = gpt.generate(ids, { maxNew: 40, temperature: 0.8, topK: 40, seed: 7 });
console.log(`generate 40 tokens: ${(performance.now() - t0).toFixed(0)} ms ->`, JSON.stringify(prompt + tok.decode(g.newIds)));

const s = gpt.surprisal(tok.encode('Lily went to the park with her mom. She saw a big purple elephant eating spaghetti.'));
console.log(`surprisal: mean ${s.meanBits.toFixed(2)} bits/token, perplexity ${s.perplexity.toFixed(2)}`);
console.log(s.tokens.map((x) => `${tok.tokenToDisplay(x.id)}[${x.rank ?? '-'}]`).join(''));

const f = gpt.forward(tok.encode('word '.repeat(80)));
console.log('truncation report:', f.truncated);
