// Parity test: the TypeScript TinyGPT (site/src/lib/models/tinygpt.ts) against PyTorch.
//  1. every traced tensor on the 3 reference prompts (pass bar max |diff| <= 1e-4)
//  2. truncation to the context window, the PRNG, and generate() vs the Python sampler
//  3. a float16 checkpoint: vs PyTorch on the same fp16-rounded weights (<= 1e-4) and vs the
//     original fp32 checkpoint (fp16 tolerance)
//  4. replay.json: the JS model on every fp16 checkpoint reproduces the precomputed top-10,
//     greedy and sampled continuations
// Writes the measured numbers into spec.json "parity".
// --no-write: run every check but leave spec.json untouched; prints the record it would write as one
// `PARITY_JSON {...}` line instead (the DevKit always runs legacy parity this way).
// Run: node pipeline/parity/tinygpt.parity.ts [--no-write]
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { load, mulberry32, type FetchLike, type TinyGPT } from '../../site/src/lib/models/tinygpt.ts';

const dir = fileURLToPath(new URL('../../site/static/models/tinygpt/', import.meta.url)).split('\\').join('/');
const fetchFile: FetchLike = async (url: string) => {
  const buf = readFileSync(url);
  return {
    ok: true,
    json: async () => JSON.parse(buf.toString('utf8')),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer,
  };
};
const readJSON = (name: string) => JSON.parse(readFileSync(dir + name, 'utf8'));

type B64 = { shape: number[]; b64: string };
function f32(t: B64): Float32Array {
  const b = Buffer.from(t.b64, 'base64');
  const u8 = new Uint8Array(b.length);
  u8.set(b);
  return new Float32Array(u8.buffer);
}
function maxDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Infinity;
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] === b[i]) continue; // also covers -Infinity === -Infinity
    const d = Math.abs(a[i] - b[i]);
    if (Number.isNaN(d) || d === Infinity) return Infinity; // NaN or unmatched infinity fails at once
    if (d > m) m = d;
  }
  return m;
}
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

const problems: string[] = [];
const ref = readJSON('reference.json');
const model = await load(fetchFile, dir);
const TOL = 1e-4;
const NO_WRITE = process.argv.includes('--no-write');

// 1) every traced tensor -------------------------------------------------------------------------
let overall = 0;
let nTensors = 0;
const worst: { name: string; d: number }[] = [];
for (const p of ref.prompts) {
  const t0 = performance.now();
  const tr = model.forward(p.ids);
  const ms = performance.now() - t0;
  if (!same(tr.order, p.order)) problems.push(`trace order differs for ${JSON.stringify(p.text)}`);
  let promptMax = 0;
  for (const name of p.order) {
    const r = p.tensors[name];
    const mine = tr.t[name];
    if (!mine) {
      problems.push(`missing ${name}`);
      continue;
    }
    if (!same(mine.shape, r.shape)) problems.push(`${name}: shape ${mine.shape} vs ${r.shape}`);
    const d = maxDiff(mine.data, f32(r));
    nTensors++;
    worst.push({ name: `${name} (T=${p.ids.length})`, d });
    promptMax = Math.max(promptMax, d);
  }
  overall = Math.max(overall, promptMax);
  console.log(`prompt T=${p.ids.length}: ${p.order.length} tensors, max |diff| ${promptMax.toExponential(2)}, forward ${ms.toFixed(1)} ms`);
}
worst.sort((a, b) => b.d - a.d);
console.log('largest diffs:', worst.slice(0, 4).map((w) => `${w.name} ${w.d.toExponential(2)}`).join('; '));
if (!(overall <= TOL)) problems.push(`trace max |diff| ${overall} > ${TOL}`);

// 2) truncation, PRNG, sampling ------------------------------------------------------------------
const tr = model.forward(ref.truncation.ids);
const C = ref.truncation.context;
const truncDiff = maxDiff(tr.t.logits.data.subarray((C - 1) * model.vocabSize), f32(ref.truncation.lastLogits));
const truncOk = tr.truncated.dropped === ref.truncation.keptFrom && tr.truncated.kept === C && truncDiff <= TOL;
console.log(`truncation: ${ref.truncation.ids.length} ids -> kept ${tr.truncated.kept}, dropped ${tr.truncated.dropped}; last logits max |diff| ${truncDiff.toExponential(2)}`);
if (!truncOk) problems.push('truncation check failed');
const r = mulberry32(ref.prng.seed);
const prngOk = ref.prng.values.every((v: number) => r() === v);
console.log(`mulberry32 matches Python: ${prngOk}`);
if (!prngOk) problems.push('PRNG differs');
let sampOk = 0;
for (const s of ref.sampling) {
  const g = model.generate(s.ids, { maxNew: s.maxNew, temperature: s.temperature, topK: s.topK, topP: s.topP, seed: s.seed });
  if (same(g.newIds, s.newIds)) sampOk++;
  else problems.push(`generate (T=${s.temperature}, topK=${s.topK}, topP=${s.topP}) differs from Python`);
}
console.log(`generate vs Python sampler: ${sampOk}/${ref.sampling.length} identical`);

// surprisal agrees with the traced probabilities
const p3 = ref.prompts[2];
const probs = f32(p3.tensors.probs);
const V = model.vocabSize;
const sur = model.surprisal(p3.ids);
let surDiff = 0;
for (let i = 1; i < p3.ids.length; i++) {
  const bits = -Math.log2(probs[(i - 1) * V + p3.ids[i]]);
  surDiff = Math.max(surDiff, Math.abs(bits - (sur.tokens[i].bits as number)));
}
console.log(`surprisal vs PyTorch probs: max |diff| ${surDiff.toExponential(2)} bits; mean ${sur.meanBits.toFixed(3)} bits/token, ppl ${sur.perplexity.toFixed(2)}`);
if (!(surDiff < 1e-3)) problems.push('surprisal differs');
// long text goes through sliding windows
const longIds = ref.truncation.ids.concat(ref.truncation.ids);
const surLong = model.surprisal(longIds);
console.log(`surprisal on ${longIds.length} tokens: ${surLong.windows} windows, ${surLong.scored} scored`);
if (surLong.scored !== longIds.length - 1) problems.push('sliding-window surprisal missed tokens');

// 3) float16 checkpoint ----------------------------------------------------------------------------
const ckRef = ref.checkpoint;
const ck = await load(fetchFile, dir, { checkpoint: ckRef.step });
const ckLogits = ck.forward(ckRef.ids).t.logits.data;
const dFp16 = maxDiff(ckLogits, f32(ckRef.logitsFp16Weights));
const dFp32 = maxDiff(ckLogits, f32(ckRef.logitsFp32Weights));
console.log(`checkpoint step ${ckRef.step} (fp16): logits vs PyTorch on the same fp16 weights ${dFp16.toExponential(2)}, vs original fp32 weights ${dFp32.toExponential(2)}`);
if (!(dFp16 <= TOL)) problems.push(`checkpoint fp16 logits diff ${dFp16}`);
if (!(dFp32 <= 0.05)) problems.push(`checkpoint vs fp32 weights diff ${dFp32} > 0.05`);

// 4) replay.json -------------------------------------------------------------------------------------
const replay = readJSON('replay.json');
let rTop = 0;
let rGreedy = 0;
let rSampled = 0;
let rTotal = 0;
const sset = replay.settings.sampled;
for (const c of replay.checkpoints) {
  const m: TinyGPT = await load(fetchFile, dir, { checkpoint: c.step });
  replay.probeIds.forEach((ids: number[], k: number) => {
    const e = c.probes[k];
    rTotal++;
    const { probs } = m.nextTokenProbs(ids);
    if (e.top.every((t: { id: number; p: number }) => Math.abs(probs[t.id] - t.p) < 1e-4)) rTop++;
    const g = m.generate(ids, { maxNew: 24, temperature: 0 });
    if (same(g.newIds, e.greedy.ids)) rGreedy++;
    const s = m.generate(ids, { maxNew: sset.maxNew, temperature: sset.temperature, topK: sset.topK, topP: sset.topP, seed: sset.seed });
    if (same(s.newIds, e.sampled.ids)) rSampled++;
  });
}
console.log(`replay (${replay.checkpoints.length} checkpoints x ${replay.probeIds.length} probes): top-10 ${rTop}/${rTotal}, greedy ${rGreedy}/${rTotal}, sampled ${rSampled}/${rTotal} reproduced in JS`);
if (rTop !== rTotal) problems.push('replay top-10 probabilities differ');
if (rGreedy !== rTotal || rSampled !== rTotal) problems.push('replay continuations differ (see counts)');

// write spec.parity ------------------------------------------------------------------------------------
const spec = readJSON('spec.json');
// keep the verifier's fresh-prompt numbers (written by pipeline/parity/verify_tinygpt/verify_model.check.ts)
const freshPromptCheck = spec.parity?.freshPromptCheck;
const shippedMax = spec.parity?.max_abs_diff;
spec.parity = {
  against: 'pytorch-fp32',
  max_abs_diff: Number(overall.toExponential(3)),
  checked: new Date().toISOString().slice(0, 10),
  tolerance: TOL,
  tensorsChecked: nTensors,
  prompts: ref.prompts.length,
  promptLengths: ref.prompts.map((p: { ids: number[] }) => p.ids.length),
  truncation: { ok: truncOk, lastLogitsMaxAbsDiff: Number(truncDiff.toExponential(3)) },
  sampling: `${sampOk}/${ref.sampling.length} generate() runs identical to the Python sampler`,
  checkpointFp16: {
    step: ckRef.step,
    maxAbsDiffLogitsVsSameFp16Weights: Number(dFp16.toExponential(3)),
    maxAbsDiffLogitsVsOriginalFp32Weights: Number(dFp32.toExponential(3)),
  },
  replay: { checkpoints: replay.checkpoints.length, probes: replay.probeIds.length, top10: rTop, greedy: rGreedy, sampled: rSampled, of: rTotal },
  script: 'pipeline/parity/tinygpt.parity.ts',
  passed: problems.length === 0,
  ...(freshPromptCheck ? { freshPromptCheck } : {}),
};
if (NO_WRITE) {
  console.log(`--no-write: spec.json left unchanged (shipped max_abs_diff ${shippedMax}, measured ${spec.parity.max_abs_diff})`);
  console.log('PARITY_JSON ' + JSON.stringify(spec.parity));
} else {
  writeFileSync(dir + 'spec.json', JSON.stringify(spec, null, 1));
}
console.log(problems.length ? 'PROBLEMS:\n  ' + problems.join('\n  ') : 'TINYGPT PARITY OK');
process.exit(problems.length ? 1 : 0);
