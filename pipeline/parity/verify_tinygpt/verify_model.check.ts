// Verification: tinygpt.ts against PyTorch on fresh prompts written by the verifier
// (verify_model.py -> verify_ref.json): every traced tensor, truncation, greedy and sampled
// continuations, surprisal with sliding windows, every fp16 checkpoint (step 0 near-uniform).
// Run: python pipeline/parity/verify_tinygpt/verify_model.py && node pipeline/parity/verify_tinygpt/verify_model.check.ts
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { load, type FetchLike } from '../../../site/src/lib/models/tinygpt.ts';

const dir = fileURLToPath(new URL('../../../site/static/models/tinygpt/', import.meta.url)).split('\\').join('/');
const here = fileURLToPath(new URL('./', import.meta.url)).split('\\').join('/');
const fetchFile: FetchLike = async (url: string) => {
  const buf = readFileSync(url);
  return {
    ok: true,
    json: async () => JSON.parse(buf.toString('utf8')),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer,
  };
};
const ref = JSON.parse(readFileSync(here + 'verify_ref.json', 'utf8'));
const spec = JSON.parse(readFileSync(dir + 'spec.json', 'utf8'));
const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
function f32(b64: string): Float32Array {
  const b = Buffer.from(b64, 'base64');
  const u8 = new Uint8Array(b.length);
  u8.set(b);
  return new Float32Array(u8.buffer);
}
/** max |a - b|; any NaN, or a non-finite value that is not matched exactly, gives Infinity */
function maxDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) return Infinity;
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    if (Number.isNaN(a[i]) || Number.isNaN(b[i])) return Infinity;
    if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) {
      if (a[i] !== b[i]) return Infinity;
      continue;
    }
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}
const problems: string[] = [];
const TOL = 1e-4;
const model = await load(fetchFile, dir);

// 1) traces on fresh prompts
let worst = 0;
let worst64 = 0;
let worstName = '';
let n = 0;
for (const p of ref.prompts) {
  const tr = model.forward(p.ids);
  const expectKept = p.ids.slice(-model.context);
  if (!same(tr.ids, expectKept)) problems.push(`${p.name}: kept ids differ`);
  if (tr.truncated.kept !== p.kept || tr.truncated.dropped !== p.ids.length - p.kept || tr.truncated.inputLength !== p.ids.length)
    problems.push(`${p.name}: truncation info ${JSON.stringify(tr.truncated)}`);
  if (!same(tr.order, p.order)) problems.push(`${p.name}: order differs`);
  let pm = 0;
  for (const name of p.order) {
    const mine = tr.t[name];
    if (!mine || !same(mine.shape, p.t32[name].shape)) {
      problems.push(`${p.name} ${name}: missing or shape`);
      continue;
    }
    const d = maxDiff(mine.data, f32(p.t32[name].b64));
    const d64 = maxDiff(mine.data, f32(p.t64[name]));
    n++;
    if (d > pm) pm = d;
    if (d > worst) {
      worst = d;
      worstName = `${name} (${p.name})`;
    }
    if (d64 > worst64) worst64 = d64;
  }
  console.log(`  ${p.name.padEnd(15)} ids ${String(p.ids.length).padStart(3)} kept ${tr.truncated.kept} dropped ${tr.truncated.dropped}: max |diff| vs PyTorch fp32 ${pm.toExponential(2)}`);
}
console.log(`traces: ${n} tensors on ${ref.prompts.length} fresh prompts, worst ${worst.toExponential(2)} at ${worstName}; vs PyTorch fp64 ${worst64.toExponential(2)}`);
if (!(worst <= TOL)) problems.push(`trace max |diff| ${worst} > ${TOL}`);
// nextTokenProbs = last row of forward probs
{
  const ids = ref.prompts[6].ids;
  const tr = model.forward(ids);
  const V = model.vocabSize;
  const T = tr.ids.length;
  const d = maxDiff(model.nextTokenProbs(ids).probs, tr.t.probs.data.subarray((T - 1) * V, T * V));
  console.log(`nextTokenProbs vs forward last row: ${d.toExponential(2)}`);
  if (!(d < 1e-6)) problems.push('nextTokenProbs differs from forward');
}

// 2) continuations
let gOk = 0, gN = 0, sOk = 0, sN = 0;
let surSummary = '';
const t0 = performance.now();
for (const g of ref.generate) {
  const r = model.generate(g.ids, { maxNew: g.maxNew, temperature: g.temperature, topK: g.topK, topP: g.topP, seed: g.seed, stopAtEot: g.stopAtEot });
  const ok = same(r.newIds, g.newIds);
  if (g.temperature === 0) { gN++; if (ok) gOk++; } else { sN++; if (ok) sOk++; }
  if (!ok) {
    let k = 0;
    while (k < g.newIds.length && r.newIds[k] === g.newIds[k]) k++;
    problems.push(`generate T=${g.temperature} topK=${g.topK} topP=${g.topP} seed=${g.seed} on ${JSON.stringify(g.prompt.slice(0, 30))}: first difference at new token ${k}`);
  }
}
const genMs = performance.now() - t0;
console.log(`generate (60 new tokens each, sliding window past 64): greedy ${gOk}/${gN} identical, sampled ${sOk}/${sN} identical (${(genMs / ref.generate.length).toFixed(0)} ms per run)`);

// 3) surprisal with sliding windows
{
  const s = ref.surprisal;
  const r = model.surprisal(s.ids, { stride: s.stride });
  let d = 0;
  let rankDiff = 0;
  for (let i = 1; i < s.ids.length; i++) {
    d = Math.max(d, Math.abs((r.tokens[i].bits as number) - s.bits[i]));
    if (r.tokens[i].rank !== s.ranks[i]) rankDiff++;
  }
  const meanPy = s.bits.slice(1).reduce((a: number, b: number) => a + b, 0) / (s.ids.length - 1);
  console.log(`surprisal on ${s.ids.length} tokens (${r.windows} windows): max |bits diff| ${d.toExponential(2)}, ranks differ at ${rankDiff}, mean bits ts ${r.meanBits.toFixed(5)} py ${meanPy.toFixed(5)}`);
  surSummary = `${s.ids.length} tokens, ${r.windows} windows: max |bits diff| ${d.toExponential(2)}, ranks identical ${s.ids.length - 1 - rankDiff}/${s.ids.length - 1}`;
  if (!(d < 1e-3) || rankDiff > 0 || r.scored !== s.ids.length - 1) problems.push('surprisal differs');
}

// 4) every fp16 checkpoint loads; step 0 is near-uniform
const index = JSON.parse(readFileSync(dir + 'checkpoints/index.json', 'utf8'));
const replay = JSON.parse(readFileSync(dir + 'replay.json', 'utf8'));
for (const c of index.checkpoints) {
  const m = await load(fetchFile, dir, { checkpoint: c.step });
  let ent = 0;
  let pmax = 0;
  for (const ids of replay.probeIds) {
    const { probs } = m.nextTokenProbs(ids);
    let e = 0;
    for (const p of probs) {
      if (p > 0) e -= p * Math.log(p);
      if (p > pmax) pmax = p;
    }
    ent += e / replay.probeIds.length;
  }
  console.log(`  checkpoint ${String(c.step).padStart(5)} (${c.file}): loads, mean entropy on 6 probes ${ent.toFixed(3)} nats, max p ${pmax.toFixed(4)}`);
  if (c.step === 0 && !(Math.abs(ent - Math.log(model.vocabSize)) < 0.05 && pmax < 0.02)) problems.push('step 0 is not near-uniform');
}
const fin = await load(fetchFile, dir, { checkpoint: 'final' });
console.log(`checkpoint 'final' -> step ${fin.checkpoint?.step}`);
if (fin.checkpoint?.step !== 30000) problems.push("checkpoint 'final' is not step 30000");
let threw = false;
try { await load(fetchFile, dir, { checkpoint: 123 }); } catch { threw = true; }
if (!threw) problems.push('unknown checkpoint did not throw');
for (const bad of [[], [1024], [-1], [1.5]]) {
  let t = false;
  try { model.forward(bad as number[]); } catch { t = true; }
  if (!t) problems.push(`forward(${JSON.stringify(bad)}) did not throw`);
}
// spec.parity agrees with the numbers measured by the shipped parity script
console.log(`spec.parity: max_abs_diff ${spec.parity.max_abs_diff}, passed ${spec.parity.passed}, tensors ${spec.parity.tensorsChecked}`);
if (!(spec.parity.passed === true && spec.parity.max_abs_diff <= TOL)) problems.push('spec.parity does not show a pass');

// record the result next to the shipped parity numbers (tinygpt.parity.ts keeps this field)
if (problems.length === 0) {
  const sp = JSON.parse(readFileSync(dir + 'spec.json', 'utf8'));
  sp.parity.freshPromptCheck = {
    script: 'pipeline/parity/verify_tinygpt/verify_model.check.ts (references from verify_model.py)',
    checked: new Date().toLocaleDateString('sv-SE'),
    prompts: ref.prompts.length,
    promptLengths: ref.prompts.map((p: { kept: number }) => p.kept),
    note: 'fresh prompts written by the verifier, incl. full 64-token context, T=1, Chinese + emoji, <|endoftext|> inside, and a 186-token input cut to the last 64',
    tensorsChecked: n,
    max_abs_diff: Number(worst.toExponential(3)),
    worstTensor: worstName,
    maxAbsDiffVsPytorchFp64: Number(worst64.toExponential(3)),
    generate: `greedy ${gOk}/${gN}, sampled ${sOk}/${sN} identical to an independent Python loop (60 new tokens each)`,
    surprisal: surSummary,
    passed: true,
  };
  writeFileSync(dir + 'spec.json', JSON.stringify(sp, null, 1));
  console.log('wrote spec.parity.freshPromptCheck');
}
console.log(problems.length ? 'PROBLEMS:\n  ' + problems.join('\n  ') : 'TS-SIDE VERIFICATION OK');
process.exit(problems.length ? 1 : 0);
