/**
 * TinyGPT forward pass in plain TypeScript (no DOM, no Svelte): runs in the page, a Worker or Node.
 *
 * GPT-2-style decoder: token embedding + learned position embedding; N pre-LayerNorm blocks
 * [LN -> causal multi-head self-attention -> residual add; LN -> MLP (4x, GELU tanh) -> residual add];
 * final LN; logits through the tied token embedding.
 *
 * forward(ids) records every intermediate tensor under the layer ids of spec.json. Values are stored
 * as float32; sums are accumulated in float64. pipeline/parity/tinygpt.parity.ts checks every
 * traced tensor against PyTorch (max |diff| <= 1e-4).
 *
 * Shapes (row-major): sequences [T, D]; L{i}.attn.q/k/v are [T, D] with head h in columns
 * h*dHead .. (h+1)*dHead-1; attention scores/masked/weights are [nHead, T, T] (query row, key column);
 * scores = q k^T / sqrt(dHead); masked = scores with -Infinity above the diagonal (future keys);
 * L{i}.mlp.fc and L{i}.mlp.gelu are [T, 4D]; logits and probs are [T, vocabSize].
 */

export type Tensor = { shape: number[]; data: Float32Array };
export type Trace = { order: string[]; t: Record<string, Tensor> };

export type FetchLike = (url: string) => Promise<{
  ok: boolean;
  status?: number;
  json(): Promise<unknown>;
  arrayBuffer(): Promise<ArrayBuffer>;
}>;

export type TensorEntry = { name: string; shape: number[]; offset: number; length: number };

export type TinyGPTSpec = {
  slug: string;
  vocabSize: number;
  context: number;
  nLayer: number;
  nHead: number;
  dModel: number;
  lnEps?: number;
  tensors: TensorEntry[];
  layers: { id: string; op: string; label: { en: string; zh: string }; in: string[]; outShape: (number | string)[] }[];
  [key: string]: unknown;
};

export type CheckpointInfo = {
  step: number;
  file: string;
  tokens: number;
  trainLoss: number;
  valLoss: number;
  label?: { en: string; zh: string };
};

export type CheckpointIndex = {
  dtype: 'float16';
  tensors: TensorEntry[];
  checkpoints: CheckpointInfo[];
};

/** How the input was cut to the context window. */
export type Truncation = { context: number; inputLength: number; kept: number; dropped: number };

export type ForwardResult = Trace & { ids: number[]; truncated: Truncation };

export type TopEntry = { id: number; p: number };

export type GenerateOptions = {
  maxNew?: number; // tokens to generate (default 24)
  temperature?: number; // 0 = greedy (argmax); default 1
  topK?: number; // keep only the k most likely tokens (0 = off)
  topP?: number; // keep the smallest set of tokens whose probability sums to >= topP (1 = off)
  seed?: number; // seed of the deterministic PRNG (mulberry32); default 42
  stopAtEot?: boolean; // stop after generating <|endoftext|> (default true)
  eotId?: number; // id of <|endoftext|> (default vocabSize - 1)
  topN?: number; // how many entries of the distribution to report per step (default 10)
};

export type GenerateStep = {
  token: number; // chosen token id
  p: number; // its probability in the distribution it was sampled from (after temperature/top-k/top-p)
  pModel: number; // its probability under the plain model (temperature 1, no cut-offs)
  top: TopEntry[]; // the topN most likely tokens of the sampling distribution (plain model distribution when greedy)
  u: number | null; // the random number used (null for greedy)
  contextStart: number; // index (in the full id list) of the first token the model saw at this step
};

export type GenerateResult = {
  ids: number[]; // prompt + generated
  newIds: number[];
  steps: GenerateStep[];
  promptTruncated: Truncation;
};

export type SurprisalToken = {
  pos: number;
  id: number;
  p: number | null; // probability the model gave this token (null for the first token: nothing before it)
  bits: number | null; // surprisal = -log2 p
  rank: number | null; // 1 = the model's top guess
  topId: number | null; // the model's top guess at this position
};

export type SurprisalResult = {
  tokens: SurprisalToken[];
  meanBits: number; // average surprisal over scored tokens
  meanNats: number; // = cross-entropy loss in nats
  perplexity: number; // = exp(meanNats)
  scored: number;
  windows: number; // how many context windows were run (> 1 for text longer than the context)
  truncated: Truncation; // tokens beyond opts.maxTokens are dropped and reported here
};

export type TinyGPT = {
  spec: TinyGPTSpec;
  checkpoint: CheckpointInfo | null; // null = the final fp32 weights (weights.bin)
  vocabSize: number;
  context: number;
  nLayer: number;
  nHead: number;
  dModel: number;
  weight(name: string): Tensor;
  forward(ids: number[]): ForwardResult;
  nextTokenProbs(ids: number[]): { probs: Float32Array; logits: Float32Array; truncated: Truncation };
  generate(ids: number[], opts?: GenerateOptions): GenerateResult;
  surprisal(ids: number[], opts?: { maxTokens?: number; stride?: number }): SurprisalResult;
};

// ------------------------------------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------------------------------------

function joinUrl(base: string, path: string): string {
  return base.replace(/\/+$/, '') + '/' + path.replace(/^\/+/, '');
}

async function getJSON<T>(fetchFn: FetchLike, url: string): Promise<T> {
  const r = await fetchFn(url);
  if (!r.ok) throw new Error(`could not load ${url} (${r.status ?? 'error'})`);
  return (await r.json()) as T;
}

async function getBytes(fetchFn: FetchLike, url: string): Promise<ArrayBuffer> {
  const r = await fetchFn(url);
  if (!r.ok) throw new Error(`could not load ${url} (${r.status ?? 'error'})`);
  return await r.arrayBuffer();
}

/** IEEE 754 half -> float (no Float16Array needed). */
export function halfToFloat(h: number): number {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24; // subnormal
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

export function decodeFloat16(buf: ArrayBuffer): Float32Array {
  const u16 = new Uint16Array(buf);
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) out[i] = halfToFloat(u16[i]);
  return out;
}

/** mulberry32: tiny deterministic PRNG returning floats in [0, 1). Same code in Python (export.py). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** y[t, o] = sum_i x[t, i] W[o, i] + b[o]   (W is [out, in], like PyTorch nn.Linear) */
function linear(x: Float32Array, T: number, nIn: number, W: Float32Array, b: Float32Array | null, nOut: number): Float32Array {
  const y = new Float32Array(T * nOut);
  for (let t = 0; t < T; t++) {
    const xo = t * nIn;
    for (let o = 0; o < nOut; o++) {
      const wo = o * nIn;
      let s = b ? b[o] : 0;
      for (let i = 0; i < nIn; i++) s += x[xo + i] * W[wo + i];
      y[t * nOut + o] = s;
    }
  }
  return y;
}

function layerNorm(x: Float32Array, T: number, D: number, g: Float32Array, b: Float32Array, eps: number): Float32Array {
  const y = new Float32Array(T * D);
  for (let t = 0; t < T; t++) {
    const o = t * D;
    let mean = 0;
    for (let i = 0; i < D; i++) mean += x[o + i];
    mean /= D;
    let v = 0;
    for (let i = 0; i < D; i++) v += (x[o + i] - mean) ** 2;
    v /= D; // biased variance, as in PyTorch LayerNorm
    const inv = 1 / Math.sqrt(v + eps);
    for (let i = 0; i < D; i++) y[o + i] = (x[o + i] - mean) * inv * g[i] + b[i];
  }
  return y;
}

function add(a: Float32Array, b: Float32Array): Float32Array {
  const y = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) y[i] = a[i] + b[i];
  return y;
}

const GELU_C = Math.sqrt(2 / Math.PI);
function gelu(x: Float32Array): Float32Array {
  const y = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    y[i] = 0.5 * v * (1 + Math.tanh(GELU_C * (v + 0.044715 * v * v * v)));
  }
  return y;
}

/** softmax over the last axis of rows of length n (float64 inside). -Infinity entries get 0. */
function softmaxRows(x: Float32Array, rows: number, n: number): Float32Array {
  const y = new Float32Array(rows * n);
  const tmp = new Float64Array(n);
  for (let r = 0; r < rows; r++) {
    const o = r * n;
    let m = -Infinity;
    for (let i = 0; i < n; i++) if (x[o + i] > m) m = x[o + i];
    let s = 0;
    for (let i = 0; i < n; i++) {
      const e = Math.exp(x[o + i] - m);
      tmp[i] = e;
      s += e;
    }
    for (let i = 0; i < n; i++) y[o + i] = tmp[i] / s;
  }
  return y;
}

function topEntries(p: ArrayLike<number>, n: number): TopEntry[] {
  const idx = Array.from({ length: p.length }, (_, i) => i);
  idx.sort((a, b) => p[b] - p[a] || a - b);
  return idx.slice(0, n).map((id) => ({ id, p: p[id] }));
}

// ------------------------------------------------------------------------------------------------
// model
// ------------------------------------------------------------------------------------------------

export async function load(
  fetchFn: FetchLike,
  baseUrl: string,
  opts: { checkpoint?: number | string } = {},
): Promise<TinyGPT> {
  const spec = await getJSON<TinyGPTSpec>(fetchFn, joinUrl(baseUrl, 'spec.json'));
  let table: TensorEntry[] = spec.tensors;
  let flat: Float32Array;
  let checkpoint: CheckpointInfo | null = null;
  if (opts.checkpoint === undefined || opts.checkpoint === null || opts.checkpoint === 'weights') {
    flat = new Float32Array(await getBytes(fetchFn, joinUrl(baseUrl, 'weights.bin')));
  } else {
    const index = await getJSON<CheckpointIndex>(fetchFn, joinUrl(baseUrl, 'checkpoints/index.json'));
    const c = opts.checkpoint;
    const found =
      c === 'final'
        ? index.checkpoints[index.checkpoints.length - 1]
        : index.checkpoints.find((k) => k.step === Number(c) || k.file === c);
    if (!found) throw new Error(`no checkpoint ${c}; have ${index.checkpoints.map((k) => k.step).join(', ')}`);
    checkpoint = found;
    table = index.tensors;
    flat = decodeFloat16(await getBytes(fetchFn, joinUrl(baseUrl, 'checkpoints/' + found.file)));
  }
  return fromWeights(spec, table, flat, checkpoint);
}

/** Build the model from a spec, a tensor table and the flat float32 weights (also used by tests). */
export function fromWeights(
  spec: TinyGPTSpec,
  table: TensorEntry[],
  flat: Float32Array,
  checkpoint: CheckpointInfo | null = null,
): TinyGPT {
  const W: Record<string, Tensor> = {};
  for (const e of table) {
    if (e.offset + e.length > flat.length) throw new Error(`tensor ${e.name} runs past the end of the weights`);
    W[e.name] = { shape: e.shape, data: flat.subarray(e.offset, e.offset + e.length) };
  }
  const V = spec.vocabSize;
  const C = spec.context;
  const L = spec.nLayer;
  const H = spec.nHead;
  const D = spec.dModel;
  const hd = D / H;
  const eps = spec.lnEps ?? 1e-5;
  const w = (n: string) => {
    const t = W[n];
    if (!t) throw new Error(`missing tensor ${n}`);
    return t.data;
  };

  function fit(ids: number[]): { ids: number[]; truncated: Truncation } {
    if (ids.length === 0) throw new Error('TinyGPT needs at least one token');
    for (const id of ids) if (!(id >= 0 && id < V && Number.isInteger(id))) throw new Error(`token id ${id} out of range`);
    const kept = ids.length > C ? ids.slice(ids.length - C) : ids.slice();
    return { ids: kept, truncated: { context: C, inputLength: ids.length, kept: kept.length, dropped: ids.length - kept.length } };
  }

  /** The forward pass. With record=false nothing is stored; with lastOnly the logits are only
   *  computed for the last position (for generation). */
  function run(ids: number[], record: boolean, lastOnly: boolean): { trace: Trace; logits: Float32Array; T: number } {
    const T = ids.length;
    const order: string[] = [];
    const t: Record<string, Tensor> = {};
    const rec = (name: string, shape: number[], data: Float32Array) => {
      if (record) {
        order.push(name);
        t[name] = { shape, data };
      }
      return data;
    };
    const tokW = w('tok_emb.weight');
    const posW = w('pos_emb.weight');
    const tok = new Float32Array(T * D);
    const pos = new Float32Array(T * D);
    for (let i = 0; i < T; i++) {
      tok.set(tokW.subarray(ids[i] * D, ids[i] * D + D), i * D);
      pos.set(posW.subarray(i * D, i * D + D), i * D);
    }
    rec('tok_emb', [T, D], tok);
    rec('pos_emb', [T, D], pos);
    let x = rec('embed', [T, D], add(tok, pos));
    const scale = 1 / Math.sqrt(hd);
    for (let l = 0; l < L; l++) {
      const p = `L${l}.`;
      const h = rec(p + 'ln1', [T, D], layerNorm(x, T, D, w(p + 'ln1.weight'), w(p + 'ln1.bias'), eps));
      const q = rec(p + 'attn.q', [T, D], linear(h, T, D, w(p + 'attn.q.weight'), w(p + 'attn.q.bias'), D));
      const k = rec(p + 'attn.k', [T, D], linear(h, T, D, w(p + 'attn.k.weight'), w(p + 'attn.k.bias'), D));
      const v = rec(p + 'attn.v', [T, D], linear(h, T, D, w(p + 'attn.v.weight'), w(p + 'attn.v.bias'), D));
      const scores = new Float32Array(H * T * T);
      const masked = new Float32Array(H * T * T);
      for (let hh = 0; hh < H; hh++) {
        for (let i = 0; i < T; i++) {
          for (let j = 0; j < T; j++) {
            let s = 0;
            for (let c = 0; c < hd; c++) s += q[i * D + hh * hd + c] * k[j * D + hh * hd + c];
            const o = (hh * T + i) * T + j;
            scores[o] = s * scale;
            masked[o] = j > i ? -Infinity : scores[o];
          }
        }
      }
      rec(p + 'attn.scores', [H, T, T], scores);
      rec(p + 'attn.masked', [H, T, T], masked);
      const att = rec(p + 'attn.weights', [H, T, T], softmaxRows(masked, H * T, T));
      const heads = new Float32Array(T * D);
      for (let hh = 0; hh < H; hh++) {
        for (let i = 0; i < T; i++) {
          for (let c = 0; c < hd; c++) {
            let s = 0;
            for (let j = 0; j <= i; j++) s += att[(hh * T + i) * T + j] * v[j * D + hh * hd + c];
            heads[i * D + hh * hd + c] = s;
          }
        }
      }
      rec(p + 'attn.heads', [T, D], heads);
      const proj = rec(p + 'attn.proj', [T, D], linear(heads, T, D, w(p + 'attn.proj.weight'), w(p + 'attn.proj.bias'), D));
      x = rec(p + 'resid1', [T, D], add(x, proj));
      const h2 = rec(p + 'ln2', [T, D], layerNorm(x, T, D, w(p + 'ln2.weight'), w(p + 'ln2.bias'), eps));
      const fc = rec(p + 'mlp.fc', [T, 4 * D], linear(h2, T, D, w(p + 'mlp.fc.weight'), w(p + 'mlp.fc.bias'), 4 * D));
      const g = rec(p + 'mlp.gelu', [T, 4 * D], gelu(fc));
      const m = rec(p + 'mlp.proj', [T, D], linear(g, T, 4 * D, w(p + 'mlp.proj.weight'), w(p + 'mlp.proj.bias'), D));
      x = rec(p + 'resid2', [T, D], add(x, m));
    }
    const xf = rec('ln_f', [T, D], layerNorm(x, T, D, w('ln_f.weight'), w('ln_f.bias'), eps));
    let logits: Float32Array;
    if (lastOnly) {
      logits = linear(xf.subarray((T - 1) * D, T * D), 1, D, tokW, null, V);
    } else {
      logits = rec('logits', [T, V], linear(xf, T, D, tokW, null, V));
      rec('probs', [T, V], softmaxRows(logits, T, V));
    }
    return { trace: { order, t }, logits, T };
  }

  function forward(ids: number[]): ForwardResult {
    const f = fit(ids);
    const r = run(f.ids, true, false);
    return { ...r.trace, ids: f.ids, truncated: f.truncated };
  }

  function nextTokenProbs(ids: number[]) {
    const f = fit(ids);
    const r = run(f.ids, false, true);
    return { probs: softmaxRows(r.logits, 1, V), logits: r.logits, truncated: f.truncated };
  }

  function generate(ids: number[], opts: GenerateOptions = {}): GenerateResult {
    const maxNew = opts.maxNew ?? 24;
    const temperature = opts.temperature ?? 1;
    const topK = opts.topK ?? 0;
    const topP = opts.topP ?? 1;
    const stopAtEot = opts.stopAtEot ?? true;
    const eot = opts.eotId ?? V - 1;
    const topN = opts.topN ?? 10;
    const rand = mulberry32(opts.seed ?? 42);
    const all = ids.slice();
    const promptTruncated = fit(ids).truncated;
    const steps: GenerateStep[] = [];
    for (let s = 0; s < maxNew; s++) {
      const start = Math.max(0, all.length - C);
      const r = run(all.slice(start), false, true);
      const logits = r.logits;
      const pModel = softmaxRows(logits, 1, V);
      let token: number;
      let dist: Float64Array;
      let u: number | null = null;
      if (temperature <= 0) {
        token = 0;
        for (let i = 1; i < V; i++) if (logits[i] > logits[token]) token = i;
        dist = new Float64Array(V);
        dist[token] = 1;
      } else {
        dist = samplingDistribution(logits, temperature, topK, topP);
        u = rand();
        token = -1;
        let cum = 0;
        for (let i = 0; i < V; i++) {
          if (dist[i] === 0) continue;
          cum += dist[i];
          token = i;
          if (cum > u) break;
        }
      }
      // greedy: report the plain model distribution (the sampling distribution would be one-hot)
      const shown = temperature <= 0 ? pModel : dist;
      steps.push({ token, p: dist[token], pModel: pModel[token], top: topEntries(shown, topN), u, contextStart: start });
      all.push(token);
      if (stopAtEot && token === eot) break;
    }
    return { ids: all, newIds: all.slice(ids.length), steps, promptTruncated };
  }

  function surprisal(ids: number[], opts: { maxTokens?: number; stride?: number } = {}): SurprisalResult {
    if (ids.length === 0) throw new Error('TinyGPT needs at least one token');
    const maxTokens = opts.maxTokens ?? 1024;
    const use = ids.slice(0, maxTokens);
    const truncated: Truncation = { context: C, inputLength: ids.length, kept: use.length, dropped: ids.length - use.length };
    const stride = Math.max(1, Math.min(C, opts.stride ?? C / 2));
    const out: SurprisalToken[] = [{ pos: 0, id: use[0], p: null, bits: null, rank: null, topId: null }];
    let windows = 0;
    // window k covers [start, start + C); it scores the positions not scored by earlier windows
    let scoredUpTo = 1; // next position to score
    for (let start = 0; scoredUpTo < use.length; start += stride) {
      const win = use.slice(start, start + C);
      const r = run(win, false, false);
      windows++;
      const probs = softmaxRows(r.logits, win.length, V);
      const end = start + win.length;
      for (let pos = scoredUpTo; pos < end; pos++) {
        const row = pos - 1 - start; // prediction for position pos comes from row pos-1
        const o = row * V;
        const p = probs[o + use[pos]];
        let rank = 1;
        let topId = 0;
        for (let i = 0; i < V; i++) {
          if (probs[o + i] > p) rank++;
          if (probs[o + i] > probs[o + topId]) topId = i;
        }
        out.push({ pos, id: use[pos], p, bits: -Math.log2(p), rank, topId });
      }
      scoredUpTo = end;
      if (end >= use.length) break;
    }
    const scored = out.filter((s) => s.bits !== null);
    const meanBits = scored.length ? scored.reduce((a, s) => a + (s.bits as number), 0) / scored.length : NaN;
    const meanNats = meanBits * Math.LN2;
    return { tokens: out, meanBits, meanNats, perplexity: Math.exp(meanNats), scored: scored.length, windows, truncated };
  }

  return {
    spec,
    checkpoint,
    vocabSize: V,
    context: C,
    nLayer: L,
    nHead: H,
    dModel: D,
    weight: (name: string) => {
      const t = W[name];
      if (!t) throw new Error(`missing tensor ${name}`);
      return t;
    },
    forward,
    nextTokenProbs,
    generate,
    surprisal,
  };
}

/**
 * The distribution generation samples from: softmax(logits / temperature) in float64, then
 * top-k (keep the k most likely), then top-p (keep the smallest most-likely set whose sum >= p),
 * renormalized. Ties are broken by the smaller token id. Same steps in Python (export.py).
 */
export function samplingDistribution(logits: ArrayLike<number>, temperature: number, topK = 0, topP = 1): Float64Array {
  const V = logits.length;
  const p = new Float64Array(V);
  let m = -Infinity;
  for (let i = 0; i < V; i++) if (logits[i] / temperature > m) m = logits[i] / temperature;
  let s = 0;
  for (let i = 0; i < V; i++) {
    p[i] = Math.exp(logits[i] / temperature - m);
    s += p[i];
  }
  for (let i = 0; i < V; i++) p[i] /= s;
  if ((topK > 0 && topK < V) || topP < 1) {
    const idx = Array.from({ length: V }, (_, i) => i).sort((a, b) => p[b] - p[a] || a - b);
    let keep = topK > 0 && topK < V ? topK : V;
    if (topP < 1) {
      let cum = 0;
      let n = 0;
      while (n < keep) {
        cum += p[idx[n]];
        n++;
        if (cum >= topP) break;
      }
      keep = Math.max(1, n);
    }
    for (let r = keep; r < V; r++) p[idx[r]] = 0;
    let z = 0;
    for (let i = 0; i < V; i++) z += p[i];
    for (let i = 0; i < V; i++) p[i] /= z;
  }
  return p;
}
