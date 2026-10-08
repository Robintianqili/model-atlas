/**
 * Logit lens for TinyGPT (pure TS: runs in the Worker, in the page and in Node).
 *
 * The model's real output is  logits = LN_f(x_last) · Eᵀ  where x_last is the residual stream after
 * block 4 and E is the token-embedding table (the output head is tied to it). The logit lens applies
 * the same final LayerNorm and the same head to the residual stream at earlier points (after the
 * embedding, after each attention sub-layer, after each MLP sub-layer) and asks "what would the model
 * predict if it stopped here?". The last point (L3.resid2) is exactly the model's output.
 *
 * Arithmetic copies tinygpt.ts (float32 storage, float64 sums), so the last row reproduces the
 * traced probabilities bit for bit.
 */

export type LensInput = {
	T: number;
	D: number;
	V: number;
	eps: number;
	/** residual stream at each lens point, [T, D] row-major */
	points: { id: string; data: Float32Array }[];
	lnW: Float32Array;
	lnB: Float32Array;
	/** token embedding [V, D] (= output head) */
	emb: Float32Array;
};

export type Lens = {
	T: number;
	/** top-k kept per cell */
	K: number;
	/** trace ids of the lens points, in order */
	points: string[];
	/** [P, T, K] token ids, most likely first */
	topIds: Int32Array;
	/** [P, T, K] their probabilities */
	topP: Float32Array;
	/** [T] the model's final top-1 token at each position (argmax of the last point) */
	finalTop: Int32Array;
	/** [P, T] probability of finalTop[t] at this point */
	pFinal: Float32Array;
	/** [P, T] rank of finalTop[t] at this point (1 = top) */
	rankFinal: Int32Array;
	/** [P, T] entropy of the distribution at this point, in nats */
	entropy: Float32Array;
};

export const LENS_K = 5;

/** Lens points for an nLayer model: embed, then after attention and after MLP of every block. */
export function lensPoints(nLayer: number): string[] {
	const out = ['embed'];
	for (let l = 0; l < nLayer; l++) out.push(`L${l}.resid1`, `L${l}.resid2`);
	return out;
}

export function logitLens(inp: LensInput, K = LENS_K): Lens {
	const { T, D, V, eps, points, lnW, lnB, emb } = inp;
	const P = points.length;
	const topIds = new Int32Array(P * T * K);
	const topP = new Float32Array(P * T * K);
	const pFinal = new Float32Array(P * T);
	const rankFinal = new Int32Array(P * T);
	const entropy = new Float32Array(P * T);
	const finalTop = new Int32Array(T);
	// all probabilities, kept to look up the final token afterwards: [P, T, V]
	const probsAll = new Float32Array(P * T * V);
	const y = new Float32Array(D);
	const logits = new Float32Array(V);
	const tmp = new Float64Array(V);
	for (let p = 0; p < P; p++) {
		const x = points[p].data;
		for (let t = 0; t < T; t++) {
			const o = t * D;
			// final LayerNorm (same steps as tinygpt.ts layerNorm)
			let mean = 0;
			for (let i = 0; i < D; i++) mean += x[o + i];
			mean /= D;
			let v = 0;
			for (let i = 0; i < D; i++) v += (x[o + i] - mean) ** 2;
			v /= D;
			const inv = 1 / Math.sqrt(v + eps);
			for (let i = 0; i < D; i++) y[i] = (x[o + i] - mean) * inv * lnW[i] + lnB[i];
			// tied head
			for (let w = 0; w < V; w++) {
				const eo = w * D;
				let s = 0;
				for (let i = 0; i < D; i++) s += y[i] * emb[eo + i];
				logits[w] = s;
			}
			// softmax (float64 inside)
			let m = -Infinity;
			for (let w = 0; w < V; w++) if (logits[w] > m) m = logits[w];
			let z = 0;
			for (let w = 0; w < V; w++) {
				const e = Math.exp(logits[w] - m);
				tmp[w] = e;
				z += e;
			}
			const po = (p * T + t) * V;
			let H = 0;
			for (let w = 0; w < V; w++) {
				const q = tmp[w] / z;
				probsAll[po + w] = q;
				if (q > 0) H -= q * Math.log(q);
			}
			entropy[p * T + t] = H;
			// top-K by insertion (ties: smaller id first, like tinygpt.ts)
			const ko = (p * T + t) * K;
			let n = 0;
			for (let w = 0; w < V; w++) {
				const q = probsAll[po + w];
				if (n === K && q <= topP[ko + K - 1]) continue;
				let j = n < K ? n++ : K - 1;
				while (j > 0 && topP[ko + j - 1] < q) {
					topP[ko + j] = topP[ko + j - 1];
					topIds[ko + j] = topIds[ko + j - 1];
					j--;
				}
				topP[ko + j] = q;
				topIds[ko + j] = w;
			}
		}
	}
	for (let t = 0; t < T; t++) finalTop[t] = topIds[((P - 1) * T + t) * K];
	for (let p = 0; p < P; p++) {
		for (let t = 0; t < T; t++) {
			const po = (p * T + t) * V;
			const q = probsAll[po + finalTop[t]];
			pFinal[p * T + t] = q;
			let r = 1;
			for (let w = 0; w < V; w++) if (probsAll[po + w] > q) r++;
			rankFinal[p * T + t] = r;
		}
	}
	return { T, K, points: points.map((q) => q.id), topIds, topP, finalTop, pFinal, rankFinal, entropy };
}

/** The buffers of a Lens, for postMessage transfer lists. */
export function lensBuffers(l: Lens): ArrayBuffer[] {
	return [l.topIds, l.topP, l.finalTop, l.pFinal, l.rankFinal, l.entropy].map((a) => a.buffer as ArrayBuffer);
}
