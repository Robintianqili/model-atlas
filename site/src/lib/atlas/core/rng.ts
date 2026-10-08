// The only random source of the DevKit (design §1 goal 4: `Math.random` is banned outside this file).
// mulberry32, as in tinygpt.ts (whose copy matches the Python sampler) and types/core.ts `Rng`:
//   next()      in [0, 1)
//   int(n)      floor(next() · n)
//   fork(label) mulberry32((seed ^ fnv1a32(label)) >>> 0): depends only on the seed and the label, never on how much
//               this generator has consumed. fnv1a32 hashes the label's UTF-8 bytes (offset 0x811c9dc5, prime 0x01000193).
//   normal()    Box–Muller: u1 = 1 − next(), u2 = next(), z = sqrt(−2 ln u1) · cos(2π u2); no caching.
// Python ak.Rng is bit-identical for next(), int() and fork(); normal() may differ in the last ulp (libm), so processes
// that use it record their draws (RecordOpts.recordDraws).
// Test vectors (rng.test.ts): mulberry32(42) → 0.6011037519, 0.4482905590, 0.8524657935, 0.6697340414;
// mulberry32(1) → 0.6270739406, 0.0027357212, 0.5274470400; fnv1a32('') = 0x811c9dc5, fnv1a32('a') = 0xe40c292c.
import type { Rng } from '../types/index.ts';

/** One mulberry32 step on state `a` (a uint32 or int32 number). Returns [newState, value in [0,1)]. */
function step(a: number): [number, number] {
	a = (a + 0x6d2b79f5) | 0;
	let t = Math.imul(a ^ (a >>> 15), 1 | a);
	t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
	return [a, ((t ^ (t >>> 14)) >>> 0) / 4294967296];
}

/** Plain mulberry32 closure (same as tinygpt.ts `mulberry32`). */
export function mulberry32(seed: number): () => number {
	let a = seed >>> 0;
	return () => {
		const r = step(a);
		a = r[0];
		return r[1];
	};
}

const enc = new TextEncoder();

/** 32-bit FNV-1a over the UTF-8 bytes of `s`, as an unsigned integer. */
export function fnv1a32(s: string): number {
	let h = 0x811c9dc5;
	for (const b of enc.encode(s)) {
		h ^= b;
		h = Math.imul(h, 0x01000193) >>> 0;
	}
	return h >>> 0;
}

/** A seeded generator implementing the frozen `Rng` contract. */
export function createRng(seed: number): Rng {
	const s0 = seed >>> 0;
	let a = s0;
	const rng: Rng = {
		seed: s0,
		next() {
			const r = step(a);
			a = r[0];
			return r[1];
		},
		normal() {
			const u1 = 1 - rng.next();
			const u2 = rng.next();
			return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
		},
		int(n: number) {
			return Math.floor(rng.next() * n);
		},
		fork(label: string) {
			return createRng((s0 ^ fnv1a32(label)) >>> 0);
		},
		getState() {
			return a >>> 0;
		},
		setState(s: number) {
			a = s >>> 0;
		}
	};
	return rng;
}

/** k distinct integers from [0, n) in draw order (partial Fisher–Yates; k is clamped to n). */
export function sampleIndices(rng: Rng, n: number, k: number): number[] {
	const m = Math.min(k, n);
	const pool = new Map<number, number>();
	const out: number[] = [];
	for (let i = 0; i < m; i++) {
		const j = i + rng.int(n - i);
		const vj = pool.get(j) ?? j;
		const vi = pool.get(i) ?? i;
		pool.set(j, vi);
		out.push(vj);
	}
	return out;
}
