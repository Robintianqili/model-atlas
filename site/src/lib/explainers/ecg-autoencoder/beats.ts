// ecg-autoencoder: the synthetic heartbeats (pure TS). The presets, the processes and the training data all come from here.
// A beat is N = 140 numbers: the height of an ECG-like trace at 140 moments of one heartbeat, made of five bumps (bell
// curves): the P wave, the Q, R and S of the QRS spike, the T wave, plus a small baseline offset and tilt and noise.
// Anomalies change one thing, by a random amount: a P wave shrunk to 0-60% of its height, a QRS 1.25-2 times as wide,
// an ST segment (between S and T) raised by 0.02-0.12, or noise of 0.02-0.06 instead of 0.008-0.024.
// Every kind draws the same numbers in the same order, so beatOf('noP', 1) is beatOf('normal', 1) without its P wave.
import type { I18n, Rng } from '../../atlas/types/index.ts';
import { createRng } from '../../atlas/core/rng.ts';

export const N = 140;
export type Kind = 'normal' | 'noP' | 'wideQRS' | 'stUp' | 'noisy';
export const KINDS: Kind[] = ['normal', 'noP', 'wideQRS', 'stUp', 'noisy'];
export const I = (en: string, zh: string): I18n => ({ en, zh });
export const KIND_LABEL: Record<Kind, I18n> = {
	normal: I('normal', '正常'),
	noP: I('missing P wave', '缺 P 波'),
	wideQRS: I('wide QRS', 'QRS 增宽'),
	stUp: I('raised ST segment', 'ST 段抬高'),
	noisy: I('noisy recording', '噪声大的记录')
};

type Wave = { c: number; a: number; w: number }; // centre (share of the beat), height, width (standard deviation)
export interface BeatParams { P: Wave; Q: Wave; R: Wave; S: Wave; T: Wave; st: number; base: number; tilt: number; noise: number }

const u = (r: Rng, lo: number, hi: number) => lo + (hi - lo) * r.next();

/** The shape of one beat of a kind: 14 uniform draws, the same for every kind; k is how strong
 *  an anomaly is (from barely visible to clear). */
export function paramsOf(kind: Kind, r: Rng): BeatParams {
	const s = u(r, -0.02, 0.02);
	const P = { c: 0.22 + s + u(r, -0.01, 0.01), a: 0.15 * u(r, 0.8, 1.2), w: 0.025 * u(r, 0.85, 1.15) };
	const Q = { c: 0.36 + s, a: -0.1 * u(r, 0.7, 1.3), w: 0.009 };
	const R = { c: 0.385 + s, a: u(r, 0.85, 1.15), w: 0.012 * u(r, 0.9, 1.1) };
	const S = { c: 0.41 + s, a: -0.22 * u(r, 0.7, 1.3), w: 0.011 };
	const T = { c: 0.64 + s + u(r, -0.02, 0.02), a: 0.3 * u(r, 0.8, 1.2), w: 0.045 * u(r, 0.85, 1.15) };
	const base = u(r, -0.03, 0.03), tilt = u(r, -0.03, 0.03), k = r.next();
	let noise = u(r, 0.008, 0.024), st = 0;
	if (kind === 'noP') P.a *= 0.6 * (1 - k);
	else if (kind === 'wideQRS') {
		const f = 1.25 + 0.75 * k;
		Q.w *= f; R.w *= f; S.w *= f;
		Q.c -= 0.012 * k; S.c += 0.012 * k; R.a *= 1 - 0.2 * k; S.a *= 1 + 0.4 * k;
	} else if (kind === 'stUp') st = 0.02 + 0.1 * k;
	else if (kind === 'noisy') noise = 0.02 + 0.04 * k;
	return { P, Q, R, S, T, st, base, tilt, noise };
}

const bump = (t: number, w: Wave) => w.a * Math.exp(-0.5 * ((t - w.c) / w.w) ** 2);
const sig = (x: number) => 1 / (1 + Math.exp(-x));

/** The 140 heights of a beat (noise: one normal draw per point, after the shape draws). */
export function render(p: BeatParams, r: Rng): Float32Array {
	const y = new Float32Array(N);
	for (let i = 0; i < N; i++) {
		const t = i / (N - 1);
		let v = p.base + p.tilt * (t - 0.5) + bump(t, p.P) + bump(t, p.Q) + bump(t, p.R) + bump(t, p.S) + bump(t, p.T);
		if (p.st) v += p.st * sig((t - p.S.c - 0.02) / 0.006) * sig((p.T.c - t) / 0.03);
		y[i] = v + p.noise * r.normal();
	}
	return y;
}

export const beat = (kind: Kind, r: Rng) => render(paramsOf(kind, r), r);
export const beatOf = (kind: Kind, seed: number) => beat(kind, createRng(seed));

/** Where each wave peaks (index of the point nearest its centre) in a beat with these parameters. */
export const wavesAt = (p: BeatParams) => Object.fromEntries((['P', 'Q', 'R', 'S', 'T'] as const).map((k) => [k, Math.round(p[k].c * (N - 1))])) as Record<'P' | 'Q' | 'R' | 'S' | 'T', number>;

/** The test set's layout: nTest normal beats, then nOdd of each anomaly kind (kinds index KINDS). */
export function testKinds(nTest: number, nOdd: number): number[] {
	return [...Array(nTest).fill(0), ...KINDS.slice(1).flatMap((_, k) => Array(nOdd).fill(k + 1))];
}

/** CRC-32 (IEEE, as zlib.crc32) of a float32 array's bytes: train.py checks the bytes it read against it. */
const CRC = Array.from({ length: 256 }, (_, n) => {
	let c = n;
	for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
	return c >>> 0;
});
export function crcBytes(a: Float32Array): number {
	const b = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
	let c = 0xffffffff;
	for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 255] ^ (c >>> 8);
	return (c ^ 0xffffffff) >>> 0;
}
