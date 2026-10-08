// NaN-safe comparison (design §4.3 W1-A note, §6.2 item 8).
//   maxDiff(a, b, { relative? }) -> { max, at, nan }
//     absolute: |a − b|; relative: |a − b| / max(1, |b|)   (b is the reference)
//     −∞ == −∞ and +∞ == +∞ (masked attention scores); NaN anywhere, or an infinity on one side only,
//     fails at once with max = Infinity and nan = true; different lengths give max = Infinity, at = −1.
//   allclose(a, b, { atol, rtol }): the golden rule |a − b| ≤ atol + rtol·|b|, with the worst offender.

export interface DiffResult {
	/** largest difference (Infinity on NaN, an unmatched infinity or a length mismatch) */
	max: number;
	/** flat index of the largest difference (−1 when the arrays are equal or lengths differ) */
	at: number;
	/** true when a NaN or an unmatched infinity was found */
	nan: boolean;
}

export function maxDiff(a: ArrayLike<number>, b: ArrayLike<number>, o: { relative?: boolean } = {}): DiffResult {
	if (a.length !== b.length) return { max: Infinity, at: -1, nan: false };
	let max = 0;
	let at = -1;
	const rel = !!o.relative;
	for (let i = 0; i < a.length; i++) {
		const x = a[i];
		const y = b[i];
		if (x === y) continue; // covers equal infinities
		if (Number.isNaN(x) || Number.isNaN(y) || !Number.isFinite(x) || !Number.isFinite(y)) return { max: Infinity, at: i, nan: true };
		let d = Math.abs(x - y);
		if (rel) d /= Math.max(1, Math.abs(y));
		if (d > max) {
			max = d;
			at = i;
		}
	}
	return { max, at, nan: false };
}

export interface CloseResult {
	ok: boolean;
	/** largest |a − b| / (atol + rtol·|b|); ≤ 1 passes */
	ratio: number;
	/** the worst element, or null when every element is identical */
	worst: { i: number; a: number; b: number; diff: number; allowed: number } | null;
}

/** |a − b| ≤ atol + rtol·|b| elementwise (b is the reference). Equal infinities pass; NaN fails. */
export function allclose(a: ArrayLike<number>, b: ArrayLike<number>, o: { atol: number; rtol: number }): CloseResult {
	if (a.length !== b.length) return { ok: false, ratio: Infinity, worst: null };
	let ratio = 0;
	let worst: CloseResult['worst'] = null;
	for (let i = 0; i < a.length; i++) {
		const x = a[i];
		const y = b[i];
		if (x === y) continue;
		const allowed = o.atol + o.rtol * Math.abs(y);
		const diff = Math.abs(x - y);
		const r = Number.isNaN(diff) || !Number.isFinite(diff) ? Infinity : diff / allowed;
		if (worst === null || r > ratio) {
			ratio = r;
			worst = { i, a: x, b: y, diff, allowed };
		}
	}
	return { ok: ratio <= 1, ratio, worst };
}

/** Bitwise equality of two float arrays (−0 ≠ +0, NaN payloads ignored: NaN == NaN). */
export function bitEqual(a: ArrayLike<number>, b: ArrayLike<number>): { equal: boolean; at: number } {
	if (a.length !== b.length) return { equal: false, at: -1 };
	for (let i = 0; i < a.length; i++) {
		const x = a[i];
		const y = b[i];
		if (Object.is(x, y)) continue;
		return { equal: false, at: i };
	}
	return { equal: true, at: -1 };
}
