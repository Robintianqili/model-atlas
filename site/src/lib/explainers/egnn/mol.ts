// egnn: the molecules of the page and of the training data, and the toy energy that scores them. The page's presets and
// the training sets (pipeline/egnn/make_data.ts) both come from here, so PyTorch trains on the bytes the page computes.
//   KINDS / R0 / DEPTH   three atom kinds (0, 1, 2): the bond length of kinds a and b is SIZE[a] + SIZE[b]
//   energy(pos, kind)    the toy energy: one Morse term per pair, E = Σ_{i<j} V(r_ij), and the exact force on every atom,
//                        F_i = −∂E/∂x_i (a made-up potential, not real chemistry)
//   sample(n, rng)       a random molecule: n atoms placed one by one near an earlier atom, centred on their average
//   molOf(raw)           the learner's atoms: painted points (x, y, class) in plane units × SCALE, at most MAX_ATOMS,
//                        turned by raw.turn degrees and mirrored (raw.flip) about their centre
import type { I18n, Rng } from '../../atlas/types/index.ts';
import type { PointsRaw } from '../../atlas/adapters/points2d/adapter.ts';

export const K = 3;
export const SIZE = [0.5, 0.6, 0.75];
export const DEPTH = [1, 0.7, 0.4];
export const ALPHA = 2;
/** molecule units per unit of the input plane (the points editor spans −1.25 … 1.25) */
export const SCALE = 2;
/** the most atoms the model reads (spec dim N); the training molecules have NMIN … NMAX atoms */
export const MAX_ATOMS = 12;
export const NMIN = 2;
export const NMAX = 10;

export const KIND_LABEL = (k: number): I18n => ({ en: `kind ${k}`, zh: `第 ${k} 类` });
export const r0 = (a: number, b: number) => SIZE[a] + SIZE[b];
export const depth = (a: number, b: number) => Math.sqrt(DEPTH[a] * DEPTH[b]);

export type MolRaw = PointsRaw & { turn?: number; flip?: boolean };
export type Mol = { n: number; pos: Float64Array; kind: Int32Array };

/** E = Σ_{i<j} D (1 − e^{−α(r − r0)})² − D and F_i = −∂E/∂x_i, in float64 */
export function energy(pos: ArrayLike<number>, kind: ArrayLike<number>) {
	const n = kind.length;
	const F = new Float64Array(2 * n);
	let E = 0;
	for (let i = 0; i < n; i++)
		for (let j = i + 1; j < n; j++) {
			const dx = pos[2 * i] - pos[2 * j], dy = pos[2 * i + 1] - pos[2 * j + 1];
			const r = Math.hypot(dx, dy) || 1e-9;
			const D = depth(kind[i], kind[j]), e = Math.exp(-ALPHA * (r - r0(kind[i], kind[j])));
			E += D * (1 - e) ** 2 - D;
			const dV = 2 * D * ALPHA * e * (1 - e); // dV/dr
			const fx = (-dV * dx) / r, fy = (-dV * dy) / r;
			(F[2 * i] += fx), (F[2 * i + 1] += fy), (F[2 * j] -= fx), (F[2 * j + 1] -= fy);
		}
	return { E, F };
}

/** a random molecule of n atoms: each new atom at 0.85–1.25 bond lengths from a random earlier one, never closer than
 *  0.8 bond lengths to any atom; centred on the average position */
export function sample(n: number, rng: Rng): Mol {
	const pos = new Float64Array(2 * n), kind = new Int32Array(n);
	kind[0] = rng.int(K);
	for (let t = 1; t < n; t++) {
		const k = rng.int(K);
		let x = 0, y = 0;
		for (let tries = 0; tries < 200; tries++) {
			const p = rng.int(t), a = 2 * Math.PI * rng.next(), r = r0(kind[p], k) * (0.85 + 0.4 * rng.next());
			(x = pos[2 * p] + r * Math.cos(a)), (y = pos[2 * p + 1] + r * Math.sin(a));
			let ok = true;
			for (let q = 0; q < t && ok; q++) ok = Math.hypot(x - pos[2 * q], y - pos[2 * q + 1]) >= 0.8 * r0(kind[q], k);
			if (ok) break;
		}
		(pos[2 * t] = x), (pos[2 * t + 1] = y), (kind[t] = k);
	}
	return { n, pos: centred(pos), kind };
}

export function centre(pos: ArrayLike<number>) {
	const n = pos.length / 2;
	let cx = 0, cy = 0;
	for (let i = 0; i < n; i++) (cx += pos[2 * i] / n), (cy += pos[2 * i + 1] / n);
	return { cx, cy };
}
const centred = (pos: Float64Array) => {
	const { cx, cy } = centre(pos);
	return pos.map((v, k) => v - (k % 2 ? cy : cx));
};

/** positions turned by deg degrees (counter-clockwise) about their centre */
export function turn(pos: ArrayLike<number>, deg: number): Float64Array {
	const { cx, cy } = centre(pos);
	const c = Math.cos((deg * Math.PI) / 180), s = Math.sin((deg * Math.PI) / 180);
	const out = new Float64Array(pos.length);
	for (let i = 0; i < pos.length / 2; i++) {
		const x = pos[2 * i] - cx, y = pos[2 * i + 1] - cy;
		(out[2 * i] = cx + c * x - s * y), (out[2 * i + 1] = cy + s * x + c * y);
	}
	return out;
}
/** positions mirrored in the vertical line through their centre (x → 2·cx − x) */
export function mirror(pos: ArrayLike<number>): Float64Array {
	const { cx } = centre(pos);
	return Float64Array.from(pos, (v, k) => (k % 2 ? v : 2 * cx - v));
}
/** a vector (x, y) turned by deg degrees */
export const turnVec = (x: number, y: number, deg: number) => {
	const c = Math.cos((deg * Math.PI) / 180), s = Math.sin((deg * Math.PI) / 180);
	return [c * x - s * y, s * x + c * y];
};
/** the direction of (x, y) in degrees, 0 = pointing right, 90 = up, in (−180, 180] */
export const angleOf = (x: number, y: number) => (Math.atan2(y, x) * 180) / Math.PI;

const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);
/** the learner's atoms (painted points: plane units × SCALE, class = kind), at most MAX_ATOMS, turned and mirrored */
export function molOf(raw: unknown): Mol & { more: boolean; deg: number; flip: boolean } {
	const r = (raw && typeof raw === 'object' ? raw : {}) as MolRaw;
	const ps = (r.points ?? []).filter((q) => finite(q?.x) && finite(q?.y));
	const atoms = ps.slice(0, MAX_ATOMS);
	const n = atoms.length;
	let pos: Float64Array = Float64Array.from(atoms.flatMap((q) => [q.x * SCALE, q.y * SCALE]));
	const kind = Int32Array.from(atoms, (q) => (Number.isInteger(q.c) && (q.c as number) >= 0 && (q.c as number) < K ? (q.c as number) : 0));
	const deg = finite(r.turn) ? r.turn : 0;
	if (r.flip) pos = mirror(pos);
	if (deg) pos = turn(pos, deg);
	return { n, pos, kind, more: ps.length > MAX_ATOMS, deg, flip: !!r.flip };
}

/** the page's molecules (molecule units, kinds), slightly off their best shape so the forces are visible */
const PRESETS: Record<string, { label: I18n; atoms: [number, number, number][] }> = {
	bent: { label: { en: 'Bent four', zh: '弯折的四原子' }, atoms: [[-1.5, -0.35, 0], [-0.45, 0.25, 2], [0.75, -0.3, 1], [1.55, 0.55, 0]] },
	ring: { label: { en: 'Ring of six', zh: '六元环' }, atoms: Array.from({ length: 6 }, (_, i) => [1.25 * Math.cos((i * Math.PI) / 3) * (i === 0 ? 1.15 : 1), 1.25 * Math.sin((i * Math.PI) / 3), i % 2] as [number, number, number]) },
	star: { label: { en: 'Star', zh: '星形' }, atoms: [[0, 0, 2], [1.3, 0.1, 0], [-0.2, 1.15, 0], [-1.25, -0.2, 1], [0.15, -1.5, 0]] },
	chain: { label: { en: 'Chain of seven', zh: '七原子链' }, atoms: Array.from({ length: 7 }, (_, i) => [-2.35 + 0.8 * i, i % 2 ? 0.4 : -0.3, i === 3 ? 2 : i % 2] as [number, number, number]) },
	pair: { label: { en: 'Two atoms', zh: '两个原子' }, atoms: [[-0.6, 0.2, 0], [0.55, -0.15, 1]] }
};
export const PRESET_IDS = Object.keys(PRESETS);
export function presetRaw(id: string): { label: I18n; raw: MolRaw } {
	const p = PRESETS[id];
	return { label: p.label, raw: { points: p.atoms.map(([x, y, c]) => ({ x: Math.round((x / SCALE) * 1000) / 1000, y: Math.round((y / SCALE) * 1000) / 1000, c })) } };
}
