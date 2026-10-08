// Print a tape as a plain-text table (atlas run). Works on any Tape (W1-C's TapeStore or the fallback store).
import type { Change, Frame, Tape } from '../../../src/lib/atlas/types/index.ts';
import { isTensorLike } from './fallback.ts';

export function fmtNum(v: number): string {
	if (!Number.isFinite(v)) return String(v);
	if (Number.isInteger(v) && Math.abs(v) < 1e9) return String(v);
	const a = Math.abs(v);
	if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(3);
	return String(Number(v.toPrecision(4)));
}

export function fmtValue(v: unknown, depth = 0): string {
	if (typeof v === 'number') return fmtNum(v);
	if (typeof v === 'string') return JSON.stringify(v.length > 24 ? v.slice(0, 23) + '…' : v);
	if (typeof v === 'boolean' || v === null || v === undefined) return String(v);
	if (isTensorLike(v)) {
		const n = v.data.length;
		const head = Array.from(v.data.subarray(0, Math.min(n, 4)), (x) => fmtNum(Number(x))).join(' ');
		const kind = v.data instanceof Float64Array ? 'f64' : v.data instanceof Float32Array ? 'f32' : 'i32';
		return `[${v.shape.join('×')} ${kind}: ${head}${n > 4 ? ' …' : ''}]`;
	}
	if (ArrayBuffer.isView(v)) return `[${(v as unknown as ArrayLike<number>).length} values]`;
	if (Array.isArray(v)) return depth > 0 || v.length > 6 ? `[${v.length} items]` : `[${v.map((x) => fmtValue(x, depth + 1)).join(', ')}]`;
	if (typeof v === 'object') {
		const keys = Object.keys(v as object);
		if (depth > 0) return `{${keys.length} keys}`;
		return `{${keys
			.slice(0, 4)
			.map((k) => `${k}: ${fmtValue((v as Record<string, unknown>)[k], depth + 1)}`)
			.join(', ')}${keys.length > 4 ? ', …' : ''}}`;
	}
	return String(v);
}

function fmtChange(c: Change): string {
	const at = `${c.field}${c.key !== undefined ? `.${c.key}` : ''}${c.index ? `[${c.index.join(',')}]` : ''}`;
	return c.from !== undefined ? `${at} ${fmtNum(c.from)}→${fmtNum(c.to ?? NaN)}` : `${at}=${fmtNum(c.to ?? NaN)}`;
}

export function frameRow(i: number, f: Frame & { key?: boolean }): string[] {
	const metrics = Object.entries(f.metrics ?? {})
		.map(([k, v]) => `${k}=${fmtNum(v)}`)
		.join(' ');
	const state = Object.entries(f.state ?? {})
		.map(([k, v]) => `${k}=${fmtValue(v)}`)
		.join(' ');
	const changes = (f.changes ?? []).map(fmtChange).join(' ');
	const draws = f.draws?.length ? `draws=${f.draws.length}` : '';
	return [String(i), f.phase, f.step === undefined ? '' : String(f.step), f.sub === undefined ? '' : String(f.sub), metrics, [state, changes, draws].filter(Boolean).join('  ')];
}

export function renderTable(t: Tape, from: number, to: number, width = 160): string {
	const head = ['#', 'phase', 'step', 'sub', 'metrics', 'stored state / changes'];
	const rows = [] as string[][];
	for (let i = Math.max(0, from); i <= Math.min(to, t.length - 1); i++) rows.push(frameRow(i, t.frame(i)));
	const w = head.map((h, c) => Math.min(c === 5 ? 999 : 40, Math.max(h.length, ...rows.map((r) => r[c].length))));
	const line = (r: string[]) => {
		const s = r.map((x, c) => (c === 5 ? x : x.padEnd(w[c]))).join('  ');
		return s.length > width ? s.slice(0, width - 1) + '…' : s;
	};
	return [line(head), line(w.map((n) => '-'.repeat(n))), ...rows.map(line)].join('\n');
}
