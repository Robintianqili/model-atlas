// Pure helpers for TapeScrubber: which frames a play range covers, and the jump marks under the scrubber.
// Coarse training tapes alternate 'step' (metrics only) and 'snapshot' (weights) frames, so a range can be restricted to
// some phases (StepDef.play.phases), and alternating short segments are not listed one by one: when the phase runs are
// too many or too short, the marks are evenly spaced and labelled by training step (or position).
import type { I18n } from '../types/core.ts';
import type { Cursor, Frame, Tape } from '../types/process.ts';
import { resolveCursor } from '../runtime/selection.ts';

/** Tape indices from `from` to `to` (inclusive, clamped), keeping only `phases` when given. */
export function playFrames(tape: Tape | null | undefined, from: Cursor, to: Cursor, phases?: string[]): number[] {
	if (!tape || tape.length === 0) return [];
	const a = resolveCursor(tape, from).index;
	const b = resolveCursor(tape, to).index;
	const lo = Math.max(0, Math.min(a, b));
	const hi = Math.max(a, b);
	const out: number[] = [];
	const keep = phases?.length ? new Set(phases) : null;
	for (let i = lo; i <= hi; i++) if (!keep || keep.has(tape.frame(i).phase)) out.push(i);
	return out;
}

export interface Mark {
	/** position in the frames list */
	at: number;
	label: I18n;
}

function phaseLabel(tape: Tape, phase: string): I18n {
	const p = (tape as { phases?: Record<string, { label?: I18n }> }).phases?.[phase];
	return p?.label ?? { en: phase, zh: phase };
}

/** What a play range counts (StepDef.play.unit / by, K0): the word ('episode', 'move') and the frame field that holds
 *  the count ('step' by default, or 'sub'). Without a word, a count reads "step k". */
export interface CountUnit {
	unit?: I18n;
	by?: 'step' | 'sub';
}

/** K2: a unit word with its number: "episode 12", or the unit's own template where {n} goes ('第 {n} 秒' reads "第 18 秒";
 *  a plain word puts the number after it, which reads wrongly in Chinese: "秒 18"). */
export function unitText(unit: I18n, n: number | string): I18n {
	const one = (s: string) => (s.includes('{n}') ? s.replace('{n}', String(n)) : `${s} ${n}`);
	return { en: one(unit.en), zh: one(unit.zh ?? unit.en) };
}

/** "step 7" / "第 7 步", or "<unit> 7" with a unit word (K2: or its {n} template); null when the frame has no such count. */
export function countLabel(f: Frame, u?: CountUnit): I18n | null {
	const n = u?.by === 'sub' ? f.sub : f.step;
	if (n === undefined) return null;
	if (u?.unit) return unitText(u.unit, n);
	return { en: `step ${n}`, zh: `第 ${n} 步` };
}

/** A short caption of one frame: its caption, else the phase label plus its count ("Update · step 7"). */
export function frameLabel(tape: Tape, i: number, u?: CountUnit): I18n {
	const f = tape.frame(i);
	const p = phaseLabel(tape, f.phase);
	const cap = f.caption;
	if (cap) return cap;
	const c = countLabel(f, u);
	if (c) return { en: `${p.en} · ${c.en}`, zh: `${p.zh ?? p.en} · ${c.zh ?? c.en}` };
	return p;
}

/** The name of one cell of a scan ("row 3", "feature 12", "key 5"), counting from 0; null without a unit. */
export function positionLabel(tape: Tape, i: number, axes?: readonly string[]): I18n | null {
	const u = (tape.state(i) as { unit?: number[] } | undefined)?.unit;
	const ax = axes ?? (tape as { axes?: readonly string[] }).axes;
	const grid = !!ax && ax.includes('y') && ax.includes('x');
	if (Array.isArray(u) && u.length >= 2 && (grid || !ax)) return { en: `row ${u[u.length - 2]}`, zh: `第 ${u[u.length - 2]} 行` };
	if (Array.isArray(u) && u.length >= 1 && ax) {
		const w = AXIS_WORD[ax[u.length - 1]] ?? { en: 'index', zh: '索引' };
		return { en: `${w.en} ${u[u.length - 1]}`, zh: `${w.zh} ${u[u.length - 1]}` };
	}
	return null;
}

/** The name of the axis a scan walks along (its last varying axis), for its jump marks: a rows scan over a [T, D] layer
 *  walks the features of one position, so its marks say "feature 12", not "position 12" (R0). Indices count from 0. */
const AXIS_WORD: Record<string, I18n> = {
	d: { en: 'feature', zh: '特征' },
	f: { en: 'feature', zh: '特征' },
	t: { en: 'position', zh: '位置' },
	tq: { en: 'query', zh: '查询' },
	tk: { en: 'key', zh: '键' },
	x: { en: 'column', zh: '列' },
	y: { en: 'row', zh: '行' },
	c: { en: 'channel', zh: '通道' },
	h: { en: 'head', zh: '头' },
	class: { en: 'class', zh: '类别' },
	vocab: { en: 'token', zh: '词元' },
	n: { en: 'node', zh: '节点' },
	nk: { en: 'node', zh: '节点' },
	b: { en: 'item', zh: '项' },
	p: { en: 'patch', zh: '图块' }
};

/** Jump marks: one per phase run when there are 2..max runs of reasonable length, else up to `even` evenly spaced marks.
 *  A scan's marks name the cell's index along the axis the scan walks (row of a raster scan; feature, position, ... of a
 *  one-axis scan), counting from 0; `axes` are the scanned layer's axes (ScanTape.axes). */
export function scrubMarks(tape: Tape | null | undefined, frames: number[], o: { max?: number; even?: number; unit?: 'position' | 'frame'; axes?: readonly string[]; count?: CountUnit } = {}): Mark[] {
	if (!tape || frames.length < 2) return [];
	const max = o.max ?? 12;
	const even = o.even ?? 8;
	const runs: { at: number; phase: string; len: number }[] = [];
	frames.forEach((fi, k) => {
		const ph = tape.frame(fi).phase;
		const last = runs[runs.length - 1];
		if (last && last.phase === ph) last.len++;
		else runs.push({ at: k, phase: ph, len: 1 });
	});
	const minLen = Math.max(2, Math.ceil(frames.length / 40));
	if (runs.length >= 2 && runs.length <= max && runs.every((r) => r.len >= minLen || r.at === 0)) return runs.map((r) => ({ at: r.at, label: phaseLabel(tape, r.phase) }));
	// many short runs (coarse training: step, snapshot, step, …) or one long run (a scan): evenly spaced marks
	const n = Math.min(even, frames.length);
	const out: Mark[] = [];
	for (let k = 0; k < n; k++) {
		const at = Math.round((k * (frames.length - 1)) / Math.max(1, n - 1));
		if (out.length && out[out.length - 1].at === at) continue;
		const f = tape.frame(frames[at]);
		let label: I18n = { en: `output ${at + 1}`, zh: `第 ${at + 1} 个输出` };
		if (o.unit === 'position') {
			// a scan: name the cell by the axis it walks: the row of a raster scan over a grid (unit [.., row, col]),
			// else the last axis (the features of one position, the keys of one query, ...)
			label = positionLabel(tape, frames[at], o.axes) ?? label;
		} else {
			// a count ("episode 12" with play.unit), else a short caption, else the phase (a training run's 'init' frame)
			const c = o.count?.unit ? countLabel(f, o.count) : null;
			const cap = f.caption && (f.caption.en ?? '').length <= 20 ? f.caption : null;
			label = c ?? cap ?? countLabel(f, o.count) ?? phaseLabel(tape, f.phase);
		}
		out.push({ at, label });
	}
	// marks that would all read the same (a scan along one row, without axes): name them by their place in the scan
	if (o.unit === 'position' && out.length > 1 && new Set(out.map((m) => m.label.en)).size === 1)
		for (const m of out) m.label = { en: `output ${m.at + 1}`, zh: `第 ${m.at + 1} 个输出` };
	return out;
}
