// Rows and columns of the tape.table view (pure TS; meta.ts uses it for describe()).
import type { I18n, Tape } from '../../../types/index.ts';
import type { Props } from '../kit.parts/stage.ts';

export interface Col {
	key: string;
	label: string;
	from: 'metric' | 'state';
}
export interface Row {
	i: number;
	phase: string;
	phaseLabel: I18n;
	step?: number;
	cells: Record<string, number | string | undefined>;
}
export interface TableData {
	tape: Tape;
	cols: Col[];
	rows: Row[];
	at: number;
	from: number;
	to: number;
}

type PhaseInfo = { label: I18n };

/** The window of frames around the stage's frame (props.window rows, default 30; the whole tape when short). */
export function tableOf(p: Props): TableData | null {
	const tape = p.at.tape ?? p.rt.sources[p.source]?.tape ?? null;
	if (!tape || tape.length === 0) return null;
	const at = Math.max(0, Math.min(tape.length - 1, p.at.index >= 0 ? p.at.index : tape.length - 1));
	const win = typeof p.props.window === 'number' ? p.props.window : 30;
	const from = tape.length <= win ? 0 : Math.max(0, Math.min(tape.length - win, at - Math.floor(win / 2)));
	const to = Math.min(tape.length, from + win);
	const phases = (tape as { phases?: Record<string, PhaseInfo> }).phases ?? {};
	const schema = (tape as { schema?: Record<string, { kind?: string; label?: I18n }> }).schema ?? {};
	const want = Array.isArray(p.props.columns) ? (p.props.columns as string[]) : null;
	const metricKeys: string[] = [];
	const rows: Row[] = [];
	for (let i = from; i < to; i++) {
		const f = tape.frame(i);
		for (const k of Object.keys(f.metrics ?? {})) if (!metricKeys.includes(k)) metricKeys.push(k);
		rows.push({ i, phase: f.phase, phaseLabel: phases[f.phase]?.label ?? { en: f.phase, zh: f.phase }, step: f.step, cells: {} });
	}
	const stateKeys = Object.entries(schema)
		.filter(([, d]) => d.kind === 'scalar' || d.kind === 'int' || d.kind === 'index' || d.kind === 'text')
		.map(([k]) => k)
		.filter((k) => !metricKeys.includes(k));
	let cols: Col[] = [...metricKeys.map((k) => ({ key: k, label: k, from: 'metric' as const })), ...stateKeys.map((k) => ({ key: k, label: k, from: 'state' as const }))];
	if (want) cols = want.map((k) => cols.find((c) => c.key === k) ?? { key: k, label: k, from: 'state' as const });
	cols = cols.slice(0, 8);
	for (const r of rows) {
		const f = tape.frame(r.i);
		let st: Record<string, unknown> | null = null;
		for (const c of cols) {
			if (c.from === 'metric') r.cells[c.key] = f.metrics?.[c.key];
			else {
				st ??= (tape.state(r.i) ?? {}) as Record<string, unknown>;
				const v = st[c.key];
				r.cells[c.key] = typeof v === 'number' || typeof v === 'string' ? v : undefined;
			}
		}
	}
	return { tape, cols, rows, at, from, to };
}
