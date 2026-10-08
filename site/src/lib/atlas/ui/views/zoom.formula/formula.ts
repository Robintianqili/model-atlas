// Which formula zoom.formula shows (pure TS; meta.ts uses it for describe()).
import type { I18n } from '../../../types/index.ts';
import type { Frame, PhaseDef } from '../../../types/process.ts';
import { layerLabel, resolveUnit, unitWhere, axesOfId, type Props } from '../kit.parts/stage.ts';

export interface FormulaSpec {
	tex: string;
	values: Record<string, number | string>;
	caption?: I18n;
	where?: I18n;
	legend: { tex: string; text: I18n }[];
	from: 'props' | 'phase' | 'op';
}

type PhaseInfo = PhaseDef | { label: I18n; kind: PhaseDef['kind']; tex?: string };

function phaseOf(tape: unknown, phase: string): PhaseInfo | undefined {
	const phases = (tape as { phases?: Record<string, PhaseInfo> } | null)?.phases;
	return phases?.[phase];
}

function numbers(o: Record<string, unknown> | undefined): Record<string, number> {
	const out: Record<string, number> = {};
	for (const [k, v] of Object.entries(o ?? {})) if (typeof v === 'number') out[k] = v;
	return out;
}

export function formulaOf(p: Props): FormulaSpec | null {
	const legend = Array.isArray(p.props.where) ? (p.props.where as { tex: string; text: I18n }[]) : [];
	const caption = (p.props.caption as I18n | undefined) ?? undefined;
	if (typeof p.props.tex === 'string') {
		const values = (p.props.values as Record<string, number | string> | undefined) ?? numbers(p.at.frame?.metrics);
		return { tex: p.props.tex, values, caption, legend, from: 'props' };
	}
	const f: Frame | undefined = p.at.frame;
	if (f && p.at.tape) {
		const ph = phaseOf(p.at.tape, f.phase);
		const tex = ph && 'formula' in ph && ph.formula ? ph.formula.tex : (ph as { tex?: string } | undefined)?.tex;
		if (tex) {
			let values: Record<string, number | string> = numbers(f.metrics);
			if (ph && 'formula' in ph && ph.formula?.values) {
				try {
					values = { ...values, ...ph.formula.values(f) };
				} catch {
					/* keep the metrics */
				}
			}
			// f.step is the process's own step number (training step k counts from 0, as {frame.step} in the narration)
			const step = f.step !== undefined ? { en: ` · step ${f.step}`, zh: ` · 第 ${f.step} 步` } : { en: '', zh: '' };
			const label = ph?.label ?? { en: f.phase, zh: f.phase };
			return {
				tex,
				values,
				caption: caption ?? { en: `${label.en}${step.en}`, zh: `${label.zh ?? label.en}${step.zh}` },
				// an on-demand tape (recorded only as far as the lesson needed) does not know its length yet: no "of n" (R0:
				// "Frame 5 of 5" on a run of many more frames). K1: a frame with a step is named by its caption ("Update ·
				// step 7"): no second line repeating it as "Frame k of n: … step 7"
				where:
					f.step !== undefined
						? undefined
						: (p.at.tape as { done?: boolean }).done === false
							? { en: `Frame ${p.at.index + 1} of the run: ${label.en}.`, zh: `这次运行的第 ${p.at.index + 1} 帧：${label.zh ?? label.en}。` }
							: { en: `Frame ${p.at.index + 1} of ${p.at.tape.length}: ${label.en}.`, zh: `第 ${p.at.index + 1} / ${p.at.tape.length} 帧：${label.zh ?? label.en}。` },
				legend,
				from: 'phase'
			};
		}
	}
	const r = resolveUnit(p);
	if (r.e && r.op?.def.formula) {
		const out = r.trace?.t[r.e.layer];
		const w = unitWhere(r.e.unit, out ? axesOfId(r.model, r.e.layer, out.shape) : []);
		const name = layerLabel(r.model, r.e.layer);
		return { tex: r.op.def.formula, values: { ...r.e.values }, caption: caption ?? { en: `${name.en}, ${w.en}`, zh: `${name.zh ?? name.en}，${w.zh}` }, legend, from: 'op' };
	}
	return null;
}
