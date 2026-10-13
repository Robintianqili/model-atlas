// View 'regularization/units': dropout at one update. The hidden units of both layers as squares: a kept unit filled
// (its output multiplied by 1/(1 − p) for this update), a switched-off unit empty with a dashed outline. Reads the
// frame's mask (state.mask [2, width]: 0 or 1/(1 − p) per unit) and p (metrics.dropout). Played over a run with one
// frame per update, the units blink. No props.
import type { I18n, Tensor } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtInt } from '../../../../atlas/core/format.ts';
import { START } from '../../data.ts';

export interface UnitsPic {
	step: number;
	p: number;
	scale: number;
	width: number;
	/** per layer, per unit: kept */
	kept: boolean[][];
	off: number[];
}

export function unitsPic(p: Pick<StageProps, 'at'>): UnitsPic | null {
	const mask = (p.at?.state as { mask?: Tensor } | undefined)?.mask;
	const m = p.at?.frame?.metrics;
	if (!mask || !m) return null;
	const width = mask.shape[1];
	const kept = [0, 1].map((l) => Array.from({ length: width }, (_, i) => mask.data[l * width + i] !== 0));
	return { step: m.step, p: m.dropout, scale: 1 / (1 - m.dropout), width, kept, off: kept.map((row) => row.filter((k) => !k).length) };
}

export function describeUnits(pic: UnitsPic | null): I18n {
	if (!pic) return { en: 'No dropout mask in this frame (it starts with the first update).', zh: '这一帧还没有 dropout 掩码（从第一次更新开始才有）。' };
	return {
		en: `Update ${fmtInt(pic.step)}, dropout p = ${fmt3(pic.p)}: ${fmtInt(pic.off[0])} of the ${fmtInt(pic.width)} units of hidden layer 1 and ${fmtInt(pic.off[1])} of layer 2 are switched off; every other unit's output is multiplied by ${fmt3(pic.scale)}.`,
		zh: `第 ${fmtInt(pic.step)} 次更新，dropout p = ${fmt3(pic.p)}：隐藏层 1 的 ${fmtInt(pic.width)} 个单元中有 ${fmtInt(pic.off[0])} 个被关掉，隐藏层 2 中有 ${fmtInt(pic.off[1])} 个；其余每个单元的输出都乘以 ${fmt3(pic.scale)}。`
	};
}

const meta = {
	key: 'regularization/units',
	title: { en: 'Dropout: which units are switched off', zh: 'Dropout：哪些单元被关掉了' },
	kinds: ['process'],
	describe: (p: StageProps) => describeUnits(unitsPic(p)),
	demo: async () => {
		const [{ record }, { fit }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const input = { ...START };
		const tape = record(fit, input, { steps: 4, every: 1, dropout: 0.5 });
		const rt = fakeRuntime({ slug: 'regularization', input, tapes: { drop: tape } });
		return { rt, source: 'drop', at: { index: 2, frame: tape.frame(2), state: tape.state(2), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 200
} satisfies ViewMeta;

export default meta;
