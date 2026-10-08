// View 'gan-2d/game' (kind B): the kit's loss-curve with D's and G's losses of a train source (smoothed), the values
// they settle around when D says ½ everywhere, and a link to the 2-D flow matching page for the comparison.
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtInt } from '../../../../atlas/core/format.ts';
import { mean } from '../../net.ts';

export const LABELS = { dLoss: { en: 'D’s loss', zh: 'D 的损失' }, gLoss: { en: 'G’s loss', zh: 'G 的损失' } };
export const curveProps = { metrics: ['dLoss', 'gLoss'], labels: LABELS, smooth: 50 };

/** the mean of D's and G's losses over the last n recorded steps */
export function lateLosses(p: Pick<StageProps, 'rt' | 'source'>, n = 200): { steps: number; d: number; g: number } {
	const tape = p.rt.sources[p.source]?.tape;
	if (!tape?.length) return { steps: 0, d: NaN, g: NaN };
	const d = tape.metric('dLoss');
	const g = tape.metric('gLoss');
	const pick = (a: Float64Array) => mean(Array.from(a).filter(Number.isFinite).slice(-n));
	return { steps: tape.frame(tape.length - 1).step ?? 0, d: pick(d), g: pick(g) };
}

const meta = {
	key: 'gan-2d/game',
	title: { en: 'The two losses of the game', zh: '博弈的两个损失' },
	kinds: ['train'],
	describe(p: StageProps): I18n {
		const l = lateLosses(p);
		if (!l.steps) return { en: 'No training steps yet.', zh: '还没有训练步。' };
		return {
			en: `D's loss and G's loss over ${fmtInt(l.steps)} steps. Over the last 200 steps they average ${fmt3(l.d)} and ${fmt3(l.g)}; when D says ½ everywhere they are 2 ln 2 ≈ 1.39 and ln 2 ≈ 0.693.`,
			zh: `D 的损失和 G 的损失，共 ${fmtInt(l.steps)} 步。最后 200 步的平均分别是 ${fmt3(l.d)} 和 ${fmt3(l.g)}；当 D 处处说 ½ 时，它们分别是 2 ln 2 ≈ 1.39 和 ln 2 ≈ 0.693。`
		};
	},
	async demo() {
		const [{ default: arena }] = await Promise.all([import('../arena/meta.ts')]);
		return { ...(await arena.demo()), props: {} };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
