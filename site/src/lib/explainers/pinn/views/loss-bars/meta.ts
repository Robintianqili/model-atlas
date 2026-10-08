// View 'pinn/loss-bars' (kind B): the parts of the loss at the step's frame: data, λ·physics and start, one bar each on a
// log scale (1e−6 to 1) with its number, and one stacked bar of their shares of the total. A run that does not train on
// the law (the plain network) still shows its physics loss, measured on its curve, marked "not in its loss".
// Numbers: the frame's metrics (data, physics, start: the trainer's loss parts) and the derived field 'curves'.
import type { I18n, Tensor } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3, fmtInt } from '../../../../atlas/core/format.ts';
import { N, meanSq } from '../../osc.ts';

export type Bar = { key: string; label: I18n; value: number; inLoss: boolean };

export function barsOf(p: Pick<StageProps, 'at'>): { step: number; total: number; bars: Bar[] } | null {
	const tape = p.at?.tape;
	let i = p.at?.index ?? -1;
	if (!tape || i < 0) return null;
	while (i > 0 && tape.frame(i).phase !== 'snapshot' && tape.frame(i).phase !== 'init') i--;
	const f = tape.frame(i);
	const o = (tape.opt ?? {}) as { lam?: number; start?: number; physics?: boolean };
	const m = f.metrics ?? {};
	let cv: Tensor | undefined;
	try {
		cv = (tape.state(i) as { curves?: Tensor }).curves;
	} catch {
		cv = undefined;
	}
	const phys = typeof m.physics === 'number' ? m.physics : cv && cv.data.length === 4 * N ? meanSq(cv, 3) : NaN;
	const lam = o.lam ?? 0;
	const bars: Bar[] = [
		{ key: 'data', label: { en: 'data', zh: '数据' }, value: m.data === 0 ? 0 : (m.data ?? NaN), inLoss: true },
		{ key: 'physics', label: o.physics === false ? { en: 'physics (not in its loss)', zh: '物理（不在它的损失里）' } : { en: `λ × physics, λ = ${lam}`, zh: `λ × 物理，λ = ${lam}` }, value: o.physics === false ? phys : lam * phys, inLoss: o.physics !== false },
		{ key: 'start', label: (o.start ?? 0) > 0 ? { en: 'start', zh: '起点' } : { en: 'start (not in its loss)', zh: '起点（不在它的损失里）' }, value: m.start ?? NaN, inLoss: (o.start ?? 0) > 0 }
	];
	const total = bars.filter((b) => b.inLoss && Number.isFinite(b.value)).reduce((s, b) => s + b.value, 0);
	return { step: f.step ?? 0, total, bars };
}

export function describeBars(b: ReturnType<typeof barsOf>): I18n {
	if (!b) return { en: 'No loss has been recorded yet.', zh: '还没有记录损失。' };
	const shown = b.bars.filter((x) => Number.isFinite(x.value));
	return {
		en: `The loss after ${fmtInt(b.step)} steps: ${shown.map((x) => `${x.label.en} ${fmt3(x.value)}`).join(', ')}; total ${fmt3(b.total)}.`,
		zh: `训练 ${fmtInt(b.step)} 步后的损失：${shown.map((x) => `${x.label.zh} ${fmt3(x.value)}`).join('，')}；合计 ${fmt3(b.total)}。`
	};
}

const meta = {
	key: 'pinn/loss-bars',
	title: { en: 'The parts of the loss', zh: '损失的各个部分' },
	kinds: ['train'],
	describe: (p: StageProps): I18n => describeBars(barsOf(p)),
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { seriesPresetRaw }, { trainingProcess }, { pinn }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../../../atlas/adapters/series/adapter.ts'), import('../../../../atlas/train/trainer.ts'), import('../../trainer.ts')]);
		const raw = seriesPresetRaw('oscillator');
		const tape = record(trainingProcess(pinn), raw, { steps: 40 }, { seed: 1 });
		const rt = fakeRuntime({ slug: 'pinn', input: raw, tapes: { pinn: tape } });
		return { rt, source: 'pinn', at: atFor(rt, 'pinn', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 220
} satisfies ViewMeta;

export default meta;
