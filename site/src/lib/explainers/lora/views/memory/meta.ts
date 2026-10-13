// View 'lora/memory': training memory of GPT-2 small without activations (data.ts memoryOf, analytic), as stacked bars on
// one linear axis in MB (10⁶ bytes): full fine-tuning, LoRA and QLoRA at the frame's rank (metrics.r; the learner's rank
// before a frame exists). Each bar: the weights, the gradients and the optimizer state (Adam's two running averages).
// Assumptions in the footnote: 4-byte numbers, QLoRA's frozen weights at 4 bits without the per-block scales, LoRA on
// the six matrices of each block. No props.
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmtInt } from '../../../../atlas/core/format.ts';
import { MB, memoryOf, rankAt, type Memory } from '../../data.ts';

export const PARTS = ['weights', 'grads', 'optimizer'] as const;
export const PART_LABEL: Record<(typeof PARTS)[number], I18n> = {
	weights: { en: 'weights', zh: '权重' },
	grads: { en: 'gradients', zh: '梯度' },
	optimizer: { en: 'optimizer state', zh: '优化器状态' }
};
export const ROWS: { id: 'full' | 'lora' | 'qlora'; label: I18n }[] = [
	{ id: 'full', label: { en: 'Full fine-tuning', zh: '全量微调' } },
	{ id: 'lora', label: { en: 'LoRA', zh: 'LoRA' } },
	{ id: 'qlora', label: { en: 'QLoRA', zh: 'QLoRA' } }
];

const mb = (b: number) => fmtInt(Math.round(MB(b)));
const row = (m: Memory) => `${mb(m.total)} MB (${mb(m.weights)} + ${mb(m.grads)} + ${mb(m.optimizer)})`;

export function describeMemory(r: number): I18n {
	const m = memoryOf(r);
	return {
		en: `Training memory of GPT-2 small at rank ${r}, weights + gradients + optimizer state, without activations: full fine-tuning ${row(m.full)}, LoRA ${row(m.lora)}, QLoRA ${row(m.qlora)}.`,
		zh: `秩为 ${r} 时训练 GPT-2 small 所需的内存（权重 + 梯度 + 优化器状态，不含激活值）：全量微调 ${row(m.full)}，LoRA ${row(m.lora)}，QLoRA ${row(m.qlora)}。`
	};
}

const meta = {
	key: 'lora/memory',
	title: { en: 'Training memory: weights, gradients, optimizer state', zh: '训练内存：权重、梯度、优化器状态' },
	kinds: ['process'],
	describe: (p: StageProps) => describeMemory(rankAt(p)),
	demo: async () => {
		const { fakeRuntime } = await import('../../../../atlas/runtime/fake.ts');
		const rt = fakeRuntime({ slug: 'lora', input: { r: 8 }, tapes: {} });
		return { rt, source: 'fit', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 280
} satisfies ViewMeta;

export default meta;
