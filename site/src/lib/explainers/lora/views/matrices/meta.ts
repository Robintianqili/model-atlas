// View 'lora/matrices': the real matrices of a frame of the lora source as heat maps (the one diverging scale: blue
// below 0, orange above). Props:
//   mode?: 'merge' (default)  B (16 × r) times A (r × 16), times α/r, is the change ΔW; W0 (frozen) + ΔW = W
//          'target'           the learned change ΔW next to the change the new task needs, ΔW*, and what is still missing
//                             (ΔW* − ΔW), on one color scale
import type { I18n } from '../../../../atlas/types/core.ts';
import type { StageProps, ViewMeta } from '../../../../atlas/types/view.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { D, WORLD } from '../../data.ts';

type T = { shape: number[]; data: Float64Array };
export interface MatPic {
	r: number;
	scale: number;
	step: number;
	loss: number;
	A: T;
	B: T;
	dW: Float64Array;
	W: Float64Array;
	missing: Float64Array;
	normB: number;
	maxDW: number;
}

/** What the view draws (pure; also the text form). Null when the frame has no matrices. */
export function matOf(p: Pick<StageProps, 'at'>): MatPic | null {
	const s = (p.at?.state ?? {}) as { A?: T; B?: T; dW?: T };
	const m = p.at?.frame?.metrics ?? {};
	if (!s.A || !s.B || !s.dW) return null;
	const dW = s.dW.data;
	const W = WORLD.W0.map((v, i) => v + dW[i]);
	const missing = WORLD.dWstar.map((v, i) => v - dW[i]);
	return { r: s.A.shape[0], scale: m.scale, step: m.step, loss: m.loss, A: s.A, B: s.B, dW, W, missing, normB: m.normB, maxDW: Math.max(...Array.from(dW, Math.abs)) };
}

export function describeMat(pic: MatPic | null, mode: unknown): I18n {
	if (!pic) return { en: 'No matrices to draw yet.', zh: '还没有可画的矩阵。' };
	if (mode === 'target') {
		const miss = Math.max(...Array.from(pic.missing, Math.abs));
		return {
			en: `After ${pic.step} updates: the learned change ΔW (${D} × ${D}), the change the new task needs ΔW*, and what is still missing, ΔW* − ΔW, whose largest entry is ${fmt3(miss)}. The loss is ${fmt3(pic.loss)}.`,
			zh: `更新 ${pic.step} 次之后：学到的改变量 ΔW（${D} × ${D}）、新任务需要的改变量 ΔW*，以及还缺的部分 ΔW* − ΔW，其中最大的一项是 ${fmt3(miss)}。损失是 ${fmt3(pic.loss)}。`
		};
	}
	return {
		en: `After ${pic.step} updates: B (${D} × ${pic.r}, size ${fmt3(pic.normB)}) times A (${pic.r} × ${D}), times α/r = ${fmt3(pic.scale)}, gives the change ΔW, whose largest entry is ${fmt3(pic.maxDW)}. The frozen W0 plus ΔW is the layer W. The loss is ${fmt3(pic.loss)}.`,
		zh: `更新 ${pic.step} 次之后：B（${D} × ${pic.r}，大小 ${fmt3(pic.normB)}）乘以 A（${pic.r} × ${D}），再乘 α/r = ${fmt3(pic.scale)}，得到改变量 ΔW，其中最大的一项是 ${fmt3(pic.maxDW)}。冻结的 W0 加上 ΔW 就是层 W。损失是 ${fmt3(pic.loss)}。`
	};
}

const meta = {
	key: 'lora/matrices',
	title: { en: 'The adapter matrices', zh: '适配器矩阵' },
	kinds: ['process'],
	describe: (p: StageProps) => describeMat(matOf(p), p.props?.mode),
	demo: async () => {
		const [{ record }, { lora }, { fakeRuntime }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../process.ts'), import('../../../../atlas/runtime/fake.ts')]);
		const input = { r: 2 };
		const tape = record(lora, input, { steps: 3 }, { seed: 1 });
		const rt = fakeRuntime({ slug: 'lora', input, tapes: { fit: tape } });
		return { rt, source: 'fit', at: { index: 3, frame: tape.frame(3), state: tape.state(3), tape }, focus: [], sel: {}, provenance: 'illustrative', autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
