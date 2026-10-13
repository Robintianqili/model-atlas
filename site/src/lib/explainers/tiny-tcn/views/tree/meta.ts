// tiny-tcn view 'tree': the receptive field of one output (time t* = split − 2) on the step's weights, as rows of nodes
// over 17 time steps (the 16 it reads and one older): the changes at the bottom, layers 1 to 4, the head's output at the top. The nodes and lines of the
// output's tree are drawn strong, the others faint. Pick a node (tree nodes are buttons) to see its arithmetic below.
// props.mode sets the node picked first and what is drawn:
//   causal    layer 1 at t* and the two changes it reads        reach     the whole tree (hidden while a predict is open)
//   tree      the whole tree, the output picked                 residual  layer 4 at t*: skip + ReLU(conv)
//   head      the output: the head's sum and the forecast
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fakeRuntime } from '../../../../atlas/runtime/fake.ts';
import { createRng } from '../../../../atlas/core/rng.ts';
import { seriesPresetRaw } from '../../../../atlas/adapters/series/adapter.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { dataOf, RF } from '../../data.ts';
import { TCN } from '../../model.ts';
import { paramsOf } from '../../trainer.ts';
import { headAt, paramsAt, tStar } from '../../look.ts';

export function describeTree(p: Omit<StageProps, 'ctx'>): I18n {
	const P = paramsAt(p.at);
	const d = dataOf(p.rt.input);
	const t = tStar(d);
	const head = { en: `The receptive field of the output at time ${t}: the ${RF} changes from time ${t - RF + 1} to ${t} feed 8 nodes of layer 1, 4 of layer 2, 2 of layer 3 and 1 of layer 4, which the head turns into one predicted change.`, zh: `时刻 ${t} 那个输出的感受野：从时刻 ${t - RF + 1} 到 ${t} 的 ${RF} 个变化量，连到第 1 层的 8 个节点、第 2 层的 4 个、第 3 层的 2 个和第 4 层的 1 个，再由输出头变成一个预测的变化量。` };
	if (!P) return head;
	const h = headAt(P, d);
	return { en: `${head.en} Forecast: ${fmt3(h.last)} + ${fmt3(h.change)} = ${fmt3(h.forecast)}; true value ${fmt3(h.truth)}.`, zh: `${head.zh}预测：${fmt3(h.last)} + ${fmt3(h.change)} = ${fmt3(h.forecast)}；真实值 ${fmt3(h.truth)}。` };
}

const meta = {
	key: 'tiny-tcn/tree',
	title: { en: 'The receptive-field tree', zh: '感受野之树' },
	kinds: ['train'],
	describe: describeTree,
	async demo() {
		const m = new TCN(createRng(1).fork('init'));
		return { rt: fakeRuntime({ slug: 'tiny-tcn', input: seriesPresetRaw('seasonal') }), source: 'init', at: { index: 0, state: { params: paramsOf(m) } }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'tree' } };
	},
	minHeight: 520
} satisfies ViewMeta;

export default meta;
