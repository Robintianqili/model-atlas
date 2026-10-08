// layer.tensor (W1-D2): one whole tensor, displayed by its axes (design §5.1); also the thumbnail view (props.thumb).
import type { ViewMeta } from '../../../types/view.ts';
import { fmt3, type Props } from '../kit.parts/stage.ts';
import { demoStage, processStage, type DemoVariant } from '../kit.parts/demo.ts';
import { pickTensor } from './pick.ts';

const SHOWN: Record<string, { en: string; zh: string }> = {
	tiles: { en: 'as one small map per channel', zh: '每个通道画成一张小图' },
	tokens: { en: 'as one row per token', zh: '每个词元一行' },
	heads: { en: 'as one table per attention head', zh: '每个注意力头一张表' },
	dist: { en: 'as probability bars', zh: '画成概率条' },
	logits: { en: 'as a strip of scores', zh: '画成一条分数带' },
	points: { en: 'as points in the plane', zh: '画成平面上的点' },
	series: { en: 'as a line', zh: '画成一条线' },
	strip: { en: 'as a strip', zh: '画成一条带' },
	grid: { en: 'as a grid', zh: '画成网格' },
	tensor: { en: 'as a grid', zh: '画成网格' }
};

export function describeTensor(p: Props) {
	const x = pickTensor(p);
	if (!x) return { en: 'A tensor: nothing to show yet.', zh: '张量：还没有可显示的内容。' };
	let lo = Infinity;
	let hi = -Infinity;
	for (let i = 0; i < x.t.data.length; i++) {
		const v = x.t.data[i];
		if (Number.isFinite(v)) (lo = Math.min(lo, v)), (hi = Math.max(hi, v));
	}
	const how = SHOWN[x.display];
	return {
		en: `${x.label.en}: ${x.t.shape.join('×')} numbers (axes ${x.axes.join(', ')}), shown ${how.en}. They range from ${fmt3(lo)} to ${fmt3(hi)}.`,
		zh: `${x.label.zh ?? x.label.en}：${x.t.shape.join('×')} 个数（轴 ${x.axes.join('、')}），${how.zh}。数值范围从 ${fmt3(lo)} 到 ${fmt3(hi)}。`
	};
}

export async function variants(): Promise<DemoVariant[]> {
	return [
		{ label: { en: 'c,y,x feature maps', zh: 'c,y,x 特征图' }, stage: demoStage('cnn', 'relu1') },
		{ label: { en: 't,d token rows', zh: 't,d 词元行' }, stage: demoStage('gpt', 'ln') },
		{ label: { en: 'h,tq,tk attention weights', zh: 'h,tq,tk 注意力权重' }, stage: demoStage('gpt', 'weights') },
		{ label: { en: 'h,tq,tk scores with the mask', zh: '带掩码的 h,tq,tk 分数' }, stage: demoStage('gpt', 'masked') },
		{ label: { en: 'class probabilities', zh: '类别概率' }, stage: demoStage('cnn', 'probs') },
		{ label: { en: 'vocab logits', zh: '词表 logits' }, stage: demoStage('gpt', 'logits') },
		{ label: { en: 'a process field (Q estimates)', zh: '过程中的字段（Q 估计值）' }, stage: processStage(35, { field: 'Q' }) },
		{ label: { en: 'thumbnail', zh: '缩略图' }, stage: demoStage('cnn', 'conv2', undefined, { thumb: true }) }
	];
}

const meta = {
	key: 'layer.tensor',
	title: { en: 'A whole layer, drawn by its axes', zh: '按轴画出整个层' },
	kinds: ['trace', 'train', 'process'],
	accepts: (o) => Array.isArray(o.axes) && o.axes.length > 0,
	describe: describeTensor,
	demo: async () => demoStage('cnn', 'conv1'),
	minHeight: 320
} satisfies ViewMeta;

export default meta;
