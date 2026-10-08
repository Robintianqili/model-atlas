// View 'tiny-unet-seg/mask': the learner's picture with a class map over it (an opacity slider; class colours as everywhere
// on the site). props.mode: 'truth' (the classes the learner gave), 'pred' (the U-Net's answer; one pixel's 4 shares;
// props.rf draws that pixel's windows along the three paths of the U), 'compare' (without skip connections | U-Net,
// border pixels right), 'iou' (the answer, wrong pixels marked, IoU per class), 'limits' (the answer, and the measured
// mean IoU on the test scenes and on three kinds of scene unlike the training set: LIMITS, from facts.json).
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmtPct } from '../../../../atlas/core/format.ts';
import { CLASSES } from '../../scenes.ts';
import { SURE, demoRt, scoreOf, unsure } from '../../seg.ts';

type P = Omit<StageProps, 'ctx'>;
export const modeOf = (p: P) => (['truth', 'compare', 'iou', 'limits'].includes(p.props.mode as string) ? (p.props.mode as string) : 'pred');
export const LIMITS = [
	{ fact: 'testMiou', en: 'like the training set', zh: '和训练集相似' },
	{ fact: 'bigMiou', en: 'bigger shapes (size 9 to 13)', zh: '更大的形状（大小 9 到 13）' },
	{ fact: 'hollowMiou', en: 'outlines only', zh: '只有轮廓' },
	{ fact: 'darkMiou', en: 'dark shapes, light background', zh: '亮背景上的暗形状' }
];
const list = (c: number[], zh = false) => c.map((n, k) => (n ? `${n} ${zh ? CLASSES[k].zh : CLASSES[k].en}` : '')).filter(Boolean).join(zh ? '、' : ', ');
const counts = (a: ArrayLike<number>) => [0, 1, 2, 3].map((k) => Array.prototype.filter.call(a, (v: number) => v === k).length);

export function describeMask(p: P): I18n {
	const s = scoreOf(p.rt), mode = modeOf(p);
	if (!s) return { en: 'Your picture and its class map: nothing to show yet.', zh: '你的图和它的类别图：还没有内容。' };
	if (mode === 'truth') return { en: `Your picture with the class you gave each pixel tinted: ${list(counts(s.truth))} pixels.`, zh: `你的图，每个像素按你给的类别着色：${list(counts(s.truth), true)} 个像素。` };
	if (mode === 'limits') {
		const v = LIMITS.map((l) => fmtPct(p.rt.num(`fact:${l.fact}`)));
		return { en: `Mean IoU measured on scenes the model never saw: ${LIMITS.map((l, i) => `${l.en} ${v[i]}`).join(', ')}.`, zh: `在模型从没见过的场景上测得的平均 IoU：${LIMITS.map((l, i) => `${l.zh} ${v[i]}`).join('、')}。` };
	}
	const n = scoreOf(p.rt, 'noskip');
	if (mode === 'compare' && n) return {
		en: `Two maps of class shares side by side. Without skip connections: ${unsure(p.rt, 'noskip')} pixels less than ${fmtPct(SURE, 0)} sure, ${n.wrong} wrong. The U-Net: ${unsure(p.rt)} less than ${fmtPct(SURE, 0)} sure, ${s.wrong} wrong.`,
		zh: `两张类别份额图并排。去掉跳跃连接：${unsure(p.rt, 'noskip')} 个像素把握不到 ${fmtPct(SURE, 0)}，错 ${n.wrong} 个。U-Net：${unsure(p.rt)} 个把握不到 ${fmtPct(SURE, 0)}，错 ${s.wrong} 个。`
	};
	const iou = s.per.map((c) => `${CLASSES[c.k].en} ${fmtPct(c.iou)} (${c.inter} of ${c.union})`).join(', ');
	const iouZh = s.per.map((c) => `${CLASSES[c.k].zh} ${fmtPct(c.iou)}（${c.union} 个中 ${c.inter} 个）`).join('、');
	return {
		en: `The U-Net's class map over your picture: ${list(counts(s.map.k))} pixels; ${s.wrong} of 1,024 differ from your classes. IoU per class: ${iou}.`,
		zh: `U-Net 的类别图叠在你的图上：${list(counts(s.map.k), true)} 个像素；1,024 个中有 ${s.wrong} 个和你给的类别不同。各类别的 IoU：${iouZh}。`
	};
}

const meta = {
	key: 'tiny-unet-seg/mask',
	title: { en: 'Your picture and its class map', zh: '你的图和它的类别图' },
	kinds: ['trace'],
	describe: describeMask,
	async demo() {
		return { rt: demoRt(), source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'iou' } };
	},
	minHeight: 340
} satisfies ViewMeta;

export default meta;
