// View 'tiny-unet-seg/unet': the U drawn from the live trace. Each level shows one of its maps (the strongest channel)
// with its shape and, on the down path, the window of the picture one cell sees; arrows for pooling, upsampling and the
// two skip connections; the class map at the end. props.part ('down' | 'bottleneck' | 'up' | 'skip' | 'all') lights
// that part and opens a level's maps as tiles below (props.open, else the part's level; a tap on a level opens it).
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { T, demoRt } from '../../seg.ts';

type P = Omit<StageProps, 'ctx'>;
export const LEVELS = [
	{ id: 'enc1', part: 'down', en: 'down 1', zh: '下行 1' },
	{ id: 'enc2', part: 'down', en: 'down 2', zh: '下行 2' },
	{ id: 'bott', part: 'bottleneck', en: 'bottleneck', zh: '瓶颈' },
	{ id: 'dec2', part: 'up', en: 'up 2', zh: '上行 2' },
	{ id: 'dec1', part: 'up', en: 'up 1', zh: '上行 1' }
];
const OPEN: Record<string, string> = { down: 'enc2', bottleneck: 'bott', up: 'up1', skip: 'cat1', all: 'dec1' };
/** names of the layers a level's tiles can show */
export const NAMES: Record<string, { en: string; zh: string }> = {
	...Object.fromEntries(LEVELS.map((l) => [l.id, l])), up2: { en: 'the bottleneck upsampled', zh: '上采样后的瓶颈' }, cat2: { en: 'skip 2 + upsampled', zh: '跳跃连接 2 + 上采样' },
	up1: { en: 'up 2 upsampled', zh: '上采样后的上行 2' }, cat1: { en: 'skip 1 + upsampled', zh: '跳跃连接 1 + 上采样' }
};
export const openOf = (p: P) => String(p.props.open ?? OPEN[String(p.props.part ?? 'all')] ?? 'dec1');
const shp = (s: number[] | undefined) => (s ? `${s[0]} × ${s[1]}×${s[2]}` : '?');

export function describeU(p: P): I18n {
	const sh = (id: string) => shp(T(p.rt, id)?.shape);
	if (!T(p.rt, 'enc1')) return { en: 'The U-Net: nothing to show yet.', zh: 'U-Net：还没有内容。' };
	const o = openOf(p);
	return {
		en: `The U-Net as a U. Down path: ${sh('enc1')} maps, pooled to ${sh('enc2')}, pooled to the bottleneck's ${sh('bott')}. Up path: upsampled and joined with skip 2 to ${sh('dec2')}, upsampled and joined with skip 1 to ${sh('dec1')}, then a class for each pixel. Opened below: all ${sh(o)} maps of ${NAMES[o]?.en ?? o}.`,
		zh: `画成 U 形的 U-Net。下行路径：${sh('enc1')} 张图，池化成 ${sh('enc2')}，再池化成瓶颈的 ${sh('bott')}。上行路径：上采样后和跳跃连接 2 拼接，得到 ${sh('dec2')}；再上采样并和跳跃连接 1 拼接，得到 ${sh('dec1')}；最后给每个像素一个类别。下方展开：${NAMES[o]?.zh ?? o} 的全部 ${sh(o)} 张图。`
	};
}

const meta = {
	key: 'tiny-unet-seg/unet',
	title: { en: 'The U: down, across and up', zh: 'U 形：向下、横跨、向上' },
	kinds: ['trace'],
	describe: describeU,
	async demo() {
		return { rt: demoRt(), source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { part: 'skip', open: 'up1' } };
	},
	minHeight: 420
} satisfies ViewMeta;

export default meta;
