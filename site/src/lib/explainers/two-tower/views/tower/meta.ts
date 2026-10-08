// two-tower view 'tower' (kind trace): one tower's input, hidden layer and vector (View.svelte). towerText() builds its
// captions, also the screen-reader text.
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { fmt3 } from '../../../../atlas/core/format.ts';
import { MOVIES } from '../../world.ts';
import { newMovie, towerOf } from '../../recs.ts';
import { demoStage } from '../../demo.ts';

const I = (en: string, zh: string): I18n => ({ en, zh });
export function towerText(p: Omit<StageProps, 'ctx'>) {
	const user = p.props.tower === 'user', mv = typeof p.props.movie === 'number' ? MOVIES[p.props.movie] : null;
	const L = towerOf(p.rt, user ? 'user' : 'item', mv ? (p.props.movie as number) : undefined), n = L?.x.length ?? 0;
	const zeros = L ? Array.from(L.h).filter((v) => v === 0).length : 0, h = L?.h.length ?? 0, nm = newMovie(p.rt);
	return {
		head: user ? I(`User tower, on your choices: ${n} numbers`, `用户塔，读你的选择：${n} 个数`) : mv ? I(`Item tower, on ${mv.title.en}: ${n} numbers`, `物品塔，读《${mv.title.zh}》：${n} 个数`) : I(`Item tower, on your new movie: ${n} numbers`, `物品塔，读你的新电影：${n} 个数`),
		layer1: I(`${h} weighted sums of the ${n} numbers, each plus a bias, then ReLU: ${zeros} of the ${h} become 0`, `对这 ${n} 个数做 ${h} 个加权和，各加一个偏置，再过 ReLU：${h} 个里有 ${zeros} 个变成 0`),
		layer2: I(`${L?.y.length ?? 0} weighted sums of the ${h}, each plus a bias`, `对这 ${h} 个数做 ${L?.y.length ?? 0} 个加权和，各加一个偏置`),
		vec: user ? I('your vector', '你的向量') : mv ? I(`the vector of ${mv.title.en}`, `《${mv.title.zh}》的向量`) : I('your new movie’s vector', '你的新电影的向量'),
		score: nm ? I(`Its score for you: ${fmt3(nm.s)}, place ${nm.place} among the ${nm.of} movies (the ones you did not pick, and this one).`, `它对你的分数：${fmt3(nm.s)}，在 ${nm.of} 部电影（你没选过的那些，加上这一部）里排第 ${nm.place}。`) : I('', '')
	};
}

const meta = {
	key: 'two-tower/tower',
	title: { en: 'One tower at work', zh: '一座塔在工作' },
	kinds: ['trace'],
	describe: (p: StageProps): I18n => {
		const x = towerText(p), L = towerOf(p.rt, p.props.tower === 'user' ? 'user' : 'item', typeof p.props.movie === 'number' ? p.props.movie : undefined);
		if (!L) return I('A tower: nothing to show yet.', '一座塔：还没有可显示的内容。');
		const v = Array.from(L.y, (a) => fmt3(a)).join(', ');
		return I(`${x.head.en}; ${x.layer1.en}; ${x.layer2.en}: ${x.vec.en} = (${v}). ${x.score.en}`, `${x.head.zh}；${x.layer1.zh}；${x.layer2.zh}：${x.vec.zh} = (${v})。${x.score.zh}`);
	},
	demo: () => demoStage({ tower: 'item', movie: 28, table: true }),
	minHeight: 300
} satisfies ViewMeta;

export default meta;
