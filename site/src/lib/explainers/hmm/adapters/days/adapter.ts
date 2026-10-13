// hmm: local input adapter 'days' (catalog InputKind 'tokens'). The learner's raw input is an HmmRaw (hmm.ts): the
// activities, one per day (0 walk, 1 shop, 2 clean; at most MAXT days), and the model's tables in percent. It becomes the
// activity ids 'obs' [T]; the processes read dataOf(raw).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { ACTS, A0, B0, I, MAXT, obsOf, START_OBS, type HmmRaw } from '../../hmm.ts';

const say = (obs: number[]) => I(obs.map((o) => ACTS[o].en).join(', '), obs.map((o) => ACTS[o].zh).join('、'));
const PRESETS: { id: string; label: ReturnType<typeof I>; raw: HmmRaw }[] = [
	{ id: 'start', label: I('Eight mixed days', '有晴有雨的八天'), raw: { obs: START_OBS } },
	{ id: 'wet', label: I('A wet spell in the middle', '中间连着几天下雨'), raw: { obs: [0, 0, 1, 2, 2, 1, 2, 0, 0] } },
	{ id: 'outings', label: I('Out almost every day', '几乎天天出门'), raw: { obs: [0, 1, 0, 0, 2, 0, 1, 0] } },
	{ id: 'sticky', label: I('Same days, weather that rarely changes', '同样的日子，天气很少变'), raw: { obs: START_OBS, A: [[95, 5], [10, 90]] } }
];

const adapter: InputAdapter<HmmRaw> = {
	id: 'days',
	kind: 'tokens',
	preprocess(raw): Preprocessed {
		const obs = obsOf(raw), n = obs.length;
		return {
			tensors: { obs: { shape: [n], data: Float32Array.from(obs) } },
			steps: [{ id: 'obs', label: I(`Her activities: ${say(obs).en || 'none'}`, `她的活动：${say(obs).zh || '没有'}`) }],
			empty: n === 0,
			notes: n ? [] : [I('There are no days yet: add an activity.', '还没有任何一天：加一个活动吧。')]
		};
	},
	presets: () => PRESETS,
	edgeFixtures: () => [
		{ id: 'empty', label: I('No days', '没有任何一天'), raw: { obs: [] } },
		{ id: 'one', label: I('A single day: one clean', '只有一天：打扫'), raw: { obs: [2] } },
		{ id: 'long', label: I(`The longest: ${MAXT} days of cleaning`, `最长的：连续 ${MAXT} 天打扫`), raw: { obs: Array(MAXT).fill(2) } },
		{ id: 'edge', label: I('Extreme tables: walks are almost certain on sunny days', '极端的表格：晴天几乎一定去散步'), raw: { obs: [1, 0, 2, 1], pi: [95, 5], A: [[5, 95], [95, 5]], B: [[90, 5, 5], B0[1]] } },
		{ id: 'bad', label: I('Broken tables (rows that do not sum to 100) fall back to the start', '无效的表格（每行加起来不是 100）退回起始值'), raw: { obs: [0, 2], pi: [70, 70], A: [A0[0], [50, 60]], B: [[1, 2, 3], B0[1]] } }
	],
	layer0: { view: 'hmm/model', narration: { what: { en: 'Your input is {n|int} days of activities.', zh: '你的输入是 {n|int} 天的活动。' } } }
};

export default adapter;
