// kalman-filter: local input adapter 'track' (catalog InputKind 'series'). The learner's raw input is a TrackRaw
// (track.ts): a preset or a drawn true track, the seed of the sensor noise, and the filter's q and r. It becomes the true
// positions 'truth' [N] and the sensor readings 'z' [N] (true + seeded noise); the processes read dataOf(raw).
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { dataOf, HI, I, N, PRESETS, type TrackRaw } from '../../track.ts';

const adapter: InputAdapter<TrackRaw> = {
	id: 'track',
	kind: 'series',
	preprocess(raw): Preprocessed {
		const d = dataOf(raw), n = d.truth.length;
		return {
			tensors: { truth: { shape: [n], data: Float64Array.from(d.truth) }, z: { shape: [n], data: Float64Array.from(d.z) } },
			steps: [],
			empty: n === 0,
			notes: n ? [] : [I('There is no track yet: pick one, or draw one.', '还没有轨迹：选一条，或者画一条。')]
		};
	},
	presets: () => PRESETS.map((p) => ({ id: p.id, label: p.label, raw: { preset: p.id, seed: 1 } })),
	edgeFixtures: () => [
		{ id: 'empty', label: I('No track', '没有轨迹'), raw: { y: [] } },
		{ id: 'still', label: I('One value: a cart that never moves', '只有一个值：一辆从不动的小车'), raw: { y: [10] } },
		{ id: 'jumpy', label: I('A drawn track that jumps 20 m every 5 s', '画出的轨迹，每 5 秒跳 20 米'), raw: { y: Array.from({ length: N }, (_, t) => (Math.floor(t / 5) % 2 ? 25 : 5)), seed: 3 } },
		{ id: 'top', label: I('At the top edge, extreme settings', '贴着上边缘，极端的设置'), raw: { y: [HI], seed: 7, q: 1, r: 0.1 } }
	],
	layer0: { view: 'kalman-filter/track', narration: { what: { en: 'The track is {n|int} positions, one per second.', zh: '轨迹是 {n|int} 个位置，每秒一个。' } } }
};

export default adapter;
