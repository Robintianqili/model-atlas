// matrix-factorization: local input adapter 'ratings' (catalog InputKind 'ratings'). The learner's raw input is a
// RatingsRaw (world.ts): a whole number from −2 to 2, or null, for each of the 20 movies. It becomes your row of the
// rating matrix: 'ratings' [20] (0 where not rated) and 'rated' [20] (1 where rated). The trainer reads the raw input.
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { MOVIES, PRESETS, toRaw, type RatingsRaw } from '../../world.ts';

const adapter: InputAdapter<RatingsRaw> = {
	id: 'ratings',
	kind: 'ratings',
	preprocess(raw): Preprocessed {
		const r = toRaw(raw).ratings;
		const n = r.filter((x) => x !== null).length;
		return {
			tensors: { ratings: { shape: [MOVIES.length], data: Float32Array.from(r, (x) => x ?? 0) }, rated: { shape: [MOVIES.length], data: Float32Array.from(r, (x) => (x === null ? 0 : 1)) } },
			steps: [],
			empty: false,
			notes: [{ en: `Your row of the rating matrix: ${n} of ${MOVIES.length} movies rated. Training reads only these numbers, never the titles or kinds.`, zh: `你在评分矩阵中的那一行：${MOVIES.length} 部电影中评了 ${n} 部。训练只读这些数，从不读片名或类型。` }]
		};
	},
	presets: () => PRESETS,
	edgeFixtures: () => [
		{ id: 'none', label: { en: 'nothing rated', zh: '一部也没评' }, raw: { ratings: MOVIES.map(() => null) } },
		{ id: 'one', label: { en: 'one movie rated +2', zh: '只给一部电影打了 +2' }, raw: { ratings: MOVIES.map((_, j) => (j === 5 ? 2 : null)) } },
		{ id: 'all', label: { en: 'all 20 movies rated', zh: '20 部全都评了' }, raw: { ratings: MOVIES.map((m) => [2, 1, -1, -2][m.kind]) } },
		{ id: 'flat', label: { en: 'eight movies, all so-so (0)', zh: '八部电影，全是“一般”（0）' }, raw: { ratings: MOVIES.map((_, j) => (j < 8 ? 0 : null)) } }
	],
	layer0: { view: 'matrix-factorization/matrix', narration: { what: { en: 'Your ratings are one row of {n|int} numbers.', zh: '你的评分是一行 {n|int} 个数。' } } }
};

export default adapter;
