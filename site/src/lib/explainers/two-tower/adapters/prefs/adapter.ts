// two-tower: local input adapter 'prefs' (catalog InputKind 'ratings'). The learner's raw input is a Prefs (world.ts):
// the genres they like, up to 12 movies they liked, and a new movie of their own (genre, year, length). Two tensors:
//   user [15]  the user tower's input: the liked genres (6 switches), the genre shares of the picked movies (6), their
//              mean year and mean length (scaled), and 1 when any movie was picked
//   item [8]   the item tower's input for the new movie: its genre as 6 switches, its year and its length (scaled)
import type { InputAdapter, Preprocessed } from '../../../../atlas/types/index.ts';
import { EDGES, GENRES, MOVIES, PRESETS, itemFeatures, toPrefs, userFeatures, type Prefs } from '../../world.ts';

const adapter: InputAdapter<Prefs> = {
	id: 'prefs',
	kind: 'ratings',
	preprocess(raw): Preprocessed {
		const p = toPrefs(raw);
		const user = { shape: [15], data: userFeatures(p.genres, p.picks) }, item = { shape: [8], data: itemFeatures(p.item) };
		const g = p.genres.map((k) => GENRES[k]);
		return {
			tensors: { user, item },
			steps: [
				{ id: 'user', label: { en: 'What the user tower sees: 15 numbers', zh: '用户塔看到的：15 个数' }, tensor: user },
				{ id: 'item', label: { en: 'What the item tower sees of your new movie: 8 numbers', zh: '物品塔看到的你的新电影：8 个数' }, tensor: item }
			],
			empty: false,
			notes: [{
				en: `You like ${g.length ? g.map((x) => x.en).join(', ') : 'no genre yet'} and picked ${p.picks.length} of ${MOVIES.length} movies. The towers read only these numbers, never the titles.`,
				zh: `你喜欢${g.length ? g.map((x) => x.zh).join('、') : '的类型还没有选'}，在 ${MOVIES.length} 部电影里选了 ${p.picks.length} 部。两座塔只读这些数，从不读片名。`
			}]
		};
	},
	presets: () => PRESETS,
	edgeFixtures: () => EDGES,
	layer0: { view: 'two-tower/tower', narration: { what: { en: 'Your choices become {n|int} numbers.', zh: '你的选择变成 {n|int} 个数。' } } }
};

export default adapter;
