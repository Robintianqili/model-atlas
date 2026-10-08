// View 'alphazero-ttt/play' (kind C): the learner's game against the search, from the last frame of the `game` process.
// Clicking a cell adds the learner's move to the input (raw.game), which re-plays the game with the search's replies.
import type { StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { describePlay, zeroWeights } from '../../picture.ts';

const meta = {
	key: 'alphazero-ttt/play',
	title: { en: 'Play against the search', zh: '和搜索下棋' },
	kinds: ['process'],
	describe: (p: StageProps) => describePlay(p),
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { game, withWeights }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../process.ts')]);
		const tape = record(game, withWeights({ you: [4, 8] }, zeroWeights()), { sims: 8, c: 1.5 }, { seed: 1 });
		const rt = fakeRuntime({ slug: 'alphazero-ttt', input: { moves: [], game: [4, 8] }, tapes: { game: tape } });
		return { rt, source: 'game', at: atFor(rt, 'game', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 280
} satisfies ViewMeta;

export default meta;
