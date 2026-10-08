// View 'alphazero-ttt/tree' (kind C): the search tree at a frame of the `search` process (rows = moves ahead, width =
// share of the simulations, this simulation's path in the accent) and the frame's event: the PUCT table and formula of a
// choice (select), the network's answer on a new position (expand), or n and Q along the path (backup).
// props.limits: the tree's size next to the whole game tree from the learner's board.
import type { StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { describeTree, zeroWeights } from '../../picture.ts';

const meta = {
	key: 'alphazero-ttt/tree',
	title: { en: 'The search tree, one simulation at a time', zh: '搜索树，一次模拟一次模拟地长' },
	kinds: ['process'],
	describe: (p: StageProps) => describeTree(p),
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { search, withWeights }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../process.ts')]);
		const tape = record(search, withWeights({ moves: [2, 3, 7] }, zeroWeights()), { sims: 12, c: 1.5 }, { seed: 1 });
		const rt = fakeRuntime({ slug: 'alphazero-ttt', input: { moves: [2, 3, 7] }, tapes: { search: tape } });
		return { rt, source: 'search', at: atFor(rt, 'search', tape.find({ phase: 'select', step: 5 })), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 360
} satisfies ViewMeta;

export default meta;
