// View 'alphazero-ttt/board' (kinds A, C): the learner's position and, by props.mode, the network's input numbers and
// mask ('input'), a guess then the priors ('prior'), the value on a −1 … +1 scale with perfect play ('value'), or the
// search's visit counts next to the priors ('visits', 'pick'). Reads the trace of source fwd and the frame's tree state.
import type { StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { describeBoard } from '../../picture.ts';

const meta = {
	key: 'alphazero-ttt/board',
	title: { en: 'The board, with the network’s and the search’s numbers', zh: '棋盘，以及网络和搜索给出的数' },
	kinds: ['trace', 'process'],
	describe: (p: StageProps) => describeBoard(p),
	async demo() {
		const { fakeRuntime } = await import('../../../../atlas/runtime/fake.ts');
		const T = (d: number[]) => ({ shape: [d.length], data: Float32Array.from(d) });
		const prior = [0.3, 0.05, 0, 0, 0.3, 0.05, 0.2, 0, 0.1];
		const rt = fakeRuntime({ slug: 'alphazero-ttt', input: { moves: [2, 3, 7], guess: 4 }, trace: { order: ['prior', 'value'], t: { prior: T(prior), value: T([-0.3]) } } });
		return { rt, source: 'fwd', at: { index: 0 }, focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'prior' } };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
