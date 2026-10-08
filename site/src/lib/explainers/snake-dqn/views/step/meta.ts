// View 'snake-dqn/step' (kinds A, C): the learner's board and, by props.mode, the three input planes ('input'), the
// online network's Q-values ('q', from the trace), or one phase of the `dqn` process ('act', 'reward', 'target', 'loss').
import type { StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { describeStep } from '../../picture.ts';

/** All-zero weights (every Q-value 0): the illustrative demos of the views. */
export const zeroW = () => [16 * 27, 16, 32 * 144, 32, 64 * 2592, 64, 192, 3].map((n) => new Float32Array(n));
const meta = {
	key: 'snake-dqn/step',
	title: { en: 'The board, with one step of DQN', zh: '棋盘，以及 DQN 的一步' },
	kinds: ['trace', 'process'],
	describe: (p: StageProps) => describeStep(p),
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { dqn, withWeights }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../process.ts')]);
		const tape = record(dqn, withWeights({ seed: 3, moves: [] }, zeroW(), zeroW()), { eps: 0.1, gamma: 0.95 }, { seed: 1 });
		const rt = fakeRuntime({ slug: 'snake-dqn', input: { seed: 3, moves: [] }, tapes: { dqn: tape } });
		return { rt, source: 'dqn', at: atFor(rt, 'dqn', 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { mode: 'act' } };
	},
	minHeight: 320
} satisfies ViewMeta;

export default meta;
