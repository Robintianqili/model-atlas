// View 'snake-dqn/replay' (kind C): the `replay` process. The buffer's slots (each a stored move), the slot just written,
// the slots drawn for a batch, and the batch's targets and losses.
import type { StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { describeReplay } from '../../picture.ts';
import { zeroW } from '../step/meta.ts';

const meta = {
	key: 'snake-dqn/replay',
	title: { en: 'A replay buffer of stored moves', zh: '存放旧步的经验回放缓冲区' },
	kinds: ['process'],
	describe: (p: StageProps) => describeReplay(p),
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { replay, withWeights }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../process.ts')]);
		const tape = record(replay, withWeights({ seed: 3 }, zeroW(), zeroW()), { eps: 0.1, gamma: 0.95, fill: 20, cap: 12, batch: 4 }, { seed: 1 });
		const rt = fakeRuntime({ slug: 'snake-dqn', input: { seed: 3, moves: [] }, tapes: { replay: tape } });
		return { rt, source: 'replay', at: atFor(rt, 'replay', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
