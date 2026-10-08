// View 'snake-dqn/watch' (kind C): the trained agent's game (process `agent`) beside the learner's own game on the same
// food seed (raw.game, played with the buttons).
import type { StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { describeWatch } from '../../picture.ts';
import { zeroW } from '../step/meta.ts';

const meta = {
	key: 'snake-dqn/watch',
	title: { en: 'Watch the agent, then play yourself', zh: '看智能体下，再自己下' },
	kinds: ['process'],
	describe: (p: StageProps) => describeWatch(p),
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { agent, withWeights }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../process.ts')]);
		const tape = record(agent, withWeights({ seed: 3 }, zeroW()), { cap: 1500 }, { seed: 1 });
		const rt = fakeRuntime({ slug: 'snake-dqn', input: { seed: 3, moves: [], game: [0, 0, 1] }, tapes: { agent: tape } });
		return { rt, source: 'agent', at: atFor(rt, 'agent', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: {} };
	},
	minHeight: 300
} satisfies ViewMeta;

export default meta;
