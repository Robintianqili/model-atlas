// View 'matrix-factorization/map' (kinds B, C): every movie's two numbers as a dot on a plane (colored by its kind, which
// training never sees), the other people as gray dots and your two numbers as an arrow from the middle. Props:
//   people     draw the other people;  movie: a movie index to mark, or 'closest' (closest in direction among your
//   unrated);  dot: the dot product of your arrow and that movie, with its shadow on the arrow;  loss: the error
//   curve over the snapshots so far, with the error at the start and now (all known ratings, and yours);  kinds: how
//   many movies have a nearest movie of their own kind;  who: the arrow's label (default "you");  formula: the frame's
//   phase formula. The dot product's result is hidden while a predict gate makes the stage inert.
// On a 'nudge' frame it draws that step's person and movie, before (pale) and after the update.
import type { StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { describeMap } from '../../picture.ts';

const meta = {
	key: 'matrix-factorization/map',
	title: { en: 'Two numbers per movie and per person, as a map', zh: '把每部电影、每个人的两个数画成地图' },
	kinds: ['train', 'process'],
	describe: (p: StageProps) => describeMap(p),
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { trainingProcess }, { fit }, { PRESETS }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../../../atlas/train/trainer.ts'), import('../../trainer.ts'), import('../../world.ts')]);
		const tape = record(trainingProcess(fit), PRESETS[0].raw, { steps: 1000 }, { seed: 1 });
		const rt = fakeRuntime({ slug: 'matrix-factorization', input: PRESETS[0].raw, tapes: { train: tape } });
		return { rt, source: 'train', at: atFor(rt, 'train', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { people: true, movie: 'closest', dot: true, loss: true, kinds: true } };
	},
	minHeight: 420
} satisfies ViewMeta;

export default meta;
