// View 'matrix-factorization/matrix' (kinds B, C): the rating matrix, one row per movie and one column per person (you
// first). Props: fill (the empty cells show the predictions of the step's weights), rank (your three best unrated
// movies are numbered), edit (rating buttons under the matrix; they change the input), focus (a movie row to mark).
import type { I18n, StageProps, ViewMeta } from '../../../../atlas/types/index.ts';
import { N_ITEMS, N_OTHERS, OTHERS, known } from '../../world.ts';
import { describeMatrix } from '../../picture.ts';

/** grid[i][u]: person u's rating of movie i, or null */
export const GRID: (number | null)[][] = Array.from({ length: N_ITEMS }, (_, i) => Array.from({ length: N_OTHERS }, (_, u) => OTHERS.find((x) => x.u === u && x.i === i)?.r ?? null));

const meta = {
	key: 'matrix-factorization/matrix',
	title: { en: 'The rating matrix', zh: '评分矩阵' },
	kinds: ['train', 'process'],
	describe: (p: StageProps): I18n => {
		const k = known(p.rt.input);
		return describeMatrix(p, { known: k.n, yours: k.yours });
	},
	async demo() {
		const [{ record }, { fakeRuntime }, { atFor }, { trainingProcess }, { fit }, { PRESETS }] = await Promise.all([import('../../../../atlas/process/record.ts'), import('../../../../atlas/runtime/fake.ts'), import('../../../../atlas/runtime/sources.ts'), import('../../../../atlas/train/trainer.ts'), import('../../trainer.ts'), import('../../world.ts')]);
		const tape = record(trainingProcess(fit), PRESETS[0].raw, { steps: 1000 }, { seed: 1 });
		const rt = fakeRuntime({ slug: 'matrix-factorization', input: PRESETS[0].raw, tapes: { train: tape } });
		return { rt, source: 'train', at: atFor(rt, 'train', tape.length - 1), focus: [], sel: {}, provenance: 'illustrative' as const, autoplay: false, onend: () => {}, props: { fill: true, rank: true } };
	},
	minHeight: 400
} satisfies ViewMeta;

export default meta;
