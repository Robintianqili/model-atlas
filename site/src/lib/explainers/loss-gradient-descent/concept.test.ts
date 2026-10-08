// loss-gradient-descent: the concept page's own lint (atlas/ui/concept/lint.ts): the stage panels of concept.ts (toy,
// real, compare) run like lesson steps on every fixture in en and zh, and the hook, pitfalls and further reading are
// checked. `atlas check` runs it with the package's other tests.
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { assertConceptClean, lintConceptPackage } from '../../atlas/ui/concept/node.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));

test('concept page: panels, hook, pitfalls and links lint clean on every fixture in en and zh', async () => {
	assertConceptClean(await lintConceptPackage('loss-gradient-descent', SITE));
});
