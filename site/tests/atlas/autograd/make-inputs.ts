// Write the TS-side twin inputs (W1-B1):  node tests/atlas/autograd/make-inputs.ts   (from site/ or a lane's site/)
// Then run the torch side:  python pipeline/atlas_fixtures/autograd/make.py   (with ATLAS_SITE set to that site folder)
import { writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { makePinnInputs, makeTrainInputs } from './inputs.ts';

const here = dirname(fileURLToPath(import.meta.url));
const write = (name: string, v: unknown) => {
	writeFileSync(join(here, name), JSON.stringify(v) + '\n', 'utf8');
	console.log(`wrote ${join(here, name)}`);
};
write('train_inputs.json', makeTrainInputs());
write('pinn_inputs.json', makePinnInputs());
