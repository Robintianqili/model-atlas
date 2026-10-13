// tiny-encdec: tests of the local op 'greedy' (ops/greedy.ts): its shape, and that its float32 decoding writes what
// the float64 arithmetic of tf.ts writes on the start input (atlas parity checks it against PyTorch on every case).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { fsFetch } from '../../../atlas/core/io.ts';
import { createRuntime } from '../../../atlas/runtime/runtime.ts';
import { loadPackage } from '../../../../../scripts/atlas/load.ts';
import { greedy } from './greedy.ts';
import { run64, weightsOf } from '../tf.ts';

const SITE = fileURLToPath(new URL('../../../../../', import.meta.url));

test('greedy: [T, D] memory -> [steps] decoder input; a wrong rank is refused', () => {
	assert.deepEqual(greedy.inferShape([[13, 32]], { steps: 11, bos: 0, nHead: 2 }, {}), [11]);
	assert.throws(() => greedy.inferShape([[32]], { steps: 11, bos: 0, nHead: 2 }, {}));
});

test('greedy: the op (float32) feeds back the symbols the float64 run writes, starting with <s>', async () => {
	const pkg = (await loadPackage('tiny-encdec', SITE)).pkg;
	const rt = await createRuntime(pkg, { fetch: fsFetch, base: join(SITE, 'static'), input: 'March 5, 2021' });
	await rt.ensure('fwd', 'last');
	const op = Array.from(rt.sources.fwd.trace!.t.greedy.data);
	const r = run64(weightsOf(rt.model!), Array.from(rt.pre!.tensors.input.data));
	assert.deepEqual(op, r.ids);
	assert.equal(op[0], 0);
});
