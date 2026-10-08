// The Node package loader and `atlas run` against the kit modules of other packages (W1-B1 trainingProcess, W1-C
// record/saveTape, W1-A CORE_OPS). Those modules are not in this lane yet, so a throwaway repo gets small stand-ins with
// the §4 signatures; integration 1a re-runs `atlas run` on the real ones.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { loadPackage, listPackages } from '../load.ts';
import { pathToFileURL } from 'node:url';
import { SITE, cli, tempRepo } from './helpers.ts';

const TYPES = "import type { ProcessDef, TrainerDef, Frame } from '../types/index.ts';";

function standIns(site: string) {
	const atlas = join(site, 'src', 'lib', 'atlas');
	mkdirSync(join(atlas, 'train'), { recursive: true });
	mkdirSync(join(atlas, 'process'), { recursive: true });
	mkdirSync(join(atlas, 'ops'), { recursive: true });
	// W1-B1 stand-in: a 1-parameter "trainer" (w -> w - lr * 2(w - target)), fine detail adds the batch
	writeFileSync(
		join(atlas, 'train', 'trainer.ts'),
		`${TYPES}
export function trainingProcess(def: TrainerDef<any, any, any, any>): ProcessDef<any, any, any> {
	return {
		id: 'train:' + def.id, kind: 'training', title: def.title, defaults: def.defaults,
		phases: { init: { label: { en: 'init' }, kind: 'other' }, snapshot: { label: { en: 'snapshot' }, kind: 'update' }, forward: { label: { en: 'forward' }, kind: 'forward' } },
		schema: { params: { kind: 'json', label: { en: 'params' } }, batch: { kind: 'json', label: { en: 'batch' } }, step: { kind: 'int', label: { en: 'step' } } },
		*run(data, ctx) {
			let w = def.model(ctx.rng.fork('init'), ctx.opt) as number;
			yield { phase: 'init', step: 0, state: { params: { w: { shape: [1], data: Float64Array.of(w) } } } } as Frame;
			for (let s = 0; s < ctx.opt.steps; s++) {
				const b = def.batch(data, s, ctx.rng.fork('batch'), ctx.opt) as number;
				if (ctx.detail === 'fine') yield { phase: 'forward', step: s, state: { batch: { x: { shape: [1], data: Float64Array.of(b) } } } } as Frame;
				const loss = (w - b) ** 2;
				w = w - ctx.opt.lr * 2 * (w - b);
				yield { phase: 'snapshot', step: s, state: { params: { w: { shape: [1], data: Float64Array.of(w) } }, step: s }, metrics: { loss } } as Frame;
			}
		}
	};
}
`
	);
	// W1-C stand-in: record() marks its tapes so the test can see which recorder ran
	writeFileSync(
		join(atlas, 'process', 'record.ts'),
		`import type { ProcessDef, Tape, Frame, RecordOpts } from '../types/index.ts';
import { recordFallback } from '${pathToFileURL(join(SITE, 'scripts', 'atlas', 'lib', 'fallback.ts')).href}';
export function record(p: ProcessDef<any, any, any>, input: unknown, opt: any, ro: RecordOpts = {}): Tape {
	const t = recordFallback(p, input, opt, ro);
	(t as any).notes.push('recorded by the W1-C stand-in');
	return t;
}
`
	);
	writeFileSync(join(atlas, 'ops', 'index.ts'), "export const CORE_OPS = [{ op: 'stand_in_op' }];\n");
}

function trainPackage(site: string) {
	const dir = join(site, 'src', 'lib', 'explainers', 'toy');
	mkdirSync(join(dir, 'adapters', 'nums'), { recursive: true });
	writeFileSync(
		join(dir, 'explainer.ts'),
		`export default { slug: 'toy', type: 'model', kinds: ['train'], quality: 'draft', devOnly: true,
	input: { adapter: 'nums', start: [1, 2, 3] },
	sources: { fit: { kind: 'train', trainer: 'fit', detail: 'coarse' }, fwd: { kind: 'trace' } },
	page: { goals: [], prequestion: { id: 'q', question: { en: 'q' }, options: [], answer: 0, explain: { en: 'e' } } }, credits: [] };
`
	);
	writeFileSync(
		join(dir, 'trainer.ts'),
		`export const fitter = { id: 'fit', title: { en: 'Fit' }, defaults: { seed: 5, lr: 0.25, batch: 1, steps: 4, snapshotEvery: 1, optimizer: 'sgd' },
	model: (rng: any) => rng.next(), batch: (data: number[], step: number) => data[step % data.length], loss: () => ({ loss: null }) };
`
	);
	writeFileSync(
		join(dir, 'adapters', 'nums', 'adapter.ts'),
		`export default { id: 'nums', kind: 'series', preprocess: (raw: number[]) => ({ tensors: {}, steps: [], empty: raw.length === 0 }),
	edgeFixtures: () => [{ id: 'empty', label: { en: 'empty' }, raw: [] }], layer0: { view: 'x', narration: { what: { en: 'w' } } } };
`
	);
}

test('loader: trainer exports become processes train:<id>, CORE_OPS default, local adapters, notes for renamed exports', async () => {
	const t = tempRepo();
	try {
		standIns(t.site);
		trainPackage(t.site);
		assert.deepEqual(listPackages(t.site), ['toy']);
		const l = await loadPackage('toy', t.site);
		assert.deepEqual(Object.keys(l.pkg.processes), ['train:fit']);
		assert.deepEqual(l.pkg.ops, [{ op: 'stand_in_op' }]);
		assert.deepEqual(Object.keys(l.pkg.adapters), ['nums']);
		assert.equal(l.pkg.lesson, null);
		assert.ok(l.notes.some((n) => /export 'fitter' has id 'fit'/.test(n)));
	} finally {
		t.cleanup();
	}
});

test('run on a train source: W1-C record() is used when present; --save writes the tape and train_fixture.json', () => {
	const t = tempRepo();
	const env = { ATLAS_SITE: t.site, ATLAS_ROOT: t.repo };
	try {
		standIns(t.site);
		trainPackage(t.site);
		const r = cli(['run', 'toy', 'fit'], env);
		assert.equal(r.status, 0, r.stderr);
		assert.match(r.stdout, /using {3}atlas\/process\/record\.ts/);
		assert.match(r.stdout, /recorded by the W1-C stand-in/);
		assert.match(r.stdout, /metric loss: first/);
		const s = cli(['run', 'toy', 'fit', '--save'], env);
		assert.equal(s.status, 0, s.stderr);
		const fx = JSON.parse(readFileSync(join(t.site, '.atlas', 'toy', 'train_fixture.json'), 'utf8'));
		assert.equal(fx.format, 'atlas-train-fixture/1');
		assert.equal(fx.detail, 'fine'); // --save forces fine detail so batches are recorded
		assert.deepEqual(fx.batches.map((b: { batch: { x: { data: number[] } } }) => b.batch.x.data[0]), [1, 2, 3, 1]);
		assert.equal(fx.losses.length, 4);
		assert.equal(fx.init.w.data.length, 1);
		assert.ok(existsSync(join(t.site, '.atlas', 'toy', 'fit.tape.json')));
		// a fixture from the adapter's edge fixtures
		const e = cli(['run', 'toy', 'fit', '--fixture', 'empty', '--json'], env);
		assert.equal(e.status, 0, e.stderr);
		// a trace source is refused with a pointer to parity
		const tr = cli(['run', 'toy', 'fwd'], env);
		assert.equal(tr.status, 1);
		assert.match(tr.stderr, /^AK301 {2}atlas run prints process and train sources/m);
		const missing = cli(['run', 'nope', 'x'], env);
		assert.match(missing.stderr, /^AK301 {2}no package 'nope'/m);
	} finally {
		t.cleanup();
	}
});

test('run: a package file that only Vite can load is reported as AK503', () => {
	const t = tempRepo();
	try {
		const dir = join(t.site, 'src', 'lib', 'explainers', 'vite-only');
		mkdirSync(dir, { recursive: true });
		writeFileSync(join(dir, 'explainer.ts'), "import { base } from '$app/paths';\nexport default { slug: 'vite-only', base };\n");
		const r = cli(['run', 'vite-only', 'x'], { ATLAS_SITE: t.site, ATLAS_ROOT: t.repo });
		assert.equal(r.status, 1);
		assert.match(r.stderr, /^AK503 {2}explainer\.ts does not load in Node/m);
	} finally {
		t.cleanup();
	}
});
