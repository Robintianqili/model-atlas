// pinn: the TypeScript side of export.py's checks (node pipeline/pinn/ts_dump.ts <dir>; export.py runs it). ATLAS_SITE (set
// by the CLI) is the site to load. Writes into <dir>:
//   deriv_dump.json   for three weight sets (the shared start, the trained PINN, the trained plain network): the weights
//                     and what the kit's autograd computes on the 200 physics times: u, u′, u″ (grad twice with
//                     createGraph), the residual, the physics loss and its gradient for every weight
//   variant_fixture.<source>.<sgd|adam>.json   atlas-train-fixture/1 files (as `atlas parity` writes them) for every
//                     train source of the page with its own options (λ = 0, 0.01, 100, c = 0, ...): `atlas parity`
//                     runs the twin once per trainer, so export.py checks each source's options itself
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

const site = process.env.ATLAS_SITE;
if (!site) throw new Error('run through `npm run atlas -- py pipeline/pinn/export.py` (ATLAS_SITE is not set)');
const out = process.argv[2];
mkdirSync(out, { recursive: true });
const lib = (p: string) => import(pathToFileURL(join(site, 'src/lib', p)).href);
const { record } = await lib('atlas/process/record.ts');
const { trainingProcess } = await lib('atlas/train/trainer.ts');
const { seriesPresetRaw } = await lib('atlas/adapters/series/adapter.ts');
const { grad, withGradMode } = await lib('atlas/autograd/var.ts');
const op = await lib('atlas/autograd/ops.ts');
const tr = await lib('explainers/pinn/trainer.ts');
const osc = await lib('explainers/pinn/osc.ts');
const manifest = (await lib('explainers/pinn/explainer.ts')).default;

type T = { shape: number[]; data: Float64Array };
const plain = (r: Record<string, T>) => Object.fromEntries(Object.entries(r).map(([k, t]) => [k, { shape: t.shape, data: Array.from(t.data) }]));
const raw = seriesPresetRaw('oscillator');
const run = (def: unknown, opt = {}, ro = {}) => record(trainingProcess(def as never), raw, opt, { seed: 1, ...ro });

// 1. derivatives and the physics-loss gradients
const pinnRun = run(tr.pinn);
const plainRun = run(tr.plain);
const sets: Record<string, Record<string, T>> = { start: pinnRun.state(0).params, pinn: pinnRun.state(pinnRun.length - 1).params, plain: plainRun.state(plainRun.length - 1).params };
const arr = (v: { data: Float64Array }) => Array.from(v.data);
const dump: Record<string, unknown> = { format: 'pinn-deriv-dump/1', T: Array.from(osc.T), c: osc.C, k: osc.K, sets: {} };
for (const [name, params] of Object.entries(sets)) {
	const m = tr.netAt(params, {});
	withGradMode(true, () => {
		const d = osc.derivs(m, osc.T, true);
		const r = osc.residual(d, osc.C, osc.K);
		const phys = op.mean(op.square(r));
		const named = Object.entries(m.named() as Record<string, never>);
		const gs = grad(phys, named.map(([, v]) => v)) as { data: Float64Array }[];
		(dump.sets as Record<string, unknown>)[name] = { params: plain(params), u: arr(d.u), du: arr(d.du), d2u: arr(d.d2u), r: arr(r), phys: phys.item(), grads: Object.fromEntries(named.map(([k], i) => [k, arr(gs[i])])) };
	});
}
writeFileSync(join(out, 'deriv_dump.json'), JSON.stringify(dump));

// 2. one twin fixture per train source and optimizer, with that source's options (as atlas parity builds them)
const STEPS = 50;
let n = 0;
for (const [source, d] of Object.entries(manifest.sources as Record<string, { kind: string; trainer?: string; opt?: Record<string, unknown> }>)) {
	if (d.kind !== 'train') continue;
	const def = tr[d.trainer as 'pinn' | 'plain'];
	const { seed, ...srcOpt } = d.opt ?? {};
	for (const [plan, extra, bar] of [['sgd', { optimizer: 'sgd', decoupled: false, weightDecay: 0 }, 1e-10], ['adam', {}, 1e-7]] as const) {
		const opt = { ...def.defaults, ...srcOpt, ...extra, steps: STEPS, stopOnDiverge: true };
		const tape = run(def, opt, { seed: Number(seed ?? 1), detail: 'fine', maxFrames: 4 * STEPS + 8 });
		const batches: unknown[] = [];
		const losses: number[] = [];
		let init: Record<string, T> | null = null;
		let grads0: Record<string, T> | null = null;
		let final: Record<string, T> | null = null;
		for (let i = 0; i < tape.length; i++) {
			const f = tape.frame(i);
			const s = f.state;
			if (f.phase === 'init' && !init) init = s.params;
			if (f.phase === 'forward') batches.push({ step: f.step, batch: plain(s.batch) });
			if (f.phase === 'loss') losses.push(f.metrics.loss);
			if (f.phase === 'backward' && f.step === 0) grads0 = s.grads;
			if (f.phase === 'update') final = s.params;
		}
		const fx = { format: 'atlas-train-fixture/1', slug: 'pinn', source, trainer: d.trainer, seed: Number(seed ?? 1), opt, steps: STEPS, bar, init: plain(init!), batches, ts: { losses, grads0: plain(grads0!), final: plain(final!) } };
		writeFileSync(join(out, `variant_fixture.${source}.${plan}.json`), JSON.stringify(fx));
		n++;
	}
}
console.log(`ts_dump: 3 weight sets x ${osc.N} times, ${n} twin fixtures -> ${out}`);
