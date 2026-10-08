// Demo props for the kind-B views (ViewMeta.demo, the /dev/atlas/ gallery, the Lesson Lab). Each demo records a small
// real training run with the toy trainers (record() from process/record.ts) and serves it through runtime/fake.ts, so
// the numbers are computed, but the stage is labelled 'illustrative' (a demo, not the learner's run).
// Pure TS: resolves in Node (types/contract.test.ts).
import type { StageProps } from '../../types/view.ts';
import type { ExplainerRuntime } from '../../types/runtime.ts';
import type { Cursor } from '../../types/process.ts';
import { trainingProcess } from '../../train/trainer.ts';
import { record } from '../../process/record.ts';
import type { TapeStore } from '../../process/tape.ts';
import { fakeRuntime } from '../../runtime/fake.ts';
import { atFor } from '../../runtime/sources.ts';
import { resolveCursor } from '../../runtime/selection.ts';
import { preprocessPoints, presetRaw } from '../../adapters/points2d/adapter.ts';
import { preprocessSeries, seriesPresetRaw } from '../../adapters/series/adapter.ts';
import { classifierTrainer, fitTrainer } from './toys.ts';

export type DemoProps = Omit<StageProps, 'ctx'>;

const cache = new Map<string, { rt: ExplainerRuntime; tape: TapeStore }>();

/** A classifier run on two moons: coarse (snapshots every 10 steps) or fine (SGD, every phase of 4 steps). */
export function classifierRun(detail: 'coarse' | 'fine' = 'coarse', steps = 200): { rt: ExplainerRuntime; tape: TapeStore } {
	const key = `cls:${detail}:${steps}`;
	const hit = cache.get(key);
	if (hit) return hit;
	const raw = presetRaw('moons', { n: 120 });
	const def = classifierTrainer({ res: 28, defaults: detail === 'fine' ? { optimizer: 'sgd', lr: 0.5, batch: 8 } : {} });
	const tape = record(trainingProcess(def), raw, { steps, snapshotEvery: 10 }, { seed: 1, detail });
	const rt = fakeRuntime({ slug: 'demo-train', input: raw, pre: preprocessPoints(raw, null, {}), tapes: { train: tape } });
	const out = { rt, tape };
	cache.set(key, out);
	return out;
}

/** A curve fit on the seasonal series. */
export function fitRun(steps = 400): { rt: ExplainerRuntime; tape: TapeStore } {
	const key = `fit:${steps}`;
	const hit = cache.get(key);
	if (hit) return hit;
	const raw = seriesPresetRaw('seasonal', { n: 96 });
	const tape = record(trainingProcess(fitTrainer({ defaults: { lr: 0.02 } })), raw, { steps, snapshotEvery: 25 }, { seed: 2 });
	const rt = fakeRuntime({ slug: 'demo-fit', input: raw, pre: preprocessSeries(raw, null, {}), tapes: { train: tape } });
	const out = { rt, tape };
	cache.set(key, out);
	return out;
}

/** Stage props for a view on source 'train' at a cursor. */
export function demoProps(run: { rt: ExplainerRuntime; tape: TapeStore }, c: Cursor, props: Record<string, unknown> = {}, focus: StageProps['focus'] = []): DemoProps {
	const { index } = resolveCursor(run.tape, c);
	return {
		rt: run.rt,
		source: 'train',
		at: atFor(run.rt, 'train', index),
		focus,
		sel: run.rt.sel,
		provenance: 'illustrative',
		autoplay: false,
		onend: () => {},
		props
	};
}
