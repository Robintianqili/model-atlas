// The bench packages of /dev/atlas-train/, built the way ui/packages.ts builds real ones: processes from trainer.ts
// (wrapped as 'train:<id>'), the kit adapters, and the package's own worker found by a `?worker` glob (one bundle per
// package; design §4.10). Browser only (the glob is a Vite feature).
import type { ExplainerManifest, PackageModules, ProcessDef, SourceDef, TrainerDef } from '$lib/atlas/types/index.ts';
import { trainingProcess } from '$lib/atlas/train/trainer.ts';
import points2d from '$lib/atlas/adapters/points2d/adapter.ts';
import series from '$lib/atlas/adapters/series/adapter.ts';

const WORKERS = import.meta.glob('./*/worker.ts', { query: '?worker', import: 'default' }) as Record<string, () => Promise<new () => Worker>>;

/** The worker constructor of a bench package (null if it has none or its chunk cannot load). */
export async function workerOf(name: string): Promise<(() => Worker) | null> {
	const load = WORKERS[`./${name}/worker.ts`];
	if (!load) return null;
	try {
		const W = await load();
		return () => new W();
	} catch {
		return null;
	}
}

export const workerNames = (): string[] => Object.keys(WORKERS).map((k) => k.split('/')[1]);

export function benchPackage(o: {
	slug: string;
	trainers: TrainerDef<any, any, any, any>[];
	adapter: 'points2d' | 'series';
	start: unknown;
	sources: Record<string, SourceDef>;
	worker: (() => Worker) | null;
}): PackageModules {
	const processes: Record<string, ProcessDef<any, any, any>> = {};
	for (const t of o.trainers) processes[`train:${t.id}`] = trainingProcess(t);
	const manifest: ExplainerManifest = {
		slug: o.slug,
		type: 'model',
		kinds: ['train'],
		quality: 'draft',
		devOnly: true,
		input: { adapter: o.adapter, start: o.start },
		sources: o.sources,
		page: {
			goals: [{ en: 'Measure the training engine', zh: '测量训练引擎' }],
			prequestion: { id: `${o.slug}-q`, question: { en: 'Where does training run?', zh: '训练在哪里运行？' }, options: [{ en: 'In a worker', zh: '在后台线程' }], answer: 0, explain: { en: 'In the package worker.', zh: '在包自己的后台线程里。' } }
		},
		credits: []
	};
	return { slug: o.slug, manifest, lesson: null, ops: [], processes, adapters: { points2d, series }, ...(o.worker ? { worker: o.worker } : {}) };
}
