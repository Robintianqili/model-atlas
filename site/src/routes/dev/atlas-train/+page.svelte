<!--
  /dev/atlas-train/ — the training engine's performance page (DevKit W1-B2).
  Real training in the browser through the kit: createRuntime (W1-C) + engineRunner (each bench package's own worker)
  + the kind-B views. Every number is live. Engine choice: ?engine=main (main-thread recorder), ?engine=broken (a worker
  that fails to load, so the fallback takes over), default: the package workers.
  Checks on the page: steps per second, long tasks while training (PerformanceObserver), main-thread time spent storing
  frames, and "same frames as a main-thread run" (frame and state hashes).
-->
<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import { colon, t } from '$lib/i18n/i18n.svelte';
	import Section from '$lib/kit/Section.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Value from '$lib/atlas/ui/train/Value.svelte';
	import Stage from './Stage.svelte';
	import type { Focus, LiveTape, StageProps, Tape, Tensor } from '$lib/atlas/types/index.ts';
	import type { AtlasRuntime } from '$lib/atlas/runtime/runtime.ts';
	import type { EngineRunner, EngineTape } from '$lib/atlas/engine/runner.ts';
	import { atFor } from '$lib/atlas/runtime/sources.ts';
	import { presetRaw, type PointsRaw } from '$lib/atlas/adapters/points2d/adapter.ts';
	import { seriesPresetRaw, type SeriesRaw } from '$lib/atlas/adapters/series/adapter.ts';
	import PointsCapture from '$lib/atlas/adapters/points2d/Capture.svelte';
	import SeriesCapture from '$lib/atlas/adapters/series/Capture.svelte';
	import TrainControls from '$lib/atlas/ui/views/train-controls/View.svelte';
	import LossCurve from '$lib/atlas/ui/views/loss-curve/View.svelte';
	import Plane2d from '$lib/atlas/ui/views/plane2d/View.svelte';
	import Curve1d from '$lib/atlas/ui/views/curve1d/View.svelte';
	import Network from '$lib/atlas/ui/views/network/View.svelte';
	import TrainStep from '$lib/atlas/ui/views/train-step/View.svelte';
	import { BOUNDS, eulerPaths, velocityField } from '$lib/atlas/ui/train/toys.ts';

	type Mode = 'auto' | 'main' | 'broken';
	let mode = $state<Mode>('auto');
	let flowRt = $state.raw<AtlasRuntime | null>(null);
	let toysRt = $state.raw<AtlasRuntime | null>(null);
	let fitRt = $state.raw<AtlasRuntime | null>(null);
	let runner: EngineRunner | null = null;
	let error = $state<string | null>(null);
	let tick = $state(0);
	let bundles = $state<string[]>([]);

	// ---------------------------------------------------------------- long tasks
	let longTasks = $state<{ start: number; ms: number }[]>([]);
	let trainingSince = $state<number | null>(null);
	const longWhileTraining = $derived(trainingSince === null ? [] : longTasks.filter((l) => l.start >= trainingSince!));

	const flowPoints = presetRaw('smiley', { n: 1000 });
	let clsPoints = $state.raw<PointsRaw>(presetRaw('moons', { n: 200 }));
	let fitSeries = $state.raw<SeriesRaw>(seriesPresetRaw('seasonal', { n: 120 }));

	onMount(() => {
		const q = new URLSearchParams(location.search).get('engine');
		mode = q === 'main' || q === 'broken' ? q : 'auto';
		let obs: PerformanceObserver | null = null;
		try {
			obs = new PerformanceObserver((list) => {
				longTasks = [...longTasks, ...list.getEntries().map((e) => ({ start: e.startTime, ms: e.duration }))];
			});
			obs.observe({ type: 'longtask', buffered: true });
		} catch {
			obs = null;
		}
		let closed = false;
		const offs: (() => void)[] = [];
		(async () => {
			try {
				const [{ createRuntime }, { makeEngineRunner }, pk, { flow }, { cls, fit }] = await Promise.all([
					import('$lib/atlas/runtime/runtime.ts'),
					import('$lib/atlas/engine/runner.ts'),
					import('./bench/packages.ts'),
					import('./bench/flow/trainer.ts'),
					import('./bench/toys/trainer.ts')
				]);
				bundles = pk.workerNames();
				const w = async (name: string) => (mode === 'main' ? null : await pk.workerOf(mode === 'broken' ? 'broken' : name));
				const [flowW, toysW] = await Promise.all([w('flow'), w('toys')]);
				runner = makeEngineRunner({ mode: mode === 'main' ? 'main' : 'auto', readyTimeoutMs: 8000 });
				const flowPkg = pk.benchPackage({
					slug: 'bench-flow',
					trainers: [flow],
					adapter: 'points2d',
					start: flowPoints,
					sources: { flow: { kind: 'train', trainer: 'flow', opt: { steps: 3000, snapshotEvery: 50 }, detail: 'coarse' } },
					worker: flowW
				});
				const toysPkg = pk.benchPackage({
					slug: 'bench-toys',
					trainers: [cls],
					adapter: 'points2d',
					start: clsPoints,
					sources: {
						train: { kind: 'train', trainer: 'cls', opt: { steps: 2000, snapshotEvery: 10 }, detail: 'coarse' },
						sgd1: { kind: 'train', trainer: 'cls', opt: { steps: 200, optimizer: 'sgd', lr: 0.3, batch: 8 }, detail: 'fine' }
					},
					worker: toysW
				});
				const fitPkg = pk.benchPackage({
					slug: 'bench-fit',
					trainers: [fit],
					adapter: 'series',
					start: fitSeries,
					sources: { fit: { kind: 'train', trainer: 'fit', opt: { steps: 3000, snapshotEvery: 25, lr: 0.01 }, detail: 'coarse' } },
					worker: toysW
				});
				const fetchFn = fetch.bind(globalThis);
				const [a, b, c] = await Promise.all([
					createRuntime(flowPkg, { fetch: fetchFn, base, runner }),
					createRuntime(toysPkg, { fetch: fetchFn, base, runner }),
					createRuntime(fitPkg, { fetch: fetchFn, base, runner })
				]);
				if (closed) return;
				for (const rt of [a, b, c]) offs.push(rt.onChange(() => tick++));
				flowRt = a;
				toysRt = b;
				fitRt = c;
				// for the browser checks (e2e.check.ts) and for poking at the engine from the console
				(window as unknown as { __atlasTrain?: unknown }).__atlasTrain = { flowRt: a, toysRt: b, fitRt: c, runner };
				// open every source and show its first frame (training starts when the learner presses Train)
				await Promise.all([a.ensure('flow', 'first'), b.ensure('train', 'first'), b.ensure('sgd1', { phase: 'update' }), c.ensure('fit', 'first')]);
				tick++;
			} catch (e) {
				error = e instanceof Error ? e.message : String(e);
			}
		})();
		return () => {
			closed = true;
			obs?.disconnect();
			offs.forEach((f) => f());
			for (const rt of [flowRt, toysRt, fitRt]) rt?.close();
			runner?.dispose();
		};
	});

	// ---------------------------------------------------------------- stage props
	const tapeOf = (rt: AtlasRuntime | null, s: string) => (rt?.sources[s]?.tape as (LiveTape & Partial<EngineTape>) | null | undefined) ?? null;
	function sp(rt: AtlasRuntime, source: string, index: number, props: Record<string, unknown> = {}, focus: Focus[] = []): StageProps {
		return { rt, source, at: atFor(rt, source, index), focus, sel: rt.sel, provenance: rt.sources[source]?.provenance ?? 'live', autoplay: false, onend: () => {}, props };
	}
	/** the latest frame that carries weights (a snapshot) */
	function lastSnapshot(tp: Tape | null): number {
		if (!tp || tp.length === 0) return -1;
		const i = tp.find({ phase: 'snapshot', last: true });
		return i >= 0 ? i : 0;
	}

	// the flow field at time t, computed once per (weights, t)
	let fieldMemo: { params: unknown; t: number; v: Tensor } | null = null;
	function vectorsAt(time: number, s: { params?: Record<string, Tensor> }): Tensor {
		if (!s.params) return { shape: [0, 4], data: new Float64Array(0) };
		if (fieldMemo && fieldMemo.params === s.params && fieldMemo.t === time) return fieldMemo.v;
		const v = velocityField(s.params, time, BOUNDS, 15, 'v');
		fieldMemo = { params: s.params, t: time, v };
		return v;
	}

	const flowTape = $derived.by(() => {
		void tick;
		return tapeOf(flowRt, 'flow');
	});
	const flowSnap = $derived.by(() => {
		void tick;
		return lastSnapshot(flowTape);
	});
	const flowLatest = $derived.by(() => {
		void tick;
		return flowTape ? flowTape.length - 1 : -1;
	});
	const flowFieldProps = $derived(flowRt && flowSnap >= 0 ? sp(flowRt, 'flow', flowSnap, { vectorsAt, time: { min: 0, max: 1, value: 0.5, step: 0.02, label: { en: 'time t', zh: '时间 t' } } }) : null);

	// samples: Euler steps from seeded noise with the latest weights (in the worker through a 'call', else in slices here)
	let samples = $state.raw<{ paths: Tensor; samples: Tensor; step: number; where: string } | null>(null);
	let sampling = $state(false);
	async function sample() {
		if (!flowRt || !flowTape || sampling) return;
		const st = flowTape.state(flowSnap) as { params?: Record<string, Tensor>; step?: number };
		if (!st.params) return;
		sampling = true;
		const { createRng } = await import('$lib/atlas/core/rng.ts');
		const rng = createRng(5);
		const n = 300;
		const start = new Float64Array(n * 2);
		for (let i = 0; i < start.length; i++) start[i] = rng.normal();
		const K = 40;
		let paths: Tensor | null = null;
		let where = 'main';
		const client = runner?.client(flowRt.pkg);
		if (client && client.mode === 'worker') {
			paths = (await client.call<Tensor>('euler', { params: st.params, start, K }, 'euler')) ?? null;
			where = 'worker';
		}
		if (!paths) {
			// main thread, a few particles per slice so no task gets long
			const parts: Tensor[] = [];
			for (let i0 = 0; i0 < n; i0 += 40) {
				parts.push(eulerPaths(st.params, start.slice(i0 * 2, Math.min(n, i0 + 40) * 2), K));
				await new Promise((r) => setTimeout(r, 0));
			}
			const data = new Float64Array(n * (K + 1) * 2);
			let o = 0;
			for (const p of parts) {
				data.set(p.data, o);
				o += p.data.length;
			}
			paths = { shape: [n, K + 1, 2], data };
		}
		const end = new Float64Array(n * 2);
		for (let i = 0; i < n; i++) {
			end[2 * i] = paths.data[(i * (K + 1) + K) * 2];
			end[2 * i + 1] = paths.data[(i * (K + 1) + K) * 2 + 1];
		}
		samples = { paths, samples: { shape: [n, 2], data: end }, step: st.step ?? 0, where };
		sampling = false;
	}
	const sampleProps = $derived(flowRt && samples && flowSnap >= 0 ? sp(flowRt, 'flow', flowSnap, { points: null, paths: samples.paths, samples: samples.samples }) : null);

	// classifier
	const clsTape = $derived.by(() => {
		void tick;
		return tapeOf(toysRt, 'train');
	});
	const clsSnap = $derived.by(() => {
		void tick;
		return lastSnapshot(clsTape);
	});
	const clsLatest = $derived.by(() => {
		void tick;
		return clsTape ? clsTape.length - 1 : -1;
	});
	function setPoints(raw: PointsRaw) {
		clsPoints = raw;
		toysRt?.setInput(raw);
	}

	// one SGD step, slowed down: a frame scrubber over the fine source
	const sgdTape = $derived.by(() => {
		void tick;
		return tapeOf(toysRt, 'sgd1');
	});
	let sgdFrame = $state<number | null>(null);
	const sgdIndex = $derived.by(() => {
		void tick;
		const n = sgdTape?.length ?? 0;
		if (!n) return -1;
		return sgdFrame === null ? n - 1 : Math.min(n - 1, sgdFrame);
	});

	// curve fit
	const fitTape = $derived.by(() => {
		void tick;
		return tapeOf(fitRt, 'fit');
	});
	const fitSnap = $derived.by(() => {
		void tick;
		return lastSnapshot(fitTape);
	});
	function setSeries(raw: SeriesRaw) {
		fitSeries = raw;
		fitRt?.setInput(raw);
	}

	// ---------------------------------------------------------------- engine stats (flow)
	const stats = $derived.by(() => {
		void tick;
		const tp = flowTape;
		return tp ? { where: tp.where ?? 'main', frames: tp.length, messages: tp.stats?.messages ?? 0, maxHandle: tp.stats?.maxHandleMs ?? NaN, totalHandle: tp.stats?.totalHandleMs ?? NaN, notice: tp.notice ?? null } : null;
	});
	$effect(() => {
		void tick;
		const tp = flowTape;
		if (tp && trainingSince === null && tp.paused === false) trainingSince = performance.now();
	});

	// ---------------------------------------------------------------- same frames as a main-thread run
	let check = $state<{ frames: number; worker: string; local: string; states: boolean; where: string } | null>(null);
	let checking = $state(false);
	async function hashCheck() {
		if (!toysRt || checking) return;
		const tp = tapeOf(toysRt, 'train');
		if (!tp || tp.length < 2) return;
		checking = true;
		tp.control({ type: 'pause' });
		await new Promise((r) => setTimeout(r, 120)); // let the frames in flight arrive
		const { makeLocalRunner } = await import('$lib/atlas/runtime/local.ts');
		const { Hasher } = await import('$lib/atlas/process/values.ts');
		const n = tp.length;
		const d = toysRt.pkg.manifest.sources!.train as { trainer: string; opt?: Record<string, unknown>; detail?: 'coarse' | 'fine' };
		const local = makeLocalRunner({ autorun: false }).open({ slug: 'bench-toys', source: 'check', process: `train:${d.trainer}`, input: toysRt.input, opt: { ...(d.opt ?? {}) }, seed: 1, detail: d.detail ?? 'coarse' }, toysRt.pkg);
		await local.ensure(n - 1);
		const hashFrames = (x: Tape) => {
			const h = new Hasher();
			for (let i = 0; i < n; i++) h.value(x.frame(i));
			return h.hex();
		};
		const st = (x: Tape) => {
			const h = new Hasher();
			for (let i = 0; i < n; i += 7) h.value((x as unknown as { store: { rawState(i: number): unknown } }).store.rawState(i));
			return h.hex();
		};
		check = { frames: n, worker: hashFrames(tp), local: hashFrames(local), states: st(tp) === st(local), where: tp.where ?? 'main' };
		local.close();
		checking = false;
	}

	const modeLink = (m: Mode) => `${base}/dev/atlas-train/${m === 'auto' ? '' : `?engine=${m}`}`;
</script>

<svelte:head><title>{t({ en: 'Training engine', zh: '训练引擎' })} · Model Atlas</title></svelte:head>

<div class="container page">
	<header class="head">
		<p class="eyebrow">/dev/atlas-train/</p>
		<h1>{t({ en: 'Training engine', zh: '训练引擎' })}</h1>
		<p class="lede">
			{t({
				en: 'Real training in your browser, through the DevKit: each package trains in its own background thread and streams its steps here. Every number on this page is computed live.',
				zh: '在你的浏览器里真实地训练，用的是开发套件：每个包在自己的后台线程里训练，再把每一步传到这里。本页的每个数都是实时算出来的。'
			})}
		</p>
		<nav class="modes small" aria-label={t({ en: 'Where the engine runs', zh: '引擎在哪里运行' })}>
			<span>{t({ en: 'Engine', zh: '引擎' })}{colon()}</span>
			<a href={modeLink('auto')} aria-current={mode === 'auto' ? 'page' : undefined} data-sveltekit-reload>{t({ en: 'package workers', zh: '包的后台线程' })}</a>
			<a href={modeLink('main')} aria-current={mode === 'main' ? 'page' : undefined} data-sveltekit-reload>{t({ en: 'main thread', zh: '主线程' })}</a>
			<a href={modeLink('broken')} aria-current={mode === 'broken' ? 'page' : undefined} data-sveltekit-reload>{t({ en: 'a broken worker (fallback)', zh: '坏掉的后台线程（回退）' })}</a>
		</nav>
		{#if error}<p class="warn" role="alert">{error}</p>{/if}
	</header>

	<Section id="flow" title={t({ en: 'Flow matching in 2-D: 3 → 64 → 64 → 64 → 2, batch 256', zh: '二维流匹配：3 → 64 → 64 → 64 → 2，批大小 256' })} provenance="live">
		<p class="intro">
			{t({
				en: 'A velocity network learns to push random noise onto the smiley. Each step trains on 256 random (noise, data, time) triples with Adam. This is the heaviest toy in the batch, so it is the engine’s speed test.',
				zh: '一个速度网络学着把随机噪声推到笑脸上。每一步用 256 个随机的（噪声，数据，时间）三元组和 Adam 训练。这是这一批里最重的玩具，所以用来测引擎的速度。'
			})}
		</p>
		{#if flowRt && flowTape}
			<div class="grid2">
				<div class="card" data-testid="flow-controls">
					<Stage view={TrainControls} p={sp(flowRt, 'flow', flowLatest, { lrs: [0.0001, 0.0003, 0.001, 0.003, 0.01] })} />
				</div>
				<div class="card stats" data-testid="engine-stats">
					<h3 class="small">{t({ en: 'Engine', zh: '引擎' })} <ProvenanceBadge mode="live" compact /></h3>
					{#if stats}
						<dl>
							<div><dt>{t({ en: 'runs in', zh: '运行于' })}</dt><dd data-testid="where">{stats.where}</dd></div>
							<div><dt>{t({ en: 'frames received', zh: '收到的帧' })}</dt><dd class="num">{stats.frames}</dd></div>
							<div><dt>{t({ en: 'messages', zh: '消息数' })}</dt><dd class="num">{stats.messages}</dd></div>
							<div><dt>{t({ en: 'longest frame handling (main thread)', zh: '处理一批帧的最长用时（主线程）' })}</dt><dd><Value value={stats.maxHandle} unit=" ms" /></dd></div>
							<div><dt>{t({ en: 'long tasks while training (> 50 ms, any cause)', zh: '训练时的长任务（> 50 毫秒，任何原因）' })}</dt><dd class="num" data-testid="longtasks">{trainingSince === null ? '—' : `${longWhileTraining.length}${longWhileTraining.length ? ` (max ${Math.round(Math.max(...longWhileTraining.map((l) => l.ms)))} ms)` : ''}`}</dd></div>
						</dl>
						{#if stats.notice}<p class="small notice">{t(stats.notice)}</p>{/if}
					{/if}
				</div>
			</div>
			<div class="card"><Stage view={LossCurve} p={sp(flowRt, 'flow', flowLatest, { log: false, smooth: 50 })} /></div>
			<div class="grid2">
				<div class="card">
					<h3 class="small">{t({ en: 'The learned velocity field (drag the time)', zh: '学到的速度场（拖动时间）' })}</h3>
					{#if flowFieldProps}<Stage view={Plane2d} p={flowFieldProps} />{/if}
				</div>
				<div class="card">
					<h3 class="small">{t({ en: 'Sampling: noise follows the field for 40 Euler steps', zh: '采样：噪声沿着速度场走 40 个欧拉步' })}</h3>
					<button class="btn" type="button" onclick={sample} disabled={sampling || flowSnap < 0}>{sampling ? t({ en: 'Sampling…', zh: '采样中…' }) : t({ en: 'Sample with the current weights', zh: '用当前权重采样' })}</button>
					{#if sampleProps && samples}
						<p class="small muted">{t({ en: `300 points, weights of step ${samples.step}, computed in the ${samples.where === 'worker' ? 'worker' : 'page'}.`, zh: `300 个点，用第 ${samples.step} 步的权重，在${samples.where === 'worker' ? '后台线程' : '页面'}里计算。` })}</p>
						<Stage view={Plane2d} p={sampleProps} />
					{/if}
				</div>
			</div>
		{:else}
			<p class="muted">{t({ en: 'Starting the engine…', zh: '正在启动引擎…' })}</p>
		{/if}
	</Section>

	<Section id="classifier" title={t({ en: 'A classifier on 2-D points: 2 → 8 → 8 → 2, tanh', zh: '二维点上的分类器：2 → 8 → 8 → 2，tanh' })} provenance="live">
		<p class="intro">
			{t({
				en: 'Paint points or pick a set, then train. The background is the network’s answer at every place; the line is where it is 50/50. Changing the points sends the new data to the running trainer.',
				zh: '画一些点或选一个点集，然后训练。背景是网络在每个位置的回答；线是五五开的地方。改动这些点，会把新数据发给正在运行的训练。'
			})}
		</p>
		{#if toysRt && clsTape}
			<div class="grid2">
				<div class="card"><PointsCapture raw={clsPoints} onchange={setPoints} /></div>
				<div class="card"><Stage view={Plane2d} p={sp(toysRt, 'train', clsSnap)} /></div>
			</div>
			<div class="grid2">
				<div class="card"><Stage view={TrainControls} p={sp(toysRt, 'train', clsLatest)} /></div>
				<div class="card">
					<Stage view={LossCurve} p={sp(toysRt, 'train', clsLatest)} />
					<Stage view={LossCurve} p={sp(toysRt, 'train', clsLatest, { metric: 'accuracy', log: false })} />
				</div>
			</div>
			<div class="card"><Stage view={Network} p={sp(toysRt, 'train', clsSnap, { inputs: ['x', 'y'], outputs: [{ en: 'class 0', zh: '第 0 类' }, { en: 'class 1', zh: '第 1 类' }] })} /></div>
		{/if}
	</Section>

	<Section id="sgd" title={t({ en: 'One SGD step, slowed down', zh: '放慢的一步 SGD' })} provenance="live">
		<p class="intro">{t({ en: 'The same classifier trained with plain SGD (learning rate 0.3, batch 8), recorded phase by phase. Step it, then scrub through the four parts.', zh: '同一个分类器用普通 SGD（学习率 0.3，批大小 8）训练，按阶段记录。一步一步地走，然后拖动查看四个部分。' })}</p>
		{#if toysRt && sgdTape}
			<div class="card"><Stage view={TrainControls} p={sp(toysRt, 'sgd1', sgdTape.length - 1, { showReset: true })} /></div>
			<label class="scrub small">
				<span>{t({ en: 'Frame', zh: '帧' })} <span class="num">{sgdIndex + 1} / {sgdTape.length}</span></span>
				<input type="range" min="0" max={Math.max(0, sgdTape.length - 1)} value={sgdIndex} oninput={(e) => (sgdFrame = Number((e.currentTarget as HTMLInputElement).value))} />
			</label>
			<div class="grid2">
				<div class="card"><Stage view={TrainStep} p={sp(toysRt, 'sgd1', sgdIndex)} /></div>
				<div class="card"><Stage view={Network} p={sp(toysRt, 'sgd1', sgdIndex, { color: 'grad', inputs: ['x', 'y'] })} /></div>
			</div>
		{/if}
	</Section>

	<Section id="fit" title={t({ en: 'Fit a curve: 1 → 16 → 16 → 1, tanh', zh: '拟合曲线：1 → 16 → 16 → 1，tanh' })} provenance="live">
		<p class="intro">{t({ en: 'Draw over the series to change it. The network sees only (time, value) pairs.', zh: '在序列上画就能改它。网络只看到（时间，值）这样的数对。' })}</p>
		{#if fitRt && fitTape}
			<div class="grid2">
				<div class="card"><SeriesCapture raw={fitSeries} onchange={setSeries} options={{ mode: 'curve' }} /></div>
				<div class="card"><Stage view={TrainControls} p={sp(fitRt, 'fit', fitTape.length - 1, { lrs: [0.001, 0.003, 0.01, 0.03, 0.1] })} /></div>
			</div>
			<div class="card"><Stage view={Curve1d} p={sp(fitRt, 'fit', fitSnap)} /></div>
		{/if}
	</Section>

	<Section id="checks" title={t({ en: 'Engine checks', zh: '引擎检查' })}>
		<div class="card">
			<h3 class="small">{t({ en: 'Same frames as a main-thread run', zh: '与主线程运行的帧相同' })}</h3>
			<p class="small muted">{t({ en: 'Pauses the classifier, records the same number of frames on this page with the local recorder, and compares hashes of every frame and of the states.', zh: '暂停分类器，用本页的本地记录器记录同样多的帧，再比较每一帧和状态的哈希。' })}</p>
			<button class="btn" type="button" onclick={hashCheck} disabled={checking || !clsTape} data-testid="hash-check">{checking ? t({ en: 'Checking…', zh: '检查中…' }) : t({ en: 'Compare', zh: '比较' })}</button>
			{#if check}
				<p class="small" data-testid="hash-result">
					{check.frames} {t({ en: 'frames', zh: '帧' })} ({check.where}): {t({ en: 'frames', zh: '帧' })} <code>{check.worker}</code> vs <code>{check.local}</code> —
					<strong>{check.worker === check.local && check.states ? t({ en: 'equal', zh: '相同' }) : t({ en: 'DIFFERENT', zh: '不同' })}</strong>
				</p>
			{/if}
		</div>
		<div class="card">
			<h3 class="small">{t({ en: 'Worker bundles', zh: '后台线程的打包' })}</h3>
			<p class="small muted">{t({ en: 'Each bench package has its own worker entry, found by one ?worker glob; the build makes one bundle per package.', zh: '每个测试包都有自己的后台线程入口，用一个 ?worker glob 找到；构建时每个包打成一个单独的包。' })}</p>
			<ul class="small num">{#each bundles as b (b)}<li>bench/{b}/worker.ts</li>{/each}</ul>
		</div>
	</Section>
</div>

<style>
	.page {
		padding-bottom: var(--s8);
	}
	.head {
		padding: var(--s6) 0 var(--s4);
		display: grid;
		gap: var(--s2);
	}
	.modes {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 14px;
	}
	.modes a[aria-current='page'] {
		font-weight: 700;
		text-decoration: none;
		color: var(--ink);
	}
	.intro {
		max-width: var(--measure);
		margin: 0 0 var(--s3);
	}
	.grid2 {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s3);
		margin-bottom: var(--s3);
	}
	@media (min-width: 860px) {
		.grid2 {
			grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		}
	}
	.card {
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
		padding: var(--s3);
		min-width: 0;
		margin-bottom: var(--s3);
		display: grid;
		gap: var(--s2);
		align-content: start;
	}
	.grid2 .card {
		margin-bottom: 0;
	}
	h3 {
		margin: 0;
		display: flex;
		gap: 8px;
		align-items: center;
	}
	.stats dl {
		margin: 0;
		display: grid;
		gap: 6px;
	}
	.stats dl div {
		display: flex;
		justify-content: space-between;
		gap: var(--s3);
		border-bottom: 1px dashed var(--rule);
		padding-bottom: 4px;
	}
	.stats dt {
		color: var(--ink-2);
	}
	.stats dd {
		margin: 0;
		text-align: right;
	}
	.scrub {
		display: flex;
		gap: var(--s2);
		align-items: center;
		margin-bottom: var(--s3);
	}
	.scrub input {
		flex: 1;
		min-width: 0;
	}
	.warn {
		color: var(--k-text);
	}
	.notice {
		margin: 0;
		color: var(--ink-2);
	}
	code {
		font-family: var(--font-mono);
		font-size: 12px;
		word-break: break-all;
	}
	ul {
		margin: 0;
		padding-left: 1.2em;
	}
</style>
