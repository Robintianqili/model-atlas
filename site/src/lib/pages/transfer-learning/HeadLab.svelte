<!--
  Part 1 (live): freeze TinyCNN's body, train a new 2-way head on the learner's examples.
  Features of the preset drawings are computed in a Web Worker; the head trains on the main
  thread a few steps per animation frame (it takes ~5–30 ms in total), so the UI stays smooth.
-->
<script lang="ts">
	import { onDestroy, tick } from 'svelte';
	import DrawPad from '$lib/kit/DrawPad.svelte';
	import ProbBars from '$lib/kit/ProbBars.svelte';
	import StepPlayer from '$lib/kit/StepPlayer.svelte';
	import ProvenanceBadge from '$lib/kit/ProvenanceBadge.svelte';
	import Num from '$lib/kit/Num.svelte';
	import { motion } from '$lib/kit/theme.svelte';
	import { categorical } from '$lib/kit/color';
	import { themeState } from '$lib/kit/theme.svelte';
	import type { Stroke } from '$lib/kit/draw';
	import { t, i18n } from '$lib/i18n/i18n.svelte';
	import { createHeadTrainer, trainHead, predictHead, type Head } from '$lib/models/tinycnn_finetune';
	import type { Trace } from '$lib/models/tinycnn';
	import LabNetwork from './LabNetwork.svelte';
	import TrainSet from './TrainSet.svelte';
	import TrainCharts from './TrainCharts.svelte';
	import PcaMap from './PcaMap.svelte';
	import Article from './ArticlePart.svelte';
	import { BASE_CLASSES, decodeB64, simplifyStrokes, type HeldoutJson, type PcaJson, type SetItem } from './data';
	import { lab, goTo } from './state.svelte';
	import type { Engine } from './engine';
	import { presetFeatures } from './presetFeats';

	interface Props {
		engine: Engine | null;
		heldout: HeldoutJson | null;
		pca: PcaJson | null;
		tolerance?: number;
	}
	let { engine, heldout, pca, tolerance }: Props = $props();

	const NS = [1, 2, 4, 8, 16, 32];
	const EPOCHS = 200;

	interface Ex {
		id: string;
		cls: 0 | 1;
		input: Float32Array;
		feat: Float32Array | null;
		drawn: boolean;
		flipped: boolean;
	}
	interface Run {
		key: string;
		snaps: Float32Array[];
		loss: number[];
		trainAcc: number[];
		testAcc: number[] | null;
		scale: number;
		nA: number;
		nB: number;
		preset: boolean;
		done: boolean;
	}

	let mode = $state<'preset' | 'own'>('preset');
	let nPer = $state(8);
	let ownNames = $state({ a: '', b: '' });
	let flipped = $state(0);
	let drawnPreset = $state.raw<Ex[]>([]);
	let drawnOwn = $state.raw<Ex[]>([]);
	let counter = 0;

	// ---------------------------------------------------------------- preset drawings + features
	const presetIn = $derived(
		heldout ? heldout.classes.map((c) => ({ train: c.train.map(decodeB64), test: c.test.map(decodeB64) })) : null
	);
	let presetFeat = $state.raw<{ train: Float32Array[][]; test: Float32Array[][] } | null>(null);
	let featStatus = $state<'idle' | 'busy' | 'ready' | 'error'>('idle');
	let featMs = $state(0);
	let featWhere = $state<'worker' | 'main'>('worker');

	$effect(() => {
		if (engine && heldout && featStatus === 'idle') computePresetFeats(engine, heldout);
	});
	// shared with the guided lesson: the 264 preset features are computed once per page (in the worker)
	async function computePresetFeats(eng: Engine, h: HeldoutJson) {
		featStatus = 'busy';
		try {
			const r = await presetFeatures(eng, h);
			featMs = r.ms;
			featWhere = r.where;
			presetFeat = r.feats;
			featStatus = 'ready';
		} catch {
			featStatus = 'error';
		}
	}

	// ---------------------------------------------------------------- examples
	const examples = $derived.by<Ex[]>(() => {
		if (mode === 'own') return drawnOwn;
		if (!presetIn) return [];
		const out: Ex[] = [];
		const nf = Math.min(flipped, nPer - 1);
		for (let k = 0 as 0 | 1; k < 2; k = (k + 1) as 0 | 1) {
			for (let i = 0; i < nPer; i++) {
				const isFlip = k === 1 && i < nf;
				out.push({
					id: `p${k}-${i}`,
					cls: isFlip ? 0 : k,
					input: presetIn[k].train[i],
					feat: presetFeat?.train[k][i] ?? null,
					drawn: false,
					flipped: isFlip
				});
			}
		}
		return [...out, ...drawnPreset];
	});
	const names = $derived<[string, string]>(
		mode === 'preset'
			? [t({ en: 'ladder', zh: '梯子' }), t({ en: 'envelope', zh: '信封' })]
			: [ownNames.a.trim() || t({ en: 'Class A', zh: '类别 A' }), ownNames.b.trim() || t({ en: 'Class B', zh: '类别 B' })]
	);
	const dataKey = $derived(`${mode}|${examples.map((e) => `${e.id}:${e.cls}`).join(',')}`);
	const counts = $derived([examples.filter((e) => e.cls === 0).length, examples.filter((e) => e.cls === 1).length]);
	const featsReady = $derived(examples.length > 0 && examples.every((e) => e.feat));
	const canTrain = $derived(counts[0] > 0 && counts[1] > 0 && featsReady);

	// ---------------------------------------------------------------- the head and its training run
	let run = $state.raw<Run | null>(null);
	let epoch = $state(0);
	let training = $state(false);
	let destroyed = false;
	onDestroy(() => (destroyed = true));

	// Each mode keeps its own head: switching between "ladder vs envelope" and "your own two things"
	// must never show one pair's trained head under the other pair's class names.
	const savedRuns: Record<'preset' | 'own', { run: Run | null; epoch: number }> = {
		preset: { run: null, epoch: 0 },
		own: { run: null, epoch: 0 }
	};
	function setMode(m: 'preset' | 'own') {
		if (m === mode || training) return;
		savedRuns[mode] = { run, epoch };
		mode = m;
		run = savedRuns[m].run;
		epoch = savedRuns[m].epoch;
	}

	function pack(h: Head): Float32Array {
		const s = new Float32Array(514);
		s.set(h.W, 0);
		s.set(h.b, 512);
		return s;
	}
	function testAccOf(snap: Float32Array, scale: number): number | null {
		if (!presetFeat) return null;
		let ok = 0;
		let n = 0;
		for (let k = 0; k < 2; k++) {
			for (const f of presetFeat.test[k]) {
				let m = snap[512] - snap[513];
				for (let d = 0; d < 256; d++) m += ((snap[d] - snap[256 + d]) * f[d]) / scale;
				if ((m >= 0 ? 0 : 1) === k) ok++;
				n++;
			}
		}
		return ok / n;
	}
	function startRun(ex: Ex[], key: string) {
		const trainer = createHeadTrainer(
			ex.map((e) => e.feat!),
			ex.map((e) => e.cls),
			{ numClasses: 2 }
		);
		const h = trainer.head();
		const preset = mode === 'preset';
		const ta = preset ? testAccOf(pack(h), h.scale) : null;
		const r: Run = {
			key,
			snaps: [pack(h)],
			loss: [h.lossHistory[0]],
			trainAcc: [h.accHistory[0]],
			testAcc: ta === null ? null : [ta],
			scale: h.scale,
			nA: ex.filter((e) => e.cls === 0).length,
			nB: ex.filter((e) => e.cls === 1).length,
			preset,
			done: false
		};
		return { trainer, r };
	}

	// an untrained head (step 0) follows the data automatically; a trained one is kept until retrained
	$effect(() => {
		const key = dataKey;
		if (training || !canTrain) {
			if (!canTrain && run && run.snaps.length <= 1) run = null;
			return;
		}
		if (run && (run.key === key || run.snaps.length > 1)) return;
		run = startRun(examples, key).r;
		epoch = 0;
	});

	const nextFrame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));
	async function train() {
		if (training || !canTrain) return;
		training = true;
		const key = dataKey;
		const ex = examples;
		const { trainer, r } = startRun(ex, key);
		run = r;
		epoch = 0;
		// ~6 steps per frame (about half a second in all); on a slow device, where a frame takes
		// longer than ~45 ms, take bigger chunks so the whole run still finishes in about a second.
		// Reduced motion: everything in one go.
		let per = motion.reduced ? EPOCHS : 6;
		let last = performance.now();
		while (trainer.epoch() < EPOCHS) {
			await nextFrame();
			if (destroyed) return;
			const now = performance.now();
			if (now - last > 45 && per < 50) per = Math.min(50, per * 2);
			last = now;
			for (let i = 0; i < per && trainer.epoch() < EPOCHS; i++) {
				const h = trainer.step(1);
				const s = pack(h);
				r.snaps.push(s);
				r.loss.push(h.lossHistory[h.lossHistory.length - 1]);
				r.trainAcc.push(h.accHistory[h.accHistory.length - 1]);
				if (r.testAcc) r.testAcc.push(testAccOf(s, r.scale) ?? 0);
			}
			run = { ...r, snaps: r.snaps.slice(), loss: r.loss.slice(), trainAcc: r.trainAcc.slice(), testAcc: r.testAcc ? r.testAcc.slice() : null };
			epoch = r.snaps.length - 1;
		}
		r.done = true;
		run = { ...r, snaps: r.snaps.slice(), loss: r.loss.slice(), trainAcc: r.trainAcc.slice(), testAcc: r.testAcc ? r.testAcc.slice() : null };
		epoch = EPOCHS;
		training = false;
		lab.liveRun = {
			nA: r.nA,
			nB: r.nB,
			testAcc: r.testAcc ? r.testAcc[r.testAcc.length - 1] : null,
			testN: presetFeat ? presetFeat.test[0].length + presetFeat.test[1].length : 0,
			preset: r.preset && flipped === 0 && drawnPreset.length === 0
		};
	}

	// The heavy views (network zoom, KaTeX formulas, map, thumbnails) follow the step at most ~9 times
	// a second; the curves and the step label follow it exactly. Keeps phones smooth while training,
	// playing or scrubbing.
	let viewEpoch = $state(0);
	let veTimer: ReturnType<typeof setTimeout> | undefined;
	let veLast = 0;
	$effect(() => {
		const e = epoch;
		const now = performance.now();
		clearTimeout(veTimer);
		if (now - veLast > 110) {
			veLast = now;
			viewEpoch = e;
		} else
			veTimer = setTimeout(() => {
				veLast = performance.now();
				viewEpoch = e;
			}, 110 - (now - veLast));
	});
	let headCache: Head | null = null;
	const head = $derived.by<Head | null>(() => {
		if (!run) return (headCache = null);
		const e = Math.max(0, Math.min(viewEpoch, run.snaps.length - 1));
		const s = run.snaps[e];
		// same snapshot as before (the run object is replaced every frame while training): keep the
		// same Head object so nothing downstream recomputes
		if (headCache && headCache.W.buffer === s.buffer && headCache.scale === run.scale) return headCache;
		return (headCache = {
			W: s.subarray(0, 512),
			b: s.subarray(512, 514),
			K: 2,
			D: 256,
			scale: run.scale,
			lossHistory: [],
			accHistory: [],
			epochs: e,
			trainableParams: 514
		});
	});
	const trainedSteps = $derived(run ? run.snaps.length - 1 : 0);
	const stale = $derived(!!run && run.snaps.length > 1 && run.key !== dataKey);
	const items = $derived<SetItem[]>(
		examples.map((e) => ({
			id: e.id,
			cls: e.cls,
			input: e.input,
			drawn: e.drawn,
			flipped: e.flipped,
			pA: head && e.feat ? predictHead(head, e.feat).probs[0] : null
		}))
	);
	const testNow = $derived.by(() => {
		if (!run || !run.testAcc || !presetFeat || !head) return null;
		const per = [0, 0];
		const n = [presetFeat.test[0].length, presetFeat.test[1].length];
		for (let k = 0; k < 2; k++) for (const f of presetFeat.test[k]) if (predictHead(head, f).top === k) per[k]++;
		return { acc: (per[0] + per[1]) / (n[0] + n[1]), per, n };
	});

	// ---------------------------------------------------------------- the pad (new examples and tests)
	let strokes = $state<Stroke[]>([]);
	let pad = $state<ReturnType<typeof DrawPad>>();
	let padIn = $state.raw<Float32Array | null>(null);
	let padTrace = $state.raw<Trace | null>(null);
	let padTimer: ReturnType<typeof setTimeout> | undefined;
	function runPad(s: Stroke[], now = false) {
		if (!engine) return;
		clearTimeout(padTimer);
		// while drawing: coarser point list, at most every 100 ms; at the end of a stroke: full detail
		const go = () => {
			const r = engine!.lib.preprocessStrokes(simplifyStrokes(s, now ? 0.004 : 0.012));
			padIn = r.empty ? null : r.input;
			padTrace = padIn ? engine!.model.forward(padIn) : null;
			if (padIn && now) inspect = 'pad';
			else if (!padIn && inspect === 'pad') inspect = examples[0]?.id ?? '';
		};
		if (now) go();
		else padTimer = setTimeout(go, 100);
	}
	const padFeat = $derived(padTrace?.t.flatten?.data ?? null);
	const padNew = $derived(head && padFeat ? predictHead(head, padFeat) : null);
	const padOld = $derived.by(() => {
		if (!padTrace) return null;
		const p = padTrace.t.softmax.data;
		let top = 0;
		for (let i = 1; i < p.length; i++) if (p[i] > p[top]) top = i;
		return { top, p: p[top] };
	});
	const classes = $derived(engine?.spec.classes ?? BASE_CLASSES);

	function addPad(cls: 0 | 1) {
		if (!padIn || !padFeat) return;
		const ex: Ex = { id: `d${++counter}`, cls, input: padIn, feat: Float32Array.from(padFeat), drawn: true, flipped: false };
		if (mode === 'preset') drawnPreset = [...drawnPreset, ex];
		else drawnOwn = [...drawnOwn, ex];
		pad?.clear();
		padIn = null;
		padTrace = null;
		inspect = ex.id;
	}
	function remove(id: string) {
		drawnPreset = drawnPreset.filter((e) => e.id !== id);
		drawnOwn = drawnOwn.filter((e) => e.id !== id);
		if (inspect === id) inspect = examples[0]?.id ?? '';
	}

	// ---------------------------------------------------------------- inspection + linking
	let inspect = $state('p0-0');
	let hover = $state<string | null>(null);
	let current = $state('head');
	const inspected = $derived(inspect === 'pad' ? null : (examples.find((e) => e.id === inspect) ?? examples[0] ?? null));
	const inspectedTrace = $derived.by<Trace | null>(() => {
		if (inspect === 'pad' && padTrace) return padTrace;
		if (!engine || !inspected) return null;
		return engine.model.forward(inspected.input);
	});
	const inspectedLabel = $derived.by(() => {
		if (inspect === 'pad' && padTrace) return t({ en: 'the drawing on your pad', zh: '你画板上的画' });
		if (!inspected) return '–';
		const k = examples.filter((e) => e.cls === inspected.cls).indexOf(inspected) + 1;
		return `${names[inspected.cls]} ${k}${inspected.drawn ? t({ en: ' (yours)', zh: '（你画的）' }) : ''}`;
	});

	// ---------------------------------------------------------------- Try this
	let tryMsg = $state<string | null>(null);
	async function tryOne() {
		setMode('preset');
		flipped = 0;
		nPer = 1;
		await tick();
		await train();
		tryMsg = 'one';
	}
	let cleanAcc = $state<number | null>(null);
	async function tryFlip() {
		setMode('preset');
		nPer = 8;
		flipped = 0;
		await tick();
		// the same 8 + 8 drawings with correct labels, for comparison (a few ms)
		if (presetFeat) {
			const ex = examples;
			const h = trainHead(
				ex.map((e) => e.feat!),
				ex.map((e) => e.cls),
				{ numClasses: 2 }
			);
			cleanAcc = testAccOf(pack(h), h.scale);
		}
		flipped = 2;
		await tick();
		await train();
		tryMsg = 'flip';
	}
	async function tryOwn() {
		setMode('own');
		tryMsg = 'own';
		await tick();
		document.getElementById('tl-name-a')?.focus();
	}
	function tryRewind() {
		if (run && run.snaps.length > 1) epoch = 0;
		tryMsg = 'rewind';
		goTo('tl-train');
	}
	const pctTxt = (v: number | null | undefined) => (v == null ? '–' : (v * 100).toFixed(1) + '%');
</script>

<div class="lab">
	<!-- 1. choose the data -->
	<div class="controls">
		<div class="segmented" role="group" aria-label={t({ en: 'Which two classes', zh: '哪两个类别' })}>
			<button type="button" aria-pressed={mode === 'preset'} disabled={training} onclick={() => setMode('preset')}>{t({ en: 'Ladder vs envelope', zh: '梯子 vs 信封' })}</button>
			<button type="button" aria-pressed={mode === 'own'} disabled={training} onclick={() => setMode('own')}>{t({ en: 'Your own two things', zh: '你自己的两样东西' })}</button>
		</div>
		{#if mode === 'preset'}
			<div class="nsel">
				<span class="muted small" id="tl-n-label">{t({ en: 'Preset drawings per class:', zh: '每类预置图数量：' })}</span>
				<div class="segmented" role="group" aria-labelledby="tl-n-label">
					{#each NS as n (n)}
						<button type="button" class="num" aria-pressed={nPer === n} onclick={() => (nPer = n)}>{n}</button>
					{/each}
				</div>
				{#if flipped > 0}
					<button type="button" class="btn ghost small" onclick={() => (flipped = 0)}>{t({ en: 'Fix the wrong labels', zh: '改正错误标签' })}</button>
				{/if}
			</div>
		{:else}
			<div class="names">
				<label><span class="dot a"></span>{t({ en: 'Class A', zh: '类别 A' })}
					<input id="tl-name-a" type="text" maxlength="24" bind:value={ownNames.a} placeholder={t({ en: 'e.g. cup', zh: '例如：杯子' })} />
				</label>
				<label><span class="dot b"></span>{t({ en: 'Class B', zh: '类别 B' })}
					<input type="text" maxlength="24" bind:value={ownNames.b} placeholder={t({ en: 'e.g. key', zh: '例如：钥匙' })} />
				</label>
			</div>
		{/if}
	</div>

	<div class="cols">
		<div class="left">
			<TrainSet {items} {names} {inspect} bind:hover onselect={(id) => (inspect = id)} onremove={remove} />
			<div class="padbox">
				<DrawPad
					bind:this={pad}
					bind:strokes
					size={220}
					brush={0.055}
					showBrush={false}
					label={{ en: 'Draw a new example or a test drawing', zh: '画一个新样本或测试图' }}
					ondraw={(s) => runPad(s)}
					onchange={(s) => runPad(s, true)}
				/>
				<div class="padside">
					<p class="small strong">{t({ en: 'Draw, then add it to a class — or just test the head on it.', zh: '画好后加到某个类别里，或者只是用它测试输出头。' })}</p>
					<div class="addrow">
						<button type="button" class="btn add a" disabled={!padIn} onclick={() => addPad(0)}><span class="dot a"></span>{t({ en: `Add to ${names[0]}`, zh: `加到「${names[0]}」` })}</button>
						<button type="button" class="btn add b" disabled={!padIn} onclick={() => addPad(1)}><span class="dot b"></span>{t({ en: `Add to ${names[1]}`, zh: `加到「${names[1]}」` })}</button>
					</div>
					{#if padIn}
						<div class="padres">
							{#if padNew}
								<p class="small">{t({ en: 'New head ', zh: '新输出头' })}<span class="muted">{t({ en: `(after step ${viewEpoch})`, zh: `（第 ${viewEpoch} 步之后）` })}</span></p>
								<ProbBars
									items={[
										{ label: names[0], p: padNew.probs[0], id: 0, color: 'var(--div-pos)' },
										{ label: names[1], p: padNew.probs[1], id: 1, color: 'var(--div-neg)' }
									]}
									sort={false}
									chosen={padNew.top}
								/>
							{/if}
							{#if padOld}
								<p class="small muted">
									{t({ en: 'Old 10-class head would say: ', zh: '旧的 10 类输出头会说：' })}<strong>{i18n.lang === 'zh' ? classes[padOld.top].zh : classes[padOld.top].en}</strong>{i18n.lang === 'zh' ? '（' : ' ('}<Num value={padOld.p * 100} unit="%" />{i18n.lang === 'zh' ? '）' : ')'}
									<span class="sw" style:background={categorical(padOld.top, themeState.resolved)}></span>
								</p>
							{/if}
						</div>
					{:else}
						<p class="muted small">{t({ en: 'The pad’s drawing is only a test until you add it: it is not used for training.', zh: '没加进类别之前，画板上的画只用于测试，不参与训练。' })}</p>
					{/if}
				</div>
			</div>
		</div>

		<!-- 2. train -->
		<div class="right" id="tl-train">
			<div class="trainbar">
				<button type="button" class="btn primary big" disabled={!canTrain || training} onclick={train}>
					{training ? t({ en: 'Training…', zh: '训练中…' }) : run && run.snaps.length > 1 && !stale ? t({ en: 'Train again', zh: '再训练一次' }) : t({ en: 'Train the new head', zh: '训练新输出头' })}
				</button>
				<span class="muted small">
					{t({ en: `${counts[0] + counts[1]} examples · 514 trainable parameters · ${EPOCHS} steps of gradient descent`, zh: `${counts[0] + counts[1]} 个样本 · 514 个可训练参数 · ${EPOCHS} 步梯度下降` })}
				</span>
				<ProvenanceBadge mode="live" precision="fp32" {tolerance} against="PyTorch fp32" compact note={{ en: 'Features come from the shipped TinyCNN (JS forward pass, parity-checked against PyTorch). The head is trained right here, in your browser.', zh: '特征来自发布的 TinyCNN（JS 前向计算，与 PyTorch 对齐校验）。输出头就在你的浏览器里训练。' }} />
			</div>
			{#if featStatus === 'busy' || (!featsReady && mode === 'preset')}
				<p class="muted small">{t({ en: 'Computing frozen features for the preset drawings…', zh: '正在为预置图计算冻结特征…' })}</p>
			{:else if featStatus === 'error'}
				<p class="small">{t({ en: 'Could not compute the features. Please reload the page.', zh: '特征计算失败，请刷新页面。' })}</p>
			{/if}
			{#if !canTrain && featsReady}
				<p class="muted small">{t({ en: 'Each class needs at least one example.', zh: '每个类别至少需要一个样本。' })}</p>
			{/if}
			{#if stale}
				<p class="stale small" role="status">{t({ en: 'Your examples changed since this head was trained. The numbers below are for the old head; train again to update them.', zh: '自上次训练以来，你的样本变了。下面的数字仍属于旧的输出头；再训练一次即可更新。' })}</p>
			{/if}
			{#if run}
				<div class="tl-player">
					<StepPlayer count={run.snaps.length} bind:step={epoch} interval={35} label={{ en: 'Replay the training, one step at a time', zh: '逐步回放训练过程' }}>
						{#snippet children()}
							<TrainCharts loss={run!.loss} trainAcc={run!.trainAcc} testAcc={run!.testAcc} total={EPOCHS} bind:epoch busy={training} />
						{/snippet}
					</StepPlayer>
					<p class="steplbl num muted" aria-live="polite">
						{trainedSteps === 0
							? t({ en: 'Step 0: the new head is untrained (small random weights). Press Train.', zh: '第 0 步：新输出头还没训练（权重是很小的随机数）。请点“训练”。' })
							: t({ en: `after step ${epoch} of ${trainedSteps}`, zh: `第 ${epoch} / ${trainedSteps} 步之后` })}
					</p>
				</div>
				<div class="results" aria-live="polite">
					<div class="stat">
						<span class="lbl">{t({ en: `Training loss after step ${epoch}`, zh: `第 ${epoch} 步后的训练损失` })}</span>
						<span class="val num">{run.loss[Math.min(epoch, run.loss.length - 1)].toFixed(3)}</span>
					</div>
					<div class="stat">
						<span class="lbl">{t({ en: 'Training examples right', zh: '训练样本判对' })}</span>
						<span class="val num">{pctTxt(run.trainAcc[Math.min(epoch, run.trainAcc.length - 1)])}</span>
					</div>
					{#if testNow}
						<div class="stat main">
							<span class="lbl">{t({ en: 'Held-out test drawings right', zh: '留出测试图判对' })}</span>
							<span class="val num">{pctTxt(testNow.acc)}</span>
							<span class="sub num">{names[0]} {testNow.per[0]}/{testNow.n[0]} · {names[1]} {testNow.per[1]}/{testNow.n[1]}</span>
						</div>
					{:else if mode === 'own'}
						<p class="small muted own-note">
							{t({
								en: 'There is no held-out test set for your own classes. Test the head honestly: draw new examples on the pad that you did not add to training.',
								zh: '你自己的类别没有留出的测试集。想诚实地测试输出头，就在画板上画几张没有加进训练集的新图。'
							})}
						</p>
					{/if}
				</div>
				{#if testNow && run.preset}
					<p class="small muted">
						{t({
							en: `Live, on 200 held-out drawings (100 per class). The precomputed curves in Part 2 use 3,000 test drawings and Adam instead of plain gradient descent, so expect small differences.`,
							zh: `实时结果，基于 200 张留出图（每类 100 张）。第 2 部分的预计算曲线用的是 3,000 张测试图和 Adam 优化器（而不是普通梯度下降），所以数字会略有不同。`
						})}
					</p>
				{/if}
			{/if}
		</div>
	</div>

	<!-- 3. zoom into the network -->
	<Article part="toy-zoom" />
	<LabNetwork trace={inspectedTrace} {head} {names} epoch={viewEpoch} trained={trainedSteps} {inspectedLabel} bind:current />

	<!-- 4. the feature space -->
	<Article part="toy-map" />
	{#if pca}
		<PcaMap
			{pca}
			items={examples.map((e) => ({ id: e.id, cls: e.cls, feat: e.feat, drawn: e.drawn, flipped: e.flipped }))}
			pad={padFeat}
			{head}
			trained={trainedSteps > 0}
			preset={mode === 'preset'}
			{names}
			testFeats={presetFeat && mode === 'preset' ? [...presetFeat.test[0], ...presetFeat.test[1]] : null}
			testLabels={presetFeat && mode === 'preset' ? [...presetFeat.test[0].map(() => 0), ...presetFeat.test[1].map(() => 1)] : null}
			bind:hover
			onselect={(id) => (inspect = id)}
		/>
	{/if}

	<!-- 5. Try this -->
	<div class="try">
		<h3>{t({ en: 'Try this', zh: '试一试' })}</h3>
		<ol class="tries">
			<li>
				<p><strong>{t({ en: 'One drawing per class.', zh: '每类只给一张图。' })}</strong> {t({ en: 'Predict first: will the head get more or less than 90% of the 200 test drawings right?', zh: '先预测：输出头在 200 张测试图上的正确率会高于还是低于 90%？' })}</p>
				<button type="button" class="btn" onclick={tryOne} disabled={training}>{t({ en: 'Set 1 per class and train', zh: '设为每类 1 张并训练' })}</button>
				{#if tryMsg === 'one' && testNow}
					<p class="ans small">{t({ en: `Result: ${pctTxt(testNow.acc)}. Chance is 50%. Two drawings go a long way because the frozen features already separate ladders from envelopes. Which two matters: over 5 random choices of the single drawing per class, the precomputed average is 93.1% and the worst is 88.3% (Part 2).`, zh: `结果：${pctTxt(testNow.acc)}。瞎猜是 50%。两张图就能走这么远，是因为冻结特征本来就把梯子和信封分开了。抽到哪两张也有影响：预计算中随机换 5 次每类那一张图，平均 93.1%，最差 88.3%（见第 2 部分）。` })}</p>
				{/if}
			</li>
			<li>
				<p><strong>{t({ en: 'Two wrong labels.', zh: '两个错误标签。' })}</strong> {t({ en: 'Put 2 of 8 envelopes into the ladder class on purpose. How much does test accuracy drop, and where do the mislabeled points sit on the map?', zh: '故意把 8 个信封中的 2 个放进梯子类。测试正确率会降多少？标错的点在地图上的什么位置？' })}</p>
				<button type="button" class="btn" onclick={tryFlip} disabled={training}>{t({ en: 'Mislabel 2 and train', zh: '标错 2 个并训练' })}</button>
				{#if tryMsg === 'flip' && testNow}
					<p class="ans small">{t({ en: `Result: ${pctTxt(testNow.acc)} on the test drawings, against ${pctTxt(cleanAcc)} with the same drawings labeled correctly, while ${pctTxt(run?.trainAcc[run.trainAcc.length - 1])} of the training examples (wrong labels included) are “right”. In 256 dimensions even a straight boundary can be tilted to fit 2 odd points out of 16, and that tilt costs accuracy on new drawings. The flipped points (dotted outline) still sit among the envelopes on the map: their features did not change, only their labels. Now pick 32 per class and train again: 2 wrong labels out of 64 are outvoted.`, zh: `结果：测试图正确率 ${pctTxt(testNow.acc)}；同样这些图标对时是 ${pctTxt(cleanAcc)}。而训练样本（包括标错的）有 ${pctTxt(run?.trainAcc[run.trainAcc.length - 1])} 被判“对”。在 256 维里，即使是一条直的分界线，也能倾斜到把 16 个点里的 2 个怪点也拟合进去，而这一倾斜会降低新图上的正确率。地图上，标错的点（虚线描边）依然落在信封堆里：它们的特征没变，变的只是标签。现在选每类 32 张再训练一次：64 个样本里的 2 个错误标签就会被“少数服从多数”压下去。` })}</p>
				{/if}
			</li>
			<li>
				<p><strong>{t({ en: 'Your own two things.', zh: '你自己的两样东西。' })}</strong> {t({ en: 'Name two things TinyCNN never learned (say, cup and key), draw 3 of each, train, then test with new drawings. Where do your points land on the map?', zh: '给两样 TinyCNN 从没学过的东西起名（比如杯子和钥匙），各画 3 张，训练后再画新图测试。你的点落在地图的哪里？' })}</p>
				<button type="button" class="btn" onclick={tryOwn} disabled={training}>{t({ en: 'Switch to your own pair', zh: '切换到自己的一对' })}</button>
			</li>
			<li>
				<p><strong>{t({ en: 'Rewind to step 0.', zh: '倒回第 0 步。' })}</strong> {t({ en: 'Before training, the head’s weights are tiny random numbers. What does it say about your drawings, and where is its boundary?', zh: '训练前，输出头的权重是很小的随机数。它会怎么判断你的画？分界线在哪？' })}</p>
				<button type="button" class="btn" onclick={tryRewind} disabled={training || !run || run.snaps.length < 2}>{t({ en: 'Go to step 0', zh: '跳到第 0 步' })}</button>
				{#if tryMsg === 'rewind'}
					<p class="ans small">{t({ en: 'Every probability is close to 50%, and the boundary on the map is arbitrary. Press Play in the training replay to watch it swing into place.', zh: '每个概率都接近 50%，地图上的分界线也是随意的。在训练回放里按“播放”，看它怎样摆到正确的位置。' })}</p>
				{/if}
			</li>
		</ol>
	</div>
	{#if featMs > 0}
		<p class="muted tiny">{t({ en: `Features for ${presetIn ? presetIn[0].train.length * 2 + presetIn[0].test.length * 2 : 0} preset drawings computed in ${featMs.toFixed(0)} ms (${featWhere === 'worker' ? 'in a background worker' : 'on the main thread'}).`, zh: `${presetIn ? presetIn[0].train.length * 2 + presetIn[0].test.length * 2 : 0} 张预置图的特征计算用时 ${featMs.toFixed(0)} 毫秒（${featWhere === 'worker' ? '后台线程' : '主线程'}）。` })}</p>
	{/if}
</div>

<style>
	.lab {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s4);
	}
	.controls {
		display: flex;
		flex-wrap: wrap;
		gap: 10px 20px;
		align-items: center;
	}
	.nsel {
		display: flex;
		flex-wrap: wrap;
		gap: 6px 10px;
		align-items: center;
	}
	.names {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 16px;
	}
	.names label {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		font-size: var(--fs-sm);
		font-weight: 600;
	}
	.names input {
		width: 10em;
		min-height: 34px;
		padding: 4px 8px;
		border: 1px solid var(--rule-strong);
		border-radius: var(--r1);
		background: var(--surface);
		color: var(--ink);
		font: inherit;
		font-weight: 400;
	}
	.cols {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s5);
		align-items: start;
	}
	@media (min-width: 980px) {
		.cols {
			grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
		}
	}
	.left,
	.right {
		display: grid;
		grid-template-columns: minmax(0, 1fr);
		gap: var(--s3);
		min-width: 0;
	}
	.padbox {
		display: flex;
		flex-wrap: wrap;
		gap: 12px 16px;
		align-items: flex-start;
	}
	.padside {
		flex: 1 1 200px;
		display: grid;
		gap: 8px;
		min-width: 0;
	}
	.addrow {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}
	.add {
		display: inline-flex;
		align-items: center;
		gap: 6px;
	}
	.dot {
		display: inline-block;
		width: 10px;
		height: 10px;
		border-radius: 50%;
		flex: 0 0 auto;
	}
	.dot.a {
		background: var(--div-pos);
	}
	.dot.b {
		background: var(--div-neg);
		border-radius: 2px;
	}
	.sw {
		display: inline-block;
		width: 9px;
		height: 9px;
		border-radius: 2px;
		vertical-align: middle;
		margin-left: 2px;
	}
	.padres {
		display: grid;
		gap: 4px;
	}
	.small {
		font-size: var(--fs-sm);
		margin: 0;
	}
	.tiny {
		font-size: var(--fs-xs);
		margin: 0;
	}
	.strong {
		font-weight: 600;
	}
	.trainbar {
		display: flex;
		flex-wrap: wrap;
		gap: 8px 12px;
		align-items: center;
	}
	.big {
		min-height: 42px;
		padding-inline: 18px;
		font-weight: 650;
	}
	/* the kit player counts steps from 1 ("Step 201 of 201"); here step 0 is the untrained head,
	   so the page shows its own "after step e of 200" line instead */
	.tl-player :global(.pos) {
		display: none;
	}
	.steplbl {
		margin: 4px 0 0;
		font-size: var(--fs-xs);
	}
	.stale {
		padding: 8px 12px;
		border-left: 3px solid var(--illus);
		background: var(--illus-soft);
		border-radius: var(--r1);
	}
	.results {
		display: flex;
		flex-wrap: wrap;
		gap: 10px;
	}
	.stat {
		display: grid;
		gap: 2px;
		padding: 8px 12px;
		border: 1px solid var(--rule);
		border-radius: var(--r2);
		background: var(--surface);
		min-width: 130px;
	}
	.stat.main {
		border-color: var(--accent);
	}
	.stat .lbl {
		font-size: var(--fs-xs);
		color: var(--ink-2);
	}
	.stat .val {
		font-size: var(--fs-xl);
		font-weight: 650;
	}
	.stat .sub {
		font-size: var(--fs-xs);
		color: var(--muted);
	}
	.own-note {
		flex: 1 1 220px;
	}
	.try {
		padding: var(--s4);
		border-radius: var(--r3);
		background: var(--surface-2);
	}
	.try h3 {
		margin: 0 0 8px;
		font-size: var(--fs-lg);
	}
	.tries {
		margin: 0;
		padding-left: 1.2em;
		display: grid;
		gap: 12px;
	}
	.tries p {
		margin: 0 0 6px;
	}
	.ans {
		padding: 8px 12px;
		border-left: 3px solid var(--live);
		background: var(--surface);
		border-radius: var(--r1);
		max-width: var(--measure);
	}
</style>
