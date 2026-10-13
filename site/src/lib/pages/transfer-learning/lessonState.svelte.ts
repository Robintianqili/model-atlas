/**
 * State of the guided lesson ("Step by step"). It is separate from the explore views (Part 1–3),
 * so the lesson never moves them and they never move the lesson. Everything here is computed in
 * the browser from the shipped TinyCNN (live), except what the stages label as precomputed.
 *
 *   L.input / L.trace   the learner's drawing (a held-out test drawing or their own) through TinyCNN
 *   L.run               a new 2-way head trained on 8 ladders + 8 envelopes (seed-0 presets), 200 steps,
 *                       every snapshot kept, test accuracy on the 200 held-out drawings at every step
 *   L.headStep          which snapshot the side panel uses for "your drawing" (follows the lesson step)
 */
import type { Stroke } from '$lib/kit/draw';
import type { Text } from '$lib/i18n/i18n.svelte';
import type { Trace } from '$lib/models/tinycnn';
import { createHeadTrainer, predictHead, type Head } from '$lib/models/tinycnn_finetune';
import { decodeB64, simplifyStrokes, type FinetuneJson, type HeldoutJson, type PcaJson } from './data';
import type { Engine } from './engine';
import { presetFeatures, type PresetFeats } from './presetFeats';
import type { CurveSel } from './state.svelte';

/** preset drawings per class used to train the lesson's head (the seed-0 subset, as in Part 1's default) */
export const LESSON_N = 8;
/** gradient-descent steps (full batch), as in Part 1 */
export const LESSON_STEPS = 200;
/** createHeadTrainer's defaults, shown in step 5's update formula */
export const LR = 0.5;
export const L2 = 1e-4;

export type Source = { kind: 'test'; cls: 0 | 1; i: number } | { kind: 'drawn' };

export interface LessonRun {
	/** W (512) + b (2) after each step; index 0 = the untrained head */
	snaps: Float32Array[];
	loss: number[];
	trainAcc: number[];
	testAcc: number[];
	/** test drawings right per class [ladder, envelope] after each step */
	testPer: [number, number][];
	testN: [number, number];
	scale: number;
	feats: Float32Array[];
	labels: (0 | 1)[];
	inputs: Float32Array[];
}

export interface TrailItem {
	n: number;
	feat: Float32Array;
	input: Float32Array;
	label: Text;
}

const CLS = [
	{ en: 'ladder', zh: '梯子' },
	{ en: 'envelope', zh: '信封' }
] as const;
export const clsName = (k: 0 | 1) => CLS[k];

function pack(h: Head): Float32Array {
	const s = new Float32Array(514);
	s.set(h.W, 0);
	s.set(h.b, 512);
	return s;
}

/** the head's decision on many feature vectors, from a packed snapshot (margin z_A − z_B) */
function rightPerClass(snap: Float32Array, scale: number, test: Float32Array[][]): [number, number] {
	const per: [number, number] = [0, 0];
	for (let k = 0; k < 2; k++)
		for (const f of test[k]) {
			let m = snap[512] - snap[513];
			for (let d = 0; d < 256; d++) m += ((snap[d] - snap[256 + d]) * f[d]) / scale;
			if ((m >= 0 ? 0 : 1) === k) per[k]++;
		}
	return per;
}

export function trainLessonRun(p: PresetFeats): LessonRun {
	const feats = [...p.feats.train[0].slice(0, LESSON_N), ...p.feats.train[1].slice(0, LESSON_N)];
	const inputs = [...p.inputs.train[0].slice(0, LESSON_N), ...p.inputs.train[1].slice(0, LESSON_N)];
	const labels: (0 | 1)[] = [...feats.slice(0, LESSON_N).map(() => 0 as const), ...feats.slice(LESSON_N).map(() => 1 as const)];
	const trainer = createHeadTrainer(feats, labels, { numClasses: 2 });
	let h = trainer.head();
	const testN: [number, number] = [p.feats.test[0].length, p.feats.test[1].length];
	const run: LessonRun = { snaps: [], loss: [], trainAcc: [], testAcc: [], testPer: [], testN, scale: h.scale, feats, labels, inputs };
	const record = (hh: Head) => {
		const s = pack(hh);
		const per = rightPerClass(s, hh.scale, p.feats.test);
		run.snaps.push(s);
		run.loss.push(hh.lossHistory[hh.lossHistory.length - 1]);
		run.trainAcc.push(hh.accHistory[hh.accHistory.length - 1]);
		run.testPer.push(per);
		run.testAcc.push((per[0] + per[1]) / (testN[0] + testN[1]));
	};
	record(h);
	for (let e = 0; e < LESSON_STEPS; e++) {
		h = trainer.step(1);
		record(h);
	}
	return run;
}

/** a Head view of snapshot e (shares the snapshot's memory) */
export function headOf(run: LessonRun, e: number): Head {
	const s = run.snaps[Math.max(0, Math.min(run.snaps.length - 1, e))];
	return { W: s.subarray(0, 512), b: s.subarray(512, 514), K: 2, D: 256, scale: run.scale, lossHistory: [], accHistory: [], epochs: e, trainableParams: 514 };
}

/**
 * Step 5: one step of full-batch gradient descent, by hand, for one training example.
 * Everything in float64 from the stored snapshots; `check` is the largest difference between this
 * hand-computed update and the trainer's own weights after step 1.
 */
export interface StepDetail {
	ex: number;
	y: 0 | 1;
	fs: Float64Array;
	sum: [number, number];
	b: [number, number];
	z: [number, number];
	p: [number, number];
	loss: number;
	dz: [number, number];
	/** this example's gradient for W (512) */
	gEx: Float32Array;
	/** the gradient averaged over all training examples (512), and for b */
	gMean: Float32Array;
	gbMean: [number, number];
	/** W after the update minus W before (512) */
	dW: Float32Array;
	/** the feature followed in the update formula: this example's largest feature */
	d: number;
	w0: number;
	w1: number;
	check: number;
	/** this example after the step */
	p1: number;
	loss1: number;
}

export function stepDetail(run: LessonRun, ex: number): StepDetail {
	const s0 = run.snaps[0];
	const s1 = run.snaps[1];
	const n = run.feats.length;
	const scale = run.scale;
	const f = run.feats[ex];
	const y = run.labels[ex];
	const fs = Float64Array.from(f, (v) => v / scale);
	const sum: [number, number] = [0, 0];
	for (let k = 0; k < 2; k++) for (let d = 0; d < 256; d++) sum[k] += s0[k * 256 + d] * fs[d];
	const b: [number, number] = [s0[512], s0[513]];
	const z: [number, number] = [sum[0] + b[0], sum[1] + b[1]];
	const m = Math.max(z[0], z[1]);
	const e0 = Math.exp(z[0] - m);
	const e1 = Math.exp(z[1] - m);
	const p: [number, number] = [e0 / (e0 + e1), e1 / (e0 + e1)];
	const loss = -Math.log(p[y]);
	const dz: [number, number] = [p[0] - (y === 0 ? 1 : 0), p[1] - (y === 1 ? 1 : 0)];
	const gEx = new Float32Array(512);
	for (let k = 0; k < 2; k++) for (let d = 0; d < 256; d++) gEx[k * 256 + d] = dz[k] * fs[d];
	// the full batch: every example's gradient, averaged
	const gSum = new Float64Array(512);
	const gb: [number, number] = [0, 0];
	for (let i = 0; i < n; i++) {
		const fi = run.feats[i];
		let za = s0[512];
		let zb = s0[513];
		for (let d = 0; d < 256; d++) {
			za += (s0[d] * fi[d]) / scale;
			zb += (s0[256 + d] * fi[d]) / scale;
		}
		const mm = Math.max(za, zb);
		const ea = Math.exp(za - mm);
		const eb = Math.exp(zb - mm);
		const g = [ea / (ea + eb) - (run.labels[i] === 0 ? 1 : 0), eb / (ea + eb) - (run.labels[i] === 1 ? 1 : 0)];
		gb[0] += g[0];
		gb[1] += g[1];
		for (let k = 0; k < 2; k++) for (let d = 0; d < 256; d++) gSum[k * 256 + d] += (g[k] * fi[d]) / scale;
	}
	const gMean = Float32Array.from(gSum, (v) => v / n);
	const dW = new Float32Array(512);
	let check = 0;
	for (let j = 0; j < 512; j++) {
		const w1 = s0[j] - LR * (gSum[j] / n + L2 * s0[j]);
		dW[j] = w1 - s0[j];
		check = Math.max(check, Math.abs(w1 - s1[j]));
	}
	for (let k = 0; k < 2; k++) check = Math.max(check, Math.abs(s0[512 + k] - (LR * gb[k]) / n - s1[512 + k]));
	let d = 0;
	for (let i = 1; i < 256; i++) if (fs[i] > fs[d]) d = i;
	const j = y * 256 + d;
	const after = predictHead(headOf(run, 1), f);
	return {
		ex,
		y,
		fs,
		sum,
		b,
		z,
		p,
		loss,
		dz,
		gEx,
		gMean,
		gbMean: [gb[0] / n, gb[1] / n],
		dW,
		d,
		w0: s0[j],
		w1: s1[j],
		check,
		p1: after.probs[y],
		loss1: -Math.log(after.probs[y])
	};
}

/** Euclidean distance in the full 256-number feature space */
function dist(a: ArrayLike<number>, b: ArrayLike<number>): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) {
		const v = a[i] - b[i];
		s += v * v;
	}
	return Math.sqrt(s);
}

export class LessonState {
	engine = $state.raw<Engine | null>(null);
	heldout = $state.raw<HeldoutJson | null>(null);
	pca = $state.raw<PcaJson | null>(null);
	finetune = $state.raw<FinetuneJson | null>(null);
	preset = $state.raw<PresetFeats | null>(null);
	presetError = $state(false);

	/** the default is ladder test drawing 3, the hook's typical ladder (its top guess, house, is the most common one) */
	source = $state<Source>({ kind: 'test', cls: 0, i: 2 });
	drawnInput = $state.raw<Float32Array | null>(null);
	/** the lesson pad's strokes (kept here, so the input sheet can close and reopen with the drawing) */
	strokes = $state.raw<Stroke[]>([]);
	trail = $state.raw<TrailItem[]>([]);
	#trailN = 0;

	/** snapshot used for "your drawing" in the side panel */
	headStep = $state(0);
	/** step 6's replay position */
	replay = $state(0);
	/** step 5's example: index into run.feats (0 = ladder 1, LESSON_N = envelope 1) */
	example = $state(0);
	/** step 7's chart selection (the explore chart keeps its own) */
	curve = $state<CurveSel>({ part2N: 1, part2Zoom: false, part2Seeds: false });

	testCount = $derived(this.heldout ? this.heldout.classes[0].test.length : 100);

	input = $derived.by<Float32Array | null>(() => {
		const s = this.source;
		if (s.kind === 'drawn') return this.drawnInput;
		const h = this.heldout;
		return h ? decodeB64(h.classes[s.cls].test[s.i]) : null;
	});
	trace = $derived.by<Trace | null>(() => (this.engine && this.input ? this.engine.model.forward(this.input) : null));
	feat = $derived(this.trace?.t.flatten.data ?? null);
	/** the old 10-class head's answer */
	old = $derived.by(() => {
		const p = this.trace?.t.softmax.data;
		if (!p) return null;
		let top = 0;
		for (let i = 1; i < p.length; i++) if (p[i] > p[top]) top = i;
		return { probs: p, top, p: p[top] };
	});
	run = $derived.by<LessonRun | null>(() => (this.preset ? trainLessonRun(this.preset) : null));
	/** the new head's answer for the learner's drawing at `headStep` */
	fresh = $derived.by(() => (this.run && this.feat ? predictHead(headOf(this.run, this.headStep), this.feat) : null));
	detail = $derived.by<StepDetail | null>(() => (this.run ? stepDetail(this.run, this.example) : null));

	/** the 3 drawings closest to the learner's in all 256 features, among the 264 shipped ladders and envelopes */
	neighbours = $derived.by(() => {
		const p = this.preset;
		const f = this.feat;
		if (!p || !f) return [];
		const src = this.source;
		const all: { cls: 0 | 1; split: 'train' | 'test'; i: number; d: number; input: Float32Array }[] = [];
		for (const split of ['train', 'test'] as const)
			for (const k of [0, 1] as const)
				p.feats[split][k].forEach((g, i) => {
					// a test drawing is not its own neighbour
					if (split === 'test' && src.kind === 'test' && src.cls === k && src.i === i) return;
					all.push({ cls: k, split, i, d: dist(f, g), input: p.inputs[split][k][i] });
				});
		all.sort((a, b) => a.d - b.d);
		return all.slice(0, 3);
	});

	init(engine: Engine, heldout: HeldoutJson, pca: PcaJson, finetune: FinetuneJson) {
		this.engine = engine;
		this.heldout = heldout;
		this.pca = pca;
		this.finetune = finetune;
		presetFeatures(engine, heldout).then(
			(p) => (this.preset = p),
			() => (this.presetError = true)
		);
	}

	label(): Text {
		const s = this.source;
		if (s.kind === 'drawn') return { en: 'your drawing', zh: '你画的' };
		return { en: `${CLS[s.cls].en} test drawing ${s.i + 1}`, zh: `${CLS[s.cls].zh}测试图 ${s.i + 1}` };
	}

	/** keep the current drawing as a numbered mark on the map before it is replaced */
	#commit() {
		const f = this.feat;
		const x = this.input;
		if (!f || !x) return;
		const last = this.trail[this.trail.length - 1];
		if (last && last.input === x) return;
		this.trail = [...this.trail, { n: ++this.#trailN, feat: Float32Array.from(f), input: x, label: this.label() }].slice(-5);
	}

	pickTest(cls: 0 | 1, i?: number) {
		const n = this.testCount;
		const cur = this.source;
		// default: test drawing 3 of each class, the hook's typical one (its top guess is the class's most
		// common guess over all 1,500 test drawings: house for the ladder, fish for the envelope)
		const j = i ?? (cur.kind === 'test' && cur.cls === cls ? cur.i : 2);
		const next: Source = { kind: 'test', cls, i: ((j % n) + n) % n };
		if (cur.kind === next.kind && cur.cls === next.cls && cur.i === next.i) return;
		this.#commit();
		this.source = next;
	}
	stepTest(d: number) {
		const s = this.source;
		if (s.kind !== 'test') return;
		this.pickTest(s.cls, s.i + d);
	}

	/** the pad, each frame while drawing (coarse) and at the end of each stroke (full detail) */
	#padTimer: ReturnType<typeof setTimeout> | undefined;
	drawing(strokes: Stroke[], final = false) {
		if (!this.engine) return;
		if (this.source.kind === 'test' && strokes.length) {
			this.#commit();
			this.source = { kind: 'drawn' };
		}
		clearTimeout(this.#padTimer);
		const go = () => {
			if (!strokes.length) {
				// cleared: keep the finished drawing on the map
				if (this.drawnInput) this.#commit();
				this.drawnInput = null;
				return;
			}
			const r = this.engine!.lib.preprocessStrokes(simplifyStrokes(strokes, final ? 0.004 : 0.012));
			this.drawnInput = r.empty ? null : r.input;
		};
		if (final) go();
		else this.#padTimer = setTimeout(go, 90);
	}
}
