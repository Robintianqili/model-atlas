// TrainSession: reactive state and controls for one live train (or process) source, for train-controls, loss-curve and
// pages that drive training. It wraps any LiveTape (engineRunner's EngineTape, localRunner's recorder) and reads its
// latest frame on every (animation-frame throttled) update. Nothing here computes training numbers: step, loss and lr
// are read from the recorded frames, so what the controls show is what ran.
//
//   const s = new TrainSession(rt.sources.train.tape);   // or s.bind(tape) later
//   s.play(); s.pause(); s.stepBy(10); s.setLr(0.03); s.reset(); s.runTo(300)
//   s.step, s.loss, s.lr, s.length, s.paused, s.done, s.rate (steps per second), s.where, s.notice, s.error
import type { I18n } from '../../types/core.ts';
import type { Cursor, LiveTape, Tape } from '../../types/process.ts';

type Extra = { paused?: boolean; where?: string; notice?: I18n | null; error?: string | null; truncated?: boolean; stats?: { frames: number; messages: number } };

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

function lastNum(t: Tape, name: string): number {
	for (let i = t.length - 1; i >= 0 && i >= t.length - 64; i--) {
		const v = t.frame(i).metrics?.[name];
		if (typeof v === 'number') return v;
	}
	return NaN;
}

export class TrainSession {
	tape = $state.raw<LiveTape | null>(null);
	/** bumps on every update (read it to re-derive anything from the tape) */
	version = $state(0);
	length = $state(0);
	step = $state(NaN);
	loss = $state(NaN);
	lr = $state(NaN);
	paused = $state(true);
	done = $state(false);
	/** training steps per second over the last second of updates (0 while paused) */
	rate = $state(0);
	where = $state<string>('main');
	notice = $state.raw<I18n | null>(null);
	error = $state<string | null>(null);
	diverged = $state(false);

	private unsub: (() => void) | null = null;
	private samples: { t: number; step: number }[] = [];
	// plain mirrors: methods may run inside an $effect, so they never read the $state fields they write
	private cur: LiveTape | null = null;
	private v = 0;
	private lastRate = 0;
	private isPaused = true;

	constructor(tape?: LiveTape | Tape | null) {
		if (tape) this.bind(tape);
	}

	/** Follow another tape (a source reopened after new input, a reset that replaced the tape). */
	bind(tape: LiveTape | Tape | null) {
		if ((tape as LiveTape | null) === this.cur) return;
		this.unsub?.();
		this.unsub = null;
		this.samples = [];
		this.cur = (tape as LiveTape | null) ?? null;
		this.tape = this.cur;
		const live = tape as LiveTape | null;
		if (live && typeof live.subscribe === 'function') this.unsub = live.subscribe(() => this.refresh());
		this.refresh();
	}

	/** Re-read the tape (called on every throttled tape update). */
	refresh() {
		const t = this.cur;
		const x = (t ?? {}) as Extra;
		this.length = t?.length ?? 0;
		this.version = ++this.v;
		if (!t || t.length === 0) {
			this.step = NaN;
			this.loss = NaN;
			this.lr = NaN;
		} else {
			const f = t.frame(t.length - 1);
			this.step = typeof f.step === 'number' ? f.step : lastNum(t, 'step');
			this.loss = lastNum(t, 'loss');
			this.lr = lastNum(t, 'lr');
			this.diverged = f.metrics?.diverged === 1;
		}
		this.isPaused = x.paused ?? true;
		this.paused = this.isPaused;
		const doneNow = (t as LiveTape | null)?.done ?? true;
		this.done = doneNow;
		this.where = x.where ?? 'main';
		this.notice = x.notice ?? null;
		this.error = x.error ?? null;
		const ts = now();
		if (Number.isFinite(this.step)) this.samples.push({ t: ts, step: this.step });
		while (this.samples.length > 2 && ts - this.samples[0].t > 1000) this.samples.shift();
		const a = this.samples[0];
		const b = this.samples[this.samples.length - 1];
		const idle = this.isPaused || doneNow;
		this.lastRate = a && b && b.t - a.t > 150 && !idle ? Math.max(0, ((b.step - a.step) / (b.t - a.t)) * 1000) : idle ? 0 : this.lastRate;
		this.rate = this.lastRate;
	}

	private send(c: Parameters<LiveTape['control']>[0]) {
		const t = this.cur;
		if (!t || typeof t.control !== 'function') return;
		t.control(c);
		this.refresh();
	}

	play() {
		this.send({ type: 'resume' });
	}
	pause() {
		this.send({ type: 'pause' });
	}
	toggle() {
		if (this.isPaused) this.play();
		else this.pause();
	}
	/** n more frames (also while paused) */
	stepBy(n = 1) {
		this.send({ type: 'step', n });
	}
	/** a fresh run from new weights (same seed unless given) */
	reset(seed?: number) {
		this.samples = [];
		this.send(seed === undefined ? { type: 'reset' } : { type: 'reset', seed });
	}
	/** takes effect at the next training step (the step being computed keeps its lr) */
	setLr(lr: number) {
		this.send({ type: 'set', opt: { lr } });
	}
	set(opt: Record<string, unknown>) {
		this.send({ type: 'set', opt });
	}
	setInput(input: unknown) {
		this.send({ type: 'input', input });
	}
	/** Run until a frame with this training step exists (or the run ends); resolves with its index. */
	runTo(step: number): Promise<number> {
		return this.ensure({ step });
	}
	ensure(c: Cursor): Promise<number> {
		const t = this.cur;
		if (!t || typeof t.ensure !== 'function') return Promise.resolve(-1);
		return t.ensure(c);
	}
	/** A metric over the whole tape (NaN where a frame does not record it). */
	metric(name: string): Float64Array {
		return this.cur ? this.cur.metric(name) : new Float64Array(0);
	}

	close() {
		this.unsub?.();
		this.unsub = null;
	}
}
