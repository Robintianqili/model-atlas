// W1-B2 engine tests (design §4.10, §8.2): host + client + engineRunner over a MessageChannel (the real structured clone
// and transfer lists), against localRunner.
//   acceptance 4: frames from the "worker" give the same TapeStore states as localRunner (frame hash and state hash equal)
//                 for a coarse trainer, a fine trainer (record-valued params, prev, grads, batch, acts, Adam moments),
//                 and a kind-C process with track: 'changes', recordDraws, keep and a small keyEvery; also with controls
//                 (set lr, input, reset) applied at the same frames.
//   acceptance 2: the main-thread fallback (no worker, a throwing factory, an error status, a silent worker, an unknown
//                 process, a crash mid-run).
//   transfer rules: every posted typed array has its own exactly sized, once-listed buffer; nothing of the process's
//                 memory is detached; derived fields are never sent but are recomputed on the main thread.
//   also: supersede, calls, pause/resume/step, the hidden-tab pause, maxFps batching, portable traces (explain on a
//         cloned TinyGPT trace).
// Run from site/: node --test src/lib/atlas/engine/engine.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { LiveTape, PackageModules, ProcessDef } from '../types/index.ts';
import { trainingProcess } from '../train/trainer.ts';
import { makeLocalRunner } from '../runtime/local.ts';
import type { Recorder } from '../process/record.ts';
import { walker } from '../process/fixtures/walker.ts';
import { fromParts } from '../graph/executor.ts';
import { CORE_OPS } from '../ops/index.ts';
import { makeEngineRunner, NOTICES, type EngineTape } from './runner.ts';
import { channelWorker, framesSent, type ChannelWorker } from './fixtures/channel.ts';
import { arenaProcess, clsTrainer, DATA } from './fixtures/trainers.ts';
import { registerModules } from './host.ts';
import { FramePacker, checkTransfer, schemaJson } from './protocol.ts';
import { portableTrace } from './trace.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
const trainers = { clsTrainer };
const processes = { walker, arenaProcess };
const mods = { trainers, processes };

function pkgWith(worker?: () => ChannelWorker): PackageModules {
	return {
		slug: 'engine-test',
		manifest: { slug: 'engine-test', type: 'model', kinds: ['train', 'process'], quality: 'draft', devOnly: true, page: { goals: [], prequestion: { id: 'q', question: { en: 'q' }, options: [], answer: 0, explain: { en: 'e' } } }, credits: [] },
		lesson: null,
		ops: [],
		processes: registerModules(mods) as Record<string, ProcessDef<any, any, any>>,
		adapters: {},
		...(worker ? { worker: worker as unknown as () => Worker } : {})
	};
}

type Case = { name: string; process: string; input: unknown; opt: Record<string, unknown>; detail: 'coarse' | 'fine'; recordDraws?: boolean; keyEvery?: number; keep?: { every: number } };
const CASES: Case[] = [
	{ name: 'coarse trainer', process: 'train:cls', input: DATA, opt: { steps: 80, snapshotEvery: 10 }, detail: 'coarse' },
	{ name: 'fine trainer (Adam)', process: 'train:cls', input: DATA, opt: { steps: 12 }, detail: 'fine' },
	{ name: 'fine trainer (SGD)', process: 'train:cls', input: DATA, opt: { steps: 8, optimizer: 'sgd', lr: 0.3 }, detail: 'fine' },
	{ name: 'kind C: changes + draws + keep + keyEvery 8', process: 'walker', input: { start: 3 }, opt: { steps: 300 }, detail: 'coarse', recordDraws: true, keyEvery: 8, keep: { every: 3 } },
	{ name: 'arena views', process: 'arena', input: null, opt: { steps: 40 }, detail: 'coarse', keyEvery: 4 }
];

const open = (r: { open: (o: any, p: PackageModules) => LiveTape }, pkg: PackageModules, c: Case, seed = 7) =>
	r.open({ slug: pkg.slug, source: c.name, process: c.process, input: c.input, opt: c.opt, seed, detail: c.detail, recordDraws: c.recordDraws }, pkg);

const storeOf = (t: LiveTape) => (t as unknown as { store: { frameHash(): string; stateHash(): string; length: number } }).store;

async function both(c: Case, drive: (t: LiveTape) => Promise<void>) {
	let w: ChannelWorker | null = null;
	const pkg = pkgWith(() => (w = channelWorker(mods)));
	const eng = makeEngineRunner({ keyEvery: c.keyEvery, keep: c.keep });
	const loc = makeLocalRunner({ autorun: false, keyEvery: c.keyEvery, keep: c.keep });
	const a = open(eng, pkg, c) as EngineTape;
	const b = open(loc, pkg, c);
	await drive(a);
	await drive(b);
	const res = { remote: storeOf(a), local: storeOf(b), where: a.where, w: w as ChannelWorker | null, a, b };
	return { ...res, done: () => { a.close(); b.close(); eng.dispose(); } };
}

for (const c of CASES) {
	test(`acceptance 4: worker frames = localRunner frames (${c.name})`, async () => {
		const r = await both(c, async (t) => {
			await t.ensure('last');
		});
		try {
			assert.equal(r.where, 'worker');
			assert.ok(r.remote.length > 3, 'frames recorded');
			assert.equal(r.remote.length, r.local.length, 'same length');
			assert.equal(r.remote.frameHash(), r.local.frameHash(), 'frame hash');
			assert.equal(r.remote.stateHash(), r.local.stateHash(), 'state hash');
			assert.deepEqual(r.w!.problems, [], 'transfer rules');
		} finally {
			r.done();
		}
	});
}

test('acceptance 4 with controls: set lr, input and reset at the same frames give equal stores', async () => {
	const c = CASES[0];
	const half = { ...DATA, xy: DATA.xy.slice(0, 64), label: DATA.label.slice(0, 32), n: 32 };
	const drive = async (t: LiveTape) => {
		t.control({ type: 'step', n: 12 });
		await t.ensure(11);
		t.control({ type: 'set', opt: { lr: 0.01 } });
		t.control({ type: 'step', n: 10 });
		await t.ensure(21);
		t.control({ type: 'input', input: half });
		t.control({ type: 'step', n: 10 });
		await t.ensure(31);
		const before = storeOf(t).length;
		assert.ok(before >= 32);
		t.control({ type: 'reset', seed: 11 });
		t.control({ type: 'step', n: 15 });
		await t.ensure(14);
	};
	const r = await both(c, drive);
	try {
		assert.equal(r.remote.length, r.local.length);
		assert.equal(r.remote.frameHash(), r.local.frameHash());
		assert.equal(r.remote.stateHash(), r.local.stateHash());
		assert.equal(r.a.seed, 11);
	} finally {
		r.done();
	}
});

test('on demand: numeric cursors record exactly; query cursors find the same frame as localRunner', async () => {
	const c = CASES[0];
	const r = await both(c, async (t) => {
		assert.equal(await t.ensure(9), 9);
		assert.equal(storeOf(t).length, 10, 'exactly the frames the cursor needs');
	});
	try {
		assert.equal(r.remote.frameHash(), r.local.frameHash());
		const q = { phase: 'snapshot', step: 40 };
		const ia = await r.a.ensure(q);
		const ib = await r.b.ensure(q);
		assert.equal(ia, ib);
		assert.equal(r.a.frame(ia).step, 40);
		assert.ok(r.a.length >= r.b.length && r.a.length - r.b.length <= 64, `overshoot ${r.a.length - r.b.length} frames`);
		for (let i = 0; i <= ia; i += 7) assert.deepEqual(r.a.frame(i).metrics, r.b.frame(i).metrics);
	} finally {
		r.done();
	}
});

test('transfer rules: own exact buffers, listed once, copied at yield time, nothing detached, derived fields not sent', async () => {
	const c = CASES[4];
	const r = await both(c, async (t) => {
		await t.ensure('last');
	});
	try {
		const w = r.w!;
		assert.deepEqual(w.problems, []);
		const frames = framesSent(w);
		assert.ok(frames.length > 0);
		for (const f of frames) assert.ok(!('total' in (f.state as object)), 'derived field sent');
		// the same view twice in a frame is one buffer (shared after the clone)
		const st = r.a.store;
		for (let i = 0; i < st.length; i++) {
			const s = st.state(i) as { view: { data: Float64Array }; twice: { data: Float64Array }; total: number; table: { data: Float64Array } };
			assert.equal(s.view.data, s.twice.data, 'dedupe by identity');
			assert.ok(s.view.data.every((v) => v >= 0 && v < 1), `frame ${i}: values mutated after yield leaked in`);
			assert.equal(s.total, s.table.data.reduce((a, b) => a + b, 0), 'derived field recomputed on the main thread');
		}
		assert.equal(r.remote.frameHash(), r.local.frameHash());
	} finally {
		r.done();
	}
});

test('fine trainer: record-valued fields arrive whole; shared grads re-used within a batch', async () => {
	const c = CASES[1];
	const r = await both(c, async (t) => {
		await t.ensure('last');
	});
	try {
		const st = r.a.store;
		const u = st.find({ phase: 'update', step: 3 });
		const s = st.state(u) as { params: Record<string, { data: Float64Array }>; prev: Record<string, unknown>; grads: Record<string, unknown>; opt: Record<string, { m: unknown; v: unknown }>; batch: Record<string, unknown>; acts: Record<string, unknown> };
		assert.deepEqual(Object.keys(s.params).sort(), ['net.0.bias', 'net.0.weight', 'net.2.bias', 'net.2.weight', 'net.4.bias', 'net.4.weight']);
		assert.deepEqual(Object.keys(s.prev).sort(), Object.keys(s.params).sort());
		assert.deepEqual(Object.keys(s.grads).sort(), Object.keys(s.params).sort());
		assert.ok(s.opt['net.0.weight'].m && s.opt['net.0.weight'].v, 'Adam moments');
		assert.ok(Object.keys(s.batch).length > 0 && Object.keys(s.acts).length > 0, 'batch and activations');
		assert.ok(r.w!.host.stats.reused > 0, 'grads shared by the backward and update frames are sent once');
		assert.ok(r.w!.transferred.buffers > 0);
	} finally {
		r.done();
	}
});

test('FramePacker: copies at add() time; checkTransfer flags shared and oversized buffers', () => {
	const big = new Float64Array(100);
	const p = new FramePacker(['d']);
	const v = big.subarray(10, 14);
	v.set([1, 2, 3, 4]);
	p.add({ phase: 'x', state: { a: { shape: [4], data: v }, b: v, d: 5 } });
	v[0] = 99; // mutated after add: the batch keeps 1
	p.add({ phase: 'x', state: { a: { shape: [4], data: v } } });
	const { frames, transfer } = p.take();
	assert.deepEqual(checkTransfer(frames, transfer), []);
	assert.equal(transfer.length, 2, 'two versions of the view: two buffers');
	assert.equal((frames[0].state as any).a.data[0], 1);
	assert.equal((frames[1].state as any).a.data[0], 99);
	assert.equal((frames[0].state as any).a.data, (frames[0].state as any).b, 'same view, same copy');
	assert.ok(!('d' in (frames[0].state as object)));
	assert.equal(big.buffer.byteLength, 800, 'the source buffer is never transferred');
	const bad = { phase: 'x', state: { a: big.subarray(0, 2) } };
	assert.ok(checkTransfer([bad], [big.buffer]).some((s) => s.includes('whole buffer')));
	assert.ok(checkTransfer([bad], [big.buffer, big.buffer]).some((s) => s.includes('twice')));
	assert.equal(schemaJson(walker.schema).best.derived, true);
});

// ------------------------------------------------------------------ acceptance 2: main-thread fallback
const FALLBACKS: { name: string; worker?: () => ChannelWorker; mid?: boolean; timeout?: number; process?: string }[] = [
	{ name: 'no worker' },
	{
		name: 'factory throws',
		worker: () => {
			throw new Error('blocked');
		}
	},
	{ name: 'error status', worker: () => channelWorker(mods, { statusError: 'module failed' }) },
	{ name: 'silent worker', worker: () => channelWorker(mods, { silent: true }), timeout: 50 },
	{ name: 'unknown process in the worker', worker: () => channelWorker({ processes: { walker } }) },
	{ name: 'crash mid-run', worker: () => channelWorker(mods, { crashAfterFrames: 1, sliceMs: 1 }), mid: true }
];

for (const f of FALLBACKS) {
	test(`acceptance 2: main-thread fallback (${f.name}) records the same frames`, async () => {
		const c = CASES[0];
		const pkg = pkgWith(f.worker);
		const eng = makeEngineRunner({ readyTimeoutMs: f.timeout ?? 2000 });
		const loc = makeLocalRunner({ autorun: false });
		const a = open(eng, pkg, c) as EngineTape;
		const b = open(loc, pkg, c);
		try {
			await a.ensure('last');
			await b.ensure('last');
			assert.equal(a.where, 'main', 'runs on the main thread');
			assert.equal(storeOf(a).frameHash(), storeOf(b).frameHash());
			assert.equal(storeOf(a).length, storeOf(b).length);
			if (f.mid) assert.equal(a.notice, NOTICES.restarted);
			if (f.worker && !f.mid && f.name !== 'unknown process in the worker') assert.equal(eng.client(pkg)!.mode, 'main');
		} finally {
			a.close();
			b.close();
			eng.dispose();
		}
	});
}

test('controls given before a fallback are replayed on the main thread', async () => {
	const c = CASES[0];
	const pkg = pkgWith(() => channelWorker(mods, { silent: true }));
	const eng = makeEngineRunner({ readyTimeoutMs: 40 });
	const a = open(eng, pkg, c) as EngineTape;
	a.control({ type: 'set', opt: { lr: 0.002 } });
	a.control({ type: 'step', n: 5 });
	const loc = makeLocalRunner({ autorun: false });
	const b = open(loc, pkg, c);
	b.control({ type: 'set', opt: { lr: 0.002 } });
	b.control({ type: 'step', n: 5 });
	try {
		await a.ensure(4);
		await b.ensure(4);
		assert.equal(a.where, 'main');
		assert.equal(a.frame(1).metrics?.lr, 0.002);
		assert.equal(storeOf(a).stateHash(), storeOf(b).stateHash());
	} finally {
		a.close();
		b.close();
		eng.dispose();
	}
});

// ------------------------------------------------------------------ live behaviour
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
/** K1/K2: wait for what should happen (up to 10 s), never a fixed sleep: on a loaded machine the worker answers later */
const until = async (ok: () => boolean, ms = 10_000) => {
	const t0 = Date.now();
	while (!ok() && Date.now() - t0 < ms) await sleep(10);
	return ok();
};
/** K2: wait until `read()` stops changing for `quiet` ms (frames already in flight have arrived), up to 10 s */
const settled = async (read: () => number, quiet = 150, ms = 10_000) => {
	const t0 = Date.now();
	let v = read();
	let since = Date.now();
	while (Date.now() - t0 < ms) {
		await sleep(15);
		const w = read();
		if (w !== v) {
			v = w;
			since = Date.now();
		} else if (Date.now() - since >= quiet) break;
	}
	return v;
};

test('autorun, pause, resume and step act on the worker; pause stops within one slice', async () => {
	let w: ChannelWorker | null = null;
	const pkg = pkgWith(() => (w = channelWorker(mods)));
	const eng = makeEngineRunner({ autorun: true });
	const a = open(eng, pkg, { ...CASES[0], opt: { steps: 100000, snapshotEvery: 50 } }) as EngineTape;
	// K1: wait for what should happen, not a fixed 60 ms: on a loaded machine the worker answered later and the test
	// failed at "resumed" although nothing was wrong. K2: the pause too (frames in flight settle first), no fixed sleeps
	try {
		await a.ensure(30);
		assert.equal(a.paused, false);
		a.control({ type: 'pause' });
		assert.equal(a.paused, true);
		// the worker has the pause, and every frame it made before the pause has arrived
		assert.ok(await until(() => w!.host.controls.some((r) => r.type === 'pause')), 'the worker received the pause');
		const pr = w!.host.controls.find((r) => r.type === 'pause')!;
		await until(() => a.length >= (pr.receivedAt ?? 0));
		const n1 = await settled(() => a.length);
		await sleep(100);
		assert.equal(a.length, n1, 'no frames after the pause settled');
		a.control({ type: 'step', n: 3 });
		await until(() => a.length >= n1 + 3);
		assert.equal(await settled(() => a.length), n1 + 3, 'step 3 while paused');
		a.control({ type: 'resume' });
		await until(() => a.length > n1 + 3);
		assert.ok(a.length > n1 + 3, 'resumed');
		a.control({ type: 'set', opt: { lr: 0.001 } });
		const sentAt = a.length;
		const lrNow = () => Array.from(storeOf(a) ? a.metric('lr') : []);
		await until(() => lrNow().includes(0.001));
		const lrs = lrNow();
		const first = lrs.findIndex((v) => v === 0.001);
		assert.ok(first >= sentAt - 1 && first >= 0, 'the new lr shows up');
		assert.ok(w!.host.stats.messages < w!.host.stats.frames, 'frames are batched');
	} finally {
		a.close();
		eng.dispose();
	}
});

test('control log: set takes effect in the first frame produced after it arrives; pause stops before the next frame', async () => {
	let w: ChannelWorker | null = null;
	const pkg = pkgWith(() => (w = channelWorker(mods)));
	const eng = makeEngineRunner();
	const a = open(eng, pkg, { ...CASES[0], opt: { steps: 400, snapshotEvery: 50 } }) as EngineTape;
	try {
		a.control({ type: 'step', n: 10 });
		await a.ensure(9);
		a.control({ type: 'set', opt: { lr: 0.002 } });
		a.control({ type: 'step', n: 5 });
		await a.ensure(14);
		a.control({ type: 'resume' });
		await a.ensure(40);
		a.control({ type: 'pause' });
		// K2: until the worker has the pause and the frames before it have arrived (was a fixed 40 ms)
		const log = w!.host.controls;
		assert.ok(await until(() => log.some((r) => r.type === 'pause')), 'the worker received the pause');
		await until(() => a.length >= (log.find((r) => r.type === 'pause')!.receivedAt ?? 0));
		await settled(() => a.length);
		const set = log.find((r) => r.type === 'set')!;
		assert.equal(set.receivedAt, 10);
		assert.equal(set.appliedAt, 10, 'applied by the very next frame');
		assert.equal(a.frame(10).metrics?.lr, 0.002, 'and that frame trains with the new lr');
		assert.equal(a.frame(9).metrics?.lr, 0.05);
		const pause = log.find((r) => r.type === 'pause')!;
		const n = a.length;
		await sleep(60);
		assert.equal(a.length, n);
		assert.equal(pause.receivedAt, n, 'no frame was produced after the pause arrived');
		const s = (await eng.client(pkg)!.call('stats', null, 's')) as { controls: unknown[]; frames: number };
		assert.ok(s.controls.length >= 2 && s.frames >= n);
	} finally {
		a.close();
		eng.dispose();
	}
});

test('hidden tab: a running source pauses with a notice and resumes', async () => {
	const pkg = pkgWith(() => channelWorker(mods));
	const eng = makeEngineRunner({ autorun: true });
	const a = open(eng, pkg, { ...CASES[0], opt: { steps: 100000, snapshotEvery: 50 } }) as EngineTape & { setHidden(h: boolean): void };
	try {
		await a.ensure(10);
		a.setHidden(true);
		assert.equal(a.notice, NOTICES.hidden);
		// K2: frames already in flight settle first; then none come (were fixed 50 and 60 ms sleeps)
		const n = await settled(() => a.length);
		await sleep(100);
		assert.equal(a.length, n);
		a.setHidden(false);
		assert.equal(a.notice, null);
		assert.ok(await until(() => a.length > n), 'resumes once visible');
	} finally {
		a.close();
		eng.dispose();
	}
});

test('supersede: a newer stream on a channel closes the older tape; a newer call resolves the older with null', async () => {
	const pkg = pkgWith(() => channelWorker(mods));
	const eng = makeEngineRunner();
	const a = open(eng, pkg, CASES[0]);
	const pa = a.ensure('last');
	const b = open(eng, pkg, CASES[0]);
	try {
		const ia = await pa;
		assert.equal(a.done, true, 'the older tape is closed');
		assert.ok(ia >= -1);
		await b.ensure('last');
		assert.ok(b.length > 10);
		const client = eng.client(pkg)!;
		const p1 = client.call('ping', null, 'c');
		const p2 = client.call('ping', null, 'c');
		assert.equal(await p1, null);
		assert.deepEqual(((await p2) as { processes: string[] }).processes.sort(), ['arena', 'train:cls', 'walker']);
		await assert.rejects(client.call('nope'), /unknown-method/);
	} finally {
		a.close();
		b.close();
		eng.dispose();
	}
});

test('e2e step cap: capSteps (or ?atlas-e2e=1) caps opt.steps of training sources only', async () => {
	const pkg = pkgWith(() => channelWorker(mods));
	const eng = makeEngineRunner({ capSteps: 20 });
	const a = open(eng, pkg, { ...CASES[0], opt: { steps: 500, snapshotEvery: 10 } }) as EngineTape;
	const k = open(eng, pkg, CASES[3]) as EngineTape;
	try {
		await a.ensure('last');
		assert.equal(a.frame(a.length - 1).step, 20);
		await k.ensure('last');
		assert.ok(k.length > 50, 'the kind-C process ran to its own end');
		assert.equal((k.opt as { steps: number }).steps, 300, 'processes are not capped');
	} finally {
		a.close();
		k.close();
		eng.dispose();
	}
});

test('a process error ends the tape with the message', async () => {
	const boom: ProcessDef<null, { n: number }, { x: number }> = {
		id: 'boom',
		kind: 'algorithm',
		title: { en: 'Boom', zh: '出错' },
		phases: { p: { label: { en: 'p', zh: 'p' }, kind: 'other' } },
		schema: { x: { kind: 'scalar', label: { en: 'x', zh: 'x' } } },
		defaults: { n: 3 },
		*run() {
			yield { phase: 'p', state: { x: 1 } };
			throw new Error('kaput');
		}
	};
	const m = { processes: { boom } };
	const pkg = { ...pkgWith(() => channelWorker(m)), processes: registerModules(m) as Record<string, ProcessDef<any, any, any>> };
	const eng = makeEngineRunner();
	const a = open(eng, pkg, { name: 'b', process: 'boom', input: null, opt: {}, detail: 'coarse' }) as EngineTape;
	try {
		const i = await a.ensure('last');
		assert.equal(i, 0);
		assert.equal(a.error, 'kaput');
		assert.equal(a.done, true);
		const loc = makeLocalRunner({ autorun: false }).open({ slug: 'x', source: 'b', process: 'boom', input: null, opt: {}, seed: 7, detail: 'coarse' }, pkg) as Recorder;
		await loc.ensure('last');
		assert.equal(loc.error, 'kaput', 'the local recorder reports the same error');
	} finally {
		a.close();
		eng.dispose();
	}
});

// ------------------------------------------------------------------ integration 1a carry-over: traces crossing threads
test('portableTrace: a cloned TinyGPT trace still explains tok_emb (ids folded into trace.t)', () => {
	const dir = join(SITE, 'static', 'models', 'tinygpt');
	const spec = JSON.parse(readFileSync(join(dir, 'spec.json'), 'utf8'));
	const b = readFileSync(join(dir, 'weights.bin'));
	const model = fromParts(spec, b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer, CORE_OPS);
	const ids = Int32Array.from([5, 77, 300, 12]);
	const trace = model.forward({ input: { shape: [4], data: ids, dtype: 'i32' } });
	const want = model.explain(trace, 'tok_emb', [2, 7]);
	const lost = structuredClone(trace);
	assert.throws(() => model.explain(lost, 'tok_emb', [2, 7]), 'a plain clone has lost the ids');
	const sent = structuredClone(portableTrace(trace));
	assert.deepEqual(sent.order, trace.order);
	const got = model.explain(sent, 'tok_emb', [2, 7]);
	assert.equal(got.result, want.result);
	assert.equal(got.traced, want.traced);
	assert.deepEqual(got.values, want.values);
});
