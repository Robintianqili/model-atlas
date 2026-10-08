// tinymamba claim tests (design §4.6 "Claims are tests"): every sentence of the lesson and the page that is not computed
// live is asserted here, over every fixture (the lesson's, the adapter's empty / single ? / long, the 6 presets) through
// withFixtures. `atlas check` runs this file (AK408). Run alone: node --test src/lib/explainers/tinymamba/claims.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { withFixtures } from '../../atlas/lesson/fixtures.ts';
import { compileLesson } from '../../atlas/lesson/compile.ts';
import { render } from '../../atlas/lesson/narrate.ts';
import { stepAt } from '../../atlas/runtime/sources.ts';
import { fsFetch } from '../../atlas/core/io.ts';
import { loadPackage } from '../../../../scripts/atlas/load.ts';
import { ALPHABET, ASK, TASK, answersOf, batch, isLetter, textOf } from './task.ts';
import { answers, asks, dec, deltaAt, dims, idsOf, pickCh, T, worked, writeRead, type RT } from './tm.ts';
import { follow, row, stepOf, TRIES } from './lesson.ts';
import { memory, unrolled } from './views/memory/meta.ts';
import { softplus } from './ops/scan.ts';

const SITE = fileURLToPath(new URL('../../../../', import.meta.url));
/** the repo root: the nearest folder above the site with CONTRACTS.md (the main tree or a lane) */
let ROOT = SITE;
while (!existsSync(join(ROOT, 'CONTRACTS.md')) && dirname(ROOT) !== ROOT) ROOT = dirname(ROOT);
const opts = { fetch: fsFetch, base: join(SITE, 'static') };
const pkgP = loadPackage('tinymamba', SITE).then((r) => r.pkg);
const json = (p: string) => JSON.parse(readFileSync(join(SITE, 'static', p), 'utf8'));
const t3 = (rt: RT, id: string) => T(rt, id)!;

test('step "task": the page’s answer key is the training target, and task.ts draws the same rows as data.py', () => {
	for (const e of batch(1, 500)) {
		const a = answersOf(e.ids).asks;
		e.targets.forEach((y, t) => (y >= 0 ? assert.equal(a.find((x) => x.at === t)!.want, y) : assert.notEqual(e.ids[t], ASK)));
		assert.ok(e.ids.filter(isLetter).length >= 1 && e.ids.filter(isLetter).length <= TASK.kMax);
	}
	const ref = JSON.parse(readFileSync(join(ROOT, 'pipeline/tinymamba/data_check.json'), 'utf8')) as { seed: number; examples: { text: string; targets: number[] }[] };
	batch(ref.seed, ref.examples.length).forEach((e, i) => {
		assert.equal(textOf(e.ids), ref.examples[i].text);
		assert.deepEqual(e.targets, ref.examples[i].targets);
	});
});

test('step "vectors": one table row per token of the alphabet, and no table of positions', async () => {
	const spec = json('models/tinymamba/spec.json') as { layers: { id: string; op: string }[]; tensors: { name: string; shape: number[] }[]; alphabet: string };
	assert.equal(spec.alphabet, ALPHABET);
	assert.equal(spec.tensors.find((x) => x.name === 'emb.weight')!.shape[0], ALPHABET.length);
	assert.ok(!spec.layers.some((l) => l.op === 'position_embedding'));
	assert.deepEqual(spec.layers.filter((l) => l.op === 'layernorm').map((l) => l.id), ['B1.norm', 'B2.norm', 'norm']);
});

test('steps "state" and "memory": the state starts at 0 and no token reads a later one (changing later tokens changes nothing before them)', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const ids = idsOf(rt);
		if (ids.length < 2) return;
		await rt.ensure('scan', 'first');
		const h0 = rt.sources.scan.tape!.state(0).h.data as ArrayLike<number>;
		assert.ok(Array.from(h0).every((v) => v === 0));
		const cut = Math.floor(ids.length / 2);
		const before = Float64Array.from(t3(rt, 'probs').data.subarray(0, cut * 10));
		rt.setInput(textOf(ids.map((id, t) => (t < cut ? id : (id + 3) % 10))));
		const after = t3(rt, 'probs').data.subarray(0, cut * 10);
		assert.ok(before.every((v, i) => v === after[i]));
	}, opts);
});

test('steps "delta", "write-read" and "step": Δ = softplus(dt) ≥ 0, A < 0, h = exp(Δ·A)·h_old + Δ·B·u and y = Σ C·h + D·u, on the trace', async () => {
	await withFixtures(await pkgP, (rt) => {
		const { D, N, L } = dims(rt);
		const n = idsOf(rt).length;
		if (!n) return;
		for (let b = 1; b <= L; b++) {
			const w = rt.model!.weights;
			const [u, dt, Bm, Cm, dl, h, y] = ['u', 'dt', 'B', 'C', 'scan#delta', 'scan#h', 'scan'].map((k) => t3(rt, `B${b}.${k}`).data);
			for (let t = 0; t < n; t++)
				for (let c = 0; c < D; c++) {
					assert.ok(dl[t * D + c] >= 0 && Math.abs(dl[t * D + c] - softplus(dt[t * D + c])) < 1e-6);
					let s = 0;
					for (let k = 0; k < N; k++) {
						const A = -Math.exp(w[`B${b}.scan.A_log`].data[c * N + k]);
						assert.ok(A < 0);
						const old = t ? h[((t - 1) * D + c) * N + k] : 0;
						const now = Math.exp(dl[t * D + c] * A) * old + dl[t * D + c] * Bm[t * N + k] * u[t * D + c];
						assert.ok(Math.abs(now - h[(t * D + c) * N + k]) < 1e-5 * Math.max(1, Math.abs(now)));
						s += Cm[t * N + k] * h[(t * D + c) * N + k];
					}
					const want = s + w[`B${b}.scan.D`].data[c] * u[t * D + c];
					assert.ok(Math.abs(want - y[t * D + c]) < 1e-5 * Math.max(1, Math.abs(want)));
				}
		}
	}, opts);
});

test('step "delta" (predict): Δ jumps up at the letters in a few channels (not in all), and stays near 0 at the dots in almost every channel', async () => {
	const f = json('models/tinymamba/facts.json').facts as Record<string, { value: number }>;
	assert.ok(f.deltaLetter1.value > 5 * f.deltaDot1.value, 'over 5,000 new rows');
	await withFixtures(await pkgP, (rt) => {
		const r = row(rt);
		if (!r.k || !r.dots) return;
		const { D } = dims(rt);
		const mean = (c: number, kind: (id: number) => boolean) => {
			const ts = r.ids.flatMap((id, t) => (kind(id) ? [t] : []));
			return ts.reduce((s, t) => s + deltaAt(rt, 1, t, c), 0) / ts.length;
		};
		let jump = 0;
		let near0 = 0;
		let let1 = 0;
		let dot1 = 0;
		for (let c = 0; c < D; c++) {
			const [l, d] = [mean(c, isLetter), mean(c, (id) => id === 0)];
			if (l > 5 * d && l > 0.05) jump++;
			if (d < 0.01) near0++;
			assert.ok(d < 0.05, `channel ${c}: Δ at the dots ${d}`);
			(let1 += l / D), (dot1 += d / D);
		}
		assert.ok(jump >= 2 && jump <= D / 2, `channels where Δ jumps up at the letters: ${jump} of ${D} (a few, not all)`);
		assert.ok(near0 >= 0.75 * D, `channels with Δ near 0 at the dots: ${near0} of ${D}`);
		assert.ok(let1 > 5 * dot1, `letters ${let1} dots ${dot1}`);
		const ch = pickCh(rt, 1);
		const fl = follow(rt);
		assert.equal(fl.ch, ch);
		assert.ok(fl.letDelta > fl.dotDelta && fl.letMove > fl.dotMove, 'the followed channel: letters move it more than dots');
	}, opts);
});

test('step "whole": block 1 sees only the token, so every dot gets the same Δ (and every ? the same Δ, B and C)', async () => {
	await withFixtures(await pkgP, (rt) => {
		const { D, N } = dims(rt);
		const a = asks(rt);
		for (const [kind, ids] of [['dot', 0], ['?', ASK]] as const) {
			const ts = idsOf(rt).flatMap((id, t) => (id === ids ? [t] : []));
			for (const t of ts.slice(1)) {
				assert.deepEqual(a.row('B1.scan#delta', t, D), a.row('B1.scan#delta', ts[0], D), `${kind} at token ${t}: Δ`);
				assert.deepEqual(a.row('B1.C', t, N), a.row('B1.C', ts[0], N), `${kind} at token ${t}: C`);
				assert.deepEqual(a.row('B1.B', t, N), a.row('B1.B', ts[0], N), `${kind} at token ${t}: B`);
			}
		}
	}, opts);
});

test('step "output": the question marks differ only through the state: block 1’s output changes from one ? to the next, and block 2 reads with a different C at each ?', async () => {
	await withFixtures(await pkgP, (rt) => {
		const { D, N } = dims(rt);
		const a = asks(rt);
		for (let i = 1; i < a.at.length; i++) {
			const [p, q] = [a.at[i - 1], a.at[i]];
			const y = [a.row('B1.scan', p, D), a.row('B1.scan', q, D)];
			assert.ok(a.dist(y[0], y[1]) > 0.2 * a.norm(y[0]), `block 1 output at ? ${p} and ${q}: ${a.dist(y[0], y[1])}`);
			const c = [a.row('B2.C', p, N), a.row('B2.C', q, N)];
			assert.ok(a.dist(c[0], c[1]) > 0.3 * a.norm(c[0]), `block 2 C at ? ${p} and ${q}: ${a.dist(c[0], c[1])}`);
		}
	}, opts);
});

test('steps "write-read" and "step": every printed line can be checked by hand, and the result is the model’s new state (to its last printed digit)', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const ids = idsOf(rt);
		if (!ids.length) return;
		await rt.ensure('scan', 'last');
		const tape = rt.sources.scan.tape!;
		const t1 = Math.max(0, ids.findIndex(isLetter));
		const m = tape.frame(t1 + 1).metrics!;
		assert.equal(m.t, t1);
		const s = stepOf(m);
		const P = (k: string) => Number(s.values[k].replace('−', '-'));
		assert.equal(s.values.decay, dec(Math.exp(P('delta') * P('A'))));
		assert.equal(s.values.write, dec(P('delta') * P('B') * P('u')));
		assert.equal(s.values.kept, dec(P('decay') * P('hOld')));
		assert.equal(s.values.hNew, dec(P('kept') + P('write'), 3));
		const unit = 10 ** (Math.floor(Math.log10(Math.abs(m.hNew))) - 2);
		assert.ok(Math.abs(P('hNew') - m.hNew) <= unit, `printed ${s.values.hNew}, the model's ${m.hNew}`);
		const w = writeRead(rt);
		if (w.tq > w.t) assert.equal(dec(worked({ delta: w.d, A: 0, B: w.B, u: w.u, hOld: 0 }).write), s.values.write, 'step 5 prints the same write as step 6');
	}, opts);
});

test('step "whole": in the followed channel each dot moves the state far less than a letter, and at the first ? the state still points the way the last letter left it, with most of its size', async () => {
	await withFixtures(await pkgP, (rt) => {
		const r = row(rt);
		const f = follow(rt);
		if (r.k && r.dots) assert.ok(f.dotMove < f.letMove / 5, `dots ${f.dotMove} letters ${f.letMove}`);
		const tq = r.ids.indexOf(ASK);
		const tl = r.ids.slice(0, Math.max(0, tq)).findLastIndex(isLetter);
		if (tq < 0 || tl < 0) return;
		const { D, N } = dims(rt);
		const h = t3(rt, 'B1.scan#h').data;
		const at = (t: number) => Array.from({ length: N }, (_, n) => h[(t * D + f.ch) * N + n]);
		const [a, b] = [at(tl), at(tq - 1)];
		const dot = (x: number[], y: number[]) => x.reduce((s, v, i) => s + v * y[i], 0);
		const cos = dot(a, b) / Math.sqrt(dot(a, a) * dot(b, b));
		assert.ok(cos > 0.9 && Math.sqrt(dot(b, b) / dot(a, a)) > 0.6, `kept from token ${tl} to ${tq - 1}: cos ${cos}`);
	}, opts);
});

test('step "step": the process numbers are the trace’s, and h_new = decay × h_old + write exactly', async () => {
	await withFixtures(await pkgP, async (rt) => {
		const ids = idsOf(rt);
		if (!ids.length) return;
		await rt.ensure('scan', 'last');
		const tape = rt.sources.scan.tape!;
		const { ch, slot } = tape.frame(0).metrics!;
		const { D, N } = dims(rt);
		for (let i = 1; i < tape.length; i++) {
			const m = tape.frame(i).metrics!;
			assert.equal(m.decay * m.hOld + m.write, m.hNew);
			assert.equal(m.decay, Math.exp(m.delta * m.A));
			assert.ok(Math.abs(m.hNew - t3(rt, 'B1.scan#h').data[(m.t * D + ch) * N + slot]) < 1e-4 * Math.max(1, Math.abs(m.hNew)));
		}
	}, opts);
});

test('step "memory": the unrolled look-back weights give the same output as the scan (y − D·u = Σ_s M[t][s]·u_s)', async () => {
	await withFixtures(await pkgP, (rt) => {
		const { D, L } = dims(rt);
		if (!idsOf(rt).length) return;
		for (let b = 1; b <= L; b++)
			for (const c of [0, pickCh(rt, b)]) {
				const M = unrolled(rt, b, c)!;
				const u = t3(rt, `B${b}.u`).data;
				const y = t3(rt, `B${b}.scan`).data;
				M.forEach((r, t) => {
					const s = r.reduce((a, m, j) => a + m * u[j * D + c], 0) + rt.model!.weights[`B${b}.scan.D`].data[c] * u[t * D + c];
					assert.ok(Math.abs(s - y[t * D + c]) < 1e-4 * Math.max(1, Math.abs(s)), `block ${b} channel ${c} token ${t}`);
					r.forEach((m, j) => j > t && assert.equal(m, 0));
				});
			}
		const m = memory(rt);
		const z = dims(rt);
		assert.equal(m.state, z.L * z.D * z.N);
		assert.equal(m.kv, 2 * z.d * idsOf(rt).length * z.L);
		// step "limits": the state never grows with the row; attention's keys and values do
		assert.equal(memory(rt, 1).state, memory(rt, 1000).state);
		assert.ok(memory(rt, 1000).kv > memory(rt, 999).kv);
	}, opts);
});

test('steps "output" and "limits", try-this cards: the measured accuracy falls with more letters than training had; the six-letter row gets one wrong; the other presets, the far one included, are all right', async () => {
	const f = json('models/tinymamba/facts.json').facts as Record<string, { value: number }>;
	assert.equal(f.kMax.value, TASK.kMax);
	assert.ok(f.accK4.value > f.accK6.value && f.accK6.value > f.accK8.value);
	assert.ok(f.accK1.value > 0.99 && f.accK3.value > 0.97);
	await withFixtures(await pkgP, (rt) => {
		for (const r of TRIES) {
			rt.setInput(r.text);
			assert.ok(answers(rt).some((a) => a.want >= 0 && !a.ok), r.text);
		}
	}, { ...opts, only: ['six'] });
	await withFixtures(await pkgP, (rt) => {
		assert.ok(answers(rt).every((a) => a.ok));
	}, { ...opts, only: ['start', 'one', 'four', 'twice', 'far'] });
	const far = (json('models/tinymamba/presets.json').items as { id: string; text: string }[]).find((p) => p.id === 'far')!.text;
	assert.ok(far.split('b')[1].replace('?', '').length > TASK.bodyMax - 1, 'try-this “far”: more dots after the letter than any training row had');
	assert.ok(f.accGap38.value === 1 && f.accGap30.value === 1, 'one letter is still given back after 30 and 38 dots (misconception card)');
});

test('every narration renders without “…” or NaN on every fixture, in English and Chinese', async () => {
	const pkg = await pkgP;
	const lesson = compileLesson(pkg.lesson, { spec: json('models/tinymamba/spec.json'), ops: pkg.ops, manifest: pkg.manifest } as never);
	await withFixtures(pkg, async (rt) => {
		if (!idsOf(rt).length) return;
		for (const s of lesson.steps) {
			const pos = await stepAt(rt, s);
			for (const lang of ['en', 'zh'] as const) {
				const txt = render({ ...s.src.narration } as never, rt, pos.at, lang).text;
				assert.ok(!/NaN|…|undefined/.test(txt), `${s.id} ${lang}: ${txt}`);
			}
		}
	}, opts);
});
