// DevKit K4: the kit fixes from batches 7 and 8 (LOG.md "K4"). One test per fix with kit code.
// Run from site/:  node --test src/lib/atlas/integration/k4.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { At, Frame, StepDef, Tensor, TrainOpt, TrainState, TrainerDef } from '../types/index.ts';
import { alignTex, breakSum, texValues, texWidth } from '../ui/views/kit.parts/tex.ts';
import { axisNames, gridRows, keyTexts, nameText, tokenTexts, unitWhere, type Props } from '../ui/views/kit.parts/stage.ts';
import { updateDigits } from '../ui/views/train-step/meta.ts';
import { extraEllipses, formatValue, gapMarks, FORMATS } from '../lesson/narrate.ts';
import { fmtSci, fmtValue } from '../core/format.ts';
import { compileStep } from '../lesson/compile.ts';
import { clamp, maximum, minimum, mul, sum } from '../autograd/ops.ts';
import { grad, input, param, type Var } from '../autograd/var.ts';
import { trainingProcess } from '../train/trainer.ts';
import { lastParams, updateOf } from '../train/helpers.ts';
import { findRepoRoot } from '../process/fixtures/paths.ts';
import { runProcess } from '../../../../tests/atlas/autograd/tape-check.ts';

// ------------------------------------------------------------------ formulas
test('alignTex: a first right-hand side of three long terms no longer loops (Array.map passed the index as per)', () => {
	const tex = 'p(y) = {{a}} \\cdot {{b}} + {{c}} \\cdot {{d}} + {{e}} \\cdot {{f}} = {{s}} = {{y}}';
	const out = alignTex(tex);
	assert.match(out, /^\\begin\{aligned\}/);
	assert.match(out, /\\\\ &\\quad \+/, 'the long first right-hand side is broken after its second term');
	for (const n of ['a', 'b', 'c', 'd', 'e', 'f', 's', 'y']) assert.ok(out.includes(`{{${n}}}`), n);
	// as map callbacks they never get an index as an option
	assert.equal(['x = a + b + c = y'].map(alignTex as (s: string) => string).length, 1);
	assert.equal(breakSum('{{a}} \\cdot {{b}} + {{c}} \\cdot {{d}} + {{e}} \\cdot {{f}}', 0), breakSum('{{a}} \\cdot {{b}} + {{c}} \\cdot {{d}} + {{e}} \\cdot {{f}}'));
});

test('alignTex narrow: a long two-term line (tinymoe step 5) breaks after its +; wide cards keep it on one line', () => {
	const tex = 'y_t = g_a E_a(x_t) + g_b E_b(x_t) = {{ga}} \\times {{xa}} + {{gb}} \\times {{xb}} = {{y}}';
	assert.ok(!alignTex(tex).includes('&\\quad +'), 'wide: one line');
	const narrow = alignTex(tex, 3, { narrow: true });
	assert.match(narrow, /\{\{ga\}\} \\times \{\{xa\}\} \\\\ &\\quad \+ \{\{gb\}\} \\times \{\{xb\}\}/);
	assert.ok(texWidth('{{a}} \\times {{b}}') < 16);
});

test('texValues: a short exact value prints as itself (0.25, not 0.250); measured numbers keep their digits', () => {
	const v = texValues({ s: 0.25, h: 0.5, n: 4, x: 27.3721, r: 0.125 });
	assert.equal(v.s, '0.25');
	assert.equal(v.h, '0.5');
	assert.equal(v.n, '4');
	assert.equal(v.x, 27.3721, 'a measured number stays a number (the card formats it)');
	assert.equal(v.r, 0.125, 'three significant digits print the same either way');
	assert.equal(texValues({ x: 0.25 }, 1).x, 0.25, 'digits 1: no exact form');
});

test('train-step: the update shows enough digits for old and new to differ (props.digits overrides)', () => {
	assert.equal(updateDigits({ old: 0.5123, now: 0.5117 }), 4);
	assert.equal(updateDigits({ old: 0.5, now: 0.4 }), 3);
	assert.equal(updateDigits({ old: 0.5, now: 0.4 }, 5), 5);
	assert.equal(updateDigits({ old: 1, now: 1 }), 3);
});

// ------------------------------------------------------------------ names of rows, keys and axes
const props = (p: Record<string, unknown>, pre: Record<string, unknown> | null = null, spec: Record<string, unknown> | null = null): Props =>
	({ rt: { pre, spec } as never, source: 'fwd', at: null as never, focus: [], sel: {} as never, provenance: 'live', autoplay: false, onend: () => {}, props: p }) as unknown as Props;

test('tokenTexts: runtime lists name only a layer just as long; props.tokens may be bilingual', () => {
	const p = props({}, { tokens: ['a', 'b', 'c'] });
	assert.deepEqual(tokenTexts(p, 3), ['a', 'b', 'c']);
	assert.deepEqual(tokenTexts(p, 2, [7, 8]), ['#7', '#8'], 'a shorter layer (another sequence) is not named by the input tokens');
	assert.deepEqual(tokenTexts(props({}, null, { rowNames: ['[CLS]', 'p0'] }), 3), ['0', '1', '2'], "TinyViT's rowNames do not name a 3-row layer");
	const names = props({}, { tokens: ['#1', '#2'], names: [{ en: 'Alien', zh: '异形' }, { en: 'Heat', zh: '盗火线' }] });
	assert.deepEqual(tokenTexts(names, 2, null, 'zh'), ['异形', '盗火线'], 'pre.names win over tokens, in the page language');
	assert.deepEqual(tokenTexts(props({ tokens: [{ en: 'one', zh: '一' }, 'two', 'three'] }), 2, null, 'zh'), ['一', 'two'], 'props.tokens: at least T, sliced');
	assert.equal(nameText({ text: 'x' }), 'x');
});

test('keyTexts: a cross-attention layer names its keys from props.keys (the other sequence)', () => {
	const p = props({ tokens: ['<s>', 'd', 'a'], keys: ['1', '2', ' ', 'M', 'a', 'r'] });
	assert.deepEqual(keyTexts(p, 6), ['1', '2', '␣', 'M', 'a', 'r']);
	assert.deepEqual(tokenTexts(p, 3), ['<s>', 'd', 'a']);
	assert.deepEqual(keyTexts(props({ tokens: ['x', 'y'] }), 2), ['x', 'y'], 'self-attention: the tokens');
});

test('unitWhere: axis words and item names from props (FNO mode k, real/imag), cross-attention keys', () => {
	const names = axisNames({ props: { axisWords: { k: { en: 'mode', zh: '模式' } }, axisItems: { xy: [{ en: 'real part', zh: '实部' }, { en: 'imaginary part', zh: '虚部' }] } } });
	const w = unitWhere([1, 2, 0], ['k', 'c', 'xy'], undefined, undefined, names);
	assert.equal(w.en, 'mode 1, channel 2, real part (counting from 0)');
	assert.equal(w.zh, '模式 1，通道 2，实部（从 0 数起）');
	assert.equal(unitWhere([3, 1], ['n', 'xy'], undefined, undefined, axisNames({ props: { axisItems: { xy: ['x', 'y'] } } })).en, 'node 3, y (counting from 0)', 'xy named by the step (egnn: coordinates)');
	assert.equal(unitWhere([3, 1], ['n', 'xy']).en, 'node 3, index 1 (counting from 0)', 'xy without names: an index (it means different things)');
	const k = unitWhere([0, 2, 4], ['h', 'tq', 'tk'], ['<s>', 'a', 'b'], undefined, axisNames({ props: { keys: ['1', '2', '-', '0', '5'] } }));
	assert.equal(k.en, 'head 0, query 2 “b”, key 4 “5” (counting from 0)');
	assert.equal(unitWhere([2], ['k']).en, 'index 2 (counting from 0)', 'unchanged without names');
});

test('gridRows: the most square grid that holds exactly n cells (no NaN cell in zoom.reduce)', () => {
	assert.deepEqual([1, 3, 4, 6, 7, 8, 9, 12, 72].map(gridRows), [1, 1, 2, 2, 1, 2, 3, 3, 8]);
});

// ------------------------------------------------------------------ narration
test('gap marks: only a standalone "…" is an unresolved value; a literal range or word is text (TransE, behavior-cloning)', () => {
	assert.equal(gapMarks('the loss is ….'), 1);
	assert.equal(gapMarks('token “…” here'), 1);
	assert.equal(gapMarks('是…的首都'), 0);
	assert.equal(gapMarks('scaled to −1…1'), 0);
	assert.equal(gapMarks('Training…'), 0);
	assert.equal(extraEllipses('Canada 是…的首都', 'Canada {rel}'), 0, 'a vars() value with a literal ellipsis is not a gap');
	assert.equal(extraEllipses('the loss is …', 'the loss is {x}'), 1);
});

test('formats pct2 and sci: 0.99983 is 99.98%, not 100.0%; 1.71e−4 reads 1.71 × 10⁻⁴', () => {
	assert.ok((FORMATS as readonly string[]).includes('pct2') && (FORMATS as readonly string[]).includes('sci'));
	assert.equal(formatValue(0.99983, 'pct2', 'en'), '99.98%');
	assert.equal(formatValue(0.000171, 'sci', 'en'), '1.71 × 10⁻⁴');
	assert.equal(formatValue(-123456, 'sci', 'zh'), '−1.23 × 10⁵');
	assert.equal(fmtSci(0.5), '0.500', 'ordinary sizes print as fmt3');
	assert.equal(fmtValue(0.000171, 'sci'), '1.71 × 10⁻⁴');
});

test('predictAt: compiled through; without predict it is a compile error', () => {
	const base: StepDef = { id: 's', title: { en: 'T', zh: 'T' }, narration: { en: 'Which one?', zh: '哪一个？' }, view: 'tape.table', provenance: 'live' };
	const withQ: StepDef = {
		...base,
		predict: { id: 'q', question: { en: 'Which?', zh: '哪个？' }, options: [{ en: 'a', zh: '甲' }, { en: 'b', zh: '乙' }], answer: 0, explain: { en: 'a', zh: '甲' } },
		predictAt: () => ({ answer: 1, explain: { en: 'b, live', zh: '乙，实时' } })
	};
	const c = compileStep(withQ, { origin: 'author', defaultSource: 'fwd' });
	assert.equal(c.predictAt?.({} as never, {} as At).answer, 1);
});

// ------------------------------------------------------------------ autograd: clamp, minimum, maximum
const g1 = (f: (x: Var) => Var, xs: number[]) => {
	const x = param('x', { shape: [xs.length], data: xs });
	const [gx] = grad(sum(f(x)), [x]);
	return Array.from(gx!.data);
};

test('clamp passes the gradient on the closed range (as torch.clamp); minimum/maximum split a tie evenly', () => {
	assert.deepEqual(Array.from(clamp(input({ shape: [4], data: [-2, -1, 0.5, 3] }), -1, 1).data), [-1, -1, 0.5, 1]);
	assert.deepEqual(g1((x) => clamp(x, -1, 1), [-2, -1, 0.5, 3]), [0, 1, 1, 0]);
	assert.deepEqual(g1((x) => clamp(x, null, 0), [-1, 2]), [1, 0]);
	const b = input({ shape: [3], data: [1, 1, 1] });
	assert.deepEqual(g1((x) => minimum(x, b), [0, 1, 2]), [1, 0.5, 0]);
	assert.deepEqual(g1((x) => maximum(x, b), [0, 1, 2]), [0, 0.5, 1]);
	// broadcasting: a scalar side gets the sum of its shares
	const s = param('s', { shape: [1], data: [1] });
	const [gs] = grad(sum(minimum(input({ shape: [3], data: [0, 1, 2] }), s)), [s]);
	assert.deepEqual(Array.from(gs!.data), [1.5]);
	assert.deepEqual(Array.from(minimum(2, input({ shape: [2], data: [1, 3] })).data), [1, 2]);
	assert.ok(Number.isNaN(minimum(input({ shape: [1], data: [NaN] }), 1).data[0]));
	void mul;
});

// ------------------------------------------------------------------ TrainerDef.project and lastParams
type M = { w: Var; params(): Var[]; named(): Record<string, Var> };
const proj: TrainerDef<null, M, null, TrainOpt> = {
	id: 'unit',
	title: { en: 'Unit length', zh: '单位长度' },
	defaults: { seed: 1, lr: 0.1, batch: 1, steps: 3, snapshotEvery: 1, optimizer: 'sgd' },
	model: () => {
		const w = param('w', { shape: [2], data: [3, 4] });
		return { w, params: () => [w], named: () => ({ w }) };
	},
	batch: () => null,
	loss: (m) => ({ loss: sum(mul(m.w, input({ shape: [2], data: [1, -1] }))) }),
	project(m) {
		const n = Math.hypot(m.w.data[0], m.w.data[1]);
		m.w.data[0] /= n;
		m.w.data[1] /= n;
	}
};

test('TrainerDef.project runs after every step; fine update frames say so and updateOf refuses the SGD rule', () => {
	const coarse = runProcess(trainingProcess(proj), null, { seed: 1 }) as Frame<TrainState>[];
	const last = coarse[coarse.length - 1].state.params!.w.data;
	assert.ok(Math.abs(Math.hypot(last[0], last[1]) - 1) < 1e-12, 'unit length after the steps');
	const fine = runProcess(trainingProcess(proj), null, { seed: 1, detail: 'fine', opt: { steps: 1 } }) as Frame<TrainState>[];
	const upd = fine.find((f) => f.phase === 'update')!;
	assert.equal(upd.metrics!.projected, 1);
	assert.throws(() => updateOf({ frame: upd, state: upd.state, index: 0 } as unknown as At, 'w', 0), /projects/);
	const plain = runProcess(trainingProcess({ ...proj, project: undefined }), null, { seed: 1, detail: 'fine', opt: { steps: 1 } }) as Frame<TrainState>[];
	assert.equal(plain.find((f) => f.phase === 'update')!.metrics!.projected, undefined);
});

test('lastParams: the last frame of a train source (score-2d and realnvp-2d each wrote it)', () => {
	const frames = runProcess(trainingProcess(proj), null, { seed: 1 }) as Frame<TrainState>[];
	const tape = { length: frames.length, state: (i: number) => frames[i].state };
	const p = lastParams({ sources: { train: { tape } } }, 'train') as Record<string, Tensor>;
	assert.deepEqual(Array.from(p.w.data), Array.from(frames[frames.length - 1].state.params!.w.data));
	assert.equal(lastParams({ sources: {} }, 'train'), null);
	assert.equal(lastParams({ sources: { train: { tape: { length: 0, state: () => ({}) } } } }, 'train'), null);
});

// ------------------------------------------------------------------ the repo root outside the CLI
test('findRepoRoot: ATLAS_ROOT, else a lane.json root (a lane on another drive), else the walk up', () => {
	const base = mkdtempSync(join(tmpdir(), 'atlas-k4-'));
	try {
		const repo = join(base, 'repo');
		mkdirSync(join(repo, 'pipeline'), { recursive: true });
		writeFileSync(join(repo, 'CONTRACTS.md'), '#\n');
		const lane = join(base, 'lanes', 'x', 'site');
		mkdirSync(lane, { recursive: true });
		assert.equal(findRepoRoot(lane, {}), null);
		writeFileSync(join(base, 'lanes', 'x', 'lane.json'), JSON.stringify({ format: 'atlas-lane/1', name: 'x', root: repo }));
		assert.equal(findRepoRoot(lane, {}), repo);
		assert.equal(findRepoRoot(lane, { ATLAS_ROOT: base }), base, 'the environment wins');
		assert.equal(findRepoRoot(join(repo, 'site'), {}), repo);
	} finally {
		rmSync(base, { recursive: true, force: true });
	}
});
